import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# GEOMETRY & CPP SOLVER
# ==========================================================

def compute_polyline_length(pts):
    """Compute total Euclidean arc length of a 2D polyline."""
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))


def edge_direction(pts):
    """Return unit chord vector from start to end of a polyline."""
    vec = pts[-1] - pts[0]
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return np.array([1.0, 0.0])
    return vec / norm


def angle_between(v1, v2):
    """Return unsigned deflection angle (radians) between two vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    dot = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
    return float(np.arccos(dot))


def traversal_direction(pts, current_pos):
    """Orient polyline away from current_pos and return unit direction."""
    if np.linalg.norm(pts[-1] - current_pos) < np.linalg.norm(pts[0] - current_pos):
        pts = pts[::-1]
    return edge_direction(pts)


def solve_chinese_postman(final_graph):
    """Solve undirected CPP via Edmonds-Johnson and extract min-turn Eulerian circuit."""
    G = nx.MultiGraph()
    for n, attrs in final_graph.nodes(data=True):
        G.add_node(n, **attrs)
    for u, v, k, data in final_graph.edges(keys=True, data=True):
        attrs = dict(data)
        attrs["weight"] = compute_polyline_length(data["pts"])
        G.add_edge(u, v, key=k, **attrs)

    odd_nodes = [n for n in G.nodes if G.degree(n) % 2 == 1]
    E = G.copy()

    if odd_nodes:
        shortest = {}
        for i, u in enumerate(odd_nodes):
            for v in odd_nodes[i + 1:]:
                shortest[(u, v)] = {
                    "distance": nx.dijkstra_path_length(G, u, v, weight="weight"),
                    "path": nx.dijkstra_path(G, u, v, weight="weight"),
                }

        matching_graph = nx.Graph()
        for (u, v), info in shortest.items():
            matching_graph.add_edge(u, v, weight=-info["distance"])

        matching = nx.max_weight_matching(matching_graph, maxcardinality=True)

        for u, v in matching:
            lookup = (u, v) if (u, v) in shortest else (v, u)
            path = shortest[lookup]["path"]
            for a, b in zip(path[:-1], path[1:]):
                edge_data = min(E.get_edge_data(a, b).values(), key=lambda d: d["weight"])
                dup_attrs = dict(edge_data)
                dup_attrs["duplicated"] = True
                E.add_edge(a, b, **dup_attrs)

    # Angle-guided Fleury circuit extraction
    R = E.copy()
    current = min(E.nodes)
    previous_dir = None
    route = []

    def is_bridge(u, v, k):
        if R.degree(u) == 1:
            return False
        data = dict(R[u][v][k])
        R.remove_edge(u, v, key=k)
        connected = nx.has_path(R, u, v)
        R.add_edge(u, v, key=k, **data)
        return not connected

    while R.number_of_edges() > 0:
        candidates = []
        for _, nbr, k in R.edges(current, keys=True):
            pts = R[current][nbr][k]["pts"]
            direction = traversal_direction(pts, E.nodes[current]["pos"])
            deflection = 0.0 if previous_dir is None else angle_between(previous_dir, direction)
            candidates.append((deflection, nbr, k, direction))

        safe = [c for c in candidates if not is_bridge(current, c[1], c[2])]
        pool = safe if safe else candidates
        _, nbr, k, direction = min(pool, key=lambda c: c[0])

        route.append((current, nbr, k))
        R.remove_edge(current, nbr, key=k)
        previous_dir = direction
        current = nbr

    return E, route


# ==========================================================
# BORDERLESS PUBLICATION PLOTTING
# ==========================================================

def _offset_segment(p0, p1, offset):
    """Shift segment (p0 -> p1) laterally by `offset` units for parallel edges."""
    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length < 1e-12 or abs(offset) < 1e-12:
        return p0, p1
    normal = np.array([-vec[1], vec[0]]) / length
    return p0 + normal * offset, p1 + normal * offset


def save_eulerised_graph(E, out_path):
    """Save clean borderless plot of the Eulerised graph (red = duplicated edges)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.4), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Group parallel edges between same unordered node pair to separate visually
    pair_groups = {}
    for u, v, k, data in E.edges(keys=True, data=True):
        pair = (min(u, v), max(u, v))
        pair_groups.setdefault(pair, []).append((u, v, k, data))

    for (a, b), edges in pair_groups.items():
        n_edges = len(edges)
        for idx, (u, v, k, data) in enumerate(edges):
            pts = data["pts"]
            offset = (idx - (n_edges - 1) / 2.0) * 0.14
            p0, p1 = _offset_segment(pts[0], pts[-1], offset)
            is_dup = data.get("duplicated", False)
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color="#d62728" if is_dup else "#111111",
                linewidth=3.4 if is_dup else 2.2,
                linestyle="--" if is_dup else "-",
                solid_capstyle="round",
                zorder=3 if is_dup else 2,
            )

    for n, attrs in E.nodes(data=True):
        x, y = attrs["pos"]
        ax.scatter(x, y, color="#1f4e78", edgecolors="white", linewidths=1.2, s=110, zorder=4)
        ax.text(
            x, y + 0.22, str(n),
            fontsize=10, fontweight="bold", color="#111111",
            ha="center", va="bottom", zorder=5,
        )

    ax.set_xlim(-0.6, 7.6)
    ax.set_ylim(-0.6, 4.7)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_eulerian_walk(E, route, out_path):
    """Save clean borderless plot of the directed minimum-angle Eulerian walk."""
    fig, ax = plt.subplots(figsize=(6.5, 4.4), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cmap = plt.cm.viridis
    total_steps = len(route)

    # Track multiplicity per unordered pair so parallel traversals do not overlap
    pair_counts = {}
    for u, v, k in route:
        pair = (min(u, v), max(u, v))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    pair_seen = {}

    for step_idx, (u, v, k) in enumerate(route):
        p_start = E.nodes[u]["pos"]
        p_end = E.nodes[v]["pos"]
        pair = (min(u, v), max(u, v))
        occ = pair_seen.get(pair, 0)
        pair_seen[pair] = occ + 1
        mult = pair_counts[pair]

        # Lateral offset (0.24) + longitudinal stagger (t_pos) for duplicated edges
        canon_dir = E.nodes[pair[1]]["pos"] - E.nodes[pair[0]]["pos"]
        canon_len = np.linalg.norm(canon_dir)
        canon_norm = np.array([-canon_dir[1], canon_dir[0]]) / canon_len
        shift = (occ - (mult - 1) / 2.0) * 0.24 * canon_norm

        q0 = p_start + shift
        q1 = p_end + shift
        colour = cmap(step_idx / max(total_steps - 1, 1))

        ax.annotate(
            "",
            xy=q1,
            xytext=q0,
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.55,head_width=0.32",
                color=colour,
                lw=2.5,
                shrinkA=5,
                shrinkB=5,
            ),
            zorder=3,
        )

        # Stagger badge position along edge when mult > 1 so badges never collide
        t_pos = 0.5 if mult == 1 else (0.40 if occ == 0 else 0.60)
        badge_pos = (1.0 - t_pos) * q0 + t_pos * q1
        ax.text(
            badge_pos[0], badge_pos[1], str(step_idx + 1),
            fontsize=7.5, fontweight="bold", color="#111111",
            ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.14", fc="white", ec=colour, lw=1.0),
            zorder=5,
        )

    for n, attrs in E.nodes(data=True):
        x, y = attrs["pos"]
        ax.scatter(x, y, color="#111111", edgecolors="white", linewidths=1.0, s=65, zorder=4)

    ax.set_xlim(-0.6, 7.6)
    ax.set_ylim(-0.6, 4.7)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("Route-mapping", exist_ok=True)

    test_graph = nx.MultiGraph()
    positions = {
        0: (0, 0), 1: (2, 0), 2: (4, 0), 3: (6, 0),
        4: (1, 2), 5: (3, 2), 6: (5, 2), 7: (7, 2),
        8: (0, 4), 9: (2, 4), 10: (4, 4), 11: (6, 4),
    }
    for n, pos in positions.items():
        test_graph.add_node(n, pos=np.array(pos, dtype=float))

    edges = [
        (0, 1), (1, 2), (2, 3),
        (4, 5), (5, 6), (6, 7),
        (8, 9), (9, 10), (10, 11),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (0, 5), (2, 5), (2, 7),
        (4, 9), (6, 9), (6, 11),
    ]
    for a, b in edges:
        test_graph.add_edge(a, b, pts=np.array([positions[a], positions[b]], dtype=float))

    euler_graph, euler_route = solve_chinese_postman(test_graph)
    save_eulerised_graph(euler_graph, "Route-mapping/Eulerised-graph.png")
    save_eulerian_walk(euler_graph, euler_route, "Route-mapping/Eulerian-walk.png")
    print("Saved Route-mapping/Eulerised-graph.png and Route-mapping/Eulerian-walk.png")