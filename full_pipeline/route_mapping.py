"""
route_mapping.py
Chinese Postman Routing, Turn-Penalised Eulerian Circuit Generation,
and Pre-Fourier Path Descriptor Extraction (Sections 2.2.2 - 2.3).
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# GEOMETRY & TANGENT FUNCTIONS
# ==========================================================

def compute_polyline_length(pts):
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))


def _initial_tangent(pts):
    """Unit tangent vector in the direction of travel at the start of an edge polyline."""
    pts = np.asarray(pts, dtype=float)
    for i in range(1, len(pts)):
        vec = pts[i] - pts[0]
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            return vec / norm
    return np.array([1.0, 0.0])


def _terminal_tangent(pts):
    """Unit tangent vector in the direction of travel at the end of an edge polyline."""
    pts = np.asarray(pts, dtype=float)
    for i in range(len(pts) - 2, -1, -1):
        vec = pts[-1] - pts[i]
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            return vec / norm
    return np.array([1.0, 0.0])


def orient_edge_pts(E, u, v, pts, prev_tangent=None):
    """
    Returns `pts` oriented from node u towards node v.
    For self-loops (u == v), if `prev_tangent` is provided, chooses the loop
    orientation that minimises the entry turning angle.
    """
    pts = np.asarray(pts, dtype=float)
    if u == v:
        if prev_tangent is not None and len(pts) >= 2:
            tan_fwd = _initial_tangent(pts)
            tan_rev = _initial_tangent(pts[::-1])
            if angle_between(prev_tangent, tan_rev) < angle_between(prev_tangent, tan_fwd):
                return pts[::-1].copy()
        return pts.copy()

    pos_u = np.asarray(E.nodes[u]["pos"], dtype=float)
    pos_v = np.asarray(E.nodes[v]["pos"], dtype=float)
    d_fwd = np.linalg.norm(pts[0] - pos_u) + np.linalg.norm(pts[-1] - pos_v)
    d_rev = np.linalg.norm(pts[-1] - pos_u) + np.linalg.norm(pts[0] - pos_v)
    return pts[::-1].copy() if d_rev < d_fwd else pts.copy()


def edge_direction(pts):
    """Overall chord direction from start to end of `pts`."""
    vec = pts[-1] - pts[0]
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return _initial_tangent(pts)
    return vec / norm


def angle_between(v1, v2):
    """ Unsigned deflection angle in radians [0, pi] between vectors v1 and v2. """
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    dot = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
    return float(np.arccos(dot))


# ==========================================================
# ROUTE QUALITY & ANGULAR DESCRIPTOR ANALYSIS
# ==========================================================

def compute_total_turn_angle(E, route):
    """
    Computes cumulative exterior turning angle (in degrees) across all consecutive
    edge transitions in the Eulerian circuit using true terminal and initial tangents.
    """
    if len(route) < 2:
        return 0.0

    total_angle = 0.0
    prev_tan = None

    # Orient first edge
    u0, v0, k0 = route[0]
    pts0 = orient_edge_pts(E, u0, v0, E[u0][v0][k0]["pts"])
    prev_tan = _terminal_tangent(pts0)

    for i in range(1, len(route)):
        u, v, k = route[i]
        pts = orient_edge_pts(E, u, v, E[u][v][k]["pts"], prev_tangent=prev_tan)
        init_tan = _initial_tangent(pts)
        total_angle += np.degrees(angle_between(prev_tan, init_tan))
        prev_tan = _terminal_tangent(pts)

    return float(total_angle)


def compute_polyline_turn_angles(polyline):
    """
    Computes exterior deflection angles Delta_theta_i in radians [0, pi]
    at every interior vertex of the stitched RDP polyline.
    Ignores zero-length steps.
    """
    pts = np.asarray(polyline, dtype=float)
    if len(pts) < 3:
        return np.zeros(1, dtype=float)

    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    valid_vecs = diffs[seg_lens > 1e-9]
    valid_lens = seg_lens[seg_lens > 1e-9]

    if len(valid_vecs) < 2:
        return np.zeros(1, dtype=float)

    unit_vecs = valid_vecs / valid_lens[:, None]
    dots = np.clip(np.sum(unit_vecs[:-1] * unit_vecs[1:], axis=1), -1.0, 1.0)
    return np.arccos(dots)  # Exterior deflection angle: 0 = straight, pi = 180-deg U-turn


def compute_route_descriptors(E, route, polyline, canvas_diag=None):
    """
    Computes scale-invariant geometric, topological, and angular descriptors of
    the Eulerian route prior to Fourier reconstruction:
      - doubled_path_prop: L_duplicated / L_total (proportion of path doubled up)
      - mst_bridge_len_prop: L_bridges / L_total (proportion of path on MST bridges)
      - tortuosity_len_to_diag: L_total / D (normalised path length)
      - rms_vertex_turn_norm: RMS exterior turning angle across RDP vertices normalised to [0, 1]
      - mean_junction_turn_norm: Mean turning angle at route junctions normalised to [0, 1]
      - acute_turn_fraction: Fraction of polyline vertex turns >= 60 degrees
    """
    total_len = 0.0
    duplicated_len = 0.0
    bridge_len = 0.0
    original_stroke_len = 0.0

    for u, v, k in route:
        edge_data = E[u][v][k]
        seg_len = compute_polyline_length(edge_data["pts"])
        total_len += seg_len

        if edge_data.get("duplicated", False):
            duplicated_len += seg_len
        elif edge_data.get("bridge", False):
            bridge_len += seg_len
        else:
            original_stroke_len += seg_len

    safe_total_len = max(total_len, 1e-9)
    doubled_path_prop = duplicated_len / safe_total_len
    mst_bridge_len_prop = bridge_len / safe_total_len

    # Estimate canvas diagonal from polyline bounding box if not passed explicitly
    if canvas_diag is None or canvas_diag <= 0:
        span_x = float(np.ptp(polyline[:, 0])) if len(polyline) > 0 else 1.0
        span_y = float(np.ptp(polyline[:, 1])) if len(polyline) > 0 else 1.0
        canvas_diag = max(float(np.hypot(span_x, span_y)), 1e-6)

    tortuosity_len_to_diag = total_len / max(float(canvas_diag), 1e-6)

    # Angular metrics across both route junctions and all stitched RDP vertices
    total_junction_turn_deg = compute_total_turn_angle(E, route)
    mean_junction_turn_deg = total_junction_turn_deg / max(1, len(route))
    mean_junction_turn_norm = mean_junction_turn_deg / 180.0

    vertex_turns_rad = compute_polyline_turn_angles(polyline)
    rms_vertex_turn_rad = float(np.sqrt(np.mean(vertex_turns_rad ** 2)))
    rms_vertex_turn_deg = float(np.degrees(rms_vertex_turn_rad))
    rms_vertex_turn_norm = rms_vertex_turn_rad / np.pi  # In [0, 1]
    acute_turn_fraction = float(np.mean(np.degrees(vertex_turns_rad) >= 60.0))

    return {
        "total_polyline_len": total_len,
        "original_stroke_len": original_stroke_len,
        "duplicated_len": duplicated_len,
        "mst_bridge_len": bridge_len,
        "doubled_path_prop": doubled_path_prop,
        "mst_bridge_len_prop": mst_bridge_len_prop,
        "tortuosity_len_to_diag": tortuosity_len_to_diag,
        "total_turn_angle_deg": total_junction_turn_deg,
        "mean_junction_turn_deg": mean_junction_turn_deg,
        "mean_junction_turn_norm": mean_junction_turn_norm,
        "rms_vertex_turn_deg": rms_vertex_turn_deg,
        "rms_vertex_turn_norm": rms_vertex_turn_norm,
        "acute_turn_fraction": acute_turn_fraction,
    }


# ==========================================================
# CPP FUNCTIONS
# ==========================================================

def build_cpp_graph(final_graph):
    G = nx.MultiGraph()
    for n, attrs in final_graph.nodes(data=True):
        G.add_node(n, **attrs)

    for u, v, k, data in final_graph.edges(keys=True, data=True):
        attrs = dict(data)
        attrs["weight"] = max(compute_polyline_length(data["pts"]), 1e-6)
        G.add_edge(u, v, key=k, **attrs)

    return G


def find_odd_nodes(G):
    return [n for n in G.nodes if G.degree(n) % 2 == 1]


def compute_odd_node_shortest_paths(G, odd_nodes):
    shortest = {}
    for i, u in enumerate(odd_nodes):
        lengths, paths = nx.single_source_dijkstra(G, source=u, weight="weight")
        for v in odd_nodes[i + 1 :]:
            if v in lengths:
                shortest[(u, v)] = {"distance": lengths[v], "path": paths[v]}
    return shortest


def minimum_weight_matching(odd_nodes, shortest):
    matching_graph = nx.Graph()
    for (u, v), info in shortest.items():
        matching_graph.add_edge(u, v, weight=-info["distance"])
    return nx.max_weight_matching(matching_graph, maxcardinality=True)


def eulerise_graph(G, matching, shortest):
    E = G.copy()
    for u, v in matching:
        lookup = (u, v) if (u, v) in shortest else (v, u)
        path = shortest[lookup]["path"]

        for a, b in zip(path[:-1], path[1:]):
            edge_data = min(E.get_edge_data(a, b).values(), key=lambda d: d["weight"])
            duplicated_attrs = dict(edge_data)
            duplicated_attrs["duplicated"] = True
            E.add_edge(a, b, **duplicated_attrs)

    return E


# ==========================================================
# TURN-PENALISED EULER ROUTE & POLYLINE BUILDING
# ==========================================================

def build_euler_route(E, penalise_turns=True):
    """
    Extracts an Eulerian circuit from multigraph E.
    When penalise_turns=True, executes a Turn-Minimising Hierholzer algorithm
    that greedily chooses the unvisited outgoing edge with minimum angular
    deflection at every junction and penalises immediate 180-deg U-turns back
    onto parallel duplicated edges whenever another branch is available.
    """
    if not penalise_turns or E.number_of_edges() == 0:
        return list(nx.eulerian_circuit(E, keys=True))

    unused_adj = {n: [] for n in E.nodes()}
    unused_set = set()
    for u, v, k in E.edges(keys=True):
        eid = (min(u, v), max(u, v), k)
        unused_set.add(eid)
        unused_adj[u].append((v, k, eid))
        if u != v:
            unused_adj[v].append((u, k, eid))

    start_node = next(iter(E.nodes()))
    # Stack stores (current_node, incoming_edge_triple, incoming_terminal_tangent)
    stack = [(start_node, None, None)]
    circuit = []

    while stack:
        u, in_edge, in_tan = stack[-1]
        candidates = [item for item in unused_adj[u] if item[2] in unused_set]

        if not candidates:
            if in_edge is not None:
                circuit.append(in_edge)
            stack.pop()
        else:
            if in_tan is None or len(candidates) == 1:
                best_v, best_k, best_eid = candidates[0]
                pts_out = orient_edge_pts(
                    E, u, best_v, E[u][best_v][best_k]["pts"], prev_tangent=in_tan
                )
            else:
                best_cost = float("inf")
                best_v, best_k, best_eid, pts_out = None, None, None, None
                prev_u = in_edge[0]

                for cand_v, cand_k, cand_eid in candidates:
                    c_pts = orient_edge_pts(
                        E, u, cand_v, E[u][cand_v][cand_k]["pts"], prev_tangent=in_tan
                    )
                    out_tan = _initial_tangent(c_pts)
                    turn_rad = angle_between(in_tan, out_tan)  # in [0, pi]

                    # Penalise immediate 180-degree U-turns onto parallel edges
                    if cand_v == prev_u and np.degrees(turn_rad) > 150.0:
                        turn_rad += np.pi

                    if turn_rad < best_cost:
                        best_cost = turn_rad
                        best_v, best_k, best_eid = cand_v, cand_k, cand_eid
                        pts_out = c_pts

            unused_set.remove(best_eid)
            next_in_tan = _terminal_tangent(pts_out)
            stack.append((best_v, (u, best_v, best_k), next_in_tan))

    return circuit[::-1]


def route_to_polyline(E, route):
    drawing_pts = []
    prev_tan = None

    for u, v, k in route:
        pts = orient_edge_pts(E, u, v, E[u][v][k]["pts"], prev_tangent=prev_tan)

        if len(drawing_pts) == 0:
            drawing_pts.extend(pts)
            prev_tan = _terminal_tangent(pts)
            continue

        # Ensure spatial continuity for self-loops (u == v)
        if np.linalg.norm(drawing_pts[-1] - pts[-1]) < np.linalg.norm(drawing_pts[-1] - pts[0]):
            pts = pts[::-1]

        drawing_pts.extend(pts[1:])
        prev_tan = _terminal_tangent(pts)

    return np.array(drawing_pts, dtype=float)


def chinese_postman(final_graph, penalise_turns=True, canvas_diag=None):
    print("Building weighted graph...")
    G = build_cpp_graph(final_graph)
    odd_nodes = find_odd_nodes(G)
    print(f"Odd vertices found: {len(odd_nodes)}")

    if len(odd_nodes) == 0:
        E = G
    else:
        print("Finding shortest paths between odd-degree vertices...")
        shortest = compute_odd_node_shortest_paths(G, odd_nodes)
        print("Solving minimum-weight perfect matching...")
        matching = minimum_weight_matching(odd_nodes, shortest)
        print(f"Matched pairs: {len(matching)}")
        print("Eulerising graph...")
        E = eulerise_graph(G, matching, shortest)

    print(f"Computing Euler circuit (penalise_turns={penalise_turns})...")
    route = build_euler_route(E, penalise_turns=penalise_turns)
    polyline = route_to_polyline(E, route)
    descriptors = compute_route_descriptors(E, route, polyline, canvas_diag=canvas_diag)

    print(
        f"Route summary: Doubled-up path = {descriptors['doubled_path_prop']*100:.1f}% | "
        f"MST bridge path = {descriptors['mst_bridge_len_prop']*100:.1f}% | "
        f"RMS vertex turn = {descriptors['rms_vertex_turn_deg']:.1f} deg"
    )

    return {
        "euler_graph": E,
        "route": route,
        "polyline": polyline,
        "descriptors": descriptors,
    }


# ==========================================================
# SAVING PLOTS TO DISK
# ==========================================================

def plot_original_graph(G, save_path="original_graph.png"):
    plt.figure(figsize=(8, 7))
    for _, _, data in G.edges(data=True):
        pts = data["pts"]
        plt.plot(pts[:, 0], pts[:, 1], "k-", linewidth=1.8)

    for n, attrs in G.nodes(data=True):
        x, y = attrs["pos"]
        plt.scatter(x, y, color="royalblue", s=30, zorder=5)

    plt.title(f"Extracted Base Graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved original graph plot to '{save_path}'")


def plot_eulerised_graph(E, save_path="eulerised_graph.png"):
    plt.figure(figsize=(8, 7))
    for _, _, _, data in E.edges(keys=True, data=True):
        pts = data["pts"]
        if data.get("duplicated", False):
            plt.plot(pts[:, 0], pts[:, 1], color="crimson", linewidth=3.0, alpha=0.85, label="CPP Duplicated")
        elif data.get("bridge", False):
            plt.plot(pts[:, 0], pts[:, 1], "c--", linewidth=1.8, label="MST Bridge")
        else:
            plt.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.5)

    for n, attrs in E.nodes(data=True):
        x, y = attrs["pos"]
        plt.scatter(x, y, color="royalblue", s=20, zorder=5)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.title("Eulerised Graph (Red = CPP Duplicated Edges)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Eulerised graph plot to '{save_path}'")


def plot_route(E, route, save_path="euler_route.png"):
    plt.figure(figsize=(9, 8))
    cmap = plt.cm.viridis
    N = len(route)
    prev_tan = None

    for i, (u, v, k) in enumerate(route):
        pts = orient_edge_pts(E, u, v, E[u][v][k]["pts"], prev_tangent=prev_tan)
        prev_tan = _terminal_tangent(pts)
        colour = cmap(i / max(N, 1))
        plt.plot(pts[:, 0], pts[:, 1], color=colour, linewidth=2.2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=N))
    plt.colorbar(sm, ax=plt.gca(), fraction=0.03, pad=0.04, label="Step Index in Euler Tour")

    plt.title("Continuous Single-Line Euler Traversal Order")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Euler route plot to '{save_path}'")


# ==========================================================
# STANDALONE TEST (ONLY RUNS WHEN EXECUTED DIRECTLY)
# ==========================================================

if __name__ == "__main__":
    test_G = nx.MultiGraph()
    positions = {
        0: (0, 0), 1: (2, 0), 2: (4, 0), 3: (6, 0),
        4: (1, 2), 5: (3, 2), 6: (5, 2), 7: (7, 2),
        8: (0, 4), 9: (2, 4), 10: (4, 4), 11: (6, 4),
    }
    for n, pos in positions.items():
        test_G.add_node(n, pos=np.array(pos, dtype=float))

    for a, b in [
        (0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7),
        (8, 9), (9, 10), (10, 11), (0, 4), (1, 5), (2, 6),
        (3, 7), (4, 8), (5, 9), (6, 10), (7, 11),
    ]:
        test_G.add_edge(a, b, pts=np.array([positions[a], positions[b]], dtype=float))

    res = chinese_postman(test_G, penalise_turns=True)
    print("Extracted Descriptors:", res["descriptors"])
    plot_original_graph(test_G, "test_original.png")
    plot_eulerised_graph(res["euler_graph"], "test_eulerised.png")
    plot_route(res["euler_graph"], res["route"], "test_route.png")