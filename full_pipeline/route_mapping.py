"""
Chinese Postman Routing & Eulerian Circuit Generation (Sections 2.2.2 - 2.2.3)
Matches odd-degree vertices via minimum-weight perfect matching, duplicates
shortest connecting paths, extracts an Eulerian circuit, and saves diagnostic plots.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# GEOMETRY FUNCTIONS
# ==========================================================

def compute_polyline_length(pts):
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))


def orient_edge_pts(E, u, v, pts):
    """Returns pts oriented from node u towards node v."""
    if u == v:
        return pts.copy()
    pos_u = np.asarray(E.nodes[u]["pos"], dtype=float)
    pos_v = np.asarray(E.nodes[v]["pos"], dtype=float)
    d_fwd = np.linalg.norm(pts[0] - pos_u) + np.linalg.norm(pts[-1] - pos_v)
    d_rev = np.linalg.norm(pts[-1] - pos_u) + np.linalg.norm(pts[0] - pos_v)
    return pts[::-1].copy() if d_rev < d_fwd else pts.copy()


def edge_direction(pts):
    vec = pts[-1] - pts[0]
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return np.array([1.0, 0.0])
    return vec / norm


def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    dot = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
    return float(np.arccos(dot))


# ==========================================================
# ROUTE QUALITY ANALYSIS
# ==========================================================

def compute_total_turn_angle(E, route):
    total_angle = 0.0
    for i in range(len(route) - 1):
        u1, v1, k1 = route[i]
        u2, v2, k2 = route[i + 1]

        pts1 = orient_edge_pts(E, u1, v1, E[u1][v1][k1]["pts"])
        pts2 = orient_edge_pts(E, u2, v2, E[u2][v2][k2]["pts"])

        dir1 = edge_direction(pts1)
        dir2 = edge_direction(pts2)
        total_angle += np.degrees(angle_between(dir1, dir2))

    return total_angle


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
# EULER ROUTE & POLYLINE BUILDING
# ==========================================================

def build_euler_route(E):
    return list(nx.eulerian_circuit(E, keys=True))


def route_to_polyline(E, route):
    drawing_pts = []

    for u, v, k in route:
        pts = orient_edge_pts(E, u, v, E[u][v][k]["pts"])

        if len(drawing_pts) == 0:
            drawing_pts.extend(pts)
            continue

        # Ensure smooth orientation for self-loops (u == v)
        if np.linalg.norm(drawing_pts[-1] - pts[-1]) < np.linalg.norm(drawing_pts[-1] - pts[0]):
            pts = pts[::-1]

        drawing_pts.extend(pts[1:])

    return np.array(drawing_pts, dtype=float)


def chinese_postman(final_graph):
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

    print("Computing Euler circuit...")
    route = build_euler_route(E)
    polyline = route_to_polyline(E, route)

    return {
        "euler_graph": E,
        "route": route,
        "polyline": polyline,
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

    for i, (u, v, k) in enumerate(route):
        pts = orient_edge_pts(E, u, v, E[u][v][k]["pts"])
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

    res = chinese_postman(test_G)
    plot_original_graph(test_G, "test_original.png")
    plot_eulerised_graph(res["euler_graph"], "test_eulerised.png")
    plot_route(res["euler_graph"], res["route"], "test_route.png")