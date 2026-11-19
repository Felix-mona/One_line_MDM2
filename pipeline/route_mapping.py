"""
route_mapping.py

Step 3: Route Pathfinding via Chinese Postman Problem (CPP) and Minimum-Angle
Fleury Eulerian Circuit Extraction (Report Section 2.3).
Pairs odd-degree vertices via all-pairs Dijkstra shortest paths and minimum-weight
perfect matching, then extracts a single continuous Eulerian circuit that greedily
minimises angular deflection at every vertex while avoiding bridges.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ALPHA is used only to report an optional angle-weighted cost after routing
ALPHA: float = 0.0


# ==============================================================================
# 1. GEOMETRY & DEFLECTION HELPERS
# ==============================================================================
def compute_polyline_length(pts: np.ndarray) -> float:
    """Computes total Euclidean arc length of a 2D polyline."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))


def edge_direction(pts: np.ndarray) -> np.ndarray:
    """Computes the unit chord direction vector from the start to the end of `pts`."""
    vec = pts[-1] - pts[0]
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.array([1.0, 0.0])
    return vec / norm


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes the unsigned deflection angle in radians [0, pi] between `v1` and `v2`."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    dot = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
    return float(np.arccos(dot))


def traversal_direction(pts: np.ndarray, current_pos: np.ndarray) -> np.ndarray:
    """Orients `pts` away from `current_pos` and returns its unit chord direction."""
    d1 = np.linalg.norm(pts[0] - current_pos)
    d2 = np.linalg.norm(pts[-1] - current_pos)
    if d2 < d1:
        pts = pts[::-1]
    return edge_direction(pts)


# ==============================================================================
# 2. ROUTE METRICS & PRE-FOURIER DESCRIPTORS
# ==============================================================================
def compute_angle_weighted_cost(
    distance: float,
    total_angle_deg: float,
    alpha: float = ALPHA,
) -> float:
    """Computes angle-weighted route cost W = L * (1 + alpha * sin(theta / 2))."""
    theta = np.radians(total_angle_deg)
    return float(distance * (1.0 + alpha * np.sin(theta / 2.0)))


def compute_total_turn_angle(euler_graph: nx.MultiGraph, route: list[tuple[int, int, int]]) -> float:
    """Computes cumulative exterior turning angle in degrees across `route`."""
    total_angle = 0.0
    previous_dir = None

    for u, v, k in route:
        pts = euler_graph[u][v][k]["pts"]
        direction = traversal_direction(pts, euler_graph.nodes[u]["pos"])
        if previous_dir is not None:
            total_angle += float(np.degrees(angle_between(previous_dir, direction)))
        previous_dir = direction

    return float(total_angle)


def compute_polyline_turn_angles(polyline: np.ndarray) -> np.ndarray:
    """Computes exterior deflection angles in radians [0, pi] at each polyline vertex."""
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
    return np.arccos(dots)


def compute_route_descriptors(
    euler_graph: nx.MultiGraph,
    route: list[tuple[int, int, int]],
    polyline: np.ndarray,
    canvas_diag: float | None = None,
) -> dict[str, float]:
    """
    Computes scale-invariant geometric, topological, and angular descriptors of
    the Eulerian route prior to Fourier reconstruction.
    """
    total_len = 0.0
    duplicated_len = 0.0
    bridge_len = 0.0
    original_stroke_len = 0.0

    for u, v, k in route:
        edge_data = euler_graph[u][v][k]
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

    if canvas_diag is None or canvas_diag <= 0:
        span_x = float(np.ptp(polyline[:, 0])) if len(polyline) > 0 else 1.0
        span_y = float(np.ptp(polyline[:, 1])) if len(polyline) > 0 else 1.0
        canvas_diag = max(float(np.hypot(span_x, span_y)), 1e-6)

    tortuosity_len_to_diag = total_len / max(float(canvas_diag), 1e-6)

    total_junction_turn_deg = compute_total_turn_angle(euler_graph, route)
    mean_junction_turn_deg = total_junction_turn_deg / max(1, len(route))
    mean_junction_turn_norm = mean_junction_turn_deg / 180.0

    vertex_turns_rad = compute_polyline_turn_angles(polyline)
    rms_vertex_turn_rad = float(np.sqrt(np.mean(vertex_turns_rad ** 2)))
    rms_vertex_turn_deg = float(np.degrees(rms_vertex_turn_rad))
    rms_vertex_turn_norm = rms_vertex_turn_rad / np.pi
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


# ==============================================================================
# 3. CHINESE POSTMAN PROBLEM (SECTION 2.3.1)
# ==============================================================================
def build_cpp_graph(bridged_graph: nx.MultiGraph) -> nx.MultiGraph:
    """Assigns Euclidean arc-length weights to every edge of `bridged_graph`."""
    weighted = nx.MultiGraph()
    for n, attrs in bridged_graph.nodes(data=True):
        weighted.add_node(n, **attrs)

    for u, v, k, data in bridged_graph.edges(keys=True, data=True):
        attrs = dict(data)
        attrs["weight"] = compute_polyline_length(data["pts"])
        weighted.add_edge(u, v, key=k, **attrs)

    return weighted


def find_odd_nodes(graph: nx.MultiGraph) -> list[int]:
    """Returns all vertices in `graph` with odd degree."""
    return [n for n in graph.nodes if graph.degree(n) % 2 == 1]


def compute_odd_node_shortest_paths(
    graph: nx.MultiGraph,
    odd_nodes: list[int],
) -> dict[tuple[int, int], dict]:
    """Computes all-pairs Dijkstra shortest paths and distances between `odd_nodes`."""
    shortest: dict[tuple[int, int], dict] = {}
    for i, u in enumerate(odd_nodes):
        for v in odd_nodes[i + 1 :]:
            path = nx.dijkstra_path(graph, u, v, weight="weight")
            distance = nx.dijkstra_path_length(graph, u, v, weight="weight")
            shortest[(u, v)] = {"distance": float(distance), "path": path}
    return shortest


def minimum_weight_matching(
    odd_nodes: list[int],
    shortest: dict[tuple[int, int], dict],
) -> set[tuple[int, int]]:
    """Solves minimum-weight perfect matching over `odd_nodes` using negative distances."""
    matching_graph = nx.Graph()
    for (u, v), info in shortest.items():
        matching_graph.add_edge(u, v, weight=-info["distance"])
    return nx.max_weight_matching(matching_graph, maxcardinality=True)


def eulerise_graph(
    graph: nx.MultiGraph,
    matching: set[tuple[int, int]],
    shortest: dict[tuple[int, int], dict],
) -> nx.MultiGraph:
    """Duplicates shortest-path edges between matched odd vertices so all degrees become even."""
    euler_graph = graph.copy()
    for u, v in matching:
        lookup = (u, v) if (u, v) in shortest else (v, u)
        path = shortest[lookup]["path"]
        for a, b in zip(path[:-1], path[1:]):
            edge_data = min(
                euler_graph.get_edge_data(a, b).values(),
                key=lambda d: d["weight"],
            )
            duplicated_attrs = dict(edge_data)
            duplicated_attrs["duplicated"] = True
            euler_graph.add_edge(a, b, **duplicated_attrs)
    return euler_graph


# ==============================================================================
# 4. FLEURY + MINIMUM-ANGLE EULERIAN CIRCUIT & POLYLINE STITCHING (SECTION 2.3.2)
# ==============================================================================
def build_euler_route(
    euler_graph: nx.MultiGraph,
    start: int | None = None,
) -> list[tuple[int, int, int]]:
    """
    Extracts an Eulerian circuit traversing every edge of `euler_graph` once.
    At each vertex, selects the untraversed outgoing edge with the smallest
    exterior turning angle relative to the incoming direction of travel while
    using Fleury's bridge check so remaining edges are never disconnected.

    Args:
        euler_graph: Connected Eulerian multigraph (all vertex degrees even).
        start: Optional starting vertex ID (defaults to `min(euler_graph.nodes)`).

    Returns:
        Ordered list of traversed edge tuples `(u, v, key)`.
    """
    remaining = euler_graph.copy()
    if start is None:
        start = min(euler_graph.nodes)

    current = start
    previous_dir = None
    route: list[tuple[int, int, int]] = []

    def is_bridge(u: int, v: int, k: int) -> bool:
        if remaining.degree(u) == 1:
            return False
        data = dict(remaining[u][v][k])
        remaining.remove_edge(u, v, key=k)
        connected = nx.has_path(remaining, u, v)
        remaining.add_edge(u, v, key=k, **data)
        return not connected

    while remaining.number_of_edges() > 0:
        candidates = []
        for _, nbr, k in remaining.edges(current, keys=True):
            pts = remaining[current][nbr][k]["pts"]
            direction = traversal_direction(pts, euler_graph.nodes[current]["pos"])
            deflection = 0.0 if previous_dir is None else angle_between(previous_dir, direction)
            candidates.append((deflection, nbr, k, direction))

        safe = [c for c in candidates if not is_bridge(current, c[1], c[2])]
        pool = safe if safe else candidates

        _, nbr, k, direction = min(pool, key=lambda c: c[0])
        route.append((current, nbr, k))
        remaining.remove_edge(current, nbr, key=k)
        previous_dir = direction
        current = nbr

    return route


def route_to_polyline(
    euler_graph: nx.MultiGraph,
    route: list[tuple[int, int, int]],
) -> np.ndarray:
    """Stitches oriented edge polylines along `route` into one continuous 2D trajectory."""
    drawing_pts: list[np.ndarray] = []

    for u, v, k in route:
        pts = euler_graph[u][v][k]["pts"].copy()
        if len(drawing_pts) == 0:
            if np.linalg.norm(pts[-1] - euler_graph.nodes[u]["pos"]) < np.linalg.norm(
                pts[0] - euler_graph.nodes[u]["pos"]
            ):
                pts = pts[::-1]
            drawing_pts.extend(pts)
            continue

        start_dist = np.linalg.norm(drawing_pts[-1] - pts[0])
        end_dist = np.linalg.norm(drawing_pts[-1] - pts[-1])
        if end_dist < start_dist:
            pts = pts[::-1]
        drawing_pts.extend(pts[1:])

    return np.array(drawing_pts, dtype=float)


def chinese_postman(
    bridged_graph: nx.MultiGraph,
    canvas_diag: float | None = None,
) -> dict:
    """
    Executes Step 3 (Report Section 2.3): solves the Chinese Postman Problem on
    `bridged_graph` and constructs an angle-guided Eulerian trajectory.

    Args:
        bridged_graph: Connected spatial multigraph from Step 2.
        canvas_diag: Optional image bounding diagonal D for descriptor scaling.

    Returns:
        Dictionary containing 'euler_graph', 'route', 'polyline', and 'descriptors'.
    """
    print("Building weighted graph...")
    weighted = build_cpp_graph(bridged_graph)
    odd_nodes = find_odd_nodes(weighted)
    print(f"Odd vertices found: {len(odd_nodes)}")

    if len(odd_nodes) == 0:
        euler_graph = weighted
    else:
        print("Finding shortest paths (distance only)...")
        shortest = compute_odd_node_shortest_paths(weighted, odd_nodes)
        print("Solving minimum weight matching...")
        matching = minimum_weight_matching(odd_nodes, shortest)
        print(f"Matched pairs: {len(matching)}")
        print("Eulerising graph...")
        euler_graph = eulerise_graph(weighted, matching, shortest)

    print("Computing angle-guided Euler circuit...")
    route = build_euler_route(euler_graph)
    polyline = route_to_polyline(euler_graph, route)
    descriptors = compute_route_descriptors(euler_graph, route, polyline, canvas_diag=canvas_diag)

    return {
        "euler_graph": euler_graph,
        "route": route,
        "polyline": polyline,
        "descriptors": descriptors,
    }


# ==============================================================================
# 5. UNIFORM VISUALISATION HELPERS
# ==============================================================================
def plot_original_graph(graph: nx.MultiGraph, save_path: str = "original_graph.png") -> None:
    """Saves a clean borderless plot of the extracted base multigraph."""
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    for _, _, data in graph.edges(data=True):
        pts = data["pts"]
        ax.plot(pts[:, 0], pts[:, 1], "k-", linewidth=1.8)
    for _, attrs in graph.nodes(data=True):
        x, y = attrs["pos"]
        ax.scatter(x, y, color="royalblue", s=24, zorder=5)

    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print(f"Saved original graph plot to '{save_path}'")


def plot_eulerised_graph(euler_graph: nx.MultiGraph, save_path: str = "eulerised_graph.png") -> None:
    """Saves a clean borderless plot highlighting CPP-duplicated edges in crimson."""
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    for _, _, _, data in euler_graph.edges(keys=True, data=True):
        pts = data["pts"]
        if data.get("duplicated", False):
            ax.plot(pts[:, 0], pts[:, 1], color="crimson", linewidth=2.8, alpha=0.85)
        elif data.get("bridge", False):
            ax.plot(pts[:, 0], pts[:, 1], "c--", linewidth=1.8)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.8)
    for _, attrs in euler_graph.nodes(data=True):
        x, y = attrs["pos"]
        ax.scatter(x, y, color="royalblue", s=20, zorder=5)

    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print(f"Saved Eulerised graph plot to '{save_path}'")


def plot_route(
    euler_graph: nx.MultiGraph,
    route: list[tuple[int, int, int]],
    save_path: str = "euler_route.png",
) -> None:
    """Saves a clean borderless plot colour-coded by traversal order along `route`."""
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    cmap = plt.cm.viridis
    n_steps = len(route)
    for i, (u, v, k) in enumerate(route):
        pts = euler_graph[u][v][k]["pts"].copy()
        if np.linalg.norm(pts[-1] - euler_graph.nodes[u]["pos"]) < np.linalg.norm(
            pts[0] - euler_graph.nodes[u]["pos"]
        ):
            pts = pts[::-1]
        colour = cmap(i / max(n_steps, 1))
        ax.plot(pts[:, 0], pts[:, 1], color=colour, linewidth=2.0, solid_capstyle="round")

    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print(f"Saved Euler route plot to '{save_path}'")


# ==============================================================================
# STANDALONE VERIFICATION TEST
# ==============================================================================
if __name__ == "__main__":
    test_graph = nx.MultiGraph()
    positions = {
        0: (0, 0), 1: (2, 0), 2: (4, 0), 3: (6, 0),
        4: (1, 2), 5: (3, 2), 6: (5, 2), 7: (7, 2),
        8: (0, 4), 9: (2, 4), 10: (4, 4), 11: (6, 4),
    }
    for n, pos in positions.items():
        test_graph.add_node(n, pos=np.array(pos, dtype=float))

    for a, b in [
        (0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7),
        (8, 9), (9, 10), (10, 11), (0, 4), (1, 5), (2, 6),
        (3, 7), (4, 8), (5, 9), (6, 10), (7, 11), (0, 5),
        (2, 5), (2, 7), (4, 9), (6, 9), (6, 11),
    ]:
        test_graph.add_edge(a, b, pts=np.array([positions[a], positions[b]], dtype=float))

    result = chinese_postman(test_graph)
    euler_g = result["euler_graph"]
    tour = result["route"]

    assert len(tour) == euler_g.number_of_edges(), "Some edges were missed!"
    assert len({(min(u, v), max(u, v), k) for u, v, k in tour}) == len(tour), "Duplicate edge!"
    assert nx.is_eulerian(euler_g), "Graph is not Eulerian!"

    total_dist = sum(euler_g[u][v][k]["weight"] for u, v, k in tour)
    turn_deg = compute_total_turn_angle(euler_g, tour)
    print(f"Total route distance: {total_dist:.2f} | Total turn angle: {turn_deg:.2f} deg")
    plot_original_graph(test_graph, "test_original.png")
    plot_eulerised_graph(euler_g, "test_eulerised.png")
    plot_route(euler_g, tour, "test_route.png")
