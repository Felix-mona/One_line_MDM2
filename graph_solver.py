import copy
import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist


def parse_svg_paths(svg_path: str):
    """Parses raw polyline paths and canvas bounds from an SVG file."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    width = float(root.attrib.get("width", 0))
    height = float(root.attrib.get("height", 0))
    if "viewBox" in root.attrib:
        vb = [float(v) for v in root.attrib["viewBox"].replace(",", " ").split()]
        width, height = vb[2], vb[3]

    if width == 0 or height == 0:
        width, height = 500.0, 500.0

    namespaces = {"svg": "http://www.w3.org/2000/svg"}
    path_elements = root.findall(".//svg:path", namespaces) or root.findall(".//path")

    raw_polylines = []
    for elem in path_elements:
        d = elem.attrib.get("d", "")
        tokens = d.replace(",", " ").split()
        if not tokens:
            continue

        current_line = []
        i = 0
        while i < len(tokens):
            cmd = tokens[i]
            if cmd == "M":
                if current_line:
                    raw_polylines.append(current_line)
                    current_line = []
                current_line.append((float(tokens[i + 1]), float(tokens[i + 2])))
                i += 3
            elif cmd == "L":
                current_line.append((float(tokens[i + 1]), float(tokens[i + 2])))
                i += 3
            elif cmd == "Z":
                if current_line and current_line[0] != current_line[-1]:
                    current_line.append(current_line[0])
                raw_polylines.append(current_line)
                current_line = []
                i += 1
            else:
                i += 1

        if current_line:
            raw_polylines.append(current_line)

    return raw_polylines, width, height


def compute_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes the deflection angle between two consecutive segment vectors in degrees."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def split_polyline_by_angle(points: list, max_angle_deg: float = 60.0):
    """Splits a polyline into sub-polylines at vertices where directional deviation >= max_angle_deg."""
    if len(points) <= 2:
        return [points], set()

    sub_polylines = []
    corner_points = set()
    current_segment = [points[0]]

    for i in range(1, len(points) - 1):
        p_prev = np.array(points[i - 1], dtype=np.float64)
        p_curr = np.array(points[i], dtype=np.float64)
        p_next = np.array(points[i + 1], dtype=np.float64)

        angle = compute_angle_deg(p_curr - p_prev, p_next - p_curr)
        current_segment.append(points[i])

        if angle >= max_angle_deg:
            corner_points.add(points[i])
            sub_polylines.append(current_segment)
            current_segment = [points[i]]

    current_segment.append(points[-1])
    sub_polylines.append(current_segment)
    return sub_polylines, corner_points


def polyline_arc_length(points: list) -> float:
    """Calculates cumulative Euclidean arc length along a sequence of coordinate vertices."""
    pts = np.array(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def split_polyline_by_length(points: list, max_length: float) -> list:
    """Subdivides a polyline so no individual edge segment exceeds max_length."""
    total_len = polyline_arc_length(points)
    if total_len <= max_length or len(points) < 2 or max_length <= 0:
        return [points]

    pts = np.array(points, dtype=np.float64)
    seg_dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum_dists = np.insert(np.cumsum(seg_dists), 0, 0.0)

    n_splits = int(np.ceil(total_len / max_length))
    target_dists = np.linspace(0.0, total_len, n_splits + 1)

    def interpolate_at(dist_val: float):
        idx = int(np.searchsorted(cum_dists, dist_val))
        if idx <= 0:
            return (float(pts[0][0]), float(pts[0][1]))
        if idx >= len(cum_dists):
            return (float(pts[-1][0]), float(pts[-1][1]))
        prev_d = cum_dists[idx - 1]
        span = cum_dists[idx] - prev_d
        t = 0.0 if span <= 1e-12 else (dist_val - prev_d) / span
        interp = pts[idx - 1] + t * (pts[idx] - pts[idx - 1])
        return (float(interp[0]), float(interp[1]))

    sub_polylines = []
    for s in range(n_splits):
        start_d, end_d = target_dists[s], target_dists[s + 1]
        sub_pts = [interpolate_at(start_d)]

        for k in range(1, len(points) - 1):
            if start_d < cum_dists[k] < end_d:
                sub_pts.append((float(points[k][0]), float(points[k][1])))

        sub_pts.append(interpolate_at(end_d))
        sub_polylines.append(sub_pts)

    return sub_polylines


def build_graph_from_polylines(
    raw_polylines: list,
    width: float,
    height: float,
    max_angle_deg: float = 60.0,
    length_limit_pct: float = 0.10,
    decimals: int = 2,
) -> nx.MultiGraph:
    """Constructs a NetworkX MultiGraph directly from a list of polylines."""
    max_edge_len = max(width, height) * length_limit_pct

    G = nx.MultiGraph()
    G.graph["canvas_width"] = width
    G.graph["canvas_height"] = height

    def quantize(pt):
        return (round(float(pt[0]), decimals), round(float(pt[1]), decimals))

    all_corner_nodes = set()
    all_final_segments = []

    for poly in raw_polylines:
        angle_splits, corners = split_polyline_by_angle(poly, max_angle_deg=max_angle_deg)
        for c in corners:
            all_corner_nodes.add(quantize(c))

        for seg in angle_splits:
            all_final_segments.extend(split_polyline_by_length(seg, max_length=max_edge_len))

    for seg in all_final_segments:
        if len(seg) < 2:
            continue

        u = quantize(seg[0])
        v = quantize(seg[-1])
        seg_clean = [u] + [(float(p[0]), float(p[1])) for p in seg[1:-1]] + [v]

        length = polyline_arc_length(seg_clean)
        if length <= 1e-9 and u == v:
            continue

        start_tangent = (seg_clean[1][0] - seg_clean[0][0], seg_clean[1][1] - seg_clean[0][1])
        end_tangent = (seg_clean[-1][0] - seg_clean[-2][0], seg_clean[-1][1] - seg_clean[-2][1])

        if u not in G:
            G.add_node(u, pos=u, is_corner=(u in all_corner_nodes))
        else:
            G.nodes[u]["is_corner"] = G.nodes[u].get("is_corner", False) or (u in all_corner_nodes)

        if v not in G:
            G.add_node(v, pos=v, is_corner=(v in all_corner_nodes))
        else:
            G.nodes[v]["is_corner"] = G.nodes[v].get("is_corner", False) or (v in all_corner_nodes)

        G.add_edge(
            u,
            v,
            path=seg_clean,
            weight=length,
            start_tangent=start_tangent,
            end_tangent=end_tangent,
            is_bridge=False,
        )

    for n in G.nodes():
        G.nodes[n]["is_endpoint"] = (G.degree(n) == 1)

    return G


def svg_to_graph(
    svg_file: str,
    max_angle_deg: float = 60.0,
    length_limit_pct: float = 0.10,
    decimals: int = 2,
) -> nx.MultiGraph:
    """Constructs a NetworkX MultiGraph from an SVG file."""
    raw_polylines, width, height = parse_svg_paths(svg_file)
    return build_graph_from_polylines(
        raw_polylines, width, height,
        max_angle_deg=max_angle_deg,
        length_limit_pct=length_limit_pct,
        decimals=decimals,
    )


def connect_disjoint_components(
    G: nx.MultiGraph,
    interior_penalty: float = 40.0,
) -> nx.MultiGraph:
    """Connects disjoint components using a Minimum Spanning Tree with an interior node penalty."""
    if G.number_of_nodes() == 0:
        return G

    components = [list(c) for c in nx.connected_components(G)]
    k = len(components)
    if k <= 1:
        return G

    def get_penalty(node):
        if G.degree(node) != 2 or G.nodes[node].get("is_corner", False):
            return 0.0
        return interior_penalty

    # Precompute coordinate and penalty arrays per component for vectorized cdist
    comp_coords = [
        np.array([G.nodes[n]["pos"] for n in comp], dtype=np.float64)
        for comp in components
    ]
    comp_penalties = [
        np.array([get_penalty(n) for n in comp], dtype=np.float64)
        for comp in components
    ]

    H = nx.Graph()
    for i in range(k):
        H.add_node(i)

    for i in range(k):
        for j in range(i + 1, k):
            dists = cdist(comp_coords[i], comp_coords[j])
            costs = dists + comp_penalties[i][:, None] + comp_penalties[j][None, :]

            min_flat = int(np.argmin(costs))
            r, c = divmod(min_flat, len(components[j]))

            best_cost = float(costs[r, c])
            best_dist = float(dists[r, c])
            best_pair = (components[i][r], components[j][c])

            H.add_edge(i, j, weight=best_cost, actual_dist=best_dist, bridge_nodes=best_pair)

    mst = nx.minimum_spanning_tree(H, weight="weight")

    for _, _, data in mst.edges(data=True):
        u, v = data["bridge_nodes"]
        dist = data["actual_dist"]
        p1 = G.nodes[u]["pos"]
        p2 = G.nodes[v]["pos"]
        tangent = (p2[0] - p1[0], p2[1] - p1[1])

        G.add_edge(
            u,
            v,
            path=[p1, p2],
            weight=dist,
            start_tangent=tangent,
            end_tangent=tangent,
            is_bridge=True,
        )

    return G


def _match_odd_vertices_fast(odd_dist_mat: np.ndarray):
    """Pairs K-2 odd vertices and leaves 2 open endpoints using Greedy + 2-Opt refinement."""
    K = odd_dist_mat.shape[0]
    num_pairs = (K - 2) // 2

    # 1. Greedy initialization: sort all pairs by shortest distance
    iu, ju = np.triu_indices(K, k=1)
    order = np.argsort(odd_dist_mat[iu, ju])

    matched_mask = np.zeros(K, dtype=bool)
    pairs = []
    for idx in order:
        i, j = int(iu[idx]), int(ju[idx])
        if not matched_mask[i] and not matched_mask[j]:
            matched_mask[i] = True
            matched_mask[j] = True
            pairs.append([i, j])
            if len(pairs) == num_pairs:
                break

    unmatched = [i for i in range(K) if not matched_mask[i]]

    # 2. Fast 2-Opt local refinement (uncrossing pairs & optimizing open endpoints)
    for _ in range(3):
        improved = False

        # Check swaps between matched pairs and the 2 unmatched endpoints
        for p_idx in range(num_pairs):
            a, b = pairs[p_idx]
            curr_cost = odd_dist_mat[a, b]
            for u_idx in range(2):
                e = unmatched[u_idx]
                if odd_dist_mat[e, b] + 1e-6 < curr_cost:
                    pairs[p_idx][0], unmatched[u_idx] = e, a
                    a, curr_cost = e, odd_dist_mat[e, b]
                    improved = True
                elif odd_dist_mat[a, e] + 1e-6 < curr_cost:
                    pairs[p_idx][1], unmatched[u_idx] = e, b
                    b, curr_cost = e, odd_dist_mat[a, e]
                    improved = True

        # Check 2-opt swaps between pairs of matched edges
        for p1 in range(num_pairs):
            a, b = pairs[p1]
            dab = odd_dist_mat[a, b]
            for p2 in range(p1 + 1, num_pairs):
                c, d = pairs[p2]
                current = dab + odd_dist_mat[c, d]
                alt1 = odd_dist_mat[a, c] + odd_dist_mat[b, d]
                alt2 = odd_dist_mat[a, d] + odd_dist_mat[b, c]

                if alt1 + 1e-6 < current and alt1 <= alt2:
                    pairs[p1][1], pairs[p2][0] = c, b
                    b, dab = c, odd_dist_mat[a, c]
                    improved = True
                elif alt2 + 1e-6 < current:
                    pairs[p1][1], pairs[p2][1] = d, b
                    b, dab = d, odd_dist_mat[a, d]
                    improved = True

        if not improved:
            break

    return pairs, (unmatched[0], unmatched[1])


def make_semi_eulerian(G: nx.MultiGraph):
    """Augments G by duplicating shortest paths along existing edges so at most two vertices have odd degree."""
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty (no strokes detected in image).")
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected before solving. Run connect_disjoint_components first.")

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    odd_nodes = [n for n in nodes if G.degree(n) % 2 != 0]
    K = len(odd_nodes)

    if K == 0:
        return G.copy(), (nodes[0], nodes[0])
    if K == 2:
        return G.copy(), (odd_nodes[0], odd_nodes[1])

    # 1. Build sparse adjacency matrix keeping the shortest edge between any pair of adjacent nodes
    N = len(nodes)
    min_edges = {}
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        iu, iv = node_to_idx[u], node_to_idx[v]
        if iu > iv:
            iu, iv = iv, iu
        w = max(float(data.get("weight", 1.0)), 1e-6)
        if (iu, iv) not in min_edges or w < min_edges[(iu, iv)]:
            min_edges[(iu, iv)] = w

    rows, cols, weights = [], [], []
    for (iu, iv), w in min_edges.items():
        rows.extend([iu, iv])
        cols.extend([iv, iu])
        weights.extend([w, w])

    adj_csr = csr_matrix((weights, (rows, cols)), shape=(N, N))

    # 2. Run compiled C Dijkstra from all odd vertices simultaneously
    odd_indices = [node_to_idx[n] for n in odd_nodes]
    dist_matrix, predecessors = dijkstra(
        adj_csr, directed=False, indices=odd_indices, return_predecessors=True
    )
    odd_dist_mat = dist_matrix[:, odd_indices]

    # 3. Match K - 2 odd nodes and leave 2 optimal open endpoints
    matched_pairs, (end_i, end_j) = _match_odd_vertices_fast(odd_dist_mat)
    best_endpoints = (odd_nodes[end_i], odd_nodes[end_j])

    # 4. Reconstruct paths from predecessor matrix ONLY for the matched pairs
    G_semi = G.copy()
    for i, j in matched_pairs:
        src_node_idx = odd_indices[i]
        curr_idx = odd_indices[j]

        while curr_idx != src_node_idx and curr_idx >= 0:
            prev_idx = int(predecessors[i, curr_idx])
            if prev_idx < 0:
                break
            n1, n2 = nodes[prev_idx], nodes[curr_idx]
            edge_candidates = G[n1][n2]
            best_key = min(edge_candidates.keys(), key=lambda k: edge_candidates[k].get("weight", 1.0))
            dup_data = copy.deepcopy(edge_candidates[best_key])
            dup_data["is_retraced"] = True
            G_semi.add_edge(n1, n2, **dup_data)
            curr_idx = prev_idx

    return G_semi, best_endpoints


def simple_eulerian_trail(G_semi: nx.MultiGraph, start_node) -> list:
    """Finds a valid Eulerian path/circuit in O(E) time using Hierholzer's algorithm."""
    adj = {n: [] for n in G_semi.nodes()}
    for u, v, k in G_semi.edges(keys=True):
        adj[u].append((v, k))
        if u != v:
            adj[v].append((u, k))

    used_edges = set()
    stack = [start_node]
    trail = []

    while stack:
        curr = stack[-1]
        nbr_list = adj[curr]

        # Pop already-used edges in O(1) amortized time
        while nbr_list and (min(curr, nbr_list[-1][0]), max(curr, nbr_list[-1][0]), nbr_list[-1][1]) in used_edges:
            nbr_list.pop()

        if nbr_list:
            nbr, k = nbr_list.pop()
            used_edges.add((min(curr, nbr), max(curr, nbr), k))
            stack.append(nbr)
        else:
            trail.append(stack.pop())

    trail.reverse()
    return trail


def reconstruct_full_polyline(G_semi: nx.MultiGraph, node_trail: list) -> list:
    """Stitches node transitions back into the exact curved polyline coordinates."""
    if len(node_trail) < 2:
        return []

    edge_tracker = {}
    for u, v, k, data in G_semi.edges(keys=True, data=True):
        edge_id = (min(u, v), max(u, v), k)
        edge_tracker[edge_id] = {
            "path": data.get("path", [u, v]),
            "used": False,
        }

    full_coords = [node_trail[0]]

    for i in range(len(node_trail) - 1):
        u, v = node_trail[i], node_trail[i + 1]

        matched_edge_id = None
        for k in G_semi[u][v]:
            edge_id = (min(u, v), max(u, v), k)
            if not edge_tracker[edge_id]["used"]:
                matched_edge_id = edge_id
                break

        if matched_edge_id is not None:
            edge_tracker[matched_edge_id]["used"] = True
            raw_path = edge_tracker[matched_edge_id]["path"]
        else:
            best_k = min(G_semi[u][v].keys(), key=lambda k: G_semi[u][v][k].get("weight", 1.0))
            raw_path = G_semi[u][v][best_k].get("path", [u, v])

        p_start = np.array(raw_path[0])
        p_end = np.array(raw_path[-1])
        u_arr = np.array(u)

        if np.linalg.norm(p_start - u_arr) <= np.linalg.norm(p_end - u_arr):
            segment = raw_path[1:]
        else:
            segment = list(reversed(raw_path))[1:]

        full_coords.extend(segment)

    return full_coords