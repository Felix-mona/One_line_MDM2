"""
graph_formulation.py

Step 1: Pre-Processing and Graph Extraction (Report Section 2.1).
Implements grayscale conversion, foreground binarisation, optional stroke-width
normalisation, Lee's medial-axis thinning, Rutovitz crossing-number classification,
medoid junction clustering, lossless 8-connected skeleton tracing,
Ramer-Douglas-Peucker (RDP) polyline decimation, and topological graph refinement
(close-junction contraction and whisker/noise pruning).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

# Clockwise 8-connected neighbourhood offsets starting at North
NEIGHBOUR_OFFSETS_8: list[tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


# ==============================================================================
# 1. BINARISATION, BOUNDING-BOX CROPPING & RESCALING (SECTION 2.1.1)
# ==============================================================================
def load_and_scale_image(
    image_path: str,
    input_stroke_width: float = 10.0,
    target_stroke_width: float = 10.0,
    binarise_thresh: int = 128,
) -> tuple[np.ndarray, float, float]:
    """
    Loads a raster sketch, flattens any alpha channel onto white, binarises at
    `binarise_thresh` (1 = foreground, 0 = background), crops tightly to the
    active foreground bounding box (W x H, diagonal D = sqrt(W^2 + H^2)), and
    optionally downscales thick strokes prior to thinning.

    Args:
        image_path: Path to the input raster image.
        input_stroke_width: Estimated stroke thickness in pixels of the input.
        target_stroke_width: Desired working stroke thickness before thinning.
        binarise_thresh: Grayscale threshold T_bin in [0, 255] (default 128).

    Returns:
        Tuple of (padded_binary_mask, scale_factor, canvas_diagonal_D).
    """
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            gray_img = Image.alpha_composite(white_bg, rgba).convert("L")
        else:
            gray_img = img.convert("L")
        gray_arr = np.array(gray_img, dtype=np.uint8)

    binary = gray_arr < binarise_thresh
    # Ensure dark-on-light convention (foreground strokes are minority pixels)
    if np.sum(binary) > binary.size / 2:
        binary = ~binary

    active_coords = np.argwhere(binary)
    if len(active_coords) == 0:
        padded = np.pad(binary, pad_width=1, mode="constant", constant_values=False)
        return padded, 1.0, float(np.hypot(*padded.shape))

    r_min, c_min = active_coords.min(axis=0)
    r_max, c_max = active_coords.max(axis=0) + 1
    cropped = binary[r_min:r_max, c_min:c_max]

    height_orig, width_orig = cropped.shape
    canvas_diag = float(np.hypot(height_orig, width_orig))

    downsample = max(1.0, float(input_stroke_width) / max(1.0, float(target_stroke_width)))
    if downsample > 1.05:
        height_work = max(3, int(round(height_orig / downsample)))
        width_work = max(3, int(round(width_orig / downsample)))
        pil_mask = Image.fromarray((cropped.astype(np.uint8)) * 255, mode="L")
        resized = pil_mask.resize((width_work, height_work), resample=Image.Resampling.BILINEAR)
        work_mask = np.array(resized, dtype=np.uint8) >= 96
        scale_factor = canvas_diag / float(np.hypot(height_work, width_work))
    else:
        work_mask = cropped
        scale_factor = 1.0

    # Pad by 1 pixel so all foreground pixels have a valid 8-neighbourhood
    padded_mask = np.pad(work_mask, pad_width=1, mode="constant", constant_values=False)
    return padded_mask, scale_factor, canvas_diag


def run_thinning(binary_mask: np.ndarray, method: str = "lee") -> np.ndarray:
    """
    Erodes a binary foreground mask to a single-pixel medial-axis skeleton.

    Args:
        binary_mask: 2D boolean or uint8 array (nonzero = foreground).
        method: Thinning algorithm ('lee' for topology-preserving octree thinning
            or 'zhang' for Zhang-Suen thinning).

    Returns:
        2D uint8 array normalized to {0, 1}.
    """
    return (skeletonize(binary_mask, method=method) > 0).astype(np.uint8)


def run_lee_skeletonisation(binary_mask: np.ndarray) -> np.ndarray:
    """Wrapper executing Lee's medial-axis thinning normalized to {0, 1} uint8."""
    return run_thinning(binary_mask, method="lee")


# ==============================================================================
# 2. RUTOVITZ CROSSING NUMBERS & CANONICAL NODES (SECTION 2.1.2)
# ==============================================================================
def compute_rutovitz_crossing_number(skeleton: np.ndarray) -> np.ndarray:
    """
    Computes the Rutovitz crossing number X(p) for every foreground skeleton pixel
    by traversing its 8-connected perimeter (p_1, ..., p_8) clockwise:
        X(p) = 0.5 * sum_{i=1}^{8} |p_{i+1} - p_i|,  with p_9 = p_1.

    Classification:
        X(p) == 1 : Endpoint
        X(p) == 2 : Interior stroke continuation pixel
        X(p) >= 3 : Junction pixel

    Args:
        skeleton: 2D binary skeleton array ({0, 1}).

    Returns:
        2D uint8 array of crossing numbers (0 on background pixels).
    """
    skel_bin = (skeleton > 0).astype(np.int8)
    padded = np.pad(skel_bin, pad_width=1, mode="constant", constant_values=0)
    p_ring = [
        padded[:-2, 1:-1],  # N
        padded[:-2, 2:],    # NE
        padded[1:-1, 2:],   # E
        padded[2:, 2:],     # SE
        padded[2:, 1:-1],   # S
        padded[2:, :-2],    # SW
        padded[1:-1, :-2],  # W
        padded[:-2, :-2],   # NW
    ]
    transitions = sum(np.abs(p_ring[(i + 1) % 8] - p_ring[i]) for i in range(8))
    return ((transitions // 2) * skel_bin).astype(np.uint8)


def extract_canonical_nodes(
    skeleton: np.ndarray,
    crossing_number: np.ndarray,
    merge_dist_work: float,
) -> tuple[dict[int, tuple[int, int]], dict[int, str], np.ndarray, dict[int, list[tuple[int, int]]]]:
    """
    Groups contiguous junction pixels (X(p) >= 3) into single medoid vertices,
    registers endpoints (X(p) == 1), and merges vertices within `merge_dist_work`
    (corresponding to d_euclid = 0.008 * D) into their geometric medoid.

    Args:
        skeleton: 2D binary skeleton array ({0, 1}).
        crossing_number: 2D array of Rutovitz crossing numbers X(p).
        merge_dist_work: Spatial node-merge radius in working pixel coordinates.

    Returns:
        Tuple of (node_positions_rc, node_types, node_id_grid, node_member_pixels).
    """
    height, width = skeleton.shape
    node_grid = np.full((height, width), fill_value=-1, dtype=np.int32)
    raw_nodes: dict[int, tuple[float, float]] = {}
    raw_types: dict[int, str] = {}
    raw_members: dict[int, list[tuple[int, int]]] = {}
    nid = 0

    # 1. Register endpoints (X(p) == 1)
    for r, c in np.argwhere((skeleton > 0) & (crossing_number == 1)):
        raw_nodes[nid] = (float(r), float(c))
        raw_types[nid] = "endpoint"
        raw_members[nid] = [(int(r), int(c))]
        nid += 1

    # 2. Collapse contiguous junction blobs (X(p) >= 3) into medoids
    junc_mask = (skeleton > 0) & (crossing_number >= 3)
    labeled, num_blobs = ndimage.label(junc_mask, structure=np.ones((3, 3), dtype=bool))
    for blob_idx in range(1, num_blobs + 1):
        pts = np.argwhere(labeled == blob_idx)
        centroid = pts.mean(axis=0)
        best = pts[np.argmin(np.sum((pts - centroid) ** 2, axis=1))]
        raw_nodes[nid] = (float(best[0]), float(best[1]))
        raw_types[nid] = "junction"
        raw_members[nid] = [(int(r), int(c)) for r, c in pts]
        nid += 1

    # 3. Cluster spatially close vertices via KD-Tree
    final_nodes: dict[int, tuple[int, int]] = {}
    final_types: dict[int, str] = {}
    final_members: dict[int, list[tuple[int, int]]] = {}
    skel_set = set(map(tuple, np.argwhere(skeleton > 0)))

    if len(raw_nodes) > 0 and merge_dist_work > 0.0:
        raw_ids = list(raw_nodes.keys())
        coords = np.array([raw_nodes[i] for i in raw_ids])
        pairs = cKDTree(coords).query_pairs(r=merge_dist_work)
        merge_graph = nx.Graph()
        merge_graph.add_nodes_from(raw_ids)
        merge_graph.add_edges_from((raw_ids[i], raw_ids[j]) for i, j in pairs)

        for new_id, cluster in enumerate(nx.connected_components(merge_graph)):
            cluster_list = list(cluster)
            combined = {px for i in cluster_list for px in raw_members[i]}
            is_junc = any(raw_types[i] == "junction" for i in cluster_list)

            # Absorb intermediate skeleton pixels connecting merged cluster members
            if len(cluster_list) > 1:
                rep_px = raw_members[cluster_list[0]][0]
                max_steps = int(np.ceil(merge_dist_work)) + 4
                for other_id in cluster_list[1:]:
                    target_px = raw_members[other_id][0]
                    queue = [(rep_px, 0)]
                    parent: dict[tuple[int, int], tuple[int, int] | None] = {rep_px: None}
                    while queue:
                        curr, dist = queue.pop(0)
                        if curr == target_px:
                            ptr = curr
                            while ptr is not None:
                                combined.add(ptr)
                                ptr = parent[ptr]
                            break
                        if dist < max_steps:
                            for dr, dc in NEIGHBOUR_OFFSETS_8:
                                nbr = (curr[0] + dr, curr[1] + dc)
                                if nbr in skel_set and nbr not in parent:
                                    parent[nbr] = curr
                                    queue.append((nbr, dist + 1))

            pix_arr = np.array(list(combined))
            centroid = pix_arr.mean(axis=0)
            best_px = pix_arr[np.argmin(np.sum((pix_arr - centroid) ** 2, axis=1))]
            final_nodes[new_id] = (int(best_px[0]), int(best_px[1]))
            final_types[new_id] = "junction" if is_junc else "endpoint"
            final_members[new_id] = list(combined)
            for r, c in combined:
                node_grid[r, c] = new_id
    else:
        for i, (r, c) in raw_nodes.items():
            final_nodes[i] = (int(r), int(c))
            final_types[i] = raw_types[i]
            final_members[i] = raw_members[i]
            for pr, pc in raw_members[i]:
                node_grid[pr, pc] = i

    return final_nodes, final_types, node_grid, final_members


# ==============================================================================
# 3. LOSSLESS SKELETON TRACING & RDP DECIMATION (SECTION 2.1.3)
# ==============================================================================
def _bfs_cluster_path(
    start: tuple[int, int],
    end: tuple[int, int],
    valid_pixels: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Finds the shortest 8-connected pixel path between `start` and `end` inside a node cluster."""
    if start == end:
        return [start]
    queue = [start]
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        curr = queue.pop(0)
        if curr == end:
            path = []
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            return path[::-1]
        for dr, dc in NEIGHBOUR_OFFSETS_8:
            nbr = (curr[0] + dr, curr[1] + dc)
            if nbr in valid_pixels and nbr not in parent:
                parent[nbr] = curr
                queue.append(nbr)
    return [start, end]


def _rc_to_xy(rc_list: list[tuple[int, int]], scale_factor: float) -> np.ndarray:
    """Converts (row, col) pixel indices to deduplicated Cartesian (x, y) coordinates."""
    clean = [rc_list[0]]
    for pt in rc_list[1:]:
        if pt[0] != clean[-1][0] or pt[1] != clean[-1][1]:
            clean.append(pt)
    arr = np.array(clean, dtype=float)
    # x = col * scale, y = -row * scale so Cartesian plots render upright
    return np.column_stack((arr[:, 1] * scale_factor, -arr[:, 0] * scale_factor))


def trace_skeleton_to_graph(
    skeleton: np.ndarray,
    nodes: dict[int, tuple[int, int]],
    node_types: dict[int, str],
    node_grid: np.ndarray,
    node_members: dict[int, list[tuple[int, int]]],
    scale_factor: float,
) -> nx.MultiGraph:
    """
    Traces 8-connected foreground pixel chains between canonical vertices and
    initialises isolated closed loops lacking endpoints or junctions with a seed
    vertex, producing the raw spatial multigraph G = (V, E).

    Args:
        skeleton: 2D binary skeleton array ({0, 1}).
        nodes: Mapping from node ID to medoid (row, col).
        node_types: Mapping from node ID to 'endpoint' or 'junction'.
        node_grid: 2D int32 grid labelling node cluster pixels (-1 on stroke pixels).
        node_members: Mapping from node ID to member (row, col) coordinates.
        scale_factor: Coordinate multiplier restoring original image scale.

    Returns:
        Undirected spatial nx.MultiGraph with 'pos' on nodes and 'pts' on edges.
    """
    graph = nx.MultiGraph()
    for nid, (r, c) in nodes.items():
        graph.add_node(
            nid,
            pos=np.array([c * scale_factor, -r * scale_factor], dtype=float),
            kind=node_types.get(nid, "endpoint"),
        )

    height, width = skeleton.shape
    visited = np.zeros_like(skeleton, dtype=bool)
    direct_seen: set[tuple[int, int]] = set()

    # Pass 1: Trace chains radiating from canonical vertices
    for u_id, u_pos in list(nodes.items()):
        u_pixels = node_members.get(u_id, [u_pos])
        u_set = set(u_pixels)
        for p in u_pixels:
            for dr, dc in NEIGHBOUR_OFFSETS_8:
                nr, nc = p[0] + dr, p[1] + dc
                if not (0 <= nr < height and 0 <= nc < width) or not skeleton[nr, nc]:
                    continue
                nbr_id = int(node_grid[nr, nc])

                # Case A: Unvisited interior stroke pixel
                if nbr_id == -1 and not visited[nr, nc]:
                    path = list(_bfs_cluster_path(u_pos, p, u_set))
                    curr_r, curr_c = nr, nc
                    target_id, target_px = -1, None
                    steps = 0
                    left_start = False

                    while True:
                        path.append((curr_r, curr_c))
                        visited[curr_r, curr_c] = True
                        steps += 1
                        path_nbrs, other_nodes, self_nodes = [], [], []
                        for s_dr, s_dc in NEIGHBOUR_OFFSETS_8:
                            sr, sc = curr_r + s_dr, curr_c + s_dc
                            if 0 <= sr < height and 0 <= sc < width and skeleton[sr, sc]:
                                s_id = int(node_grid[sr, sc])
                                if s_id == -1 and not visited[sr, sc]:
                                    path_nbrs.append((sr, sc))
                                elif s_id != -1:
                                    if s_id != u_id:
                                        other_nodes.append(((sr, sc), s_id))
                                    else:
                                        self_nodes.append(((sr, sc), u_id))

                        if not self_nodes:
                            left_start = True

                        if other_nodes:
                            target_px, target_id = other_nodes[0]
                            break
                        if self_nodes and left_start:
                            target_px, target_id = self_nodes[0]
                            break
                        if path_nbrs:
                            # Prefer 4-connected orthogonal steps before diagonals
                            path_nbrs.sort(key=lambda pt: abs(pt[0] - curr_r) + abs(pt[1] - curr_c))
                            curr_r, curr_c = path_nbrs[0]
                        elif self_nodes and steps >= 3:
                            target_px, target_id = self_nodes[0]
                            break
                        else:
                            break

                    if target_id != -1 and target_px is not None:
                        v_pos = nodes[target_id]
                        v_set = set(node_members.get(target_id, [v_pos]))
                        path.extend(_bfs_cluster_path(target_px, v_pos, v_set))
                        pts_xy = _rc_to_xy(path, scale_factor)
                        if len(pts_xy) >= 2:
                            graph.add_edge(u_id, target_id, pts=pts_xy)

                # Case B: Directly adjacent vertex clusters
                elif nbr_id != -1 and nbr_id != u_id:
                    edge_key = (min(u_id, nbr_id), max(u_id, nbr_id))
                    if edge_key not in direct_seen:
                        direct_seen.add(edge_key)
                        v_pos = nodes[nbr_id]
                        v_set = set(node_members.get(nbr_id, [v_pos]))
                        full_rc = (
                            _bfs_cluster_path(u_pos, p, u_set)
                            + _bfs_cluster_path((nr, nc), v_pos, v_set)
                        )
                        pts_xy = _rc_to_xy(full_rc, scale_factor)
                        if len(pts_xy) >= 2:
                            graph.add_edge(u_id, nbr_id, pts=pts_xy)

    # Pass 2: Trace isolated closed loops lacking endpoints or junctions
    unvisited_loops = np.argwhere((skeleton > 0) & (~visited) & (node_grid == -1))
    for sr, sc in unvisited_loops:
        if visited[sr, sc]:
            continue
        loop_id = max(graph.nodes, default=-1) + 1
        start_rc = (int(sr), int(sc))
        path = [start_rc]
        curr_r, curr_c = start_rc
        visited[curr_r, curr_c] = True

        while True:
            nbrs = [
                (curr_r + dr, curr_c + dc)
                for dr, dc in NEIGHBOUR_OFFSETS_8
                if 0 <= curr_r + dr < height
                and 0 <= curr_c + dc < width
                and skeleton[curr_r + dr, curr_c + dc]
                and node_grid[curr_r + dr, curr_c + dc] == -1
                and not visited[curr_r + dr, curr_c + dc]
            ]
            if nbrs:
                nbrs.sort(key=lambda pt: abs(pt[0] - curr_r) + abs(pt[1] - curr_c))
                curr_r, curr_c = nbrs[0]
                path.append((curr_r, curr_c))
                visited[curr_r, curr_c] = True
            else:
                break

        if len(path) > 2:
            path.append(start_rc)
            pts_xy = _rc_to_xy(path, scale_factor)
            if len(pts_xy) >= 3:
                graph.add_node(loop_id, pos=pts_xy[0].copy(), kind="loop_anchor")
                graph.add_edge(loop_id, loop_id, pts=pts_xy)

    return graph


def rdp_simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer-Douglas-Peucker (RDP) recursive polyline decimation.
    Splits a curve at the vertex of maximum perpendicular distance from its
    chord until all vertices lie within `epsilon`.

    Args:
        points: Array of shape (N, 2) representing polyline coordinates.
        epsilon: Maximum perpendicular deviation tolerance (epsilon_RDP).

    Returns:
        Simplified coordinate array of shape (M, 2) with M <= N.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2 or epsilon <= 0.0:
        return pts
    start, end = pts[0], pts[-1]
    line_vec = end - start
    line_len = float(np.hypot(line_vec[0], line_vec[1]))
    diff = pts[1:-1] - start
    if line_len < 1e-12:
        dists = np.hypot(diff[:, 0], diff[:, 1])
    else:
        dists = np.abs(line_vec[0] * diff[:, 1] - line_vec[1] * diff[:, 0]) / line_len
    max_idx = int(np.argmax(dists)) + 1
    if dists[max_idx - 1] > epsilon:
        left = rdp_simplify(pts[: max_idx + 1], epsilon)
        right = rdp_simplify(pts[max_idx:], epsilon)
        return np.vstack((left[:-1], right))
    return np.array([start, end], dtype=float)


def apply_rdp_to_graph(graph: nx.MultiGraph, epsilon: float) -> nx.MultiGraph:
    """
    Applies RDP decimation to every edge in `graph`. Closed self-loops (u == v)
    are split at their midpoint prior to simplification so closed shapes never
    collapse into a degenerate two-point segment.

    Args:
        graph: Input spatial multigraph.
        epsilon: RDP perpendicular distance tolerance.

    Returns:
        Copy of `graph` with simplified edge polylines.
    """
    simplified_graph = graph.copy()
    for u, v, _, data in simplified_graph.edges(keys=True, data=True):
        pts = data["pts"]
        if u == v and len(pts) > 4:
            mid = len(pts) // 2
            left = rdp_simplify(pts[: mid + 1], epsilon)
            right = rdp_simplify(pts[mid:], epsilon)
            data["pts"] = np.vstack((left[:-1], right))
        else:
            data["pts"] = rdp_simplify(pts, epsilon)
    return simplified_graph


# ==============================================================================
# 4. TOPOLOGICAL GRAPH REFINEMENT (SECTION 2.1.4)
# ==============================================================================
def compute_polyline_length(pts: np.ndarray) -> float:
    """Computes total Euclidean arc length of a 2D polyline."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def dissolve_degree_two_nodes(graph: nx.MultiGraph) -> nx.MultiGraph:
    """Merges non-loop degree-2 continuation vertices along smooth strokes."""
    cleaned = graph.copy()
    changed = True
    while changed:
        changed = False
        for node in list(cleaned.nodes()):
            if cleaned.degree(node) != 2:
                continue
            edges = list(cleaned.edges(node, keys=True, data=True))
            if len(edges) != 2 or any(u == v for u, v, _, _ in edges):
                continue
            (u1, v1, k1, d1), (u2, v2, k2, d2) = edges
            nbr1 = v1 if u1 == node else u1
            nbr2 = v2 if u2 == node else u2
            node_pos = cleaned.nodes[node]["pos"]

            pts1 = d1["pts"]
            if np.linalg.norm(pts1[-1] - node_pos) > np.linalg.norm(pts1[0] - node_pos):
                pts1 = pts1[::-1]

            pts2 = d2["pts"]
            if np.linalg.norm(pts2[0] - node_pos) > np.linalg.norm(pts2[-1] - node_pos):
                pts2 = pts2[::-1]

            merged_pts = np.vstack((pts1[:-1], pts2))
            cleaned.remove_edge(u1, v1, key=k1)
            cleaned.remove_edge(u2, v2, key=k2)
            cleaned.remove_node(node)
            cleaned.add_edge(nbr1, nbr2, pts=merged_pts)
            changed = True
            break
    return cleaned


def merge_nearby_junctions(graph: nx.MultiGraph, d_edge_thresh: float) -> nx.MultiGraph:
    """
    Close-junction contraction (Section 2.1.4): contracts non-loop edges between
    junctions shorter than `d_edge_thresh` (0.02 * D) into their geometric centroid
    while strictly preserving all closed self-loops and parallel petals.

    Args:
        graph: Input spatial multigraph.
        d_edge_thresh: Maximum edge length between junctions to contract.

    Returns:
        Contracted spatial multigraph.
    """
    cleaned = dissolve_degree_two_nodes(graph)
    while True:
        short_graph = nx.Graph()
        for u, v, _, data in cleaned.edges(keys=True, data=True):
            if u != v and cleaned.degree(u) >= 3 and cleaned.degree(v) >= 3:
                if compute_polyline_length(data["pts"]) <= d_edge_thresh:
                    short_graph.add_edge(u, v)

        clusters = [list(c) for c in nx.connected_components(short_graph) if len(c) > 1]
        if not clusters:
            break

        for cluster in clusters:
            c_set = set(cluster)
            centroid = np.mean([cleaned.nodes[n]["pos"] for n in cluster], axis=0)
            rep = cluster[0]
            old_pos = {n: cleaned.nodes[n]["pos"].copy() for n in cluster}
            cleaned.nodes[rep]["pos"] = centroid
            cleaned.nodes[rep]["kind"] = "junction"

            seen = set()
            internal, external = [], []
            for u, v, k, data in list(cleaned.edges(cluster, keys=True, data=True)):
                eid = (min(u, v), max(u, v), k) if u != v else (u, v, k)
                if eid in seen:
                    continue
                seen.add(eid)
                if u in c_set and v in c_set:
                    internal.append((u, v, k, data["pts"].copy()))
                else:
                    external.append((u, v, k, data["pts"].copy()))

            for u, v, k, _ in internal + external:
                if cleaned.has_edge(u, v, key=k):
                    cleaned.remove_edge(u, v, key=k)

            # Preserve self-loops and parallel petals longer than d_edge_thresh
            pair_groups: dict[tuple[int, int], list[tuple[np.ndarray, float]]] = {}
            for u, v, _, pts in internal:
                if u == v:
                    pts[0], pts[-1] = centroid, centroid
                    cleaned.add_edge(rep, rep, pts=pts)
                else:
                    pk = (min(u, v), max(u, v))
                    pair_groups.setdefault(pk, []).append((pts, compute_polyline_length(pts)))

            for _, edge_list in pair_groups.items():
                edge_list.sort(key=lambda x: x[1])
                for pts, _ in edge_list[1:]:
                    pts[0], pts[-1] = centroid, centroid
                    cleaned.add_edge(rep, rep, pts=pts)

            # Re-anchor external edges to the new centroid
            for u, v, _, pts in external:
                inner = u if u in c_set else v
                outer = v if u in c_set else u
                in_p, out_p = old_pos[inner], cleaned.nodes[outer]["pos"]
                d_fwd = np.linalg.norm(pts[0] - in_p) + np.linalg.norm(pts[-1] - out_p)
                d_rev = np.linalg.norm(pts[-1] - in_p) + np.linalg.norm(pts[0] - out_p)
                if d_fwd <= d_rev:
                    pts[0] = centroid
                else:
                    pts[-1] = centroid
                cleaned.add_edge(rep, outer, pts=pts)

            for n in cluster[1:]:
                if n in cleaned:
                    cleaned.remove_node(n)

        cleaned = dissolve_degree_two_nodes(cleaned)

    cleaned.remove_nodes_from([n for n in list(cleaned.nodes()) if cleaned.degree(n) == 0])
    return cleaned


def prune_whiskers_and_noise(
    graph: nx.MultiGraph,
    whisker_thresh: float,
    min_comp_len: float,
) -> nx.MultiGraph:
    """
    Whisker and noise pruning (Section 2.1.4): removes degree-1 terminal spurs
    shorter than `whisker_thresh` (0.01 * D) and isolated connected components
    with total arc length below `min_comp_len` (0.015 * D).

    Args:
        graph: Input spatial multigraph.
        whisker_thresh: Minimum length for degree-1 terminal edges.
        min_comp_len: Minimum total arc length for connected components.

    Returns:
        Refined base spatial multigraph G = (V, E).
    """
    cleaned = graph.copy()
    changed = True
    while changed:
        changed = False
        for u, v, k, data in list(cleaned.edges(keys=True, data=True)):
            if u != v and (cleaned.degree(u) == 1 or cleaned.degree(v) == 1):
                if compute_polyline_length(data["pts"]) < whisker_thresh:
                    cleaned.remove_edge(u, v, key=k)
                    for n in (u, v):
                        if n in cleaned and cleaned.degree(n) == 0:
                            cleaned.remove_node(n)
                    cleaned = dissolve_degree_two_nodes(cleaned)
                    changed = True
                    break

    for comp in list(nx.connected_components(cleaned)):
        comp_len = sum(
            compute_polyline_length(d["pts"])
            for _, _, _, d in cleaned.edges(comp, keys=True, data=True)
        )
        if comp_len < min_comp_len:
            cleaned.remove_nodes_from(comp)

    return dissolve_degree_two_nodes(cleaned)


# ==============================================================================
# 5. UNIFORM VISUALISATION HELPERS & STEP 1 ENTRY POINT
# ==============================================================================
def _create_clean_canvas(
    figsize: tuple[float, float] = (10, 10),
    bg_color: str = "white",
) -> tuple[plt.Figure, plt.Axes]:
    """Creates a uniform borderless figure and axes with no ticks, labels, or frame."""
    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_color)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _save_clean_canvas(
    fig: plt.Figure,
    save_path: str,
    dpi: int = 300,
    bg_color: str = "white",
) -> None:
    """Saves a Matplotlib figure with uniform tight framing and closes it."""
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1, facecolor=bg_color)
    plt.close(fig)


def save_binary_or_skeleton_image(mask: np.ndarray, save_path: str, label: str = "image") -> None:
    """Saves a binary mask or skeleton as a black-on-white PNG image."""
    img_u8 = np.where(mask > 0, 0, 255).astype(np.uint8)
    Image.fromarray(img_u8, mode="L").save(save_path)
    print(f"Saved {label} to '{save_path}'")


def save_skeleton_image(skeleton: np.ndarray, save_path: str) -> None:
    """Saves a single-pixel skeleton as a black-on-white PNG image."""
    save_binary_or_skeleton_image(skeleton, save_path, label="skeleton image")


def plot_junction_classification(
    skeleton: np.ndarray,
    nodes: dict[int, tuple[int, int]],
    node_types: dict[int, str],
    scale_factor: float,
    save_path: str,
) -> None:
    """Saves a clean plot of the skeleton with endpoints (green) and junctions (red)."""
    fig, ax = _create_clean_canvas()
    skel_pts = np.argwhere(skeleton > 0)
    if len(skel_pts) > 0:
        ax.scatter(
            skel_pts[:, 1] * scale_factor,
            -skel_pts[:, 0] * scale_factor,
            color="lightgray",
            s=2,
        )
    ep_coords = [
        (c * scale_factor, -r * scale_factor)
        for nid, (r, c) in nodes.items()
        if node_types.get(nid) == "endpoint"
    ]
    junc_coords = [
        (c * scale_factor, -r * scale_factor)
        for nid, (r, c) in nodes.items()
        if node_types.get(nid) == "junction"
    ]
    if ep_coords:
        ep_arr = np.array(ep_coords)
        ax.scatter(ep_arr[:, 0], ep_arr[:, 1], color="limegreen", s=28, zorder=4)
    if junc_coords:
        j_arr = np.array(junc_coords)
        ax.scatter(j_arr[:, 0], j_arr[:, 1], color="crimson", s=32, zorder=5)

    _save_clean_canvas(fig, save_path)
    print(f"Saved junction classification plot to '{save_path}'")


def plot_graph_snapshot(graph: nx.MultiGraph, save_path: str, label: str = "graph snapshot") -> None:
    """Saves a clean borderless plot of a spatial MultiGraph."""
    fig, ax = _create_clean_canvas()
    for _, _, data in graph.edges(data=True):
        pts = data["pts"]
        ax.plot(pts[:, 0], pts[:, 1], "k-", linewidth=1.8)
    for _, attrs in graph.nodes(data=True):
        x, y = attrs["pos"]
        ax.scatter(x, y, color="darkorange", s=22, zorder=5)

    _save_clean_canvas(fig, save_path)
    print(f"Saved {label} to '{save_path}'")


def build_graph_from_image(
    image_path: str,
    input_stroke_width: float = 10.0,
    target_stroke_width: float = 10.0,
    binarise_thresh: int = 128,
    d_euclid_prop: float = 0.008,
    d_edge_factor: float = 2.5,
    rdp_epsilon_prop: float = 0.001,
    whisker_prop: float = 0.010,
    min_comp_prop: float = 0.015,
    thinning_method: str = "lee",
    binary_save_path: str | None = None,
    skeleton_save_path: str | None = None,
    junction_save_path: str | None = None,
    raw_graph_save_path: str | None = None,
    rdp_save_path: str | None = None,
) -> tuple[nx.MultiGraph, np.ndarray, float, float]:
    """
    Executes the full Step 1 pipeline (Report Section 2.1):
      1. Binarisation at T_bin = 128 and active bounding-box cropping (diagonal D).
      2. Medial-axis thinning via Lee's algorithm.
      3. Rutovitz crossing-number classification X(p) and medoid clustering (0.008 * D).
      4. Lossless 8-connected tracing and RDP decimation (epsilon_RDP = 0.001 * D).
      5. Close-junction contraction (0.02 * D) and whisker/noise pruning (0.01 * D, 0.015 * D).

    Args:
        image_path: Path to input sketch image.
        input_stroke_width: Estimated stroke thickness in pixels.
        target_stroke_width: Working stroke thickness before thinning.
        binarise_thresh: Grayscale binarisation threshold T_bin.
        d_euclid_prop: Vertex merge radius as a fraction of diagonal D (0.008).
        d_edge_factor: Multiplier on d_euclid for junction contraction (2.5 -> 0.020 * D).
        rdp_epsilon_prop: RDP tolerance as a fraction of diagonal D (0.001).
        whisker_prop: Minimum degree-1 spur length as a fraction of D (0.010).
        min_comp_prop: Minimum connected-component arc length as a fraction of D (0.015).
        thinning_method: Thinning algorithm ('lee' or 'zhang').
        binary_save_path: Optional output PNG path for binary mask.
        skeleton_save_path: Optional output PNG path for thinned skeleton.
        junction_save_path: Optional output PNG path for Rutovitz node plot.
        raw_graph_save_path: Optional output PNG path for raw traced graph.
        rdp_save_path: Optional output PNG path for RDP-simplified graph.

    Returns:
        Tuple of (refined_base_graph, skeleton, canvas_diagonal_D, scale_factor).
    """
    binary_mask, scale_factor, canvas_diag = load_and_scale_image(
        image_path=image_path,
        input_stroke_width=input_stroke_width,
        target_stroke_width=target_stroke_width,
        binarise_thresh=binarise_thresh,
    )
    if binary_save_path:
        save_binary_or_skeleton_image(binary_mask, binary_save_path, label="binary mask")

    skeleton = run_thinning(binary_mask, method=thinning_method)
    if skeleton_save_path:
        save_skeleton_image(skeleton, skeleton_save_path)

    d_euclid = d_euclid_prop * canvas_diag
    d_edge_thresh = d_euclid * d_edge_factor
    rdp_epsilon = rdp_epsilon_prop * canvas_diag
    whisker_thresh = whisker_prop * canvas_diag
    min_comp_len = min_comp_prop * canvas_diag

    print(f"Canvas Diagonal D = {canvas_diag:.1f}px | Scale factor = {scale_factor:.2f}x")
    print(
        f"Node merge d_euclid = {d_euclid:.2f}px | "
        f"Junction contraction d_edge = {d_edge_thresh:.2f}px | "
        f"RDP eps = {rdp_epsilon:.2f}px"
    )

    crossing_number = compute_rutovitz_crossing_number(skeleton)
    nodes, node_types, node_grid, node_members = extract_canonical_nodes(
        skeleton, crossing_number, merge_dist_work=d_euclid / scale_factor
    )
    if junction_save_path:
        plot_junction_classification(skeleton, nodes, node_types, scale_factor, junction_save_path)

    raw_graph = trace_skeleton_to_graph(
        skeleton, nodes, node_types, node_grid, node_members, scale_factor
    )
    if raw_graph_save_path:
        plot_graph_snapshot(raw_graph, raw_graph_save_path, label="raw traced graph")

    rdp_graph = apply_rdp_to_graph(raw_graph, epsilon=rdp_epsilon)
    if rdp_save_path:
        plot_graph_snapshot(rdp_graph, rdp_save_path, label="RDP simplified graph")

    contracted_graph = merge_nearby_junctions(rdp_graph, d_edge_thresh=d_edge_thresh)
    final_graph = prune_whiskers_and_noise(
        contracted_graph, whisker_thresh=whisker_thresh, min_comp_len=min_comp_len
    )
    return final_graph, skeleton, canvas_diag, scale_factor
