import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import cv2
import networkx as nx
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


# =============================================================================
# 1. IMAGE & DIAGNOSTIC EXPORT UTILITIES
# =============================================================================

def save_mask_as_image(mask_array: np.ndarray, output_path: str, invert: bool = True):
    """
    Saves a boolean or uint8 binary array as an 8-bit greyscale PNG.
    """
    preview = np.where(mask_array > 0, 0 if invert else 255, 255 if invert else 0).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(output_path)
    print(f"Saved: {output_path}")


def save_crossing_map(skeleton: np.ndarray, crossing_number: np.ndarray, output_path: str):
    """
    Saves an RGB diagnostic map of the raw skeletonised sketch on a pure white background:
      - Line strokes (CN == 2): Blue (0, 0, 255)
      - Endpoints (CN == 1): Green (0, 200, 0)
      - Raw Junction Candidates (CN >= 3): Red (255, 0, 0)
      - Isolated single-pixel dots (CN == 0 on skeleton): Magenta (255, 0, 255)
    """
    H, W = skeleton.shape
    rgb_map = np.full((H, W, 3), 255, dtype=np.uint8)
    active = skeleton > 0

    rgb_map[active & (crossing_number == 2)] = [0, 0, 255]
    rgb_map[active & (crossing_number == 1)] = [0, 200, 0]
    rgb_map[active & (crossing_number >= 3)] = [255, 0, 0]
    rgb_map[active & (crossing_number == 0)] = [255, 0, 255]

    Image.fromarray(rgb_map, mode="RGB").save(output_path)
    print(f"Saved raw crossing map: {output_path}")


def save_nodes_map(skeleton: np.ndarray, nodes: dict, node_types: dict, output_path: str):
    """
    Saves an RGB map showing the skeleton in Blue and canonical collapsed nodes as 3x3 markers:
      - Canonical Junction Nodes: Red 3x3 markers (255, 0, 0)
      - Canonical Endpoint Nodes: Green 3x3 markers (0, 200, 0)
    """
    H, W = skeleton.shape
    rgb_map = np.full((H, W, 3), 255, dtype=np.uint8)
    rgb_map[skeleton > 0] = [0, 0, 255]

    for node_id, (r, c) in nodes.items():
        color = [255, 0, 0] if node_types[node_id] == "junction" else [0, 200, 0]
        rgb_map[max(0, r - 1):min(H, r + 2), max(0, c - 1):min(W, c + 2)] = color

    Image.fromarray(rgb_map, mode="RGB").save(output_path)
    print(f"Saved collapsed canonical nodes map: {output_path}")


def save_graph_map(graph: nx.MultiGraph, width: int, height: int, output_path: str, show_split_nodes: bool = False):
    """
    Renders a topological MultiGraph as a raster PNG:
      - Regular edge polylines: Blue (0, 0, 255)
      - Bridge edges (is_bridge == True): Magenta (255, 0, 255)
      - Endpoints (deg == 1): Green 3x3 markers (0, 200, 0)
      - Junctions (deg >= 3): Red 3x3 markers (255, 0, 0)
    """
    rgb_map = np.full((height, width, 3), 255, dtype=np.uint8)

    # 1. Rasterize regular edges first (Blue), then bridges on top (Magenta)
    for is_bridge_pass in (False, True):
        for u, v, k, data in graph.edges(keys=True, data=True):
            edge_is_bridge = data.get("is_bridge", False)
            if edge_is_bridge != is_bridge_pass:
                continue
            pts = data.get("pts")
            if pts is None or len(pts) < 2:
                continue
            pts_xy = np.round(pts[:, [1, 0]]).astype(np.int32).reshape((-1, 1, 2))
            color = (255, 0, 255) if edge_is_bridge else (0, 0, 255)
            thickness = 2 if edge_is_bridge else 1
            cv2.polylines(rgb_map, [pts_xy], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_8)

    # 2. Draw 3x3 node markers
    for target_pass in ("continuation", "split", "endpoint", "junction"):
        for nid, data in graph.nodes(data=True):
            pos = data.get("pos")
            if pos is None:
                continue
            r, c = int(round(pos[0])), int(round(pos[1]))
            deg = graph.degree(nid)
            kind = data.get("kind", "")

            if target_pass == "continuation" and deg == 2 and not (show_split_nodes and kind in ("corner", "subdivision")):
                color = [0, 0, 255]
            elif target_pass == "split" and deg == 2 and show_split_nodes and kind in ("corner", "subdivision"):
                color = [255, 140, 0]
            elif target_pass == "endpoint" and deg == 1:
                color = [0, 200, 0]
            elif target_pass == "junction" and deg >= 3:
                color = [255, 0, 0]
            else:
                continue

            rgb_map[max(0, r - 1):min(height, r + 2), max(0, c - 1):min(width, c + 2)] = color

    Image.fromarray(rgb_map, mode="RGB").save(output_path)
    print(f"Saved graph map: {output_path}")

def save_graph_as_svg(graph: nx.MultiGraph, width: int, height: int, output_path: str):
    """
    Exports all edge polylines from a networkx.MultiGraph into a scalable vector graphics (SVG) file.
    Converts matrix indices (row, col) into Cartesian SVG coordinates (x=col, y=row).
    """
    svg_root = ET.Element("svg", {
        "version": "1.1",
        "xmlns": "http://www.w3.org/2000/svg",
        "width": str(width),
        "height": str(height),
        "viewBox": f"0 0 {width} {height}",
    })
    group = ET.SubElement(svg_root, "g", {
        "fill": "none",
        "stroke": "#000000",
        "stroke-width": "1.5",
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
    })

    for u, v, k, data in graph.edges(keys=True, data=True):
        pts = data.get("pts")
        if pts is None or len(pts) < 2:
            continue
        coord_strings = [f"{float(c):.2f},{float(r):.2f}" for r, c in pts]
        ET.SubElement(group, "polyline", {"points": " ".join(coord_strings), "id": f"edge_{u}_{v}_{k}"})

    pretty_xml = minidom.parseString(ET.tostring(svg_root, encoding="utf-8")).toprettyxml(indent="  ")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)
    print(f"Saved SVG: {output_path}")


# =============================================================================
# 2. IMAGE PREPROCESSING & SKELETONISATION
# =============================================================================

def import_image_as_mask(path: str, config: dict):
    """
    Imports a PNG image, thresholds it to binary, crops to tight active bounds,
    computes the true drawing dimensions, and pads with a 1-pixel False boundary.
    """
    threshold = config.get("intensity_threshold", 128)

    with Image.open(path) as img:
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba_img = img.convert("RGBA")
            white_canvas = Image.new("RGBA", rgba_img.size, (255, 255, 255, 255))
            grey_img = Image.alpha_composite(white_canvas, rgba_img).convert("L")
        else:
            grey_img = img.convert("L")

        binary_mask = np.array(grey_img, dtype=np.uint8) < threshold

    active_coords = np.argwhere(binary_mask)
    if len(active_coords) == 0:
        print("Warning: Input image contains no active foreground pixels!")
        return np.pad(binary_mask, pad_width=1, mode="constant", constant_values=False), (0, 0, 0.0)

    r_min, c_min = active_coords.min(axis=0)
    r_max, c_max = active_coords.max(axis=0) + 1
    cropped_mask = binary_mask[r_min:r_max, c_min:c_max]

    h_draw, w_draw = cropped_mask.shape
    diagonal = float(np.hypot(h_draw, w_draw))
    print(f"Tight drawing bounds: {w_draw}x{h_draw} px (Diagonal: {diagonal:.1f} px).")

    padded_mask = np.pad(cropped_mask, pad_width=1, mode="constant", constant_values=False)
    return padded_mask, (h_draw, w_draw, diagonal)


def skeletonise_mask(mask: np.ndarray, config: dict) -> np.ndarray:
    """
    Skeletonises the binary mask using Lee's medial axis thinning method to preserve homotopy.
    Normalized to binary {0, 1} uint8 across all scikit-image versions.
    """
    mode = config.get("skeleton_mode", "lee")
    before_pixels = np.count_nonzero(mask)
    skeleton = (skeletonize(mask, method=mode) > 0).astype(np.uint8)
    after_pixels = np.count_nonzero(skeleton)

    remaining = (after_pixels / before_pixels) if before_pixels > 0 else 0.0
    print(f"Skeletonised via method '{mode}': {remaining * 100:.2f}% remaining pixels ({after_pixels} px).")
    return skeleton


def skeleton_to_crossing_numbers(skeleton: np.ndarray) -> np.ndarray:
    """
    Computes the Rutovitz Crossing Number (CN) across the 2D {0, 1} skeleton array.
    """
    skel_bin = (skeleton > 0).astype(np.int8)
    padded = np.pad(skel_bin, pad_width=1, mode="constant", constant_values=0)

    p_ring = [
        padded[:-2, 1:-1],  # Top
        padded[:-2, 2:],    # Top-Right
        padded[1:-1, 2:],   # Right
        padded[2:, 2:],     # Bottom-Right
        padded[2:, 1:-1],   # Bottom
        padded[2:, :-2],    # Bottom-Left
        padded[1:-1, :-2],  # Left
        padded[:-2, :-2],   # Top-Left
    ]

    diff_sum = sum(np.abs(p_ring[(i + 1) % 8] - p_ring[i]) for i in range(8))
    return (0.5 * diff_sum * skel_bin).astype(np.uint8)


def extract_canonical_nodes(skeleton: np.ndarray, crossing_number: np.ndarray, config: dict):
    """
    Stage 1 Node Extraction:
    Collapses contiguous junction clusters (CN >= 3) into medoids, registers endpoints (CN == 1),
    and merges nodes closer than config['node_merge_distance'] into canonical nodes.
    """
    H, W = skeleton.shape
    node_grid = np.full((H, W), fill_value=-1, dtype=np.int32)
    raw_nodes, raw_node_types, node_pixel_members = {}, {}, {}
    current_node_id = 0

    # 1. Process Endpoints (CN == 1)
    endpoint_coords = np.argwhere((skeleton > 0) & (crossing_number == 1))
    for r, c in endpoint_coords:
        raw_nodes[current_node_id] = (float(r), float(c))
        raw_node_types[current_node_id] = "endpoint"
        node_pixel_members[current_node_id] = [(int(r), int(c))]
        current_node_id += 1

    # 2. Cluster & Collapse Contiguous Junction Blobs (CN >= 3)
    junction_mask = (skeleton > 0) & (crossing_number >= 3)
    labeled_blobs, num_blobs = ndimage.label(junction_mask, structure=np.ones((3, 3), dtype=bool))

    for blob_idx in range(1, num_blobs + 1):
        blob_pixels = np.argwhere(labeled_blobs == blob_idx)
        centroid = blob_pixels.mean(axis=0)
        best_px = blob_pixels[np.argmin(np.sum((blob_pixels - centroid) ** 2, axis=1))]

        raw_nodes[current_node_id] = (float(best_px[0]), float(best_px[1]))
        raw_node_types[current_node_id] = "junction"
        node_pixel_members[current_node_id] = [(int(r), int(c)) for r, c in blob_pixels]
        current_node_id += 1

    # 3. Spatial Node Merging via KD-Tree
    merge_dist = float(config.get("node_merge_distance", 0.0))
    final_nodes, final_node_types, final_node_members = {}, {}, {}

    if len(raw_nodes) > 0 and merge_dist > 0.0:
        raw_ids = list(raw_nodes.keys())
        coords = np.array([raw_nodes[nid] for nid in raw_ids])
        pairs = cKDTree(coords).query_pairs(r=merge_dist)

        merge_graph = nx.Graph()
        merge_graph.add_nodes_from(raw_ids)
        merge_graph.add_edges_from((raw_ids[i], raw_ids[j]) for i, j in pairs)

        for new_node_id, cluster in enumerate(nx.connected_components(merge_graph)):
            combined_pixels = [px for nid in cluster for px in node_pixel_members[nid]]
            is_junction = any(raw_node_types[nid] == "junction" for nid in cluster)

            pixel_arr = np.array(combined_pixels)
            centroid = pixel_arr.mean(axis=0)
            best_pixel = pixel_arr[np.argmin(np.sum((pixel_arr - centroid) ** 2, axis=1))]

            final_nodes[new_node_id] = (int(best_pixel[0]), int(best_pixel[1]))
            final_node_types[new_node_id] = "junction" if is_junction else "endpoint"
            final_node_members[new_node_id] = combined_pixels

            for r, c in combined_pixels:
                node_grid[r, c] = new_node_id

        print(f"Extracted {len(endpoint_coords)} endpoints & {num_blobs} junction blobs. "
              f"Merged within {merge_dist:.2f}px -> {len(final_nodes)} canonical nodes.")
    else:
        for nid, (r, c) in raw_nodes.items():
            final_nodes[nid] = (int(r), int(c))
            final_node_types[nid] = raw_node_types[nid]
            final_node_members[nid] = node_pixel_members[nid]
            for pr, pc in node_pixel_members[nid]:
                node_grid[pr, pc] = nid

    return final_nodes, final_node_types, node_grid, final_node_members


# =============================================================================
# 3. ROBUST EDGE TRACING, DOT CLUSTERING & GRAPH CONSTRUCTION
# =============================================================================

def get_cluster_path(start: tuple, end: tuple, valid_pixels: set) -> list:
    """
    Finds the shortest 8-connected path between two pixels within a node cluster using BFS.
    """
    if start == end:
        return [start]
    queue = [start]
    parent = {start: None}

    while queue:
        curr = queue.pop(0)
        if curr == end:
            path = []
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            return path[::-1]
        for dr, dc in OFFSETS:
            nbr = (curr[0] + dr, curr[1] + dc)
            if nbr in valid_pixels and nbr not in parent:
                parent[nbr] = curr
                queue.append(nbr)
    return [start, end]


def _clean_pts(path: list) -> np.ndarray:
    """Removes consecutive duplicate points and converts to a float32 (N, 2) array."""
    clean = [path[0]]
    for pt in path[1:]:
        if pt[0] != clean[-1][0] or pt[1] != clean[-1][1]:
            clean.append(pt)
    return np.array(clean, dtype=np.float32)


def trace_skeleton_to_graph(skeleton: np.ndarray, nodes: dict, node_grid: np.ndarray,
                            node_pixel_members: dict) -> nx.MultiGraph:
    """
    Traces 8-connected chains of interior pixels between canonical nodes to construct a MultiGraph.
    Checks `if node_nbrs:` BEFORE `elif path_nbrs:` so traced paths never jump diagonally across junctions.
    """
    G = nx.MultiGraph()
    for nid, pos in nodes.items():
        G.add_node(nid, pos=pos)

    H, W = skeleton.shape
    visited = np.zeros_like(skeleton, dtype=bool)
    direct_edges_seen = set()

    # Pass 1: Trace paths leaving each canonical node
    for u_id, u_pos in list(nodes.items()):
        u_pixels = node_pixel_members.get(u_id, [u_pos])
        u_pixel_set = set(u_pixels)

        for p in u_pixels:
            pr, pc = p
            for dr, dc in OFFSETS:
                nr, nc = pr + dr, pc + dc
                if not (0 <= nr < H and 0 <= nc < W) or not skeleton[nr, nc]:
                    continue

                nbr_grid = int(node_grid[nr, nc])

                # Case A: Neighbour is an unvisited interior stroke pixel
                if nbr_grid == -1 and not visited[nr, nc]:
                    path = list(get_cluster_path(u_pos, p, u_pixel_set))
                    curr_r, curr_c = nr, nc
                    target_node_id = -1
                    target_pixel = None
                    steps = 0

                    while True:
                        path.append((curr_r, curr_c))
                        visited[curr_r, curr_c] = True
                        steps += 1

                        path_nbrs = []
                        node_nbrs = []

                        for s_dr, s_dc in OFFSETS:
                            sr, sc = curr_r + s_dr, curr_c + s_dc
                            if 0 <= sr < H and 0 <= sc < W and skeleton[sr, sc]:
                                s_grid = int(node_grid[sr, sc])
                                if s_grid == -1 and not visited[sr, sc]:
                                    path_nbrs.append((sr, sc))
                                elif s_grid != -1:
                                    if s_grid != u_id or steps > 2:
                                        node_nbrs.append(((sr, sc), s_grid))

                        if node_nbrs:
                            node_nbrs.sort(key=lambda item: item[1] == u_id)
                            target_pixel, target_node_id = node_nbrs[0]
                            break
                        elif path_nbrs:
                            path_nbrs.sort(key=lambda pt: abs(pt[0] - curr_r) + abs(pt[1] - curr_c))
                            curr_r, curr_c = path_nbrs[0]
                        else:
                            break

                    if target_node_id != -1 and target_pixel is not None:
                        v_pos = nodes[target_node_id]
                        v_pixel_set = set(node_pixel_members.get(target_node_id, [v_pos]))
                        path.extend(get_cluster_path(target_pixel, v_pos, v_pixel_set))
                        pts_array = _clean_pts(path)
                        if len(pts_array) >= 2:
                            G.add_edge(u_id, target_node_id, pts=pts_array)

                # Case B: Directly adjacent canonical node clusters with no intermediate path pixels
                elif nbr_grid != -1 and nbr_grid != u_id:
                    v_id = nbr_grid
                    edge_key = (min(u_id, v_id), max(u_id, v_id))
                    if edge_key not in direct_edges_seen:
                        direct_edges_seen.add(edge_key)
                        v_pos = nodes[v_id]
                        v_pixel_set = set(node_pixel_members.get(v_id, [v_pos]))
                        full_pts = get_cluster_path(u_pos, p, u_pixel_set) + get_cluster_path((nr, nc), v_pos, v_pixel_set)
                        pts_array = _clean_pts(full_pts)
                        if len(pts_array) >= 2:
                            G.add_edge(u_id, v_id, pts=pts_array)

    # Pass 2: Trace isolated closed loops (e.g. letter 'O')
    unvisited_loop_pixels = np.argwhere((skeleton > 0) & (~visited) & (node_grid == -1))
    for start_r, start_c in unvisited_loop_pixels:
        if visited[start_r, start_c]:
            continue
        loop_node_id = max(G.nodes, default=-1) + 1
        canonical_pos = (int(start_r), int(start_c))
        G.add_node(loop_node_id, pos=canonical_pos)

        path = [canonical_pos]
        curr_r, curr_c = int(start_r), int(start_c)
        visited[curr_r, curr_c] = True

        while True:
            nbrs = [
                (curr_r + dr, curr_c + dc)
                for dr, dc in OFFSETS
                if 0 <= curr_r + dr < H and 0 <= curr_c + dc < W
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
            path.append(canonical_pos)
            G.add_edge(loop_node_id, loop_node_id, pts=_clean_pts(path))
        else:
            G.remove_node(loop_node_id)

    print(f"Traced raster skeleton into graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def combine_nearby_dots(graph: nx.MultiGraph, skeleton: np.ndarray, crossing_number: np.ndarray,
                        merge_distance: float) -> nx.MultiGraph:
    """
    Combines isolated dots (CN == 0 on skeleton) and tiny micro-stroke components (total length <= 3px)
    that lie within `merge_distance` of each other by connecting their actual coordinates directly
    into a closed polygonal loop before small-component pruning.
    Single isolated noise dots that do not cluster with nearby dots are removed.
    """
    H = graph.copy()
    dot_coords = [(float(r), float(c)) for r, c in np.argwhere((skeleton > 0) & (crossing_number == 0))]
    micro_comp_nodes = []

    for comp in list(nx.connected_components(H)):
        sub_edges = list(H.edges(comp, keys=True, data=True))
        comp_len = sum(compute_polyline_length(d["pts"]) for _, _, _, d in sub_edges)
        if comp_len <= 3.0:
            pts = np.array([H.nodes[n]["pos"] for n in comp], dtype=np.float64)
            center = pts.mean(axis=0)
            dot_coords.append((float(center[0]), float(center[1])))
            micro_comp_nodes.append(comp)

    if not dot_coords:
        return H

    # Remove the micro-components from H; clustered ones will be connected into a closed loop
    for comp in micro_comp_nodes:
        H.remove_nodes_from(comp)

    coords_arr = np.array(dot_coords, dtype=np.float64)
    dot_graph = nx.Graph()
    dot_graph.add_nodes_from(range(len(dot_coords)))
    if len(dot_coords) > 1 and merge_distance > 0.0:
        pairs = cKDTree(coords_arr).query_pairs(r=merge_distance)
        dot_graph.add_edges_from(pairs)

    next_nid = max(H.nodes, default=-1) + 1
    combined_loops = 0

    for cluster in nx.connected_components(dot_graph):
        if len(cluster) < 2:
            # Single isolated dot with no nearby dots -> leave removed as noise
            continue

        cluster_pts = coords_arr[list(cluster)]
        centroid = cluster_pts.mean(axis=0)

        # Sort the actual dot coordinates angularly around their centroid to form a non-self-intersecting loop
        angles = np.arctan2(cluster_pts[:, 0] - centroid[0], cluster_pts[:, 1] - centroid[1])
        ordered_pts = cluster_pts[np.argsort(angles)]

        # Close the polygonal loop by appending the first point at the end
        loop_pts = np.vstack((ordered_pts, ordered_pts[0])).astype(np.float32)
        loop_pts = _clean_pts(loop_pts.tolist())

        if len(loop_pts) >= 2:
            anchor_pos = (int(round(loop_pts[0, 0])), int(round(loop_pts[0, 1])))
            H.add_node(next_nid, pos=anchor_pos)
            H.add_edge(next_nid, next_nid, pts=loop_pts)
            next_nid += 1
            combined_loops += 1

    if combined_loops > 0:
        print(f"Connected {combined_loops} cluster(s) of nearby dots into closed polygonal loops.")
    return H


# =============================================================================
# 4. RDP SIMPLIFICATION, CONTRACTION, PRUNING & CORNER/EDGE SPLITTING
# =============================================================================

def rdp_simplify_polyline(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Simplifies a 2D polyline using the recursive Ramer–Douglas–Peucker algorithm.
    """
    if len(points) <= 2 or epsilon <= 0.0:
        return points

    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = float(np.linalg.norm(line_vec))

    if line_len == 0.0:
        dists = np.linalg.norm(points[1:-1] - start, axis=1)
    else:
        line_unit = line_vec / line_len
        vecs_to_pts = points[1:-1] - start
        perp_vecs = vecs_to_pts - np.outer(np.dot(vecs_to_pts, line_unit), line_unit)
        dists = np.linalg.norm(perp_vecs, axis=1)

    max_dist_idx = int(np.argmax(dists)) + 1
    if dists[max_dist_idx - 1] > epsilon:
        left = rdp_simplify_polyline(points[:max_dist_idx + 1], epsilon)
        right = rdp_simplify_polyline(points[max_dist_idx:], epsilon)
        return np.vstack((left[:-1], right)).astype(np.float32)
    return np.array([start, end], dtype=np.float32)


def apply_rdp_to_graph(graph: nx.MultiGraph, epsilon: float) -> nx.MultiGraph:
    """
    Creates a copy of the graph with all edge polylines simplified via RDP.
    Splits self-loops in half before RDP so closed loops never collapse to a single point.
    """
    simplified_G = graph.copy()
    for u, v, k, data in simplified_G.edges(keys=True, data=True):
        pts = data["pts"]
        if u == v and len(pts) > 4:
            mid = len(pts) // 2
            left = rdp_simplify_polyline(pts[:mid + 1], epsilon)
            right = rdp_simplify_polyline(pts[mid:], epsilon)
            data["pts"] = np.vstack((left[:-1], right)).astype(np.float32)
        else:
            data["pts"] = rdp_simplify_polyline(pts, epsilon=epsilon)
    return simplified_G


def compute_polyline_length(points: np.ndarray) -> float:
    """Calculates the cumulative Euclidean arc-length along an (N, 2) polyline."""
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def dissolve_degree_two_nodes(graph: nx.MultiGraph) -> nx.MultiGraph:
    """
    Dissolves non-loop degree-2 continuation nodes by concatenating their two incident polylines.
    Ensures that when a whisker or triangle collapses, no artificial degree-2 nodes remain.
    """
    H = graph.copy()
    changed = True
    while changed:
        changed = False
        for node in list(H.nodes()):
            if H.degree(node) != 2:
                continue
            edges = list(H.edges(node, keys=True, data=True))
            if len(edges) != 2 or any(u == v for u, v, _, _ in edges):
                continue

            (u1, v1, k1, d1), (u2, v2, k2, d2) = edges
            nbr1 = v1 if u1 == node else u1
            nbr2 = v2 if u2 == node else u2
            node_pos = np.array(H.nodes[node]["pos"], dtype=np.float32)

            pts1 = d1["pts"]
            if np.linalg.norm(pts1[-1] - node_pos) > np.linalg.norm(pts1[0] - node_pos):
                pts1 = pts1[::-1]
            pts2 = d2["pts"]
            if np.linalg.norm(pts2[0] - node_pos) > np.linalg.norm(pts2[-1] - node_pos):
                pts2 = pts2[::-1]

            merged_pts = np.vstack((pts1[:-1], pts2)).astype(np.float32)
            H.remove_edge(u1, v1, key=k1)
            H.remove_edge(u2, v2, key=k2)
            H.remove_node(node)
            H.add_edge(nbr1, nbr2, pts=merged_pts)
            changed = True
            break
    return H


def contract_short_edges(graph: nx.MultiGraph, min_edge_length: float, prune_whiskers: bool = True) -> nx.MultiGraph:
    """
    Stage 2 Topological Contraction:
      1. Prunes short dead-end whiskers (deg == 1 attached to deg >= 3 with length < min_edge_length).
      2. Finds connected components (clusters, pairs, triangles, etc.) of junctions connected by
         edges shorter than `min_edge_length` and collapses each entire cluster simultaneously
         to its geometric centroid, discarding all internal short edges.
    """
    H = dissolve_degree_two_nodes(graph)

    # Phase 1: Whisker Pruning (Dead-ends below length threshold)
    if prune_whiskers:
        changed = True
        while changed:
            changed = False
            for u, v, k, data in list(H.edges(keys=True, data=True)):
                if u != v and ((H.degree(u) == 1 and H.degree(v) >= 3) or (H.degree(v) == 1 and H.degree(u) >= 3)):
                    if compute_polyline_length(data["pts"]) < min_edge_length:
                        endpoint_node = u if H.degree(u) == 1 else v
                        H.remove_edge(u, v, key=k)
                        H.remove_node(endpoint_node)
                        H = dissolve_degree_two_nodes(H)
                        changed = True
                        break

    # Phase 2: Cluster-to-Centroid Contraction of Nearby Junctions (including triangles)
    while True:
        short_graph = nx.Graph()
        for u, v, k, data in H.edges(keys=True, data=True):
            if u != v and H.degree(u) >= 2 and H.degree(v) >= 2:
                if compute_polyline_length(data["pts"]) < min_edge_length:
                    short_graph.add_edge(u, v)

        clusters = [list(comp) for comp in nx.connected_components(short_graph) if len(comp) > 1]
        if not clusters:
            break

        for cluster_nodes in clusters:
            cluster_set = set(cluster_nodes)
            coords = np.array([H.nodes[nid]["pos"] for nid in cluster_nodes], dtype=np.float64)
            centroid_pos = tuple(np.round(coords.mean(axis=0)).astype(int))
            centroid_vec = np.array(centroid_pos, dtype=np.float32)

            rep = cluster_nodes[0]
            H.nodes[rep]["pos"] = centroid_pos

            seen_edges = set()
            to_remove, to_add = [], []

            for u, v, k, data in list(H.edges(cluster_nodes, keys=True, data=True)):
                edge_id = (min(u, v), max(u, v), k) if u != v else (u, v, k)
                if edge_id in seen_edges:
                    continue
                seen_edges.add(edge_id)

                pts = data["pts"].copy()
                u_in, v_in = u in cluster_set, v in cluster_set
                to_remove.append((u, v, k))

                if u_in and v_in:
                    # Discard internal short edges of the cluster/triangle; keep long loops
                    if compute_polyline_length(pts) >= min_edge_length and len(pts) > 2:
                        pts[0] = centroid_vec
                        pts[-1] = centroid_vec
                        to_add.append((rep, rep, _clean_pts(pts.tolist())))
                else:
                    inner = u if u_in else v
                    outer = v if u_in else u
                    inner_pos = np.array(H.nodes[inner]["pos"], dtype=np.float32)
                    outer_pos = np.array(H.nodes[outer]["pos"], dtype=np.float32)

                    d_start = np.linalg.norm(pts[0] - inner_pos) + np.linalg.norm(pts[-1] - outer_pos)
                    d_end = np.linalg.norm(pts[-1] - inner_pos) + np.linalg.norm(pts[0] - outer_pos)
                    if d_start <= d_end:
                        pts[0] = centroid_vec
                    else:
                        pts[-1] = centroid_vec

                    to_add.append((rep, outer, _clean_pts(pts.tolist())))

            for u, v, k in to_remove:
                if H.has_edge(u, v, key=k):
                    H.remove_edge(u, v, key=k)
            for u_new, v_new, pts_new in to_add:
                if len(pts_new) >= 2:
                    H.add_edge(u_new, v_new, pts=pts_new)
            for nid in cluster_nodes[1:]:
                if nid in H:
                    H.remove_node(nid)

        H = dissolve_degree_two_nodes(H)

    for nid in [n for n in list(H.nodes()) if H.degree(n) == 0]:
        H.remove_node(nid)

    return H


def prune_small_components(graph: nx.MultiGraph, min_component_ratio: float = 0.01) -> nx.MultiGraph:
    """
    Removes isolated disconnected components whose cumulative edge length is less than
    (min_component_ratio * total_length).
    """
    H = graph.copy()
    total_length = sum(compute_polyline_length(d["pts"]) for _, _, _, d in H.edges(keys=True, data=True))
    if total_length == 0.0:
        return H

    min_length_threshold = min_component_ratio * total_length
    pruned_count = 0

    for comp in list(nx.connected_components(H)):
        comp_length = sum(compute_polyline_length(d["pts"]) for _, _, _, d in H.edges(comp, keys=True, data=True))
        if comp_length < min_length_threshold:
            H.remove_nodes_from(comp)
            pruned_count += 1

    if pruned_count > 0:
        print(f"Pruned {pruned_count} small disconnected component(s) (< {min_length_threshold:.1f} px).")
    return H


def split_edges_at_sharp_corners(graph: nx.MultiGraph, max_turn_angle_deg: float = 65.0) -> nx.MultiGraph:
    """
    Splits edges at sharp corners into multiple edges connected by new degree-2 corner vertices.
    Because polylines have already been simplified via RDP (where every vertex is a true structural
    inflection point and micro-staircasing is gone), a 1-step adjacent segment deflection angle
    between (P[i] - P[i-1]) and (P[i+1] - P[i]) is exact and avoids smoothing away tight Z-bends.
    """
    if max_turn_angle_deg <= 0.0 or max_turn_angle_deg >= 180.0:
        return graph.copy()

    H = graph.copy()
    next_nid = max(H.nodes, default=-1) + 1
    split_count = 0

    for u, v, k, data in list(H.edges(keys=True, data=True)):
        pts = data.get("pts")
        if pts is None or len(pts) < 3:
            continue

        # Orient pts from u -> v so split chains connect in the right topological order
        pos_u = np.array(H.nodes[u]["pos"], dtype=np.float32)
        pos_v = np.array(H.nodes[v]["pos"], dtype=np.float32)
        if u != v and np.linalg.norm(pts[0] - pos_u) + np.linalg.norm(pts[-1] - pos_v) > np.linalg.norm(pts[-1] - pos_u) + np.linalg.norm(pts[0] - pos_v):
            pts = pts[::-1].copy()

        split_indices = [0]
        for i in range(1, len(pts) - 1):
            v_in = pts[i] - pts[i - 1]
            v_out = pts[i + 1] - pts[i]
            n_in, n_out = float(np.linalg.norm(v_in)), float(np.linalg.norm(v_out))
            if n_in < 1e-6 or n_out < 1e-6:
                continue

            cos_theta = float(np.clip(np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0))
            turn_deg = float(np.degrees(np.arccos(cos_theta)))
            if turn_deg >= max_turn_angle_deg:
                split_indices.append(i)

        if len(split_indices) == 1:
            continue

        split_indices.append(len(pts) - 1)
        H.remove_edge(u, v, key=k)

        curr_u = u
        for idx_pos in range(len(split_indices) - 1):
            i_start = split_indices[idx_pos]
            i_end = split_indices[idx_pos + 1]
            sub_pts = pts[i_start:i_end + 1].copy()

            if idx_pos < len(split_indices) - 2:
                corner_pt = pts[i_end]
                corner_nid = next_nid
                next_nid += 1
                H.add_node(corner_nid, pos=(int(round(corner_pt[0])), int(round(corner_pt[1]))), kind="corner")
                H.add_edge(curr_u, corner_nid, pts=sub_pts)
                curr_u = corner_nid
                split_count += 1
            else:
                H.add_edge(curr_u, v, pts=sub_pts)

    if split_count > 0:
        print(f"Split edges at {split_count} sharp corner(s) (turn >= {max_turn_angle_deg:.1f} deg).")
    return H


def subdivide_long_edges(graph: nx.MultiGraph, max_edge_length: float) -> nx.MultiGraph:
    """
    Splits any edge whose cumulative arc-length exceeds `max_edge_length` into `num_segments`
    equal-arc-length sub-edges separated by new degree-2 bridging nodes.
    """
    if max_edge_length <= 0.0:
        return graph.copy()

    H = graph.copy()
    next_nid = max(H.nodes, default=-1) + 1
    added_nodes = 0

    for u, v, k, data in list(H.edges(keys=True, data=True)):
        pts = data.get("pts")
        if pts is None or len(pts) < 2:
            continue

        total_len = compute_polyline_length(pts)
        if total_len <= max_edge_length:
            continue

        pos_u = np.array(H.nodes[u]["pos"], dtype=np.float32)
        pos_v = np.array(H.nodes[v]["pos"], dtype=np.float32)
        if u != v and np.linalg.norm(pts[0] - pos_u) + np.linalg.norm(pts[-1] - pos_v) > np.linalg.norm(pts[-1] - pos_u) + np.linalg.norm(pts[0] - pos_v):
            pts = pts[::-1].copy()

        num_segments = int(np.ceil(total_len / max_edge_length))
        if num_segments <= 1:
            continue

        # Cumulative arc-length parameterization along the polyline
        seg_diffs = np.diff(pts, axis=0)
        seg_lens = np.hypot(seg_diffs[:, 0], seg_diffs[:, 1])
        cum_len = np.concatenate(([0.0], np.cumsum(seg_lens)))

        H.remove_edge(u, v, key=k)
        curr_u = u
        start_dist = 0.0

        for seg_idx in range(1, num_segments + 1):
            end_dist = (seg_idx / num_segments) * total_len if seg_idx < num_segments else total_len

            # Interpolate exact start and end coordinates along the polyline
            r_start = float(np.interp(start_dist, cum_len, pts[:, 0]))
            c_start = float(np.interp(start_dist, cum_len, pts[:, 1]))
            r_end = float(np.interp(end_dist, cum_len, pts[:, 0]))
            c_end = float(np.interp(end_dist, cum_len, pts[:, 1]))

            interior_mask = (cum_len > start_dist + 1e-4) & (cum_len < end_dist - 1e-4)
            interior_pts = pts[interior_mask]

            sub_pts = np.vstack((
                [[r_start, c_start]],
                interior_pts,
                [[r_end, c_end]],
            )).astype(np.float32)

            if seg_idx < num_segments:
                mid_nid = next_nid
                next_nid += 1
                H.add_node(mid_nid, pos=(int(round(r_end)), int(round(c_end))), kind="subdivision")
                H.add_edge(curr_u, mid_nid, pts=sub_pts)
                curr_u = mid_nid
                added_nodes += 1
            else:
                H.add_edge(curr_u, v, pts=sub_pts)

            start_dist = end_dist

    if added_nodes > 0:
        print(f"Subdivided long edges (> {max_edge_length:.1f} px) by inserting {added_nodes} intermediate node(s).")
    return H


def count_graph_points(graph: nx.MultiGraph) -> int:
    """Counts total polyline vertex points across all edges in the graph."""
    return sum(len(data["pts"]) for _, _, _, data in graph.edges(keys=True, data=True))



def get_node_outgoing_tangent(graph: nx.MultiGraph, nid: int) -> np.ndarray | None:
    """
    Computes the unit tangent vector pointing OUT of the stroke at `nid` (specifically for
    endpoints of degree 1). Returns None if `nid` is not an endpoint or has no valid tangent.
    """
    if graph.degree(nid) != 1:
        return None

    node_pos = np.array(graph.nodes[nid]["pos"], dtype=np.float32)
    for u, v, k, data in graph.edges(nid, keys=True, data=True):
        pts = data.get("pts")
        if pts is None or len(pts) < 2:
            continue
        # Orient polyline so pts[-1] is at `nid` and pts[-2] is the interior point before `nid`
        if np.linalg.norm(pts[-1] - node_pos) <= np.linalg.norm(pts[0] - node_pos):
            vec = pts[-1] - pts[-2]
        else:
            vec = pts[0] - pts[1]

        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            return vec / norm
    return None


def compute_bridge_weight(
    u: int,
    v: int,
    pos_u: np.ndarray,
    pos_v: np.ndarray,
    deg_u: int,
    deg_v: int,
    tan_u: np.ndarray | None,
    tan_v: np.ndarray | None,
    alpha_deg: float = 0.15,
    beta_ang: float = 0.50,
    used_node_counts: dict[int, int] | None = None,
    used_node_Pairs: set[tuple[int, int]] | None = None,
    gamma_reuse_mult: float = 1.5,
    lambda_reuse_add: float = 25.0,
    lambda_edge_dup: float = 500.0,
) -> float:
    """
    Evaluates the Kruskal bridge weight between node `u` (in component A) and node `v` (in component B).
    - In Pass 1 (`used_node_counts=None`), computes the base weight W_1(u, v).
    - In Pass 2 (`used_node_counts` provided), applies multiplicative and additive penalties
      if `u` or `v` was already used as a connecting node in Pass 1.
    """
    vec_uv = pos_v - pos_u
    dist = float(np.linalg.norm(vec_uv))

    # 1. Degree penalty: phi_deg(u, v) = max(0, deg(u) - 1) + max(0, deg(v) - 1)
    phi_deg = float(max(0, deg_u - 1) + max(0, deg_v - 1))

    # 2. Angular deflection penalty: phi_ang(u, v) in [0, 2]
    phi_ang = 0.0
    if dist > 1e-6:
        unit_uv = vec_uv / dist
        if tan_u is not None:
            phi_ang += 0.5 * (1.0 - float(np.clip(np.dot(tan_u, unit_uv), -1.0, 1.0)))
        if tan_v is not None:
            phi_ang += 0.5 * (1.0 - float(np.clip(np.dot(tan_v, -unit_uv), -1.0, 1.0)))

    # Base weight W_1(u, v)
    w1 = dist * (1.0 + alpha_deg * phi_deg + beta_ang * phi_ang)

    # If Pass 1 (no prior used nodes), return W_1(u, v)
    if not used_node_counts:
        return w1

    # 3. Pass 2 Node-Reuse Penalties: W_2(u, v)
    k_u = used_node_counts.get(u, 0)
    k_v = used_node_counts.get(v, 0)
    ind_u = 1.0 if k_u > 0 else 0.0
    ind_v = 1.0 if k_v > 0 else 0.0

    w2 = w1 * (1.0 + gamma_reuse_mult * (ind_u + ind_v)) + lambda_reuse_add * (k_u + k_v)

    if used_node_Pairs and (min(u, v), max(u, v)) in used_node_Pairs:
        w2 += lambda_edge_dup

    return w2


def solve_component_mst_kruskal(
    graph: nx.MultiGraph,
    components: list[set[int]],
    alpha_deg: float = 0.15,
    beta_ang: float = 0.50,
    used_node_counts: dict[int, int] | None = None,
    used_node_pairs: set[tuple[int, int]] | None = None,
    gamma_reuse_mult: float = 1.5,
    lambda_reuse_add: float = 25.0,
    k_nearest_candidates: int = 8,
) -> list[tuple[int, int, float]]:
    """
    Constructs a component-level meta-graph and solves for the Minimum Spanning Tree (MST)
    using Kruskal's algorithm (`nx.minimum_spanning_edges(..., algorithm="kruskal")`).

    For each pair of components (C_a, C_b), evaluates candidate node pairs (u in C_a, v in C_b)
    and selects the minimum-weight bridge according to `compute_bridge_weight`.

    Returns
    -------
    mst_bridges : list of (u, v, weight)
        The K - 1 optimal bridging edges connecting all K components into a single tree.
    """
    num_comps = len(components)
    if num_comps <= 1:
        return []

    # Precompute node positions, degrees, and endpoint tangents
    node_pos = {n: np.array(graph.nodes[n]["pos"], dtype=np.float32) for n in graph.nodes()}
    node_deg = {n: graph.degree(n) for n in graph.nodes()}
    node_tan = {n: get_node_outgoing_tangent(graph, n) for n in graph.nodes()}

    comp_node_lists = [list(c) for c in components]
    comp_coords = [np.array([node_pos[n] for n in c_list], dtype=np.float32) for c_list in comp_node_lists]
    comp_trees = [cKDTree(coords) for coords in comp_coords]

    # Build the component-level graph where nodes are component indices 0 .. K-1
    meta_graph = nx.Graph()
    meta_graph.add_nodes_from(range(num_comps))

    for i in range(num_comps):
        nodes_i = comp_node_lists[i]
        coords_i = comp_coords[i]

        for j in range(i + 1, num_comps):
            nodes_j = comp_node_lists[j]
            tree_j = comp_trees[j]

            # Query k-nearest candidate nodes in component j for each node in component i
            # (Checking k_nearest instead of only k=1 ensures that if the closest node in j
            # was heavily penalised in Pass 1, Pass 2 can pick an unpenalised runner-up node!)
            k_query = min(k_nearest_candidates, len(nodes_j))
            _, idxs = tree_j.query(coords_i, k=k_query)
            if k_query == 1:
                idxs = idxs[:, None]

            best_w = float("inf")
            best_pair = (-1, -1)

            for idx_in_i, u in enumerate(nodes_i):
                for col_idx in range(k_query):
                    v = nodes_j[int(idxs[idx_in_i, col_idx])]
                    w = compute_bridge_weight(
                        u=u,
                        v=v,
                        pos_u=node_pos[u],
                        pos_v=node_pos[v],
                        deg_u=node_deg[u],
                        deg_v=node_deg[v],
                        tan_u=node_tan[u],
                        tan_v=node_tan[v],
                        alpha_deg=alpha_deg,
                        beta_ang=beta_ang,
                        used_node_counts=used_node_counts,
                        used_node_Pairs=used_node_pairs,
                        gamma_reuse_mult=gamma_reuse_mult,
                        lambda_reuse_add=lambda_reuse_add,
                    )
                    if w < best_w:
                        best_w = w
                        best_pair = (u, v)

            if best_pair != (-1, -1):
                meta_graph.add_edge(i, j, weight=best_w, u=best_pair[0], v=best_pair[1])

    # Run Kruskal's algorithm on the component meta-graph
    mst_meta_edges = nx.minimum_spanning_edges(meta_graph, algorithm="kruskal", weight="weight", data=True)

    mst_bridges = []
    for _comp_a, _comp_b, data in mst_meta_edges:
        mst_bridges.append((data["u"], data["v"], float(data["weight"])))

    return mst_bridges


def bridge_components_two_pass_kruskal(
    graph: nx.MultiGraph,
    config: dict,
) -> tuple[nx.MultiGraph, list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """
    Executes Kruskal MST twice to bridge disconnected components:
      1. Pass 1 (`mst_1`): Finds the primary Minimum Spanning Tree connecting all components.
      2. Pass 2 (`mst_2`): Solves the MST a second time, penalising any connecting nodes
         that were already used in Pass 1 so the second tree attaches at distinct nodes.
    """
    H = graph.copy()
    components = list(nx.connected_components(H))
    if len(components) <= 1:
        print("Graph is already a single connected component (0 bridges needed).")
        return H, [], []

    alpha_deg = float(config.get("bridge_alpha_deg", 0.15))
    beta_ang = float(config.get("bridge_beta_ang", 0.50))
    gamma_mult = float(config.get("bridge_reuse_gamma", 1.5))
    lambda_add = float(config.get("bridge_reuse_lambda", 25.0))

    # --- PASS 1: Standard Kruskal MST ---
    mst_1 = solve_component_mst_kruskal(
        graph=H,
        components=components,
        alpha_deg=alpha_deg,
        beta_ang=beta_ang,
        used_node_counts=None,
        used_node_pairs=None,
    )

    # Record how many times each node (and node pair) was used in Pass 1
    used_node_counts: dict[int, int] = {}
    used_node_pairs: set[tuple[int, int]] = set()
    for u, v, _w in mst_1:
        used_node_counts[u] = used_node_counts.get(u, 0) + 1
        used_node_counts[v] = used_node_counts.get(v, 0) + 1
        used_node_pairs.add((min(u, v), max(u, v)))

    # --- PASS 2: Penalised Kruskal MST (discouraging nodes in `used_node_counts`) ---
    mst_2 = solve_component_mst_kruskal(
        graph=H,
        components=components,
        alpha_deg=alpha_deg,
        beta_ang=beta_ang,
        used_node_counts=used_node_counts,
        used_node_pairs=used_node_pairs,
        gamma_reuse_mult=gamma_mult,
        lambda_reuse_add=lambda_add,
    )

    # Insert both sets of MST bridge edges into the MultiGraph
    for pass_idx, mst_edges in enumerate((mst_1, mst_2), start=1):
        for u, v, w in mst_edges:
            pos_u = np.array(H.nodes[u]["pos"], dtype=np.float32)
            pos_v = np.array(H.nodes[v]["pos"], dtype=np.float32)
            bridge_pts = np.vstack((pos_u, pos_v)).astype(np.float32)
            H.add_edge(u, v, pts=bridge_pts, is_bridge=True, mst_pass=pass_idx, weight=w)

    reused_nodes = [
        n for u, v, _ in mst_2 for n in (u, v) if n in used_node_counts
    ]
    print(
        f"Bridged {len(components)} components via 2-pass Kruskal MST: "
        f"{len(mst_1)} Pass-1 bridges + {len(mst_2)} Pass-2 bridges "
        f"({len(reused_nodes)}/{2 * len(mst_2)} Pass-2 endpoints reused from Pass 1)."
    )
    return H, mst_1, mst_2
# =============================================================================
# 5. PIPELINE ORCHESTRATION
# =============================================================================

def image_to_oneline(input_image: str, config: dict):
    """
    Executes the complete image processing, graph construction, simplification,
    cluster contraction, dot merging, pruning, corner/long-edge splitting, and export pipeline.
    """
    input_path = Path(input_image)
    input_stem = input_path.stem

    target_dir = Path(config.get("output_dir", input_path.parent)) / input_stem
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory initialized at: {target_dir}")

    # Step 1: Ingest, Crop to Tight Bounds, & Re-Pad
    binary_mask, (h_draw, w_draw, diagonal) = import_image_as_mask(path=input_image, config=config)

    # Resolve proportional vs explicit merge distances for Stage 1 (nodes), Stage 2 (short edges), and Long Edges
    if config.get("node_merge_distance") is None:
        prop_node = config.get("node_merge_ratio", 0.006)
        config["node_merge_distance"] = prop_node * diagonal
        print(f"Adaptive node merge radius: {config['node_merge_distance']:.2f} px ({prop_node*100:.2f}% of diagonal).")

    if config.get("edge_merge_distance") is None:
        prop_edge = config.get("edge_merge_ratio", 0.025)
        config["edge_merge_distance"] = prop_edge * diagonal
        print(f"Adaptive edge contraction threshold: {config['edge_merge_distance']:.2f} px ({prop_edge*100:.2f}% of diagonal).")

    if config.get("max_edge_length") is None:
        prop_max_edge = config.get("max_edge_ratio", 0.20)
        config["max_edge_length"] = prop_max_edge * w_draw
        print(f"Adaptive max edge subdivision length: {config['max_edge_length']:.2f} px ({prop_max_edge*100:.1f}% of drawing width).")

    if config.get("save_binary_mask", True):
        save_mask_as_image(binary_mask, str(target_dir / f"{input_stem}_binary_mask.png"), invert=True)

    # Step 2: Skeletonisation
    skel = skeletonise_mask(mask=binary_mask, config=config)
    if config.get("save_skeleton", True):
        save_mask_as_image(skel, str(target_dir / f"{input_stem}_skeleton.png"), invert=True)

    # Step 3: Rutovitz Crossing Numbers
    cn = skeleton_to_crossing_numbers(skeleton=skel)
    if config.get("save_crossing_map", True):
        save_crossing_map(skel, cn, str(target_dir / f"{input_stem}_crossing_map.png"))

    # Step 4: Extract Canonical Nodes (Blob Collapse & Spatial KD-Tree Merge)
    nodes, node_types, node_grid, node_pixel_members = extract_canonical_nodes(skel, cn, config)
    if config.get("save_nodes_map", True):
        save_nodes_map(skel, nodes, node_types, str(target_dir / f"{input_stem}_collapsed_nodes.png"))

    # Step 5: Trace Skeleton Paths into NetworkX MultiGraph
    raw_graph = trace_skeleton_to_graph(skel, nodes, node_grid, node_pixel_members)
    H_canvas, W_canvas = skel.shape

    # Step 6: Export Dense Traced Graph to SVG
    save_graph_as_svg(raw_graph, W_canvas, H_canvas, str(target_dir / f"{input_stem}_traced_dense.svg"))
    dense_point_count = count_graph_points(raw_graph)

    # Step 7: Apply Ramer–Douglas–Peucker (RDP) Simplification
    rdp_graph = apply_rdp_to_graph(raw_graph, epsilon=config.get("rdp_epsilon", 1.5))
    save_graph_as_svg(rdp_graph, W_canvas, H_canvas, str(target_dir / f"{input_stem}_rdp_simplified.svg"))
    rdp_point_count = count_graph_points(rdp_graph)

    # Step 8: Topological Short-Edge Cluster Contraction & Whisker Pruning
    contracted_graph = contract_short_edges(
        rdp_graph,
        min_edge_length=config["edge_merge_distance"],
        prune_whiskers=config.get("prune_whiskers", True),
    )

    # Step 9: Connect Nearby Dots into Closed Polygonal Loops BEFORE Small-Component Pruning
    dot_merged_graph = combine_nearby_dots(
        contracted_graph,
        skeleton=skel,
        crossing_number=cn,
        merge_distance=config["edge_merge_distance"],
    )

    # Step 10: Prune Disconnected Small Components / Tiny Dots -> Produces `simple_graph`
    simple_graph = prune_small_components(dot_merged_graph, config.get("min_component_ratio", 0.01))
    simple_graph = dissolve_degree_two_nodes(simple_graph)

    # Save Simple Graph (SVG & PNG)
    save_graph_as_svg(simple_graph, W_canvas, H_canvas, str(target_dir / f"{input_stem}_simple_graph.svg"))
    if config.get("save_simple_graph_map", True):
        save_graph_map(
            simple_graph, W_canvas, H_canvas,
            str(target_dir / f"{input_stem}_simple_graph.png"),
            show_split_nodes=False,
        )
    simple_point_count = count_graph_points(simple_graph)

    # Step 11: Split Sharp Corners & Subdivide Long Edges -> Produces `final_graph`
    corner_split_graph = split_edges_at_sharp_corners(
        simple_graph,
        max_turn_angle_deg=config.get("corner_split_angle_deg", 65.0),
    )
    final_graph = subdivide_long_edges(
        corner_split_graph,
        max_edge_length=config["max_edge_length"],
    )

    # Save Final Graph (SVG & PNG) BEFORE bridging
    final_svg_path = target_dir / f"{input_stem}_final.svg"
    save_graph_as_svg(final_graph, W_canvas, H_canvas, str(final_svg_path))
    final_point_count = count_graph_points(final_graph)

    if config.get("save_final_graph_map", True):
        save_graph_map(
            final_graph, W_canvas, H_canvas,
            str(target_dir / f"{input_stem}_final_graph.png"),
            show_split_nodes=True,
        )

    # -------------------------------------------------------------------------
    # STEP 12: Bridge Disconnected Components on Final Graph via 2-Pass Kruskal MST
    # -------------------------------------------------------------------------
    bridged_graph, mst_pass_1, mst_pass_2 = bridge_components_two_pass_kruskal(final_graph, config)

    # Save Bridged Graph (SVG & PNG)
    bridged_svg_path = target_dir / f"{input_stem}_bridged_graph.svg"
    save_graph_as_svg(bridged_graph, W_canvas, H_canvas, str(bridged_svg_path))
    bridged_point_count = count_graph_points(bridged_graph)

    if config.get("save_bridged_graph_map", True):
        save_graph_map(
            bridged_graph, W_canvas, H_canvas,
            str(target_dir / f"{input_stem}_bridged_graph.png"),
            show_split_nodes=True,
        )
        

    reduction_rdp = ((dense_point_count - rdp_point_count) / dense_point_count) * 100.0 if dense_point_count > 0 else 0.0
    reduction_simple = ((dense_point_count - simple_point_count) / dense_point_count) * 100.0 if dense_point_count > 0 else 0.0

    print("\n" + "=" * 65)
    print("PIPELINE GRAPH REFINEMENT & DECIMATION BENCHMARK")
    print(f"  - Dense Traced Points:       {dense_point_count:,} pts  ({raw_graph.number_of_nodes()} nodes, {raw_graph.number_of_edges()} edges)")
    print(f"  - RDP Simplified Points:     {rdp_point_count:,} pts  ({reduction_rdp:.2f}% decimation)")
    print(f"  - Simple Graph (Contracted): {simple_point_count:,} pts  ({simple_graph.number_of_nodes()} nodes, {simple_graph.number_of_edges()} edges, {reduction_simple:.2f}% reduction)")
    print(f"  - Final Graph (Split/Subd):  {final_point_count:,} pts  ({final_graph.number_of_nodes()} nodes, {final_graph.number_of_edges()} edges)")
    print("=" * 65 + "\n")

    return simple_graph, final_graph


def main():
    parser = argparse.ArgumentParser(description="Convert any PNG image into a clean topological graph polyline.")
    parser.add_argument("input", nargs="?", default="images/example.png", help="Path to input image")
    parser.add_argument("-o", "--output-dir", default=None, help="Base output directory")
    parser.add_argument("-t", "--threshold", type=int, default=128, help="Luminance intensity threshold (0-255)")
    parser.add_argument("--method", default="lee", choices=["lee", "zhang"], help="Skeletonisation method")
    parser.add_argument("--node-merge-dist", type=float, default=None, help="Explicit spatial distance (px) to merge close nodes.")
    parser.add_argument("--node-merge-ratio", type=float, default=0.006, help="Proportional node merge ratio of diagonal (default: 0.006)")
    parser.add_argument("--edge-merge-dist", type=float, default=None, help="Explicit arc-length threshold (px) to contract short edges.")
    parser.add_argument("--edge-merge-ratio", type=float, default=0.025, help="Proportional short-edge contraction ratio of diagonal (default: 0.025)")
    parser.add_argument("--min-component-ratio", type=float, default=0.01, help="Prune components shorter than this ratio of total length")
    parser.add_argument("--rdp-epsilon", type=float, default=1.5, help="RDP perpendicular deviation tolerance in pixels (default: 1.5)")
    parser.add_argument("--corner-angle", type=float, default=65.0, help="Split edges at turns sharper than this deflection angle in degrees (default: 65.0)")
    parser.add_argument("--max-edge-length", type=float, default=None, help="Explicit max edge length (px) before subdivision.")
    parser.add_argument("--max-edge-ratio", type=float, default=0.20, help="Max edge length as a ratio of drawing width before subdivision (default: 0.20 = 20%%)")
    parser.add_argument("--no-save-mask", action="store_true", help="Disable saving intermediate binary mask")
    parser.add_argument("--no-save-skel", action="store_true", help="Disable saving intermediate skeleton")
    parser.add_argument("--no-save-crossing", action="store_true", help="Disable saving raw crossing map")
    parser.add_argument("--no-save-nodes", action="store_true", help="Disable saving collapsed canonical nodes map")
    parser.add_argument("--no-save-simple-graph", action="store_true", help="Disable saving simple graph diagnostic map")
    parser.add_argument("--no-save-final-graph", action="store_true", help="Disable saving final graph diagnostic map")
    parser.add_argument("--bridge-alpha", type=float, default=0.15, help="Degree penalty weight for bridging (default: 0.15)")
    parser.add_argument("--bridge-beta", type=float, default=0.50, help="Angular alignment penalty weight for bridging (default: 0.50)")
    parser.add_argument("--bridge-gamma", type=float, default=1.5, help="Pass 2 multiplicative node reuse penalty factor (default: 1.5)")
    parser.add_argument("--bridge-lambda", type=float, default=25.0, help="Pass 2 additive node reuse penalty in pixels (default: 25.0)")
    parser.add_argument("--no-save-bridged-graph", action="store_true", help="Disable saving bridged graph diagnostic map")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    base_out = args.output_dir if args.output_dir is not None else str(input_path.parent / "outputs")

    config = {
        "output_dir": base_out,
        "intensity_threshold": args.threshold,
        "skeleton_mode": args.method,
        "node_merge_distance": args.node_merge_dist,
        "node_merge_ratio": args.node_merge_ratio,
        "edge_merge_distance": args.edge_merge_dist,
        "edge_merge_ratio": args.edge_merge_ratio,
        "min_component_ratio": args.min_component_ratio,
        "rdp_epsilon": args.rdp_epsilon,
        "corner_split_angle_deg": args.corner_angle,
        "max_edge_length": args.max_edge_length,
        "max_edge_ratio": args.max_edge_ratio,
        "prune_whiskers": True,
        "save_binary_mask": not args.no_save_mask,
        "save_skeleton": not args.no_save_skel,
        "save_crossing_map": not args.no_save_crossing,
        "save_nodes_map": not args.no_save_nodes,
        "save_simple_graph_map": not args.no_save_simple_graph,
        "save_final_graph_map": not args.no_save_final_graph,
        "bridge_alpha_deg": args.bridge_alpha,
        "bridge_beta_ang": args.bridge_beta,
        "bridge_reuse_gamma": args.bridge_gamma,
        "bridge_reuse_lambda": args.bridge_lambda,
        "save_bridged_graph_map": not args.no_save_bridged_graph,
    }

    try:
        image_to_oneline(input_image=str(input_path), config=config)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()