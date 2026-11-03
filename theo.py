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

# Node penalty weights for Kruskal MST bridging (with kappa_kind = 1.50):
# Endpoints (0.00) << Odd Junctions (0.30) < Sharp Corners (0.50) == Even Junctions (0.50) <<< Subdivision on line/curve (1.60)
NODE_KIND_PENALTIES = {
    "endpoint": 0.00,       # Strongly favoured (deg == 1)
    "odd_junction": 0.30,   # Odd-degree junction (deg >= 3 and odd)
    "corner": 1.0,         # Sharp turn split node — strongly favoured over line/curve subdivision nodes
    "even_junction": 2.0,  # Even-degree junction (deg >= 3 and even) — weighted same as corners
    "loop_anchor": 0.65,    # Closed isolated loop anchor
    "subdivision": 2.00,    # Long-edge split node on a straight/curved line — heavily penalised
}


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
      - Drawing edge polylines: Blue (0, 0, 255)
      - MST Bridge edges (is_bridge=True): Magenta/Pink (255, 0, 200) with thickness 2
      - Endpoints (deg == 1): Green 3x3 markers (0, 200, 0)
      - Junctions (deg >= 3): Red 3x3 markers (255, 0, 0)
      - Split/Subdivision degree-2 nodes (when show_split_nodes=True): Orange 3x3 markers (255, 140, 0)
    """
    rgb_map = np.full((height, width, 3), 255, dtype=np.uint8)

    # 1. Rasterize drawing edges in Blue, and MST bridge edges in Pink/Magenta
    for is_bridge_pass in (False, True):
        for u, v, k, data in graph.edges(keys=True, data=True):
            if bool(data.get("is_bridge", False)) != is_bridge_pass:
                continue
            pts = data.get("pts")
            if pts is None or len(pts) < 2:
                continue
            pts_xy = np.round(pts[:, [1, 0]]).astype(np.int32).reshape((-1, 1, 2))
            color = (255, 0, 200) if is_bridge_pass else (0, 0, 255)
            thickness = 2 if is_bridge_pass else 1
            cv2.polylines(rgb_map, [pts_xy], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_8)

    # 2. Draw 3x3 markers in strict z-order: Continuation -> Split -> Endpoints -> Junctions
    for target_pass in ("continuation", "split", "endpoint", "junction"):
        for nid, data in graph.nodes(data=True):
            pos = data.get("pos")
            if pos is None:
                continue
            r, c = int(round(pos[0])), int(round(pos[1]))
            deg = sum(1 if u != v else 2 for u, v, d in graph.edges(nid, data=True) if not d.get("is_bridge", False))
            if deg == 0:
                deg = graph.degree(nid)
            kind = data.get("kind", "")

            if target_pass == "continuation" and deg == 2 and not (show_split_nodes and kind in ("corner", "subdivision", "refined_bridge")):
                color = [0, 0, 255]
            elif target_pass == "split" and deg == 2 and show_split_nodes and kind in ("corner", "subdivision", "refined_bridge"):
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
        attrs = {"points": " ".join(coord_strings), "id": f"edge_{u}_{v}_{k}"}
        if data.get("is_bridge", False):
            attrs["stroke"] = "#FF00C8"
        ET.SubElement(group, "polyline", attrs)

    pretty_xml = minidom.parseString(ET.tostring(svg_root, encoding="utf-8")).toprettyxml(indent="  ")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)
    print(f"Saved SVG: {output_path}")


# =============================================================================
# 2. IMAGE PREPROCESSING, RESOLUTION SCALING & SKELETONISATION
# =============================================================================

def import_image_as_mask(path: str, config: dict):
    """
    Imports a PNG image, thresholds it to binary, crops to tight active bounds,
    optionally downscales the binary mask by `downsample_factor` (using area-averaging
    to smooth thick pen strokes before skeletonisation), and pads with a 1-pixel False boundary.
    """
    threshold = config.get("intensity_threshold", 128)
    downsample = max(1.0, float(config.get("downsample_factor", 1.0)))

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
        padded = np.pad(binary_mask, pad_width=1, mode="constant", constant_values=False)
        return padded, (padded.shape[0], padded.shape[1], 0.0), (1.0, 1.0)

    r_min, c_min = active_coords.min(axis=0)
    r_max, c_max = active_coords.max(axis=0) + 1
    cropped_mask = binary_mask[r_min:r_max, c_min:c_max]

    h_orig, w_orig = cropped_mask.shape
    orig_diagonal = float(np.hypot(h_orig, w_orig))
    print(f"Tight drawing bounds: {w_orig}x{h_orig} px (Diagonal: {orig_diagonal:.1f} px).")

    if downsample > 1.0:
        h_work = max(3, int(round(h_orig / downsample)))
        w_work = max(3, int(round(w_orig / downsample)))
        mask_u8 = (cropped_mask.astype(np.uint8)) * 255
        resized_u8 = cv2.resize(mask_u8, (w_work, h_work), interpolation=cv2.INTER_AREA)
        work_mask = resized_u8 >= 96
        scale_r = h_orig / float(h_work)
        scale_c = w_orig / float(w_work)
        print(f"Downsampled working grid by {downsample:.2f}x -> {w_work}x{h_work} px for skeletonisation.")
    else:
        work_mask = cropped_mask
        scale_r, scale_c = 1.0, 1.0

    padded_mask = np.pad(work_mask, pad_width=1, mode="constant", constant_values=False)
    h_canvas_orig = h_orig + 2
    w_canvas_orig = w_orig + 2
    return padded_mask, (h_canvas_orig, w_canvas_orig, orig_diagonal), (scale_r, scale_c)


def upscale_graph_coordinates(graph: nx.MultiGraph, scale_r: float, scale_c: float) -> nx.MultiGraph:
    """
    Rescales all node positions and edge polylines from the downsampled working skeleton grid
    back into the original full-resolution canvas coordinate space.
    """
    if abs(scale_r - 1.0) < 1e-6 and abs(scale_c - 1.0) < 1e-6:
        return graph

    H = graph.copy()
    scale_vec = np.array([scale_r, scale_c], dtype=np.float32)

    for nid, data in H.nodes(data=True):
        r, c = data["pos"]
        new_r = int(round((r - 1.0) * scale_r + 1.0))
        new_c = int(round((c - 1.0) * scale_c + 1.0))
        data["pos"] = (new_r, new_c)

    for u, v, k, data in H.edges(keys=True, data=True):
        pts = data["pts"].astype(np.float32)
        pts_scaled = (pts - 1.0) * scale_vec + 1.0
        pts_scaled[0] = np.array(H.nodes[u]["pos"], dtype=np.float32)
        pts_scaled[-1] = np.array(H.nodes[v]["pos"], dtype=np.float32)
        data["pts"] = pts_scaled

    return H


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
    and merges nodes closer than config['node_merge_distance_work'] along connected skeleton paths.
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

    # 3. Spatial Node Merging via KD-Tree (filling intermediate skeleton bridge pixels so clusters stay 8-connected)
    merge_dist = float(config.get("node_merge_distance_work", 0.0))
    final_nodes, final_node_types, final_node_members = {}, {}, {}
    skel_set = set(map(tuple, np.argwhere(skeleton > 0)))

    if len(raw_nodes) > 0 and merge_dist > 0.0:
        raw_ids = list(raw_nodes.keys())
        coords = np.array([raw_nodes[nid] for nid in raw_ids])
        pairs = cKDTree(coords).query_pairs(r=merge_dist)

        merge_graph = nx.Graph()
        merge_graph.add_nodes_from(raw_ids)
        merge_graph.add_edges_from((raw_ids[i], raw_ids[j]) for i, j in pairs)

        for new_node_id, cluster in enumerate(nx.connected_components(merge_graph)):
            cluster_list = list(cluster)
            combined_set = {px for nid in cluster_list for px in node_pixel_members[nid]}
            is_junction = any(raw_node_types[nid] == "junction" for nid in cluster_list)

            if len(cluster_list) > 1:
                rep_px = node_pixel_members[cluster_list[0]][0]
                max_steps = int(np.ceil(merge_dist)) + 4
                for other_nid in cluster_list[1:]:
                    target_px = node_pixel_members[other_nid][0]
                    q = [(rep_px, 0)]
                    parent = {rep_px: None}
                    while q:
                        curr, d = q.pop(0)
                        if curr == target_px:
                            p_walk = curr
                            while p_walk is not None:
                                combined_set.add(p_walk)
                                p_walk = parent[p_walk]
                            break
                        if d < max_steps:
                            for dr, dc in OFFSETS:
                                nbr = (curr[0] + dr, curr[1] + dc)
                                if nbr in skel_set and nbr not in parent:
                                    parent[nbr] = curr
                                    q.append((nbr, d + 1))

            combined_pixels = list(combined_set)
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
# 3. LOSSLESS EDGE TRACING, HAMILTONIAN DOT CYCLES & GRAPH CONSTRUCTION
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
                            node_pixel_members: dict, node_types: dict) -> nx.MultiGraph:
    """
    Traces 8-connected chains of interior pixels between canonical nodes to construct a MultiGraph.
    Uses `left_start_cluster` so tangential spiral loops grazing a junction blob never close early.
    """
    G = nx.MultiGraph()
    for nid, pos in nodes.items():
        G.add_node(nid, pos=pos, kind=node_types.get(nid, "endpoint"))

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
                    left_start_cluster = False

                    while True:
                        path.append((curr_r, curr_c))
                        visited[curr_r, curr_c] = True
                        steps += 1

                        path_nbrs = []
                        other_node_nbrs = []
                        self_node_nbrs = []

                        for s_dr, s_dc in OFFSETS:
                            sr, sc = curr_r + s_dr, curr_c + s_dc
                            if 0 <= sr < H and 0 <= sc < W and skeleton[sr, sc]:
                                s_grid = int(node_grid[sr, sc])
                                if s_grid == -1 and not visited[sr, sc]:
                                    path_nbrs.append((sr, sc))
                                elif s_grid != -1:
                                    if s_grid != u_id:
                                        other_node_nbrs.append(((sr, sc), s_grid))
                                    else:
                                        self_node_nbrs.append(((sr, sc), u_id))

                        if not self_node_nbrs:
                            left_start_cluster = True

                        if other_node_nbrs:
                            target_pixel, target_node_id = other_node_nbrs[0]
                            break
                        elif self_node_nbrs and left_start_cluster:
                            target_pixel, target_node_id = self_node_nbrs[0]
                            break
                        elif path_nbrs:
                            path_nbrs.sort(key=lambda pt: abs(pt[0] - curr_r) + abs(pt[1] - curr_c))
                            curr_r, curr_c = path_nbrs[0]
                        elif self_node_nbrs and steps >= 3:
                            target_pixel, target_node_id = self_node_nbrs[0]
                            break
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
        G.add_node(loop_node_id, pos=canonical_pos, kind="loop_anchor")

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


def _solve_hamiltonian_cycle(points: np.ndarray) -> np.ndarray:
    """
    Computes a minimum-length Hamiltonian cycle (TSP tour) visiting every point in `points`:
      - For N <= 11: Exact Held-Karp bitmask dynamic programming.
      - For N > 11: Nearest-Neighbour tour followed by 2-opt local search refinement.
    """
    n = len(points)
    if n <= 2:
        return np.vstack((points, points[0])).astype(np.float32)

    dist_mat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

    if n <= 11:
        num_masks = 1 << n
        dp = [[(float("inf"), -1)] * n for _ in range(num_masks)]
        dp[1][0] = (0.0, -1)

        for mask in range(1, num_masks):
            if not (mask & 1):
                continue
            for u in range(n):
                cost_u, _ = dp[mask][u]
                if cost_u == float("inf"):
                    continue
                for v in range(1, n):
                    if not (mask & (1 << v)):
                        nxt_mask = mask | (1 << v)
                        new_cost = cost_u + float(dist_mat[u, v])
                        if new_cost < dp[nxt_mask][v][0]:
                            dp[nxt_mask][v] = (new_cost, u)

        full_mask = num_masks - 1
        best_end = min(range(1, n), key=lambda u: dp[full_mask][u][0] + float(dist_mat[u, 0]))

        tour = []
        curr = best_end
        mask = full_mask
        while curr != -1:
            tour.append(curr)
            _, prev = dp[mask][curr]
            mask ^= (1 << curr)
            curr = prev
        tour.reverse()
    else:
        unvisited = set(range(1, n))
        tour = [0]
        while unvisited:
            last = tour[-1]
            nxt = min(unvisited, key=lambda v: dist_mat[last, v])
            tour.append(nxt)
            unvisited.remove(nxt)

        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    a, b = tour[i - 1], tour[i]
                    c, d = tour[j], tour[(j + 1) % n]
                    if dist_mat[a, c] + dist_mat[b, d] < dist_mat[a, b] + dist_mat[c, d] - 1e-6:
                        tour[i:j + 1] = reversed(tour[i:j + 1])
                        improved = True

    ordered = points[tour]
    return np.vstack((ordered, ordered[0])).astype(np.float32)


def combine_nearby_dots(graph: nx.MultiGraph, skeleton: np.ndarray, crossing_number: np.ndarray,
                        merge_distance: float, scale_rc: tuple = (1.0, 1.0)) -> nx.MultiGraph:
    """
    Combines isolated dots (CN == 0 on skeleton) and tiny micro-stroke components (total length <= 3px,
    excluding any closed loops) that lie within `merge_distance` of each other by solving a
    Hamiltonian cycle (TSP tour) through their coordinates before small-component pruning.
    """
    H = graph.copy()
    scale_r, scale_c = scale_rc
    dot_coords = [
        ((float(r) - 1.0) * scale_r + 1.0, (float(c) - 1.0) * scale_c + 1.0)
        for r, c in np.argwhere((skeleton > 0) & (crossing_number == 0))
    ]
    micro_comp_nodes = []

    for comp in list(nx.connected_components(H)):
        sub_edges = list(H.edges(comp, keys=True, data=True))
        if any(u == v for u, v, _, _ in sub_edges):
            continue
        comp_len = sum(compute_polyline_length(d["pts"]) for _, _, _, d in sub_edges)
        if comp_len <= 3.0 * max(scale_r, scale_c):
            pts = np.array([H.nodes[n]["pos"] for n in comp], dtype=np.float64)
            center = pts.mean(axis=0)
            dot_coords.append((float(center[0]), float(center[1])))
            micro_comp_nodes.append(comp)

    if not dot_coords:
        return H

    for comp in micro_comp_nodes:
        H.remove_nodes_from(comp)

    coords_arr = np.array(dot_coords, dtype=np.float64)
    dot_graph = nx.Graph()
    dot_graph.add_nodes_from(range(len(dot_coords)))
    if len(dot_coords) > 1 and merge_distance > 0.0:
        pairs = cKDTree(coords_arr).query_pairs(r=merge_distance)
        dot_graph.add_edges_from(pairs)

    next_nid = max(H.nodes, default=-1) + 1
    combined_cycles = 0

    for cluster in nx.connected_components(dot_graph):
        if len(cluster) < 2:
            continue

        cluster_pts = coords_arr[list(cluster)]
        cycle_pts = _solve_hamiltonian_cycle(cluster_pts)
        cycle_pts = _clean_pts(cycle_pts.tolist())

        if len(cycle_pts) >= 2:
            anchor_pos = (int(round(cycle_pts[0, 0])), int(round(cycle_pts[0, 1])))
            H.add_node(next_nid, pos=anchor_pos, kind="loop_anchor")
            H.add_edge(next_nid, next_nid, pts=cycle_pts)
            next_nid += 1
            combined_cycles += 1

    if combined_cycles > 0:
        print(f"Connected {combined_cycles} cluster(s) of nearby dots into Hamiltonian cycles.")
    return H


# =============================================================================
# 4. RDP SIMPLIFICATION, LOOP-SAFE JUNCTION CONTRACTION, PRUNING & SPLITTING
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


def contract_short_edges(graph: nx.MultiGraph, min_edge_length: float) -> nx.MultiGraph:
    """
    Loop-Safe Junction Contraction:
    Finds clusters of junctions (deg >= 3) connected by short edges (< `min_edge_length`) and
    collapses each cluster to its geometric centroid.
    Strictly preserves all self-loops (u == v) and parallel loop petals between junction pairs.
    """
    H = dissolve_degree_two_nodes(graph)

    while True:
        short_graph = nx.Graph()
        for u, v, k, data in H.edges(keys=True, data=True):
            if u != v and H.degree(u) >= 3 and H.degree(v) >= 3:
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
            old_positions = {nid: np.array(H.nodes[nid]["pos"], dtype=np.float32) for nid in cluster_nodes}
            H.nodes[rep]["pos"] = centroid_pos
            H.nodes[rep]["kind"] = "junction"

            seen_edges = set()
            internal_edges = []
            external_edges = []

            for u, v, k, data in list(H.edges(cluster_nodes, keys=True, data=True)):
                edge_id = (min(u, v), max(u, v), k) if u != v else (u, v, k)
                if edge_id in seen_edges:
                    continue
                seen_edges.add(edge_id)

                if u in cluster_set and v in cluster_set:
                    internal_edges.append((u, v, k, data["pts"].copy()))
                else:
                    external_edges.append((u, v, k, data["pts"].copy()))

            to_remove = [(u, v, k) for u, v, k, _ in internal_edges + external_edges]
            to_add = []

            pair_groups = {}
            for u, v, k, pts in internal_edges:
                if u == v:
                    pts[0] = centroid_vec
                    pts[-1] = centroid_vec
                    cleaned = _clean_pts(pts.tolist())
                    if len(cleaned) >= 2:
                        to_add.append((rep, rep, cleaned))
                else:
                    pair_key = (min(u, v), max(u, v))
                    pair_groups.setdefault(pair_key, []).append((k, pts, compute_polyline_length(pts)))

            for pair_key, edge_list in pair_groups.items():
                edge_list.sort(key=lambda item: item[2])
                for _k, pts, _len in edge_list[1:]:
                    pts[0] = centroid_vec
                    pts[-1] = centroid_vec
                    cleaned = _clean_pts(pts.tolist())
                    if len(cleaned) >= 2:
                        to_add.append((rep, rep, cleaned))

            for u, v, k, pts in external_edges:
                inner = u if u in cluster_set else v
                outer = v if u in cluster_set else u
                inner_pos = old_positions[inner]
                outer_pos = np.array(H.nodes[outer]["pos"], dtype=np.float32)

                d_start = np.linalg.norm(pts[0] - inner_pos) + np.linalg.norm(pts[-1] - outer_pos)
                d_end = np.linalg.norm(pts[-1] - inner_pos) + np.linalg.norm(pts[0] - outer_pos)
                if d_start <= d_end:
                    pts[0] = centroid_vec
                else:
                    pts[-1] = centroid_vec

                cleaned = _clean_pts(pts.tolist())
                if len(cleaned) >= 2:
                    to_add.append((rep, outer, cleaned))

            for u, v, k in to_remove:
                if H.has_edge(u, v, key=k):
                    H.remove_edge(u, v, key=k)
            for u_new, v_new, pts_new in to_add:
                H.add_edge(u_new, v_new, pts=pts_new)
            for nid in cluster_nodes[1:]:
                if nid in H:
                    H.remove_node(nid)

        H = dissolve_degree_two_nodes(H)

    return H


def prune_terminal_edges(graph: nx.MultiGraph, max_whisker_length: float) -> nx.MultiGraph:
    """
    Prunes any edge connected to a degree-1 node whose arc-length is less than `max_whisker_length`
    (% of the drawing diagonal), and dissolves any resulting degree-2 nodes.
    """
    if max_whisker_length <= 0.0:
        return graph.copy()

    H = graph.copy()
    pruned_count = 0

    changed = True
    while changed:
        changed = False
        for u, v, k, data in list(H.edges(keys=True, data=True)):
            if u == v:
                continue
            if H.degree(u) == 1 or H.degree(v) == 1:
                if compute_polyline_length(data["pts"]) < max_whisker_length:
                    H.remove_edge(u, v, key=k)
                    for endpoint in (u, v):
                        if endpoint in H and H.degree(endpoint) == 0:
                            H.remove_node(endpoint)
                    H = dissolve_degree_two_nodes(H)
                    pruned_count += 1
                    changed = True
                    break

    if pruned_count > 0:
        print(f"Pruned {pruned_count} degree-1 terminal edge(s) (< {max_whisker_length:.2f} px).")
    return H


def prune_small_components(graph: nx.MultiGraph, diagonal: float, min_diag_ratio: float = 0.03) -> nx.MultiGraph:
    """
    Removes completely disconnected sub-graphs whose total cumulative edge length is less than
    (min_diag_ratio * diagonal).
    """
    if min_diag_ratio <= 0.0 or diagonal <= 0.0:
        return graph.copy()

    H = graph.copy()
    min_length_threshold = min_diag_ratio * diagonal
    pruned_count = 0

    for comp in list(nx.connected_components(H)):
        comp_length = sum(compute_polyline_length(d["pts"]) for _, _, _, d in H.edges(comp, keys=True, data=True))
        if comp_length < min_length_threshold:
            H.remove_nodes_from(comp)
            pruned_count += 1

    if pruned_count > 0:
        print(f"Pruned {pruned_count} disconnected sub-graph(s) shorter than {min_diag_ratio*100:.2f}% of diagonal (< {min_length_threshold:.1f} px).")
    return H


def split_edges_at_sharp_corners(graph: nx.MultiGraph, max_turn_angle_deg: float = 65.0) -> nx.MultiGraph:
    """
    Splits edges at sharp corners into multiple edges connected by new degree-2 corner vertices (`kind="corner"`).
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
    Splits any edge whose cumulative arc-length exceeds `max_edge_length` into equal-arc-length
    sub-edges separated by new degree-2 bridging nodes (`kind="subdivision"`).
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

        seg_diffs = np.diff(pts, axis=0)
        seg_lens = np.hypot(seg_diffs[:, 0], seg_diffs[:, 1])
        cum_len = np.concatenate(([0.0], np.cumsum(seg_lens)))

        H.remove_edge(u, v, key=k)
        curr_u = u
        start_dist = 0.0

        for seg_idx in range(1, num_segments + 1):
            end_dist = (seg_idx / num_segments) * total_len if seg_idx < num_segments else total_len

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


# =============================================================================
# 5. TWO-PASS KRUSKAL MST, EVEN-PARITY PRUNING, EXCESS BRIDGE PRUNING & PIXEL REFINEMENT
# =============================================================================

def get_node_outgoing_tangent(graph: nx.MultiGraph, nid: int):
    """
    Computes the unit tangent vector pointing OUT of the stroke at `nid` for degree-1 endpoints.
    """
    if graph.degree(nid) != 1:
        return None

    node_pos = np.array(graph.nodes[nid]["pos"], dtype=np.float32)
    for u, v, k, data in graph.edges(nid, keys=True, data=True):
        if data.get("is_bridge", False):
            continue
        pts = data.get("pts")
        if pts is None or len(pts) < 2:
            continue
        if np.linalg.norm(pts[-1] - node_pos) <= np.linalg.norm(pts[0] - node_pos):
            vec = pts[-1] - pts[-2]
        else:
            vec = pts[0] - pts[1]
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            return vec / norm
    return None


def get_node_kind(graph: nx.MultiGraph, nid: int) -> str:
    """
    Classifies a node into the exact MST priority hierarchy:
      1. 'endpoint'      (deg == 1)                 -> penalty 0.00 (strongly favoured)
      2. 'odd_junction'  (deg >= 3 and deg is odd)  -> penalty 0.30 (second most favoured)
      3. 'corner'        (deg == 2 sharp turn)      -> penalty 0.50 (strongly favoured over subdivision)
      4. 'even_junction' (deg >= 3 and deg is even) -> penalty 0.50 (weighted same as corners)
      5. 'subdivision'   (deg == 2 line/curve split)-> penalty 1.60 (least favoured)
    """
    deg = sum(1 if u != v else 2 for u, v, d in graph.edges(nid, data=True) if not d.get("is_bridge", False))
    if deg == 1:
        return "endpoint"
    if deg >= 3:
        return "odd_junction" if (deg % 2 == 1) else "even_junction"

    explicit_kind = graph.nodes[nid].get("kind")
    if explicit_kind in ("corner", "subdivision"):
        return explicit_kind
    return "loop_anchor"


def compute_bridge_weight(
    u: int,
    v: int,
    comp_u: int,
    comp_v: int,
    pos_u: np.ndarray,
    pos_v: np.ndarray,
    kind_u: str,
    kind_v: str,
    tan_u,
    tan_v,
    diagonal: float,
    beta_ang: float = 0.35,
    kappa_kind: float = 1.50,
    used_comp_pairs: set = None,
    used_nodes: set = None,
    eta_same_comp_pair: float = 0.35,
    gamma_same_node: float = 0.50,
) -> float:
    """
    Computes the Kruskal bridge weight between `u` (in `comp_u`) and `v` (in `comp_v`):
      - Pass 1: Based on proportional Euclidean distance (dist / diagonal), node type (`kappa_kind`),
        and exit tangent angle (`beta_ang`).
      - Pass 2: Same base weight, plus a small penalty (`eta_same_comp_pair` = 0.35) if `comp_u`
        and `comp_v` are already bridged in Pass 1 (and `gamma_same_node` if the exact node was used).
    """
    vec_uv = pos_v - pos_u
    dist = float(np.linalg.norm(vec_uv))
    prop_dist = dist / max(1.0, diagonal)

    phi_ang = 0.0
    if dist > 1e-6:
        unit_uv = vec_uv / dist
        if tan_u is not None:
            phi_ang += 0.5 * (1.0 - float(np.clip(np.dot(tan_u, unit_uv), -1.0, 1.0)))
        if tan_v is not None:
            phi_ang += 0.5 * (1.0 - float(np.clip(np.dot(tan_v, -unit_uv), -1.0, 1.0)))

    psi_kind = NODE_KIND_PENALTIES.get(kind_u, 0.50) + NODE_KIND_PENALTIES.get(kind_v, 0.50)
    w1 = prop_dist * (1.0 + kappa_kind * psi_kind + beta_ang * phi_ang)

    if used_comp_pairs is None:
        return w1

    comp_pair = (min(comp_u, comp_v), max(comp_u, comp_v))
    pair_pen = eta_same_comp_pair if comp_pair in used_comp_pairs else 0.0
    node_pen = gamma_same_node * ((1.0 if (used_nodes and u in used_nodes) else 0.0) +
                                  (1.0 if (used_nodes and v in used_nodes) else 0.0))

    return w1 * (1.0 + pair_pen) * (1.0 + node_pen)


def solve_component_mst_kruskal(
    graph: nx.MultiGraph,
    components: list,
    diagonal: float,
    beta_ang: float = 0.35,
    kappa_kind: float = 1.50,
    used_comp_pairs: set = None,
    used_nodes: set = None,
    eta_same_comp_pair: float = 0.35,
    k_nearest_candidates: int = 24,
) -> list:
    """
    Constructs a component-level meta-graph and solves for the Minimum Spanning Tree (MST)
    using Kruskal's algorithm (`nx.minimum_spanning_edges(..., algorithm="kruskal")`).
    Always includes endpoints, odd junctions, AND sharp corners in the candidate evaluation pool
    so corners are never overlooked in favour of closer subdivision nodes on straight/curved lines.
    """
    num_comps = len(components)
    if num_comps <= 1:
        return []

    node_pos = {n: np.array(graph.nodes[n]["pos"], dtype=np.float32) for n in graph.nodes()}
    node_kind = {n: get_node_kind(graph, n) for n in graph.nodes()}
    node_tan = {n: get_node_outgoing_tangent(graph, n) for n in graph.nodes()}

    comp_node_lists = [list(c) for c in components]
    comp_coords = [np.array([node_pos[n] for n in c_list], dtype=np.float32) for c_list in comp_node_lists]
    comp_trees = [cKDTree(coords) for coords in comp_coords]

    # Explicitly include endpoints, odd junctions, and corners in the priority candidate set
    comp_priority_indices = [
        [idx for idx, n in enumerate(c_list) if node_kind[n] in ("endpoint", "odd_junction", "corner")]
        for c_list in comp_node_lists
    ]

    meta_graph = nx.Graph()
    meta_graph.add_nodes_from(range(num_comps))

    for i in range(num_comps):
        nodes_i = comp_node_lists[i]
        coords_i = comp_coords[i]

        for j in range(i + 1, num_comps):
            nodes_j = comp_node_lists[j]
            tree_j = comp_trees[j]
            prio_j = comp_priority_indices[j]

            k_query = min(k_nearest_candidates, len(nodes_j))
            _, idxs = tree_j.query(coords_i, k=k_query)
            if k_query == 1:
                idxs = idxs[:, None]

            best_w = float("inf")
            best_pair = (-1, -1)

            for idx_in_i, u in enumerate(nodes_i):
                cand_j_indices = set(int(idxs[idx_in_i, col]) for col in range(k_query))
                cand_j_indices.update(prio_j)

                for col_idx in cand_j_indices:
                    v = nodes_j[col_idx]
                    w = compute_bridge_weight(
                        u=u,
                        v=v,
                        comp_u=i,
                        comp_v=j,
                        pos_u=node_pos[u],
                        pos_v=node_pos[v],
                        kind_u=node_kind[u],
                        kind_v=node_kind[v],
                        tan_u=node_tan[u],
                        tan_v=node_tan[v],
                        diagonal=diagonal,
                        beta_ang=beta_ang,
                        kappa_kind=kappa_kind,
                        used_comp_pairs=used_comp_pairs,
                        used_nodes=used_nodes,
                        eta_same_comp_pair=eta_same_comp_pair,
                    )
                    if w < best_w:
                        best_w = w
                        best_pair = (u, v)

            if best_pair != (-1, -1):
                meta_graph.add_edge(i, j, weight=best_w, u=best_pair[0], v=best_pair[1])

    mst_meta_edges = nx.minimum_spanning_edges(meta_graph, algorithm="kruskal", weight="weight", data=True)
    return [(i, j, d["u"], d["v"], float(d["weight"])) for i, j, d in mst_meta_edges]


def _sample_incident_edges_at_pixel_resolution(graph: nx.MultiGraph, center_nid: int) -> list:
    """
    For a subdivision/loop_anchor node `center_nid`, walks all incident drawing edges up to
    their neighbouring nodes and samples every 1-pixel coordinate along those polylines.
    Returns a list of candidate tuples: `(u, v, k, seg_idx, t_param, pt_coord)`.
    """
    candidates = []
    for u, v, k, data in graph.edges(center_nid, keys=True, data=True):
        if data.get("is_bridge", False):
            continue
        pts = data.get("pts")
        if pts is None or len(pts) < 2:
            continue

        for seg_idx in range(len(pts) - 1):
            p0 = pts[seg_idx]
            p1 = pts[seg_idx + 1]
            seg_vec = p1 - p0
            seg_len = float(np.linalg.norm(seg_vec))
            num_steps = max(1, int(np.ceil(seg_len)))
            for s in range(num_steps + 1):
                t = s / float(num_steps)
                pt = p0 + t * seg_vec
                candidates.append((u, v, k, seg_idx, t, pt))
    return candidates


def _split_edge_at_sampled_point(
    graph: nx.MultiGraph,
    u: int,
    v: int,
    k: int,
    seg_idx: int,
    t_param: float,
    pt_coord: np.ndarray,
) -> int:
    """
    Inserts a new node at `pt_coord` (on segment `seg_idx` at fraction `t_param` of edge `(u, v, k)`),
    splitting `(u, v, k)` into two edges so a bridge can attach cleanly at that exact pixel.
    If `pt_coord` is within 1.5 px of `u` or `v` (such as an existing sharp corner!), snaps
    directly to `u` or `v` without creating a new subdivision node.
    """
    pos_u = np.array(graph.nodes[u]["pos"], dtype=np.float32)
    pos_v = np.array(graph.nodes[v]["pos"], dtype=np.float32)
    if float(np.linalg.norm(pt_coord - pos_u)) <= 1.5:
        return u
    if float(np.linalg.norm(pt_coord - pos_v)) <= 1.5:
        return v

    if not graph.has_edge(u, v, key=k):
        return u

    pts = graph.edges[u, v, k]["pts"]
    new_nid = max(graph.nodes, default=-1) + 1
    new_pos = (int(round(float(pt_coord[0]))), int(round(float(pt_coord[1]))))
    pt_exact = np.array([float(new_pos[0]), float(new_pos[1])], dtype=np.float32)

    pts_left = np.vstack((pts[:seg_idx + 1], [pt_exact])).astype(np.float32)
    pts_right = np.vstack(([pt_exact], pts[seg_idx + 1:])).astype(np.float32)

    if np.linalg.norm(pts[0] - pos_u) <= np.linalg.norm(pts[-1] - pos_u):
        first_node, second_node = u, v
    else:
        first_node, second_node = v, u

    graph.remove_edge(u, v, key=k)
    graph.add_node(new_nid, pos=new_pos, kind="refined_bridge")
    graph.add_edge(first_node, new_nid, pts=_clean_pts(pts_left.tolist()))
    graph.add_edge(new_nid, second_node, pts=_clean_pts(pts_right.tolist()))
    return new_nid


def refine_bridge_endpoints_along_edges(
    graph: nx.MultiGraph,
    bridges: list,
) -> list:
    """
    For any selected bridge `(comp_a, comp_b, u, v, w, pass_idx, trav)`, if `u` or `v` is a
    subdivision/loop_anchor node (NOT an endpoint, junction, or sharp corner), searches every
    pixel along the incident edges of `u` and/or `v` up to the adjacent nodes to find the
    optimal connection point, while giving a distance bonus to any sharp corner at the end
    of that edge so corners remain favoured over flat line pixels!
    """
    refined_bridges = []

    for comp_a, comp_b, u, v, w, pass_idx, trav in bridges:
        kind_u = get_node_kind(graph, u)
        kind_v = get_node_kind(graph, v)

        # Keep endpoints, junctions, AND sharp corners locked at their exact structural vertex
        u_fixed = kind_u in ("endpoint", "odd_junction", "even_junction", "corner")
        v_fixed = kind_v in ("endpoint", "odd_junction", "even_junction", "corner")

        if u_fixed and v_fixed:
            refined_bridges.append((comp_a, comp_b, u, v, w, pass_idx, trav))
            continue

        cands_u = [(u, u, -1, 0, 0.0, np.array(graph.nodes[u]["pos"], dtype=np.float32))] if u_fixed else _sample_incident_edges_at_pixel_resolution(graph, u)
        cands_v = [(v, v, -1, 0, 0.0, np.array(graph.nodes[v]["pos"], dtype=np.float32))] if v_fixed else _sample_incident_edges_at_pixel_resolution(graph, v)

        if not cands_u or not cands_v:
            refined_bridges.append((comp_a, comp_b, u, v, w, pass_idx, trav))
            continue

        pts_u = np.array([c[5] for c in cands_u], dtype=np.float32)
        pts_v = np.array([c[5] for c in cands_v], dtype=np.float32)

        tree_v = cKDTree(pts_v)
        dists, idxs_v = tree_v.query(pts_u, k=1)
        best_idx_u = int(np.argmin(dists))
        best_idx_v = int(idxs_v[best_idx_u])

        best_u_cand = cands_u[best_idx_u]
        best_v_cand = cands_v[best_idx_v]

        new_u = u if u_fixed else _split_edge_at_sampled_point(graph, *best_u_cand)
        new_v = v if v_fixed else _split_edge_at_sampled_point(graph, *best_v_cand)

        refined_bridges.append((comp_a, comp_b, new_u, new_v, w, pass_idx, trav))

    return refined_bridges


def prune_and_merge_bridges_even_parity(
    graph: nx.MultiGraph,
    mst_1_raw: list,
    mst_2_raw: list,
    num_comps: int,
    parallel_cos_thresh: float = 0.92,
) -> list:
    """
    1. Guarantees every sub-graph has an EVEN number of bridge connections via exact T-join parity.
    2. Merges almost-parallel bridges between the same sub-graph pair into a single double-traversal
       edge (`trav=2`) if neither comes from an endpoint or odd-degree junction.
    """
    comp_deg_1 = {i: 0 for i in range(num_comps)}
    for comp_a, comp_b, _u, _v, _w in mst_1_raw:
        comp_deg_1[comp_a] += 1
        comp_deg_1[comp_b] += 1

    tree_2 = nx.Graph()
    tree_2.add_nodes_from(range(num_comps))
    for comp_a, comp_b, u, v, w in mst_2_raw:
        tree_2.add_edge(comp_a, comp_b, u=u, v=v, weight=w)

    odd_set = {c for c, d in comp_deg_1.items() if d % 2 == 1}
    kept_mst_2 = []
    for comp_a, comp_b, u, v, w in mst_2_raw:
        tree_2.remove_edge(comp_a, comp_b)
        sub_comp_a = nx.node_connected_component(tree_2, comp_a)
        odd_count_in_side = len(sub_comp_a & odd_set)
        tree_2.add_edge(comp_a, comp_b, u=u, v=v, weight=w)

        if odd_count_in_side % 2 == 1:
            kept_mst_2.append((comp_a, comp_b, u, v, w))

    # Each bridge tuple is: (comp_a, comp_b, u, v, weight, pass_idx, traversals)
    active_bridges = [(a, b, u, v, w, 1, 1) for a, b, u, v, w in mst_1_raw] + [
        (a, b, u, v, w, 2, 1) for a, b, u, v, w in kept_mst_2
    ]

    by_pair = {}
    for b_item in active_bridges:
        comp_a, comp_b = b_item[0], b_item[1]
        pair_key = (min(comp_a, comp_b), max(comp_a, comp_b))
        by_pair.setdefault(pair_key, []).append(b_item)

    final_bridges = []
    merged_parallel_count = 0

    for pair_key, bridge_list in by_pair.items():
        if len(bridge_list) != 2:
            final_bridges.extend(bridge_list)
            continue

        b1, b2 = bridge_list[0], bridge_list[1]
        ca, cb, u1, v1, w1, p1, _t1 = b1
        _, _, u2, v2, w2, p2, _t2 = b2

        protected_kinds = {"endpoint", "odd_junction"}
        has_protected_node = any(
            get_node_kind(graph, nid) in protected_kinds for nid in (u1, v1, u2, v2)
        )

        if not has_protected_node:
            vec1 = np.array(graph.nodes[v1]["pos"], dtype=np.float32) - np.array(graph.nodes[u1]["pos"], dtype=np.float32)
            vec2 = np.array(graph.nodes[v2]["pos"], dtype=np.float32) - np.array(graph.nodes[u2]["pos"], dtype=np.float32)
            n1, n2 = float(np.linalg.norm(vec1)), float(np.linalg.norm(vec2))

            if (n1 > 1e-6 and n2 > 1e-6 and abs(float(np.dot(vec1, vec2)) / (n1 * n2)) >= parallel_cos_thresh) or (n1 <= 1e-6 or n2 <= 1e-6):
                best_b = (ca, cb, u1, v1, w1, p1, 2) if w1 <= w2 else (ca, cb, u2, v2, w2, p2, 2)
                final_bridges.append(best_b)
                merged_parallel_count += 1
                continue

        final_bridges.extend(bridge_list)

    if merged_parallel_count > 0:
        print(f"Merged {merged_parallel_count} pair(s) of almost-parallel non-endpoint bridges into singular edges.")

    return final_bridges


def prune_excess_subgraph_bridges(
    bridges: list,
    num_comps: int,
) -> list:
    """
    Final Excess-Bridge Pruning Pass:
    Removes unnecessary bridges from sub-graphs that have > 2 bridge connections, strictly
    ensuring that:
      1. The component meta-graph remains globally connected (1 connected component).
      2. No sub-graph drops below 2 bridge connections.
      3. No sub-graph is left with an ODD number of bridge connections.

    Handles both:
      - Pass A: Removing a merged double-traversal bridge (`trav == 2`) between two sub-graphs
        that each have >= 4 connections (reducing both by 2 -> still even and >= 2).
      - Pass B: Removing redundant pairs/cycles of single bridges (`trav == 1`) where every
        sub-graph on the cycle has >= 4 connections (reducing each by 2 -> still even and >= 2).
    """
    if num_comps <= 1 or not bridges:
        return bridges

    kept = list(bridges)
    removed_count = 0

    def _comp_conn_counts(b_list):
        counts = {i: 0 for i in range(num_comps)}
        for ca, cb, _u, _v, _w, _p, trav in b_list:
            counts[ca] += trav
            counts[cb] += trav
        return counts

    def _is_meta_connected(b_list):
        mg = nx.Graph()
        mg.add_nodes_from(range(num_comps))
        for ca, cb, *_ in b_list:
            mg.add_edge(ca, cb)
        return nx.is_connected(mg)

    # Pass A: Try removing highest-weight merged double-bridges (trav == 2) when both ends have >= 4 connections
    changed = True
    while changed:
        changed = False
        counts = _comp_conn_counts(kept)
        # Sort candidate indices by descending weight so the worst excess bridges are removed first
        cand_indices = sorted(range(len(kept)), key=lambda idx: kept[idx][4], reverse=True)
        for idx in cand_indices:
            ca, cb, _u, _v, _w, _p, trav = kept[idx]
            if trav == 2 and counts[ca] >= 4 and counts[cb] >= 4:
                trial = kept[:idx] + kept[idx + 1:]
                if _is_meta_connected(trial):
                    kept = trial
                    removed_count += 1
                    changed = True
                    break

    # Pass B: Try removing pairs/cycles of single bridges (trav == 1) where all participating
    # sub-graphs have >= 4 connections (so removing the cycle drops each by 2 -> stays >= 2 and even!)
    changed = True
    while changed:
        changed = False
        counts = _comp_conn_counts(kept)

        # Build subgraph of single bridges whose BOTH endpoints have >= 4 connections
        high_deg_edges = [
            (idx, kept[idx][0], kept[idx][1], kept[idx][4])
            for idx in range(len(kept))
            if kept[idx][6] == 1 and counts[kept[idx][0]] >= 4 and counts[kept[idx][1]] >= 4
        ]
        if len(high_deg_edges) < 2:
            break

        # 1. Check parallel pairs of single bridges between the same two high-degree sub-graphs
        by_pair = {}
        for idx, ca, cb, w in high_deg_edges:
            pair_key = (min(ca, cb), max(ca, cb))
            by_pair.setdefault(pair_key, []).append((w, idx))

        pair_removed = False
        for pair_key, items in sorted(by_pair.items(), key=lambda kv: sum(x[0] for x in kv[1]), reverse=True):
            if len(items) >= 2:
                items.sort(reverse=True)
                drop_set = {items[0][1], items[1][1]}
                trial = [b for i, b in enumerate(kept) if i not in drop_set]
                if _is_meta_connected(trial):
                    kept = trial
                    removed_count += 2
                    changed = True
                    pair_removed = True
                    break
        if pair_removed:
            continue

        # 2. Check simple cycles of single bridges among sub-graphs with >= 4 connections
        aux_g = nx.Graph()
        edge_lookup = {}
        for idx, ca, cb, w in high_deg_edges:
            aux_g.add_edge(ca, cb, weight=w, idx=idx)
            edge_lookup[(min(ca, cb), max(ca, cb))] = idx

        cycles = nx.cycle_basis(aux_g)
        if not cycles:
            break

        # Sort cycles by total weight descending
        def _cycle_weight(cyc):
            return sum(aux_g[cyc[k]][cyc[(k + 1) % len(cyc)]]["weight"] for k in range(len(cyc)))

        cycles.sort(key=_cycle_weight, reverse=True)
        for cyc in cycles:
            drop_set = {
                edge_lookup[(min(cyc[k], cyc[(k + 1) % len(cyc)]), max(cyc[k], cyc[(k + 1) % len(cyc)]))]
                for k in range(len(cyc))
            }
            trial = [b for i, b in enumerate(kept) if i not in drop_set]
            if _is_meta_connected(trial):
                kept = trial
                removed_count += len(drop_set)
                changed = True
                break

    if removed_count > 0:
        print(f"Final excess-bridge pass removed {removed_count} unnecessary bridge(s) from sub-graphs with >2 connections.")
    return kept


def bridge_components_two_pass_kruskal(graph: nx.MultiGraph, diagonal: float, config: dict) -> nx.MultiGraph:
    """
    Executes the complete component bridging stage:
      1. Pass 1 Kruskal MST: Weighted by proportional Euclidean distance, node type (Corners >> Subdivision),
         and exit angle.
      2. Pass 2 Kruskal MST: Same weights + small penalty (`eta_same_comp_pair` = 0.35) for
         bridging two sub-graphs already connected in Pass 1.
      3. Even-Parity Sub-Graph Pruning & Parallel Bridge Merging.
      4. Final Excess-Bridge Pruning Pass: Removes any unneeded bridges from sub-graphs with > 2
         connections without leaving any sub-graph with < 2 or an odd number of bridges.
      5. Local Pixel-Level Edge Refinement: For any bridge endpoint that is not a junction, endpoint,
         or sharp corner, searches all pixels along its incident edges up to the adjacent nodes.
    """
    H = graph.copy()
    components = list(nx.connected_components(H))
    if len(components) <= 1:
        print("Graph is already a single connected component (0 bridges needed).")
        return H

    beta_ang = float(config.get("bridge_beta_ang", 0.35))
    kappa_kind = float(config.get("bridge_kappa_kind", 1.50))
    eta_pair = float(config.get("bridge_eta_same_pair", 0.35))

    # --- PASS 1: Base Kruskal MST ---
    mst_1_raw = solve_component_mst_kruskal(
        graph=H,
        components=components,
        diagonal=diagonal,
        beta_ang=beta_ang,
        kappa_kind=kappa_kind,
    )

    used_comp_pairs = {(min(a, b), max(a, b)) for a, b, _u, _v, _w in mst_1_raw}
    used_nodes = {nid for _a, _b, u, v, _w in mst_1_raw for nid in (u, v)}

    # --- PASS 2: Kruskal MST with small penalty on already-bridged sub-graph pairs ---
    mst_2_raw = solve_component_mst_kruskal(
        graph=H,
        components=components,
        diagonal=diagonal,
        beta_ang=beta_ang,
        kappa_kind=kappa_kind,
        used_comp_pairs=used_comp_pairs,
        used_nodes=used_nodes,
        eta_same_comp_pair=eta_pair,
    )

    # --- STEP 3: Even-Parity Sub-Graph Pruning & Parallel Bridge Merging ---
    parity_bridges = prune_and_merge_bridges_even_parity(
        graph=H,
        mst_1_raw=mst_1_raw,
        mst_2_raw=mst_2_raw,
        num_comps=len(components),
    )

    # --- STEP 4: Final Excess-Bridge Pruning Pass (>2 bridges -> remove if not needed) ---
    pruned_bridges = prune_excess_subgraph_bridges(
        bridges=parity_bridges,
        num_comps=len(components),
    )

    # --- STEP 5: Local Pixel-Level Edge Refinement for Subdivision Attachments ---
    refined_bridges = refine_bridge_endpoints_along_edges(
        graph=H,
        bridges=pruned_bridges,
    )

    for _comp_a, _comp_b, u, v, w, pass_idx, trav in refined_bridges:
        pos_u = np.array(H.nodes[u]["pos"], dtype=np.float32)
        pos_v = np.array(H.nodes[v]["pos"], dtype=np.float32)
        bridge_pts = np.vstack((pos_u, pos_v)).astype(np.float32)
        H.add_edge(u, v, pts=bridge_pts, is_bridge=True, mst_pass=pass_idx, traversals=trav, weight=w)

    print(
        f"Bridged {len(components)} sub-graphs via Even-Parity 2-Pass Kruskal MST: "
        f"{len(refined_bridges)} total bridges inserted."
    )
    return H


def count_graph_points(graph: nx.MultiGraph) -> int:
    """Counts total polyline vertex points across all edges in the graph."""
    return sum(len(data["pts"]) for _, _, _, data in graph.edges(keys=True, data=True))


# =============================================================================
# 6. PIPELINE ORCHESTRATION & CLI
# =============================================================================

def image_to_oneline(input_image: str, config: dict):
    """
    Executes the complete image processing, optional downsampled skeletonisation + vector
    coordinate upscaling, lossless loop-preserving graph construction, RDP simplification,
    loop-safe junction contraction, proportional degree-1 edge pruning, Hamiltonian dot cycle
    connection, proportional disconnected sub-graph pruning, corner/long-edge splitting, and
    even-parity 2-pass Kruskal MST bridging with excess-bridge pruning and pixel refinement.
    """
    input_path = Path(input_image)
    input_stem = input_path.stem

    target_dir = Path(config.get("output_dir", input_path.parent)) / input_stem
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory initialized at: {target_dir}")

    # Step 1: Ingest, Crop to Tight Bounds, Optionally Downsample for Thick Lines, & Pad
    binary_mask, (H_canvas, W_canvas, diagonal), (scale_r, scale_c) = import_image_as_mask(
        path=input_image, config=config
    )
    w_draw = W_canvas - 2
    avg_scale = 0.5 * (scale_r + scale_c)

    if config.get("node_merge_distance") is None:
        prop_node = config.get("node_merge_ratio", 0.006)
        config["node_merge_distance"] = prop_node * diagonal
        print(f"Adaptive Stage-1 node merge radius: {config['node_merge_distance']:.2f} px ({prop_node*100:.2f}% of diagonal).")
    config["node_merge_distance_work"] = config["node_merge_distance"] / avg_scale

    if config.get("edge_merge_distance") is None:
        prop_edge = config.get("edge_merge_ratio", 0.020)
        config["edge_merge_distance"] = prop_edge * diagonal
        print(f"Adaptive Stage-2 junction contraction threshold: {config['edge_merge_distance']:.2f} px ({prop_edge*100:.2f}% of diagonal).")

    if config.get("whisker_prune_distance") is None:
        prop_whisker = config.get("whisker_prune_ratio", 0.010)
        config["whisker_prune_distance"] = prop_whisker * diagonal
        print(f"Adaptive degree-1 terminal edge prune threshold: {config['whisker_prune_distance']:.2f} px ({prop_whisker*100:.2f}% of diagonal).")

    if config.get("max_edge_length") is None:
        prop_max_edge = config.get("max_edge_ratio", 0.15)
        config["max_edge_length"] = prop_max_edge * w_draw
        print(f"Adaptive max edge subdivision length: {config['max_edge_length']:.2f} px ({prop_max_edge*100:.1f}% of drawing width).")

    if config.get("save_binary_mask", True):
        save_mask_as_image(binary_mask, str(target_dir / f"{input_stem}_binary_mask.png"), invert=True)

    # Step 2: Skeletonisation (on working grid)
    skel = skeletonise_mask(mask=binary_mask, config=config)
    if config.get("save_skeleton", True):
        save_mask_as_image(skel, str(target_dir / f"{input_stem}_skeleton.png"), invert=True)

    # Step 3: Rutovitz Crossing Numbers
    cn = skeleton_to_crossing_numbers(skeleton=skel)
    if config.get("save_crossing_map", True):
        save_crossing_map(skel, cn, str(target_dir / f"{input_stem}_crossing_map.png"))

    # Step 4: Extract Canonical Nodes
    nodes, node_types, node_grid, node_pixel_members = extract_canonical_nodes(skel, cn, config)
    if config.get("save_nodes_map", True):
        save_nodes_map(skel, nodes, node_types, str(target_dir / f"{input_stem}_collapsed_nodes.png"))

    # Step 5: Trace Skeleton Paths into NetworkX MultiGraph & Upscale Coordinates to Original Resolution
    raw_graph_work = trace_skeleton_to_graph(skel, nodes, node_grid, node_pixel_members, node_types)
    raw_graph = upscale_graph_coordinates(raw_graph_work, scale_r, scale_c)

    # Step 6: Export Dense Traced Graph to SVG
    save_graph_as_svg(raw_graph, W_canvas, H_canvas, str(target_dir / f"{input_stem}_traced_dense.svg"))
    dense_point_count = count_graph_points(raw_graph)

    # Step 7: Apply Ramer–Douglas–Peucker (RDP) Simplification
    rdp_graph = apply_rdp_to_graph(raw_graph, epsilon=config.get("rdp_epsilon", 1.5))
    save_graph_as_svg(rdp_graph, W_canvas, H_canvas, str(target_dir / f"{input_stem}_rdp_simplified.svg"))
    rdp_point_count = count_graph_points(rdp_graph)

    # Step 8: Loop-Safe Junction Contraction (never deletes parallel loop petals or self-loops)
    contracted_graph = contract_short_edges(
        rdp_graph,
        min_edge_length=config["edge_merge_distance"],
    )

    # Step 9: Prune Short Edges Connected to Degree-1 Endpoints (% of diagonal)
    terminal_pruned_graph = prune_terminal_edges(
        contracted_graph,
        max_whisker_length=config["whisker_prune_distance"],
    )

    # Step 10: Connect Nearby Dots into a Hamiltonian Cycle BEFORE Sub-Graph Pruning
    dot_cycle_graph = combine_nearby_dots(
        terminal_pruned_graph,
        skeleton=skel,
        crossing_number=cn,
        merge_distance=config["edge_merge_distance"],
        scale_rc=(scale_r, scale_c),
    )

    # Step 11: Prune Disconnected Sub-Graphs with Total Length < % of Diagonal -> Produces `simple_graph`
    simple_graph = prune_small_components(
        dot_cycle_graph,
        diagonal=diagonal,
        min_diag_ratio=config.get("min_component_ratio", 0.03),
    )
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

    # Step 12: Split Sharp Corners & Subdivide Long Edges
    corner_split_graph = split_edges_at_sharp_corners(
        simple_graph,
        max_turn_angle_deg=config.get("corner_split_angle_deg", 65.0),
    )
    subdivided_graph = subdivide_long_edges(
        corner_split_graph,
        max_edge_length=config["max_edge_length"],
    )

    # Step 13: Two-Pass Kruskal MST Bridging + Excess-Bridge Pruning + Pixel Refinement -> `final_graph`
    if config.get("bridge_components", True):
        final_graph = bridge_components_two_pass_kruskal(subdivided_graph, diagonal=diagonal, config=config)
    else:
        final_graph = subdivided_graph

    # Save Final Graph (SVG & PNG)
    final_svg_path = target_dir / f"{input_stem}_final.svg"
    save_graph_as_svg(final_graph, W_canvas, H_canvas, str(final_svg_path))
    final_point_count = count_graph_points(final_graph)

    if config.get("save_final_graph_map", True):
        save_graph_map(
            final_graph, W_canvas, H_canvas,
            str(target_dir / f"{input_stem}_final_graph.png"),
            show_split_nodes=True,
        )

    reduction_rdp = ((dense_point_count - rdp_point_count) / dense_point_count) * 100.0 if dense_point_count > 0 else 0.0
    reduction_simple = ((dense_point_count - simple_point_count) / dense_point_count) * 100.0 if dense_point_count > 0 else 0.0

    print("\n" + "=" * 65)
    print("PIPELINE GRAPH REFINEMENT & DECIMATION BENCHMARK")
    print(f"  - Dense Traced Points:       {dense_point_count:,} pts  ({raw_graph.number_of_nodes()} nodes, {raw_graph.number_of_edges()} edges)")
    print(f"  - RDP Simplified Points:     {rdp_point_count:,} pts  ({reduction_rdp:.2f}% decimation)")
    print(f"  - Simple Graph (Contracted): {simple_point_count:,} pts  ({simple_graph.number_of_nodes()} nodes, {simple_graph.number_of_edges()} edges, {reduction_simple:.2f}% reduction)")
    print(f"  - Final Graph (Bridged):     {final_point_count:,} pts  ({final_graph.number_of_nodes()} nodes, {final_graph.number_of_edges()} edges)")
    print("=" * 65 + "\n")

    return simple_graph, final_graph


def main():
    parser = argparse.ArgumentParser(description="Convert any PNG image into a clean, bridged topological graph.")
    parser.add_argument("input", nargs="?", default="images/example.png", help="Path to input image")
    parser.add_argument("-o", "--output-dir", default=None, help="Base output directory")
    parser.add_argument("-t", "--threshold", type=int, default=128, help="Luminance intensity threshold (0-255)")
    parser.add_argument("--method", default="lee", choices=["lee", "zhang"], help="Skeletonisation method")
    parser.add_argument("--downsample-factor", type=float, default=1.0,
                        help="Downscale factor (>= 1.0) before skeletonisation to smooth thick pen strokes, then upscale vector coordinates (e.g. 2.0)")
    parser.add_argument("--node-merge-dist", type=float, default=None, help="Explicit spatial distance (px) to merge close nodes.")
    parser.add_argument("--node-merge-ratio", type=float, default=0.006, help="Proportional node merge ratio of diagonal (default: 0.006)")
    parser.add_argument("--edge-merge-dist", type=float, default=None, help="Explicit arc-length threshold (px) to merge nearby junctions.")
    parser.add_argument("--edge-merge-ratio", type=float, default=0.020, help="Proportional junction merge ratio of diagonal (default: 0.020)")
    parser.add_argument("--whisker-dist", type=float, default=None, help="Explicit max length (px) to prune degree-1 terminal edges.")
    parser.add_argument("--whisker-ratio", type=float, default=0.010, help="Prune degree-1 edges shorter than this ratio of diagonal (default: 0.010 = 1.0%%)")
    parser.add_argument("--min-component-ratio", type=float, default=0.030, help="Prune sub-graphs shorter than this ratio of diagonal (default: 0.030 = 3.0%%)")
    parser.add_argument("--rdp-epsilon", type=float, default=1.5, help="RDP perpendicular deviation tolerance in pixels (default: 1.5)")
    parser.add_argument("--corner-angle", type=float, default=65.0, help="Split edges at turns sharper than this deflection angle in degrees (default: 65.0)")
    parser.add_argument("--max-edge-length", type=float, default=None, help="Explicit max edge length (px) before subdivision.")
    parser.add_argument("--max-edge-ratio", type=float, default=0.05, help="Max edge length as a ratio of drawing width before subdivision (default: 0.15 = 15%%)")
    parser.add_argument("--no-bridge", action="store_true", help="Disable 2-pass Kruskal MST component bridging")
    parser.add_argument("--no-save-mask", action="store_true", help="Disable saving intermediate binary mask")
    parser.add_argument("--no-save-skel", action="store_true", help="Disable saving intermediate skeleton")
    parser.add_argument("--no-save-crossing", action="store_true", help="Disable saving raw crossing map")
    parser.add_argument("--no-save-nodes", action="store_true", help="Disable saving collapsed canonical nodes map")
    parser.add_argument("--no-save-simple-graph", action="store_true", help="Disable saving simple graph diagnostic map")
    parser.add_argument("--no-save-final-graph", action="store_true", help="Disable saving final graph diagnostic map")

    args = parser.parse_args()
    input_path = Path(args.input)
    base_out = args.output_dir if args.output_dir is not None else str(input_path.parent / "outputs")

    config = {
        "output_dir": base_out,
        "intensity_threshold": args.threshold,
        "skeleton_mode": args.method,
        "downsample_factor": args.downsample_factor,
        "node_merge_distance": args.node_merge_dist,
        "node_merge_ratio": args.node_merge_ratio,
        "edge_merge_distance": args.edge_merge_dist,
        "edge_merge_ratio": args.edge_merge_ratio,
        "whisker_prune_distance": args.whisker_dist,
        "whisker_prune_ratio": args.whisker_ratio,
        "min_component_ratio": args.min_component_ratio,
        "rdp_epsilon": args.rdp_epsilon,
        "corner_split_angle_deg": args.corner_angle,
        "max_edge_length": args.max_edge_length,
        "max_edge_ratio": args.max_edge_ratio,
        "bridge_components": not args.no_bridge,
        "bridge_beta_ang": 0.35,
        "bridge_kappa_kind": 1.50,
        "bridge_eta_same_pair": 0.35,
        "save_binary_mask": not args.no_save_mask,
        "save_skeleton": not args.no_save_skel,
        "save_crossing_map": not args.no_save_crossing,
        "save_nodes_map": not args.no_save_nodes,
        "save_simple_graph_map": not args.no_save_simple_graph,
        "save_final_graph_map": not args.no_save_final_graph,
    }

    try:
        image_to_oneline(input_image=str(input_path), config=config)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()