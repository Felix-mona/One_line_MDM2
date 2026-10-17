import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize


def remove_small_specks(binary_mask: np.ndarray, min_area: int = 12) -> np.ndarray:
    """Removes isolated connected components smaller than min_area pixels."""
    if min_area <= 1:
        return binary_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def image_to_binary_mask(
    image_path: str,
    mode: str = "primal",
    threshold_val: int = 200,
    min_speck_area: int = 12,
    save_preview_path: str = None,
) -> np.ndarray:
    """Converts any image or digital drawing into a binary mask (255 = ink, 0 = background).

    Modes:
        - 'primal'  : Detailed centerline sketch; keeps fine linework.
        - 'minimal' : Stripped skeletal sketch; removes hatching and fine textures.
        - 'xdog'    : Extended Difference of Gaussians for natural photographs.
        - 'canny'   : Gradient-based Canny edge detector.
        - 'drawing' : Direct thresholding for clean digital line art / drawings.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")

    # Handle RGBA transparency by compositing onto white, with fallback for light-on-transparent strokes
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        alpha_f = a.astype(np.float32) / 255.0
        bgr = cv2.merge([b, g, r]).astype(np.float32)
        white_bg = np.full_like(bgr, 255.0)
        composited = (bgr * alpha_f[..., None] + white_bg * (1.0 - alpha_f[..., None])).astype(np.uint8)
        gray = cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY)

        if np.min(gray) > threshold_val and np.min(a) < 10:
            _, binary_ink = cv2.threshold(a, 10, 255, cv2.THRESH_BINARY)
            return remove_small_specks(binary_ink, min_area=min_speck_area)
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if mode == "drawing":
        _, binary_ink = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

    elif mode == "primal":
        smooth = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)
        adaptive_ink = cv2.adaptiveThreshold(
            smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=15, C=9
        )
        dist = cv2.distanceTransform(adaptive_ink, cv2.DIST_L2, 5)
        thick_interior = (dist > 4.5).astype(np.uint8) * 255
        binary_ink = cv2.subtract(adaptive_ink, thick_interior)

    elif mode == "minimal":
        smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=100, sigmaSpace=100)
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=100, sigmaSpace=100)
        adaptive_ink = cv2.adaptiveThreshold(
            smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=25, C=16
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        adaptive_ink = cv2.morphologyEx(adaptive_ink, cv2.MORPH_OPEN, kernel)
        dist = cv2.distanceTransform(adaptive_ink, cv2.DIST_L2, 5)
        thick_interior = (dist > 4.0).astype(np.uint8) * 255
        binary_ink = cv2.subtract(adaptive_ink, thick_interior)

    elif mode == "xdog":
        smooth = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)
        g1 = cv2.GaussianBlur(smooth, (3, 3), 0.8)
        g2 = cv2.GaussianBlur(smooth, (7, 7), 2.0)
        dog = cv2.subtract(g2, g1)
        _, binary_ink = cv2.threshold(dog, 7, 255, cv2.THRESH_BINARY)

    elif mode == "canny":
        smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=60, sigmaSpace=60)
        v = np.median(smooth)
        lower = int(max(0, 0.67 * v))
        upper = int(min(255, 1.33 * v))
        binary_ink = cv2.Canny(smooth, lower, upper)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'primal', 'minimal', 'xdog', 'canny', or 'drawing'.")

    cleaned_ink = remove_small_specks(binary_ink, min_area=min_speck_area)

    if save_preview_path:
        cv2.imwrite(save_preview_path, 255 - cleaned_ink)

    return cleaned_ink


def skeletonize_mask(binary_mask: np.ndarray, method: str = "lee") -> np.ndarray:
    """Reduces a binary mask (0 and 255) to a 1-pixel-wide centerline skeleton."""
    skeleton_bool = skeletonize(binary_mask > 0, method=method)
    return (skeleton_bool.astype(np.uint8)) * 255


def extract_skeleton_topology(skeleton: np.ndarray):
    """Detects endpoints, junction clusters, and traces 8-connected edge paths in one pass.

    Returns:
        endpoints (list of tuple): [(x, y), ...] degree-1 nodes.
        junctions (list of tuple): [(x, y), ...] degree >= 3 cluster centroids.
        edges (list of dict): [{'start_node': (x, y), 'end_node': (x, y), 'path': [...]}, ...]
    """
    height, width = skeleton.shape
    skel_binary = (skeleton > 0).astype(np.uint8)

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = convolve(skel_binary, kernel, mode="constant", cval=0) * skel_binary

    node_id_mask = np.zeros((height, width), dtype=np.int32)
    node_coords = {}
    node_pixels = {}
    current_id = 1

    # 1. Extract and register degree-1 endpoints
    endpoint_indices = np.argwhere(neighbor_count == 1)
    endpoints = [(int(pt[1]), int(pt[0])) for pt in endpoint_indices]
    for ex, ey in endpoints:
        node_id_mask[ey, ex] = current_id
        node_coords[current_id] = (ex, ey)
        node_pixels[current_id] = [(ex, ey)]
        current_id += 1

    # 2. Extract and cluster degree >= 3 junctions
    raw_junction_mask = (neighbor_count >= 3).astype(np.uint8)
    num_labels, labels, _, centroids = cv2.connectedComponentsWithStats(
        raw_junction_mask, connectivity=8
    )

    junctions = []
    for i in range(1, num_labels):
        pts = np.argwhere(labels == i)
        cx, cy = centroids[i]
        dists = (pts[:, 1] - cx) ** 2 + (pts[:, 0] - cy) ** 2
        best_pt = pts[np.argmin(dists)]
        rep_coord = (int(best_pt[1]), int(best_pt[0]))
        junctions.append(rep_coord)

        node_coords[current_id] = rep_coord
        cluster_xy = []
        for r, c in pts:
            node_id_mask[r, c] = current_id
            cluster_xy.append((int(c), int(r)))
        node_pixels[current_id] = cluster_xy
        current_id += 1

    # 3. Walk 8-connected paths between registered nodes
    unvisited = (skel_binary > 0) & (node_id_mask == 0)
    directions = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ]

    def get_neighbors(x, y):
        for dx, dy in directions:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < width and 0 <= ny_ < height:
                yield nx_, ny_

    edges = []
    visited_direct_node_pairs = set()

    for nid, cluster_pts in node_pixels.items():
        start_node = node_coords[nid]

        for px, py in cluster_pts:
            for nx_, ny_ in get_neighbors(px, py):
                if unvisited[ny_, nx_]:
                    path = [start_node, (nx_, ny_)]
                    unvisited[ny_, nx_] = False

                    curr_x, curr_y = nx_, ny_
                    prev_x, prev_y = px, py
                    end_node = None

                    while True:
                        nbrs = [
                            (ux, uy) for ux, uy in get_neighbors(curr_x, curr_y)
                            if unvisited[uy, ux]
                        ]
                        if nbrs:
                            nbrs.sort(key=lambda pt: abs(pt[0] - curr_x) + abs(pt[1] - curr_y))
                            next_x, next_y = nbrs[0]
                            unvisited[next_y, next_x] = False
                            path.append((next_x, next_y))
                            prev_x, prev_y = curr_x, curr_y
                            curr_x, curr_y = next_x, next_y
                        else:
                            adj_node_ids = [
                                int(node_id_mask[uy, ux])
                                for ux, uy in get_neighbors(curr_x, curr_y)
                                if node_id_mask[uy, ux] > 0 and (ux, uy) != (prev_x, prev_y)
                            ]
                            if adj_node_ids:
                                end_nid = adj_node_ids[0]
                                end_node = node_coords[end_nid]
                                if path[-1] != end_node:
                                    path.append(end_node)
                            else:
                                end_node = (curr_x, curr_y)
                            break

                    edges.append({
                        "start_node": start_node,
                        "end_node": end_node,
                        "path": path,
                    })

                elif node_id_mask[ny_, nx_] > 0 and node_id_mask[ny_, nx_] != nid:
                    other_nid = int(node_id_mask[ny_, nx_])
                    pair_key = (min(nid, other_nid), max(nid, other_nid))
                    if pair_key not in visited_direct_node_pairs:
                        visited_direct_node_pairs.add(pair_key)
                        end_node = node_coords[other_nid]
                        edges.append({
                            "start_node": start_node,
                            "end_node": end_node,
                            "path": [start_node, end_node],
                        })

    # 4. Trace isolated closed loops without endpoints or junctions
    remaining_indices = np.argwhere(unvisited)
    for row, col in remaining_indices:
        if not unvisited[row, col]:
            continue

        start_pixel = (int(col), int(row))
        path = [start_pixel]
        unvisited[row, col] = False
        curr_x, curr_y = start_pixel

        while True:
            nbrs = [
                (ux, uy) for ux, uy in get_neighbors(curr_x, curr_y)
                if unvisited[uy, ux]
            ]
            if nbrs:
                nbrs.sort(key=lambda pt: abs(pt[0] - curr_x) + abs(pt[1] - curr_y))
                next_x, next_y = nbrs[0]
                unvisited[next_y, next_x] = False
                path.append((next_x, next_y))
                curr_x, curr_y = next_x, next_y
            else:
                path.append(start_pixel)
                break

        if len(path) > 3:
            edges.append({
                "start_node": start_pixel,
                "end_node": start_pixel,
                "path": path,
            })

    return endpoints, junctions, edges


def rdp_simplify(points: list, epsilon: float = 1.0) -> list:
    """Simplifies a sequence of (x, y) coordinates using Ramer-Douglas-Peucker (RDP)."""
    if len(points) <= 2:
        return points

    if len(points) > 3 and points[0] == points[-1]:
        mid_idx = len(points) // 2
        left_branch = rdp_simplify(points[:mid_idx + 1], epsilon)
        right_branch = rdp_simplify(points[mid_idx:], epsilon)
        return left_branch[:-1] + right_branch

    pts = np.array(points, dtype=np.float64)
    start_pt, end_pt = pts[0], pts[-1]
    line_vec = end_pt - start_pt
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        dists = np.linalg.norm(pts[1:-1] - start_pt, axis=1)
    else:
        diff = start_pt - pts[1:-1]
        cross_2d = line_vec[0] * diff[:, 1] - line_vec[1] * diff[:, 0]
        dists = np.abs(cross_2d) / line_len

    if len(dists) == 0:
        return [points[0], points[-1]]

    max_idx = int(np.argmax(dists)) + 1
    if dists[max_idx - 1] > epsilon:
        left_branch = rdp_simplify(points[:max_idx + 1], epsilon)
        right_branch = rdp_simplify(points[max_idx:], epsilon)
        return left_branch[:-1] + right_branch
    return [points[0], points[-1]]


def simplify_all_edges(edges: list, epsilon: float = 1.0) -> list:
    """Applies RDP decimation across all traced edges."""
    simplified = []
    for edge in edges:
        simplified.append({
            "start_node": edge["start_node"],
            "end_node": edge["end_node"],
            "path": rdp_simplify(edge["path"], epsilon=epsilon),
        })
    return simplified


def edges_to_svg(
    edges: list,
    width: int,
    height: int,
    output_path: str,
    stroke_width: float = 1.5,
    stroke_color: str = "#000000",
) -> None:
    """Serializes a list of graph edges into a standalone SVG file."""
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "version": "1.1",
        "width": str(width),
        "height": str(height),
        "viewBox": f"0 0 {width} {height}",
    })

    path_commands = []
    for edge in edges:
        pts = edge["path"]
        if len(pts) < 2:
            continue

        is_closed = (edge["start_node"] == edge["end_node"] and len(pts) > 2)
        loop_pts = pts[:-1] if (is_closed and pts[0] == pts[-1]) else pts

        start_x, start_y = loop_pts[0]
        commands = [f"M {start_x:.2f} {start_y:.2f}"]
        for x, y in loop_pts[1:]:
            commands.append(f"L {x:.2f} {y:.2f}")
        if is_closed:
            commands.append("Z")

        path_commands.append(" ".join(commands))

    ET.SubElement(svg, "path", {
        "d": " ".join(path_commands),
        "fill": "none",
        "stroke": stroke_color,
        "stroke-width": str(stroke_width),
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
    })

    raw_xml = ET.tostring(svg, encoding="utf-8")
    pretty_xml = minidom.parseString(raw_xml).toprettyxml(indent="  ")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)


def polyline_to_svg(
    polyline: list,
    width: int,
    height: int,
    output_path: str,
    stroke_width: float = 1.5,
    stroke_color: str = "#000000",
) -> None:
    """Serializes a single continuous polyline [(x, y), ...] into an SVG file."""
    edges = []
    if len(polyline) >= 2:
        edges.append({
            "start_node": polyline[0],
            "end_node": polyline[-1],
            "path": polyline,
        })
    edges_to_svg(edges, width, height, output_path, stroke_width=stroke_width, stroke_color=stroke_color)