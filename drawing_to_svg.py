import cv2
import numpy as np
import xml.etree.ElementTree as ET
import argparse
import sys
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from xml.dom import minidom


def binarize_digital_drawing(image_path: str, threshold_val: int = 200) -> np.ndarray:
    """Binarizes a digital drawing exported from an application (e.g.

    Paint, Procreate, Photoshop). Handles both transparent PNGs (Alpha channel)
    and solid white canvas backgrounds.

    Parameters:
        image_path (str): Filepath to the image.
        threshold_val (int): Brightness cutoff (0-255) for dark strokes on white canvases.
                             Defaults to 200 to capture anti-aliased edge pixels.

    Returns:
        np.ndarray: 2D uint8 binary mask where ink = 255 and canvas/background = 0.
    """
    # Load image preserving all original channels (keeps alpha channel if present)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")

    # Case 1: PNG has an Alpha channel (BGRA - 4 channels)
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)

        # Composite the foreground onto a solid white (255) background first.
        # This handles both transparent PNGs with dark ink and opaque 32-bit PNGs
        # exported with a solid white canvas (where alpha is 255 everywhere).
        alpha_f = a.astype(np.float32) / 255.0
        bgr = cv2.merge([b, g, r]).astype(np.float32)
        white_bg = np.full_like(bgr, 255.0)
        composited = (bgr * alpha_f[..., None] + white_bg * (1.0 - alpha_f[..., None])).astype(np.uint8)
        gray = cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY)

        # Edge case: If the drawing consists of pure white/light strokes on a
        # transparent background, compositing onto white makes the image blank.
        # Fall back to thresholding the alpha channel directly in that scenario.
        if np.min(gray) > threshold_val and np.min(a) < 10:
            _, binary = cv2.threshold(a, 10, 255, cv2.THRESH_BINARY)
            return binary

    # Case 2: Standard 3-channel (BGR) image with solid background
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # Already single-channel grayscale
        gray = img

    # Inverted threshold: pixels darker than threshold_val become 255 (ink),
    # while brighter pixels become 0 (canvas background)
    _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

    return binary


def skeletonize_mask(binary_mask: np.ndarray, method: str = 'lee') -> np.ndarray:
    """Reduces a binary mask (0 and 255) to a 1-pixel-wide centerline skeleton.

    Parameters:
        binary_mask (np.ndarray): 2D array where ink is 255 and background is 0.
        method (str): 'lee' or 'zhang'. 'lee' works well for thin topological paths.

    Returns:
        np.ndarray: 2D uint8 array (0 and 255) where strokes are 1 pixel wide.
    """
    # 1. Convert mask to a boolean array (True for ink, False for background)
    bool_mask = binary_mask > 0

    # 2. Perform skeletonization
    skeleton_bool = skeletonize(bool_mask, method=method)

    # 3. Convert boolean back to uint8 image format (0 and 255)
    skeleton_uint8 = (skeleton_bool.astype(np.uint8)) * 255

    return skeleton_uint8


def detect_nodes(skeleton: np.ndarray):
    """Identifies endpoints and junctions from a 1-pixel skeleton image.

    Parameters:
        skeleton (np.ndarray): 2D uint8 array where strokes are 255 and background is 0.

    Returns:
        endpoints (list of tuple): [(x, y), ...] coordinates of degree-1 endpoints.
        junctions (list of tuple): [(x, y), ...] coordinates of clustered degree >= 3 crossings.
        node_mask (np.ndarray): Diagnostic image highlighting detected nodes.
    """
    # 1. Normalize skeleton to 1s and 0s
    skel_binary = (skeleton > 0).astype(np.uint8)

    # 2. Kernel to count the 8 surrounding neighbors
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    # 3. Convolve to get neighbor count at every pixel
    neighbor_count = convolve(skel_binary, kernel, mode='constant', cval=0)

    # Restrict counts only to foreground skeleton pixels
    neighbor_count = neighbor_count * skel_binary

    # 4. Extract Endpoints (neighbor count == 1)
    # np.argwhere returns [row, col] -> convert to (x, y) = (col, row)
    endpoint_indices = np.argwhere(neighbor_count == 1)
    endpoints = [(int(pt[1]), int(pt[0])) for pt in endpoint_indices]

    # 5. Extract Junctions (neighbor count >= 3)
    raw_junction_mask = (neighbor_count >= 3).astype(np.uint8)

    # Cluster adjacent junction pixels so 1 physical crossing = 1 node
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        raw_junction_mask, connectivity=8
    )

    junctions = []
    # Label 0 is background; iterate through detected clusters
    for i in range(1, num_labels):
        pts = np.argwhere(labels == i)  # [row, col]
        cx, cy = centroids[i]

        # Snap cluster centroid to the closest actual skeleton pixel in the cluster
        # so the node coordinate never lands on an empty background pixel
        dists = (pts[:, 1] - cx) ** 2 + (pts[:, 0] - cy) ** 2
        best_pt = pts[np.argmin(dists)]
        junctions.append((int(best_pt[1]), int(best_pt[0])))

    # 6. Create diagnostic visual mask
    # Blue = Endpoints, Red = Junctions
    diagnostic_vis = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    for x, y in endpoints:
        cv2.circle(diagnostic_vis, (x, y), radius=3, color=(255, 100, 0), thickness=-1)
    for x, y in junctions:
        cv2.circle(diagnostic_vis, (x, y), radius=4, color=(0, 0, 255), thickness=-1)

    return endpoints, junctions, diagnostic_vis


def walk_edges(skeleton: np.ndarray, endpoints: list, junctions: list):
    """Traces 8-connected paths between detected nodes to produce ordered coordinate lists.

    Parameters:
        skeleton (np.ndarray): 2D uint8 array (0 and 255) of the 1-pixel skeleton.
        endpoints (list of tuple): Degree-1 coordinate points [(x, y), ...].
        junctions (list of tuple): Clustered degree >= 3 junction centroids [(x, y), ...].

    Returns:
        edges (list of dict): List of edge dictionaries containing:
            - 'start_node': (x, y)
            - 'end_node': (x, y)
            - 'path': [(x0, y0), (x1, y1), ... (xn, yn)] in drawing order.
    """
    height, width = skeleton.shape
    skel_binary = (skeleton > 0).astype(np.uint8)

    # Recompute neighbor counts to identify full multi-pixel junction clusters
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)
    neighbor_count = convolve(skel_binary, kernel, mode='constant', cval=0) * skel_binary

    # 1. Map all node pixels (endpoints and full junction clusters) to unique Node IDs
    # node_id_mask stores 0 for normal stroke/background pixels and >0 for Node IDs
    node_id_mask = np.zeros((height, width), dtype=np.int32)
    node_coords = {}  # node_id -> representative (x, y) coordinate
    node_pixels = {}  # node_id -> list of all (x, y) pixels belonging to that node cluster

    current_id = 1

    # Register degree-1 endpoints
    for ex, ey in endpoints:
        if 0 <= ey < height and 0 <= ex < width:
            node_id_mask[ey, ex] = current_id
            node_coords[current_id] = (ex, ey)
            node_pixels[current_id] = [(ex, ey)]
            current_id += 1

    # Register multi-pixel junction clusters so paths terminate at cluster boundaries
    # rather than walking straight through non-centroid junction pixels
    raw_junction_mask = (neighbor_count >= 3).astype(np.uint8)
    num_labels, labels, _, centroids = cv2.connectedComponentsWithStats(
        raw_junction_mask, connectivity=8
    )

    for i in range(1, num_labels):
        pts = np.argwhere(labels == i)
        cx, cy = centroids[i]
        dists = (pts[:, 1] - cx) ** 2 + (pts[:, 0] - cy) ** 2
        best_pt = pts[np.argmin(dists)]
        rep_coord = (int(best_pt[1]), int(best_pt[0]))

        node_coords[current_id] = rep_coord
        cluster_xy = []
        for r, c in pts:
            node_id_mask[r, c] = current_id
            cluster_xy.append((int(c), int(r)))
        node_pixels[current_id] = cluster_xy
        current_id += 1

    # Tracking mask of remaining untraced interior stroke pixels (excludes node pixels)
    unvisited = (skel_binary > 0) & (node_id_mask == 0)

    # Relative 8-neighborhood offsets (dx, dy)
    directions = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1)
    ]

    def get_neighbors(x, y):
        """Yields valid in-bounds neighboring coordinates (nx, ny)."""
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                yield nx, ny

    edges = []
    visited_direct_node_pairs = set()

    # 2. Trace edges originating from all known nodes (endpoints and junction clusters)
    for nid, cluster_pts in node_pixels.items():
        start_node = node_coords[nid]

        # Inspect all perimeter pixels around this node cluster
        for px, py in cluster_pts:
            for nx, ny in get_neighbors(px, py):
                # Case A: Neighbor is an unvisited interior stroke pixel
                if unvisited[ny, nx]:
                    path = [start_node, (nx, ny)]
                    unvisited[ny, nx] = False

                    curr_x, curr_y = nx, ny
                    prev_x, prev_y = px, py
                    end_node = None

                    # Walk along the path until encountering a node cluster or dead end
                    while True:
                        # Inspect unvisited interior stroke neighbors
                        nbrs = [
                            (ux, uy) for ux, uy in get_neighbors(curr_x, curr_y)
                            if unvisited[uy, ux]
                        ]

                        if nbrs:
                            # Prefer 4-connected steps (Manhattan distance == 1) over
                            # diagonal steps to avoid cutting across pixel corners
                            nbrs.sort(key=lambda pt: abs(pt[0] - curr_x) + abs(pt[1] - curr_y))
                            next_x, next_y = nbrs[0]
                            unvisited[next_y, next_x] = False
                            path.append((next_x, next_y))
                            prev_x, prev_y = curr_x, curr_y
                            curr_x, curr_y = next_x, next_y
                        else:
                            # No unvisited interior pixels left; check for adjacent node clusters
                            adj_node_ids = [
                                int(node_id_mask[uy, ux])
                                for ux, uy in get_neighbors(curr_x, curr_y)
                                if node_id_mask[uy, ux] > 0 and (ux, uy) != (prev_x, prev_y)
                            ]
                            if adj_node_ids:
                                # Reached an existing node cluster
                                end_nid = adj_node_ids[0]
                                end_node = node_coords[end_nid]
                                if path[-1] != end_node:
                                    path.append(end_node)
                            else:
                                # Dead-end without a formal node
                                end_node = (curr_x, curr_y)
                            break

                    edges.append({
                        "start_node": start_node,
                        "end_node": end_node,
                        "path": path
                    })

                # Case B: Two distinct node clusters are directly adjacent (0 interior pixels)
                elif node_id_mask[ny, nx] > 0 and node_id_mask[ny, nx] != nid:
                    other_nid = int(node_id_mask[ny, nx])
                    pair_key = (min(nid, other_nid), max(nid, other_nid))
                    if pair_key not in visited_direct_node_pairs:
                        visited_direct_node_pairs.add(pair_key)
                        end_node = node_coords[other_nid]
                        edges.append({
                            "start_node": start_node,
                            "end_node": end_node,
                            "path": [start_node, end_node]
                        })

    # 3. Handle isolated closed loops (cycles without junctions or endpoints)
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
                # Close the loop
                path.append(start_pixel)
                break

        if len(path) > 3:
            edges.append({
                "start_node": start_pixel,
                "end_node": start_pixel,
                "path": path
            })

    return edges


def rdp_simplify(points: list, epsilon: float = 1.0) -> list:
    """Simplifies a sequence of (x, y) coordinates using the 
    Ramer-Douglas-Peucker (RDP) algorithm.

    Parameters:
        points (list of tuple): Ordered coordinates [(x0, y0), (x1, y1), ...].
        epsilon (float): Maximum allowed perpendicular distance in pixels.
                         Values between 0.8 and 1.5 work well for hand-drawn lines.

    Returns:
        list of tuple: Decimated list of coordinates preserving key vertices.
    """
    if len(points) <= 2:
        return points

    # Split closed loops (where start and end coincide) at the midpoint first
    # so that small loops with max_dist <= epsilon do not collapse into a single point
    if len(points) > 3 and points[0] == points[-1]:
        mid_idx = len(points) // 2
        left_branch = rdp_simplify(points[:mid_idx + 1], epsilon)
        right_branch = rdp_simplify(points[mid_idx:], epsilon)
        return left_branch[:-1] + right_branch

    pts = np.array(points, dtype=np.float64)
    start_pt = pts[0]
    end_pt = pts[-1]

    # Calculate perpendicular distance from all intermediate points to the line (start_pt -> end_pt)
    line_vec = end_pt - start_pt
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        # Start and end points coincide
        dists = np.linalg.norm(pts[1:-1] - start_pt, axis=1)
    else:
        # Cross-product method for point-to-line distance in 2D:
        # dist = |(y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1| / sqrt((y2-y1)^2 + (x2-x1)^2)
        # Evaluated via direct 2D determinant (u_x * v_y - u_y * v_x) to support NumPy 2.x+
        diff = start_pt - pts[1:-1]
        cross_2d = line_vec[0] * diff[:, 1] - line_vec[1] * diff[:, 0]
        dists = np.abs(cross_2d) / line_len

    if len(dists) == 0:
        return [points[0], points[-1]]

    max_idx = int(np.argmax(dists)) + 1  # Offset by 1 because we sliced out the first point
    max_dist = dists[max_idx - 1]

    # Recursive decimation
    if max_dist > epsilon:
        # Point exceeds tolerance: keep it as a vertex and recurse on both sub-segments
        left_branch = rdp_simplify(points[:max_idx + 1], epsilon)
        right_branch = rdp_simplify(points[max_idx:], epsilon)
        return left_branch[:-1] + right_branch
    else:
        # All points are within epsilon: discard intermediate points
        return [points[0], points[-1]]


def simplify_all_edges(edges: list, epsilon: float = 1.0) -> list:
    """Applies RDP decimation across all edges extracted in Step 4.

    Parameters:
        edges (list of dict): Edge list from Step 4.
        epsilon (float): Distance tolerance threshold in pixels.

    Returns:
        list of dict: Edge list containing simplified coordinate paths.
    """
    simplified_edges = []
    total_raw_points = 0
    total_simplified_points = 0

    for edge in edges:
        raw_path = edge["path"]
        total_raw_points += len(raw_path)

        clean_path = rdp_simplify(raw_path, epsilon=epsilon)
        total_simplified_points += len(clean_path)

        simplified_edges.append({
            "start_node": edge["start_node"],
            "end_node": edge["end_node"],
            "path": clean_path
        })

    reduction_pct = (1.0 - (total_simplified_points / max(1, total_raw_points))) * 100
    print(f"RDP Simplification Complete:")
    print(f"  Raw points: {total_raw_points}")
    print(f"  Simplified points: {total_simplified_points}")
    print(f"  Data reduction: {reduction_pct:.1f}%")

    return simplified_edges


def edges_to_svg(
    edges: list,
    width: int,
    height: int,
    output_path: str,
    stroke_width: float = 1.5,
    stroke_color: str = "#000000"
) -> None:
    """Serializes a list of simplified graph edges into a valid, standalone SVG file.

    Parameters:
        edges (list of dict): List of edge dictionaries containing 'path': [(x, y), ...].
        width (int): Canvas width in pixels.
        height (int): Canvas height in pixels.
        output_path (str): Filepath where the .svg file will be saved.
        stroke_width (float): Stroke thickness in pixels (default 1.5).
        stroke_color (str): Hex color code for the path stroke (default black).
    """
    # 1. Create the root <svg> element with viewBox and XML namespaces
    svg = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'version': '1.1',
        'width': str(width),
        'height': str(height),
        'viewBox': f'0 0 {width} {height}'
    })

    path_commands = []

    # 2. Iterate through each edge and construct SVG subpath commands
    for edge in edges:
        pts = edge["path"]
        if len(pts) < 2:
            continue

        # M: Move pen to the starting vertex of the segment
        start_x, start_y = pts[0]
        commands = [f"M {start_x:.2f} {start_y:.2f}"]

        # L: Draw line segments through each subsequent anchor point
        for x, y in pts[1:]:
            commands.append(f"L {x:.2f} {y:.2f}")

        # Z: Close path if the edge loops back onto the exact same node
        if edge["start_node"] == edge["end_node"] and len(pts) > 2:
            commands.append("Z")

        path_commands.append(" ".join(commands))

    # Combine all subpaths into a single 'd' attribute string
    full_path_d = " ".join(path_commands)

    # 3. Add the <path> element with round caps and joins to prevent corner spikes
    ET.SubElement(svg, 'path', {
        'd': full_path_d,
        'fill': 'none',
        'stroke': stroke_color,
        'stroke-width': str(stroke_width),
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round'
    })

    # 4. Format XML cleanly with indentation and save to disk
    raw_xml_str = ET.tostring(svg, encoding='utf-8')
    pretty_xml = minidom.parseString(raw_xml_str).toprettyxml(indent="  ")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    print(f"SVG saved to {output_path} ({len(edges)} paths serialized).")


def run_pipeline(
    input_image_path: str,
    output_svg_path: str,
    epsilon: float = 1.0,
    stroke_width: float = 1.5,
    threshold_val: int = 200
) -> dict:
    """Executes the full image-to-SVG vectorization pipeline.

    Returns:
        dict: Diagnostic summary metrics of the pipeline run.
    """
    print(f"[1/6] Binarizing image: {input_image_path}...")
    binary = binarize_digital_drawing(input_image_path, threshold_val=threshold_val)
    height, width = binary.shape

    print(f"[2/6] Thinning strokes via skeletonization...")
    skeleton = skeletonize_mask(binary, method='lee')

    print(f"[3/6] Detecting endpoints and junction nodes...")
    endpoints, junctions, _ = detect_nodes(skeleton)
    print(f"      Found {len(endpoints)} endpoints, {len(junctions)} junctions.")

    print(f"[4/6] Tracing edges between nodes...")
    raw_edges = walk_edges(skeleton, endpoints, junctions)
    raw_point_count = sum(len(e["path"]) for e in raw_edges)
    print(f"      Extracted {len(raw_edges)} edge paths ({raw_point_count} total pixels).")

    print(f"[5/6] Simplifying paths using RDP (epsilon={epsilon})...")
    simplified_edges = simplify_all_edges(raw_edges, epsilon=epsilon)
    simplified_point_count = sum(len(e["path"]) for e in simplified_edges)
    reduction = (1.0 - (simplified_point_count / max(1, raw_point_count))) * 100
    print(f"      Reduced to {simplified_point_count} vertices ({reduction:.1f}% reduction).")

    print(f"[6/6] Serializing paths to SVG: {output_svg_path}...")
    edges_to_svg(simplified_edges, width, height, output_svg_path, stroke_width=stroke_width)
    print("Vectorization complete.")

    return {
        "canvas_size": (width, height),
        "endpoints": len(endpoints),
        "junctions": len(junctions),
        "total_edges": len(simplified_edges),
        "raw_points": raw_point_count,
        "simplified_points": simplified_point_count,
        "compression_pct": round(reduction, 2)
    }


def main():
    """CLI entrypoint supporting command-line arguments and defaults."""
    parser = argparse.ArgumentParser(
        description="Convert a digital drawing PNG into a simplified, skeletonized SVG vector graph."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="images/drawing.png",
        help="Path to input digital drawing image (default: images/drawing.png)"
    )
    parser.add_argument(
        "-o", "--output",
        default="images/output.svg",
        help="Path to output SVG file (default: output.svg)"
    )
    parser.add_argument(
        "-e", "--epsilon",
        type=float,
        default=1.0,
        help="RDP simplification tolerance in pixels (default: 1.0)"
    )
    parser.add_argument(
        "-w", "--stroke-width",
        type=float,
        default=1.5,
        help="Rendered stroke width in SVG (default: 1.5)"
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            input_image_path=args.input,
            output_svg_path=args.output,
            epsilon=args.epsilon,
            stroke_width=args.stroke_width
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()