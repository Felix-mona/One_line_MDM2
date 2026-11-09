import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
import networkx as nx
import matplotlib.pyplot as plt


# ==========================================================
# 1. PRE-PROCESSING, MANUAL RESCALING & BINARISATION
# ==========================================================

def load_and_scale_image(
    image_path,
    input_stroke_width=4.0,
    target_stroke_width=4.0,
    threshold=128
):
    """
    Loads any 1, 2, 3, or 4-channel image, binarises it, and rescales the canvas
    based on the manual input_stroke_width relative to target_stroke_width.
    """
    img = Image.open(image_path)
    
    # Handle alpha channel (LA or RGBA)
    if img.mode in ("RGBA", "LA"):
        alpha = np.array(img.split()[-1])
        arr = np.array(img.convert("L"))
        arr[alpha < 128] = 255  # Treat transparent pixels as white background
        gray_img = Image.fromarray(arr)
    else:
        gray_img = img.convert("L")

    scale_factor = 1.0

    # Binarise rescaled image (True = stroke, False = background)
    final_arr = np.array(gray_img)
    final_binary = final_arr < threshold
    if np.sum(final_binary) > final_binary.size / 2:
        final_binary = ~final_binary

    return final_binary.astype(bool), scale_factor


# ==========================================================
# 2. SKELETONISATION & CONTENT CROPPING (2.1.1)
# ==========================================================

def run_lee_skeletonisation(binary_mask):
    """
    Lee's medial-axis thinning algorithm preserving Euler characteristic.
    """
    return skeletonize(binary_mask, method="lee").astype(np.uint8)


def crop_to_skeleton_content(skeleton, padding=2):
    """
    Crops the skeleton tightly around the active skeleton pixels.
    Returns cropped skeleton, diagonal length D of the content box, and bounding coordinates.
    """
    fg_r, fg_c = np.where(skeleton > 0)
    
    if len(fg_r) == 0:
        h, w = skeleton.shape
        return skeleton, float(np.hypot(h, w)), (0, h, 0, w)

    min_r, max_r = int(np.min(fg_r)), int(np.max(fg_r))
    min_c, max_c = int(np.min(fg_c)), int(np.max(fg_c))

    content_h = max(1, max_r - min_r)
    content_w = max(1, max_c - min_c)
    canvas_diag = float(np.hypot(content_h, content_w))

    rows, cols = skeleton.shape
    pad_min_r = max(0, min_r - padding)
    pad_max_r = min(rows, max_r + padding + 1)
    pad_min_c = max(0, min_c - padding)
    pad_max_c = min(cols, max_c + padding + 1)

    cropped_skeleton = skeleton[pad_min_r:pad_max_r, pad_min_c:pad_max_c].copy()

    return cropped_skeleton, canvas_diag, (min_r, max_r, min_c, max_c)


# ==========================================================
# 3. RUTOVITZ CROSSING NUMBER (2.1.2)
# ==========================================================

def compute_rutovitz_crossing_number(skeleton):
    """
    Computes Rutovitz crossing numbers across 8-connected neighbours:
        X(p) = 0.5 * sum(|p_{i+1} - p_i|)
    Endpoints: X(p) == 1
    Junctions: X(p) >= 3
    """
    rows, cols = skeleton.shape
    crossing_map = np.zeros((rows, cols), dtype=np.uint8)
    
    fg_r, fg_c = np.where(skeleton > 0)
    nbr_offsets = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]

    for r, c in zip(fg_r, fg_c):
        if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
            continue
            
        seq = [skeleton[r + dr, c + dc] for dr, dc in nbr_offsets]
        transitions = sum(seq[i] != seq[(i + 1) % 8] for i in range(8))
        crossing_map[r, c] = transitions // 2

    endpoints = list(zip(*np.where((crossing_map == 1) & (skeleton > 0))))
    junctions = list(zip(*np.where((crossing_map >= 3) & (skeleton > 0))))

    return crossing_map, endpoints, junctions


# ==========================================================
# 4. EDGE TRACING & RDP SIMPLIFICATION (2.1.3)
# ==========================================================

def trace_skeleton_branches(skeleton, endpoints, junctions):
    """
    Traces pixel chains radiating from endpoints/junctions, and handles
    isolated closed loops lacking junction nodes.
    """
    rows, cols = skeleton.shape
    visited_edges = set()
    node_set = set(endpoints) | set(junctions)
    edges = []

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def get_valid_nbrs(pt):
        r, c = pt
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and skeleton[nr, nc]:
                yield (nr, nc)

    # 1. Trace branches connected to junctions or endpoints
    for start_node in node_set:
        for first_step in get_valid_nbrs(start_node):
            step_fwd = (start_node, first_step)
            step_rev = (first_step, start_node)
            if step_fwd in visited_edges or step_rev in visited_edges:
                continue

            path = [start_node, first_step]
            visited_edges.add(step_fwd)
            visited_edges.add(step_rev)

            prev_pt, curr_pt = start_node, first_step
            while curr_pt not in node_set:
                candidates = [n for n in get_valid_nbrs(curr_pt) if n != prev_pt]
                if not candidates:
                    break
                next_pt = candidates[0]
                visited_edges.add((curr_pt, next_pt))
                visited_edges.add((next_pt, curr_pt))
                path.append(next_pt)
                prev_pt, curr_pt = curr_pt, next_pt

            if len(path) > 1:
                edges.append(path)

    # 2. Trace isolated closed loops
    unvisited_fg = set(zip(*np.where(skeleton > 0))) - node_set
    for pt in list(unvisited_fg):
        if any((pt, n) in visited_edges or (n, pt) in visited_edges for n in get_valid_nbrs(pt)):
            continue

        loop_path = [pt]
        curr_pt, prev_pt = pt, None
        while True:
            candidates = [n for n in get_valid_nbrs(curr_pt) if n != prev_pt]
            if not candidates:
                break
            next_pt = candidates[0]
            visited_edges.add((curr_pt, next_pt))
            visited_edges.add((next_pt, curr_pt))
            if next_pt == pt:
                loop_path.append(next_pt)
                break
            if next_pt in loop_path:
                break
            loop_path.append(next_pt)
            prev_pt, curr_pt = curr_pt, next_pt

        if len(loop_path) > 2:
            edges.append(loop_path)

    return edges


def rdp_simplify(points, epsilon):
    """
    Ramer-Douglas-Peucker (RDP) polyline decimation.
    Uses 2D cross-product arithmetic compatible with NumPy 2.0+.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2:
        return pts

    start, end = pts[0], pts[-1]
    line_vec = end - start
    line_len = np.hypot(line_vec[0], line_vec[1])

    diff = pts - start
    if line_len < 1e-12:
        dists = np.hypot(diff[:, 0], diff[:, 1])
    else:
        perp_dist = np.abs(line_vec[0] * diff[:, 1] - line_vec[1] * diff[:, 0])
        dists = perp_dist / line_len

    max_idx = np.argmax(dists)
    max_dist = dists[max_idx]

    if max_dist > epsilon:
        left = rdp_simplify(pts[:max_idx + 1], epsilon)
        right = rdp_simplify(pts[max_idx:], epsilon)
        return np.vstack((left[:-1], right))
    else:
        return np.array([start, end])


# ==========================================================
# 5. CLOSE-NODE MERGING TO CENTROIDS (2.1.4)
# ==========================================================

def merge_nearby_junctions(G, d_euclid, d_edge_factor=3.0):
    """
    Collapses clusters of nearby junction nodes into their spatial centroid.
    Nodes are clustered if:
      - Direct Euclidean distance <= d_euclid, OR
      - Connected by an edge with path length <= d_euclid * d_edge_factor
    """
    d_edge_thresh = d_euclid * d_edge_factor
    nodes = list(G.nodes())
    if not nodes:
        return G
    
    merge_graph = nx.Graph()
    merge_graph.add_nodes_from(nodes)

    for i, u in enumerate(nodes):
        pos_u = G.nodes[u]["pos"]
        
        # Euclidean proximity
        for v in nodes[i + 1:]:
            pos_v = G.nodes[v]["pos"]
            if np.hypot(pos_u[0] - pos_v[0], pos_u[1] - pos_v[1]) <= d_euclid:
                merge_graph.add_edge(u, v)

        # Short connecting edge path length
        for nbr in G.neighbors(u):
            if nbr > u:
                edge_dict = G.get_edge_data(u, nbr)
                min_len = min(
                    np.sum(np.hypot(np.diff(d["pts"][:, 0]), np.diff(d["pts"][:, 1])))
                    for d in edge_dict.values()
                )
                if min_len <= d_edge_thresh:
                    merge_graph.add_edge(u, nbr)

    clusters = list(nx.connected_components(merge_graph))
    node_to_rep = {}
    cluster_centroids = {}

    for cluster in clusters:
        cluster_list = list(cluster)
        rep = cluster_list[0]
        centroid = np.mean([G.nodes[n]["pos"] for n in cluster_list], axis=0)
        cluster_centroids[rep] = centroid
        for n in cluster_list:
            node_to_rep[n] = rep

    merged_G = nx.MultiGraph()
    for rep, centroid in cluster_centroids.items():
        merged_G.add_node(rep, pos=centroid)

    for u, v, k, data in G.edges(keys=True, data=True):
        new_u = node_to_rep[u]
        new_v = node_to_rep[v]

        if new_u == new_v:
            continue

        pts = data["pts"].copy()
        pts[0] = merged_G.nodes[new_u]["pos"]
        pts[-1] = merged_G.nodes[new_v]["pos"]

        edge_len = np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))
        if edge_len > 1e-4:
            attrs = dict(data)
            attrs["pts"] = pts
            merged_G.add_edge(new_u, new_v, **attrs)

    return merged_G


# ==========================================================
# 6. COMPLETE GRAPH FORMULATION PIPELINE
# ==========================================================

def build_graph_from_image(
    image_path,
    input_stroke_width=10.0,  # Manual setting: estimated stroke width of input in pixels
    target_stroke_width=4.0,  # Normalized target width for skeletonisation
    binarise_thresh=128,
    d_euclid_prop=0.015,      # 1.5% of content diagonal
    d_edge_factor=3.0,        # 3.0 * d_euclid
    rdp_epsilon_prop=0.003    # 0.3% of content diagonal
):
    """
    Executes Sections 2.1.1 to 2.1.4:
    1. Manually rescales image by (target_stroke_width / input_stroke_width).
    2. Runs Lee's medial-axis thinning.
    3. Crops canvas to active skeleton pixels.
    4. Detects junctions and endpoints via Rutovitz crossing numbers.
    5. Simplifies branches via RDP using rdp_epsilon_prop * D.
    6. Collapses nearby nodes to their centroids using d_euclid_prop * D.
    """
    # Step 1: Manual scaling & binarisation
    binary, scale_factor = load_and_scale_image(
        image_path,
        input_stroke_width=input_stroke_width,
        target_stroke_width=target_stroke_width,
        threshold=binarise_thresh
    )

    # Step 2: Skeletonisation (2.1.1)
    full_skeleton = run_lee_skeletonisation(binary)

    # Step 3: Crop to Content Bounding Box
    skeleton, canvas_diag, (r0, r1, c0, c1) = crop_to_skeleton_content(full_skeleton, padding=2)
    
    d_euclid = d_euclid_prop * canvas_diag
    rdp_epsilon = rdp_epsilon_prop * canvas_diag

    print(f"Input stroke: {input_stroke_width:.1f}px -> scaled {scale_factor:.2f}x (target: {target_stroke_width:.1f}px)")
    print(f"Content Bounding Box: [{c1 - c0}w x {r1 - r0}h] | Diagonal D = {canvas_diag:.1f}px")
    print(f"Merge d_euclid: {d_euclid:.2f}px ({d_euclid_prop*100:.2f}% of D)")
    print(f"RDP epsilon:    {rdp_epsilon:.2f}px ({rdp_epsilon_prop*100:.2f}% of D)")

    # Step 4: Rutovitz Crossing Numbers (2.1.2)
    crossing_map, endpoints, junctions = compute_rutovitz_crossing_number(skeleton)

    # Step 5: Tracing & RDP Simplification (2.1.3)
    raw_edges = trace_skeleton_branches(skeleton, endpoints, junctions)

    initial_G = nx.MultiGraph()
    coord_to_node = {}

    def get_or_create_node(coord_xy):
        key = (round(coord_xy[0], 2), round(coord_xy[1], 2))
        if key not in coord_to_node:
            node_id = len(coord_to_node)
            coord_to_node[key] = node_id
            initial_G.add_node(node_id, pos=np.array(coord_xy, dtype=float))
        return coord_to_node[key]

    for raw_edge in raw_edges:
        coords_xy = np.array([[c, -r] for r, c in raw_edge], dtype=float)
        simplified_pts = rdp_simplify(coords_xy, epsilon=rdp_epsilon)
        u = get_or_create_node(simplified_pts[0])
        v = get_or_create_node(simplified_pts[-1])
        initial_G.add_edge(u, v, pts=simplified_pts)

    # Step 6: Close-Node Centroid Merging (2.1.4)
    final_G = merge_nearby_junctions(initial_G, d_euclid=d_euclid, d_edge_factor=d_edge_factor)

    return final_G, skeleton, canvas_diag, scale_factor


# ==========================================================
# VERIFICATION RUN
# ==========================================================

if __name__ == "__main__":
    canvas = np.full((350, 450, 3), 255, dtype=np.uint8)
    
    # Draw sample lines ~10px wide
    for w in range(-4, 5):
        canvas[100:180, np.clip(120 + w, 0, 449)] = 0
        canvas[100:180, np.clip(200 + w, 0, 449)] = 0
        canvas[np.clip(100 + w, 0, 349), 120:200] = 0
        canvas[np.clip(180 + w, 0, 349), 120:200] = 0
        canvas[np.clip(138 + w, 0, 349), 120:200] = 0
        canvas[np.clip(142 + w, 0, 349), 120:200] = 0

    test_png = "crop_proportion_test.png"
    Image.fromarray(canvas).save(test_png)

    # Specify input stroke width manually
    graph, cropped_skel, diag, scale = build_graph_from_image(
        test_png,
        input_stroke_width=10.0,   # Set to whatever stroke width you want
        target_stroke_width=4.0,   # Target scaled thickness
        d_euclid_prop=0.02,        # 2% of content diagonal
        d_edge_factor=3.0,
        rdp_epsilon_prop=0.004     # 0.4% of content diagonal
    )

    print(f"\nFinal Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    plt.figure(figsize=(7, 7))
    for u, v, k, data in graph.edges(keys=True, data=True):
        pts = data["pts"]
        plt.plot(pts[:, 0], pts[:, 1], "k-", linewidth=2)
    for n, data in graph.nodes(data=True):
        pos = data["pos"]
        plt.scatter(pos[0], pos[1], color="red", s=50, zorder=5)

    plt.title("Cropped & Proportionally Merged Graph")
    plt.axis("equal")
    plt.grid(True)
    plt.show()