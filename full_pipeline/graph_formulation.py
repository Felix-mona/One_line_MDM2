"""
Graph Formulation (Sections 2.1.1 - 2.1.4)
Handles image binarisation, stroke-width rescaling, Lee's skeletonisation,
Rutovitz crossing numbers, junction-blob clustering, RDP simplification,
and loop-safe junction merging.
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
import networkx as nx
import matplotlib.pyplot as plt

OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


# ==========================================================
# 1. PRE-PROCESSING, RESCALING & BINARISATION (2.1.1)
# ==========================================================

def load_and_scale_image(
    image_path,
    input_stroke_width=8.0,
    target_stroke_width=4.0,
    threshold=128,
):
    """
    Loads an image, flattens alpha onto white, binarises, crops tightly to
    foreground content, and downscales if input_stroke_width > target_stroke_width
    for clean medial-axis thinning.
    """
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            gray_img = Image.alpha_composite(white, rgba).convert("L")
        else:
            gray_img = img.convert("L")

        arr = np.array(gray_img, dtype=np.uint8)

    binary = arr < threshold
    if np.sum(binary) > binary.size / 2:
        binary = ~binary

    active = np.argwhere(binary)
    if len(active) == 0:
        padded = np.pad(binary, pad_width=1, mode="constant", constant_values=False)
        return padded, 1.0, float(np.hypot(*padded.shape))

    r_min, c_min = active.min(axis=0)
    r_max, c_max = active.max(axis=0) + 1
    cropped = binary[r_min:r_max, c_min:c_max]

    h_orig, w_orig = cropped.shape
    orig_diag = float(np.hypot(h_orig, w_orig))

    # Downsample thick strokes so skeletonisation produces clean 1px centerlines
    downsample = max(1.0, float(input_stroke_width) / max(1.0, float(target_stroke_width)))
    if downsample > 1.05:
        h_work = max(3, int(round(h_orig / downsample)))
        w_work = max(3, int(round(w_orig / downsample)))
        pil_mask = Image.fromarray((cropped.astype(np.uint8)) * 255, mode="L")
        resized = pil_mask.resize((w_work, h_work), resample=Image.Resampling.BILINEAR)
        work_mask = np.array(resized, dtype=np.uint8) >= 96
        scale_factor = orig_diag / float(np.hypot(h_work, w_work))
    else:
        work_mask = cropped
        scale_factor = 1.0

    # Pad by 1px so boundary foreground pixels have a complete 8-neighbourhood
    padded_mask = np.pad(work_mask, pad_width=1, mode="constant", constant_values=False)
    return padded_mask, scale_factor, orig_diag


def run_lee_skeletonisation(binary_mask):
    """Lee's medial-axis thinning normalized to {0, 1} uint8."""
    return (skeletonize(binary_mask, method="lee") > 0).astype(np.uint8)


# ==========================================================
# 2. RUTOVITZ CROSSING NUMBERS & CANONICAL NODES (2.1.2)
# ==========================================================

def compute_rutovitz_crossing_number(skeleton):
    """
    Vectorised Rutovitz crossing number across the 8-neighbourhood:
        X(p) = 0.5 * sum(|p_{i+1} - p_i|)
    Endpoints: CN == 1 | Line pixels: CN == 2 | Junctions: CN >= 3
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


def extract_canonical_nodes(skeleton, crossing_number, merge_dist_work):
    """
    Collapses contiguous junction blobs (CN >= 3) into single medoid nodes,
    registers endpoints (CN == 1), and merges nodes within merge_dist_work.
    """
    H, W = skeleton.shape
    node_grid = np.full((H, W), fill_value=-1, dtype=np.int32)
    raw_nodes, raw_types, raw_members = {}, {}, {}
    nid = 0

    # 1. Endpoints (CN == 1)
    for r, c in np.argwhere((skeleton > 0) & (crossing_number == 1)):
        raw_nodes[nid] = (float(r), float(c))
        raw_types[nid] = "endpoint"
        raw_members[nid] = [(int(r), int(c))]
        nid += 1

    # 2. Contiguous junction blobs (CN >= 3)
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

    # 3. Merge nearby nodes via KD-Tree and bridge intermediate skeleton pixels
    final_nodes, final_types, final_members = {}, {}, {}
    skel_set = set(map(tuple, np.argwhere(skeleton > 0)))

    if len(raw_nodes) > 0 and merge_dist_work > 0.0:
        raw_ids = list(raw_nodes.keys())
        coords = np.array([raw_nodes[i] for i in raw_ids])
        pairs = cKDTree(coords).query_pairs(r=merge_dist_work)

        mg = nx.Graph()
        mg.add_nodes_from(raw_ids)
        mg.add_edges_from((raw_ids[i], raw_ids[j]) for i, j in pairs)

        for new_id, cluster in enumerate(nx.connected_components(mg)):
            cluster_list = list(cluster)
            combined = {px for i in cluster_list for px in raw_members[i]}
            is_junc = any(raw_types[i] == "junction" for i in cluster_list)

            if len(cluster_list) > 1:
                rep_px = raw_members[cluster_list[0]][0]
                max_steps = int(np.ceil(merge_dist_work)) + 4
                for other_id in cluster_list[1:]:
                    target_px = raw_members[other_id][0]
                    q = [(rep_px, 0)]
                    parent = {rep_px: None}
                    while q:
                        curr, d = q.pop(0)
                        if curr == target_px:
                            p = curr
                            while p is not None:
                                combined.add(p)
                                p = parent[p]
                            break
                        if d < max_steps:
                            for dr, dc in OFFSETS:
                                nbr = (curr[0] + dr, curr[1] + dc)
                                if nbr in skel_set and nbr not in parent:
                                    parent[nbr] = curr
                                    q.append((nbr, d + 1))

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


# ==========================================================
# 3. LOSSLESS SKELETON TRACING & RDP DECIMATION (2.1.3)
# ==========================================================

def _bfs_cluster_path(start, end, valid_pixels):
    """Shortest 8-connected walk inside a node cluster."""
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


def _rc_to_xy(rc_list, scale_factor):
    """Converts (row, col) pixel list to deduplicated Cartesian (x, y) float array."""
    clean = [rc_list[0]]
    for pt in rc_list[1:]:
        if pt[0] != clean[-1][0] or pt[1] != clean[-1][1]:
            clean.append(pt)
    arr = np.array(clean, dtype=float)
    # x = col * scale, y = -row * scale (so plots render right-side-up)
    return np.column_stack((arr[:, 1] * scale_factor, -arr[:, 0] * scale_factor))


def trace_skeleton_to_graph(skeleton, nodes, node_types, node_grid, node_members, scale_factor):
    """
    Traces 8-connected pixel chains between canonical nodes and captures
    isolated closed loops (e.g. circles/polygons without junctions).
    """
    G = nx.MultiGraph()
    for nid, (r, c) in nodes.items():
        G.add_node(
            nid,
            pos=np.array([c * scale_factor, -r * scale_factor], dtype=float),
            kind=node_types.get(nid, "endpoint"),
        )

    H, W = skeleton.shape
    visited = np.zeros_like(skeleton, dtype=bool)
    direct_seen = set()

    # Pass 1: Trace paths radiating from canonical nodes
    for u_id, u_pos in list(nodes.items()):
        u_pixels = node_members.get(u_id, [u_pos])
        u_set = set(u_pixels)

        for p in u_pixels:
            for dr, dc in OFFSETS:
                nr, nc = p[0] + dr, p[1] + dc
                if not (0 <= nr < H and 0 <= nc < W) or not skeleton[nr, nc]:
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
                        for s_dr, s_dc in OFFSETS:
                            sr, sc = curr_r + s_dr, curr_c + s_dc
                            if 0 <= sr < H and 0 <= sc < W and skeleton[sr, sc]:
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
                        elif self_nodes and left_start:
                            target_px, target_id = self_nodes[0]
                            break
                        elif path_nbrs:
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
                            G.add_edge(u_id, target_id, pts=pts_xy)

                # Case B: Directly touching node clusters
                elif nbr_id != -1 and nbr_id != u_id:
                    edge_key = (min(u_id, nbr_id), max(u_id, nbr_id))
                    if edge_key not in direct_seen:
                        direct_seen.add(edge_key)
                        v_pos = nodes[nbr_id]
                        v_set = set(node_members.get(nbr_id, [v_pos]))
                        full_rc = _bfs_cluster_path(u_pos, p, u_set) + _bfs_cluster_path((nr, nc), v_pos, v_set)
                        pts_xy = _rc_to_xy(full_rc, scale_factor)
                        if len(pts_xy) >= 2:
                            G.add_edge(u_id, nbr_id, pts=pts_xy)

    # Pass 2: Trace isolated closed loops lacking junctions (e.g. circles)
    unvisited_loops = np.argwhere((skeleton > 0) & (~visited) & (node_grid == -1))
    for sr, sc in unvisited_loops:
        if visited[sr, sc]:
            continue
        loop_id = max(G.nodes, default=-1) + 1
        start_rc = (int(sr), int(sc))
        path = [start_rc]
        curr_r, curr_c = start_rc
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
            path.append(start_rc)
            pts_xy = _rc_to_xy(path, scale_factor)
            if len(pts_xy) >= 3:
                G.add_node(loop_id, pos=pts_xy[0].copy(), kind="loop_anchor")
                G.add_edge(loop_id, loop_id, pts=pts_xy)

    return G


def rdp_simplify(points, epsilon):
    """Ramer-Douglas-Peucker (RDP) polyline decimation."""
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
        left = rdp_simplify(pts[:max_idx + 1], epsilon)
        right = rdp_simplify(pts[max_idx:], epsilon)
        return np.vstack((left[:-1], right))
    return np.array([start, end], dtype=float)


def apply_rdp_to_graph(G, epsilon):
    """
    Applies RDP to every edge in G. Splits self-loops (u == v) in half
    before simplifying so closed shapes never collapse into a 2-point line.
    """
    H = G.copy()
    for u, v, k, data in H.edges(keys=True, data=True):
        pts = data["pts"]
        if u == v and len(pts) > 4:
            mid = len(pts) // 2
            left = rdp_simplify(pts[:mid + 1], epsilon)
            right = rdp_simplify(pts[mid:], epsilon)
            data["pts"] = np.vstack((left[:-1], right))
        else:
            data["pts"] = rdp_simplify(pts, epsilon)
    return H


# ==========================================================
# 4. LOOP-SAFE JUNCTION MERGING & CLEANING (2.1.4)
# ==========================================================

def polyline_length(pts):
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def dissolve_degree_two_nodes(G):
    """Merges non-loop degree-2 continuation nodes along smooth strokes."""
    H = G.copy()
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
            node_pos = H.nodes[node]["pos"]

            pts1 = d1["pts"]
            if np.linalg.norm(pts1[-1] - node_pos) > np.linalg.norm(pts1[0] - node_pos):
                pts1 = pts1[::-1]
            pts2 = d2["pts"]
            if np.linalg.norm(pts2[0] - node_pos) > np.linalg.norm(pts2[-1] - node_pos):
                pts2 = pts2[::-1]

            merged_pts = np.vstack((pts1[:-1], pts2))
            H.remove_edge(u1, v1, key=k1)
            H.remove_edge(u2, v2, key=k2)
            H.remove_node(node)
            H.add_edge(nbr1, nbr2, pts=merged_pts)
            changed = True
            break
    return H


def merge_nearby_junctions(G, d_edge_thresh):
    """
    Loop-safe junction contraction: collapses short edges (< d_edge_thresh)
    between junctions into their centroid while strictly preserving all closed
    self-loops (u == v) and parallel loop petals.
    """
    H = dissolve_degree_two_nodes(G)

    while True:
        short_g = nx.Graph()
        for u, v, k, data in H.edges(keys=True, data=True):
            if u != v and H.degree(u) >= 3 and H.degree(v) >= 3:
                if polyline_length(data["pts"]) <= d_edge_thresh:
                    short_g.add_edge(u, v)

        clusters = [list(c) for c in nx.connected_components(short_g) if len(c) > 1]
        if not clusters:
            break

        for cluster in clusters:
            c_set = set(cluster)
            centroid = np.mean([H.nodes[n]["pos"] for n in cluster], axis=0)
            rep = cluster[0]
            old_pos = {n: H.nodes[n]["pos"].copy() for n in cluster}
            H.nodes[rep]["pos"] = centroid
            H.nodes[rep]["kind"] = "junction"

            seen = set()
            internal, external = [], []
            for u, v, k, data in list(H.edges(cluster, keys=True, data=True)):
                eid = (min(u, v), max(u, v), k) if u != v else (u, v, k)
                if eid in seen:
                    continue
                seen.add(eid)
                if u in c_set and v in c_set:
                    internal.append((u, v, k, data["pts"].copy()))
                else:
                    external.append((u, v, k, data["pts"].copy()))

            for u, v, k, _ in internal + external:
                if H.has_edge(u, v, key=k):
                    H.remove_edge(u, v, key=k)

            # Keep self-loops and parallel petals longer than d_edge_thresh
            pair_groups = {}
            for u, v, k, pts in internal:
                if u == v:
                    pts[0], pts[-1] = centroid, centroid
                    H.add_edge(rep, rep, pts=pts)
                else:
                    pk = (min(u, v), max(u, v))
                    pair_groups.setdefault(pk, []).append((pts, polyline_length(pts)))

            for pk, edge_list in pair_groups.items():
                edge_list.sort(key=lambda x: x[1])
                for pts, _ in edge_list[1:]:
                    pts[0], pts[-1] = centroid, centroid
                    H.add_edge(rep, rep, pts=pts)

            # Re-anchor external edges to the new centroid on the correct end
            for u, v, k, pts in external:
                inner = u if u in c_set else v
                outer = v if u in c_set else u
                in_p, out_p = old_pos[inner], H.nodes[outer]["pos"]
                d_fwd = np.linalg.norm(pts[0] - in_p) + np.linalg.norm(pts[-1] - out_p)
                d_rev = np.linalg.norm(pts[-1] - in_p) + np.linalg.norm(pts[0] - out_p)
                if d_fwd <= d_rev:
                    pts[0] = centroid
                else:
                    pts[-1] = centroid
                H.add_edge(rep, outer, pts=pts)

            for n in cluster[1:]:
                if n in H:
                    H.remove_node(n)

        H = dissolve_degree_two_nodes(H)

    # Remove any isolated 0-degree nodes
    H.remove_nodes_from([n for n in list(H.nodes()) if H.degree(n) == 0])
    return H


def prune_whiskers_and_noise(G, whisker_thresh, min_comp_len):
    """Prunes tiny degree-1 terminal whiskers and tiny isolated noise specks."""
    H = G.copy()

    # 1. Prune short degree-1 whiskers
    changed = True
    while changed:
        changed = False
        for u, v, k, data in list(H.edges(keys=True, data=True)):
            if u != v and (H.degree(u) == 1 or H.degree(v) == 1):
                if polyline_length(data["pts"]) < whisker_thresh:
                    H.remove_edge(u, v, key=k)
                    for n in (u, v):
                        if n in H and H.degree(n) == 0:
                            H.remove_node(n)
                    H = dissolve_degree_two_nodes(H)
                    changed = True
                    break

    # 2. Prune tiny disconnected noise components
    for comp in list(nx.connected_components(H)):
        comp_len = sum(polyline_length(d["pts"]) for _, _, _, d in H.edges(comp, keys=True, data=True))
        if comp_len < min_comp_len:
            H.remove_nodes_from(comp)

    return dissolve_degree_two_nodes(H)

# ==========================================================
# 5. COMPLETE GRAPH FORMULATION PIPELINE & SAVING
# ==========================================================

def save_binary_or_skeleton_image(mask, save_path, label="image"):
    """Saves a binary mask or skeleton as a black-on-white PNG image."""
    img_u8 = np.where(mask > 0, 0, 255).astype(np.uint8)
    Image.fromarray(img_u8, mode="L").save(save_path)
    print(f"Saved {label} to '{save_path}'")


def save_skeleton_image(skeleton, save_path):
    """Backwards-compatible wrapper for saving the binary skeleton."""
    save_binary_or_skeleton_image(skeleton, save_path, label="skeleton image")


def plot_graph_snapshot(G, save_path, title="Graph Snapshot"):
    """Saves a diagnostic plot of a MultiGraph."""
    plt.figure(figsize=(8, 7))
    for _, _, data in G.edges(data=True):
        pts = data["pts"]
        plt.plot(pts[:, 0], pts[:, 1], "k-", linewidth=1.5)
    for _, attrs in G.nodes(data=True):
        x, y = attrs["pos"]
        plt.scatter(x, y, color="darkorange", s=22, zorder=5)
    plt.title(f"{title} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {title.lower()} to '{save_path}'")


def build_graph_from_image(
    image_path,
    input_stroke_width=8.0,
    target_stroke_width=4.0,
    binarise_thresh=128,
    d_euclid_prop=0.008,
    d_edge_factor=2.5,
    rdp_epsilon_prop=0.002,
    whisker_prop=0.010,
    min_comp_prop=0.015,
    binary_save_path=None,
    skeleton_save_path=None,
    raw_graph_save_path=None,
):
    """
    Full Section 2.1 pipeline with optional intermediate saves for the
    cropped binary mask, Lee skeleton, and raw traced graph (pre-RDP).
    """
    binary_mask, scale_factor, canvas_diag = load_and_scale_image(
        image_path=image_path,
        input_stroke_width=input_stroke_width,
        target_stroke_width=target_stroke_width,
        threshold=binarise_thresh,
    )
    if binary_save_path:
        save_binary_or_skeleton_image(binary_mask, binary_save_path, label="binary mask")

    skeleton = run_lee_skeletonisation(binary_mask)
    if skeleton_save_path:
        save_skeleton_image(skeleton, skeleton_save_path)

    d_euclid = d_euclid_prop * canvas_diag
    d_edge_thresh = d_euclid * d_edge_factor
    rdp_epsilon = max(1.0, rdp_epsilon_prop * canvas_diag)
    whisker_thresh = whisker_prop * canvas_diag
    min_comp_len = min_comp_prop * canvas_diag

    print(f"Canvas Diagonal D = {canvas_diag:.1f}px | Scale factor = {scale_factor:.2f}x")
    print(f"Node merge d_euclid = {d_euclid:.2f}px | Junction edge merge = {d_edge_thresh:.2f}px | RDP eps = {rdp_epsilon:.2f}px")

    cn = compute_rutovitz_crossing_number(skeleton)
    nodes, node_types, node_grid, node_members = extract_canonical_nodes(
        skeleton, cn, merge_dist_work=d_euclid / scale_factor
    )

    raw_G = trace_skeleton_to_graph(
        skeleton, nodes, node_types, node_grid, node_members, scale_factor
    )
    if raw_graph_save_path:
        plot_graph_snapshot(raw_G, raw_graph_save_path, title="Raw Traced Graph (Pre-RDP)")

    rdp_G = apply_rdp_to_graph(raw_G, epsilon=rdp_epsilon)
    merged_G = merge_nearby_junctions(rdp_G, d_edge_thresh=d_edge_thresh)
    final_G = prune_whiskers_and_noise(merged_G, whisker_thresh=whisker_thresh, min_comp_len=min_comp_len)

    return final_G, skeleton, canvas_diag, scale_factor