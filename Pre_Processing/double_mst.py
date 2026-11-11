"""
Double MST Component Bridging (Sections 2.1.4 & 2.2.1)
Splits sharp corners and long edges, then connects disconnected sub-graphs into a
single connected graph using two passes of Kruskal's Minimum Spanning Tree with
perceptual penalties and exact T-join even-parity pruning.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ==========================================================
# 1. SHARP CORNER & LONG-EDGE SPLITTING (SECTION 2.1.4)
# ==========================================================

def split_edges_at_sharp_corners(G, max_turn_angle_deg=65.0):
    """
    Splits edges at sharp corners (deflection >= max_turn_angle_deg) so bridges
    can attach cleanly to geometric corners of closed shapes.
    """
    if max_turn_angle_deg <= 0.0 or max_turn_angle_deg >= 180.0:
        return G.copy()

    H = G.copy()
    next_id = max(H.nodes, default=-1) + 1

    for u, v, k, data in list(H.edges(keys=True, data=True)):
        pts = data["pts"]
        if len(pts) < 3:
            continue

        pos_u, pos_v = H.nodes[u]["pos"], H.nodes[v]["pos"]
        if u != v and (
            np.linalg.norm(pts[0] - pos_u) + np.linalg.norm(pts[-1] - pos_v)
            > np.linalg.norm(pts[-1] - pos_u) + np.linalg.norm(pts[0] - pos_v)
        ):
            pts = pts[::-1].copy()

        split_idxs = [0]
        for i in range(1, len(pts) - 1):
            v_in = pts[i] - pts[i - 1]
            v_out = pts[i + 1] - pts[i]
            n_in, n_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
            if n_in < 1e-9 or n_out < 1e-9:
                continue
            cos_t = np.clip(np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0)
            if np.degrees(np.arccos(cos_t)) >= max_turn_angle_deg:
                split_idxs.append(i)

        if len(split_idxs) == 1:
            continue

        split_idxs.append(len(pts) - 1)
        H.remove_edge(u, v, key=k)

        curr_u = u
        for idx in range(len(split_idxs) - 1):
            i0, i1 = split_idxs[idx], split_idxs[idx + 1]
            sub_pts = pts[i0 : i1 + 1].copy()
            if idx < len(split_idxs) - 2:
                corner_id = next_id
                next_id += 1
                H.add_node(corner_id, pos=pts[i1].copy(), kind="corner")
                H.add_edge(curr_u, corner_id, pts=sub_pts)
                curr_u = corner_id
            else:
                H.add_edge(curr_u, v, pts=sub_pts)

    return H


def subdivide_polyline(pts, max_len):
    """Splits a polyline of shape (N, 2) into equal-arc-length pieces <= max_len."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return [pts]

    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    total_len = float(np.sum(seg_lens))

    if total_len <= max_len:
        return [pts]

    num_splits = int(np.ceil(total_len / max_len))
    cum_len = np.concatenate(([0.0], np.cumsum(seg_lens)))
    sub_polylines = []

    for s in range(num_splits):
        d0 = (s / num_splits) * total_len
        d1 = ((s + 1) / num_splits) * total_len if s < num_splits - 1 else total_len

        x0 = float(np.interp(d0, cum_len, pts[:, 0]))
        y0 = float(np.interp(d0, cum_len, pts[:, 1]))
        x1 = float(np.interp(d1, cum_len, pts[:, 0]))
        y1 = float(np.interp(d1, cum_len, pts[:, 1]))

        mask = (cum_len > d0 + 1e-6) & (cum_len < d1 - 1e-6)
        piece = np.vstack(([[x0, y0]], pts[mask], [[x1, y1]]))
        sub_polylines.append(piece)

    return sub_polylines


def split_long_edges(G, max_edge_len):
    """Splits any edge in G longer than max_edge_len by inserting subdivision nodes."""
    new_G = nx.MultiGraph()
    for n, attrs in G.nodes(data=True):
        new_G.add_node(n, **attrs)

    next_id = max(G.nodes, default=-1) + 1

    for u, v, k, data in G.edges(keys=True, data=True):
        pts = data["pts"]
        pos_u, pos_v = G.nodes[u]["pos"], G.nodes[v]["pos"]
        if u != v and (
            np.linalg.norm(pts[0] - pos_u) + np.linalg.norm(pts[-1] - pos_v)
            > np.linalg.norm(pts[-1] - pos_u) + np.linalg.norm(pts[0] - pos_v)
        ):
            pts = pts[::-1].copy()

        pieces = subdivide_polyline(pts, max_edge_len)
        if len(pieces) == 1:
            new_G.add_edge(u, v, **data)
        else:
            prev_node = u
            for i, piece in enumerate(pieces):
                if i == len(pieces) - 1:
                    end_node = v
                else:
                    end_node = next_id
                    next_id += 1
                    new_G.add_node(end_node, pos=piece[-1].copy(), kind="subdivision")

                attrs = dict(data)
                attrs["pts"] = piece
                new_G.add_edge(prev_node, end_node, **attrs)
                prev_node = end_node

    return new_G


# ==========================================================
# 2. GEOMETRIC INTERSECTION CHECKS
# ==========================================================

def segments_intersect(p1, p2, p3, p4):
    """Returns True if segment (p1, p2) strictly crosses segment (p3, p4)."""
    if (
        max(p1[0], p2[0]) < min(p3[0], p4[0])
        or min(p1[0], p2[0]) > max(p3[0], p4[0])
        or max(p1[1], p2[1]) < min(p3[1], p4[1])
        or min(p1[1], p2[1]) > max(p3[1], p4[1])
    ):
        return False

    def ccw(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    return (d1 * d2 < -1e-9) and (d3 * d4 < -1e-9)


def count_edge_intersections(p_start, p_end, all_segments):
    """Counts how many existing stroke segments a candidate bridge crosses."""
    crossings = 0
    for s_pt, e_pt in all_segments:
        if (
            np.linalg.norm(s_pt - p_start) < 1e-4
            or np.linalg.norm(e_pt - p_start) < 1e-4
            or np.linalg.norm(s_pt - p_end) < 1e-4
            or np.linalg.norm(e_pt - p_end) < 1e-4
        ):
            continue
        if segments_intersect(p_start, p_end, s_pt, e_pt):
            crossings += 1
    return crossings


# ==========================================================
# 3. DOUBLE MST BRIDGING & PARITY BALANCING (SECTION 2.2.1)
# ==========================================================

def _node_penalty_multiplier(G, n, even_degree_penalty):
    """
    Prefers degree-1 endpoints (1.0) and odd junctions (1.15), then sharp
    corners (1.35), and penalises flat subdivision nodes (even_degree_penalty).
    """
    deg = G.degree(n)
    if deg == 1:
        return 1.0
    if deg % 2 == 1:
        return 1.15
    if G.nodes[n].get("kind") == "corner":
        return 1.35
    return even_degree_penalty


def double_mst_bridging(
    G,
    canvas_diag,
    max_edge_len_prop=0.10,
    corner_angle_deg=65.0,
    even_degree_penalty=1.8,
    intersection_penalty=15.0,
    edge_reuse_penalty=100.0,
    node_reuse_penalty=2.5,
):
    """
    1. Splits sharp corners and subdivides long edges.
    2. Connects disconnected components using a 2-pass Kruskal MST.
    3. Applies T-join parity pruning on Pass 2 so every component has an even
       number of bridges (>= 2) while keeping the graph connected.
    """
    G = split_edges_at_sharp_corners(G, max_turn_angle_deg=corner_angle_deg)
    G = split_long_edges(G, max_edge_len=max_edge_len_prop * canvas_diag)

    components = [list(c) for c in nx.connected_components(G)]
    num_comp = len(components)
    if num_comp <= 1:
        print("Graph is already connected. No MST bridging needed.")
        return G

    print(f"Subdivided graph: {G.number_of_nodes()} nodes across {num_comp} disconnected components.")

    all_segments = []
    for _, _, d in G.edges(data=True):
        pts = d["pts"]
        for k in range(len(pts) - 1):
            all_segments.append((pts[k], pts[k + 1]))

    node_mult = {n: _node_penalty_multiplier(G, n, even_degree_penalty) for n in G.nodes()}
    candidate_bridges = {}

    for i in range(num_comp):
        comp_i = components[i]
        pts_i = np.array([G.nodes[u]["pos"] for u in comp_i])

        for j in range(i + 1, num_comp):
            comp_j = components[j]
            pts_j = np.array([G.nodes[v]["pos"] for v in comp_j])

            # Vectorised distance matrix to pre-filter the closest candidate pairs
            dist_mat = np.hypot(
                pts_i[:, 0, None] - pts_j[None, :, 0],
                pts_i[:, 1, None] - pts_j[None, :, 1],
            )
            pen_i = np.array([node_mult[u] for u in comp_i])[:, None]
            pen_j = np.array([node_mult[v] for v in comp_j])[None, :]
            approx_cost = dist_mat * pen_i * pen_j

            flat_order = np.argsort(approx_cost.ravel())[:16]
            candidates = []
            for flat_idx in flat_order:
                idx_u, idx_v = np.unravel_index(flat_idx, dist_mat.shape)
                u, v = comp_i[idx_u], comp_j[idx_v]
                pos_u, pos_v = pts_i[idx_u], pts_j[idx_v]
                dist = float(dist_mat[idx_u, idx_v])

                crossings = count_edge_intersections(pos_u, pos_v, all_segments)
                cost = float(approx_cost[idx_u, idx_v]) * (1.0 + intersection_penalty * crossings)

                candidates.append({
                    "u": u,
                    "v": v,
                    "ci": i,
                    "cj": j,
                    "dist": dist,
                    "base_cost": cost,
                    "intersections": crossings,
                })

            candidates.sort(key=lambda c: c["base_cost"])
            candidate_bridges[(i, j)] = candidates[:8]

    def build_component_mst(cost_fn):
        best_edges = []
        for (ci, cj), cand_list in candidate_bridges.items():
            best_cand = min(cand_list, key=cost_fn)
            best_edges.append((ci, cj, cost_fn(best_cand), best_cand))

        best_edges.sort(key=lambda x: x[2])
        dsu = nx.utils.UnionFind()
        selected = []
        for ci, cj, _, cand in best_edges:
            if dsu[ci] != dsu[cj]:
                dsu.union(ci, cj)
                selected.append(cand)
                if len(selected) == num_comp - 1:
                    break
        return selected

    # Pass 1: Primary MST
    mst1_bridges = build_component_mst(lambda c: c["base_cost"])
    print(f"MST Pass 1 selected {len(mst1_bridges)} primary bridges.")

    used_pairs = {tuple(sorted((b["u"], b["v"]))) for b in mst1_bridges}
    used_comp_pairs = {(min(b["ci"], b["cj"]), max(b["ci"], b["cj"])) for b in mst1_bridges}
    used_nodes = {n for b in mst1_bridges for n in (b["u"], b["v"])}

    # Pass 2: Return MST (penalising duplicate bridges and reused nodes)
    def pass2_cost(cand):
        cost = cand["base_cost"]
        if tuple(sorted((cand["u"], cand["v"]))) in used_pairs:
            cost *= edge_reuse_penalty
        if (min(cand["ci"], cand["cj"]), max(cand["ci"], cand["cj"])) in used_comp_pairs:
            cost *= 1.35
        if cand["u"] in used_nodes:
            cost *= node_reuse_penalty
        if cand["v"] in used_nodes:
            cost *= node_reuse_penalty
        return cost

    mst2_bridges = build_component_mst(pass2_cost)
    print(f"MST Pass 2 selected {len(mst2_bridges)} candidate return bridges.")

    # Step 4: Exact T-Join Parity Pruning on Pass 2
    # Keeps the minimal subset of Pass 2 bridges so every component has EVEN bridge degree (>= 2)
    comp_deg_1 = {c: 0 for c in range(num_comp)}
    for b in mst1_bridges:
        comp_deg_1[b["ci"]] += 1
        comp_deg_1[b["cj"]] += 1

    odd_set = {c for c, d in comp_deg_1.items() if d % 2 == 1}
    tree_2 = nx.Graph()
    tree_2.add_nodes_from(range(num_comp))
    for b in mst2_bridges:
        tree_2.add_edge(b["ci"], b["cj"])

    kept_mst2 = []
    for b in mst2_bridges:
        ci, cj = b["ci"], b["cj"]
        tree_2.remove_edge(ci, cj)
        sub_a = nx.node_connected_component(tree_2, ci)
        if len(sub_a & odd_set) % 2 == 1:
            kept_mst2.append(b)
        tree_2.add_edge(ci, cj)

    final_bridges = mst1_bridges + kept_mst2

    comp_degrees = {c: 0 for c in range(num_comp)}
    for b in final_bridges:
        comp_degrees[b["ci"]] += 1
        comp_degrees[b["cj"]] += 1

    print(f"After parity pruning: {len(final_bridges)} total bridges retained.")
    print(f"Bridge degrees per component (all even): {list(comp_degrees.values())}")

    # Step 5: Inject bridges into the graph
    bridged_G = G.copy()
    for b in final_bridges:
        pos_u = bridged_G.nodes[b["u"]]["pos"]
        pos_v = bridged_G.nodes[b["v"]]["pos"]
        bridged_G.add_edge(
            b["u"],
            b["v"],
            pts=np.array([pos_u, pos_v], dtype=float),
            weight=b["dist"],
            bridge=True,
        )

    print(f"Final bridged graph connected: {nx.is_connected(bridged_G)}")
    return bridged_G


# ==========================================================
# 4. SAVING VISUALISATION HELPER
# ==========================================================

def plot_bridged_graph(G, save_path="bridged_graph.png"):
    """Saves a plot of the bridged graph (black = original strokes, cyan = MST bridges)."""
    plt.figure(figsize=(9, 8))

    for u, v, k, data in G.edges(keys=True, data=True):
        pts = data["pts"]
        if data.get("bridge", False):
            plt.plot(pts[:, 0], pts[:, 1], "c--", linewidth=2.2, label="MST Bridge", zorder=4)
        else:
            plt.plot(pts[:, 0], pts[:, 1], "k-", linewidth=1.5, zorder=2)

    for n, data in G.nodes(data=True):
        pos = data["pos"]
        plt.scatter(pos[0], pos[1], color="royalblue", s=18, zorder=5)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.title("Double MST Bridged Graph (Cyan = Bridges)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved bridged graph plot to '{save_path}'")