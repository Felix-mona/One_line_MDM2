"""
Double MST Component Bridging (Sections 2.1.4 & 2.2.1)
Connects disconnected sub-graphs into a single connected graph using two rounds
of Minimum Spanning Trees with perceptual penalties and parity balancing.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ==========================================================
# 1. LONG-EDGE SUBDIVISION (SECTION 2.1.4)
# ==========================================================

def subdivide_polyline(pts, max_len):
    """
    Subdivides a polyline array of shape (N, 2) so that no individual segment
    or total arc length of any sub-piece exceeds max_len.
    Returns a list of sub-polyline arrays.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return [pts]

    # Arc lengths between consecutive polyline vertices
    diffs = pts[1:] - pts[:-1]
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    total_len = np.sum(seg_lengths)

    if total_len <= max_len:
        return [pts]

    # Determine number of equal-sized sub-divisions
    num_splits = int(np.ceil(total_len / max_len))
    target_sub_len = total_len / num_splits

    # Cumulative distance along points
    cum_dist = np.insert(np.cumsum(seg_lengths), 0, 0.0)

    sub_polylines = []
    current_start_idx = 0
    current_start_pt = pts[0]

    for k in range(1, num_splits):
        target_dist = k * target_sub_len
        
        # Locate the segment where target_dist falls
        idx = np.searchsorted(cum_dist, target_dist) - 1
        idx = min(idx, len(pts) - 2)

        d0 = cum_dist[idx]
        d1 = cum_dist[idx + 1]
        seg_len = d1 - d0

        t = (target_dist - d0) / max(seg_len, 1e-12)
        interp_pt = pts[idx] + t * (pts[idx + 1] - pts[idx])

        # Slice sub-polyline from current start up to interpolation point
        sub_pts = [current_start_pt]
        if idx + 1 > current_start_idx + 1:
            sub_pts.extend(pts[current_start_idx + 1 : idx + 1])
        sub_pts.append(interp_pt)

        sub_polylines.append(np.array(sub_pts, dtype=float))
        current_start_idx = idx
        current_start_pt = interp_pt

    # Final piece to end
    final_pts = [current_start_pt]
    if len(pts) > current_start_idx + 1:
        final_pts.extend(pts[current_start_idx + 1 :])
    sub_polylines.append(np.array(final_pts, dtype=float))

    return sub_polylines


def split_long_edges(G, max_edge_len):
    """
    Splits any edge in G whose total polyline length > max_edge_len,
    inserting intermediate nodes along the stroke.
    """
    new_G = nx.MultiGraph()
    for n, attrs in G.nodes(data=True):
        new_G.add_node(n, **attrs)

    next_node_id = max(G.nodes) + 1 if len(G.nodes) > 0 else 0

    for u, v, k, data in G.edges(keys=True, data=True):
        pts = data["pts"]
        sub_pieces = subdivide_polyline(pts, max_edge_len)

        if len(sub_pieces) == 1:
            new_G.add_edge(u, v, **data)
        else:
            prev_node = u
            for i, piece in enumerate(sub_pieces):
                if i == len(sub_pieces) - 1:
                    end_node = v
                else:
                    end_node = next_node_id
                    new_G.add_node(end_node, pos=piece[-1].copy())
                    next_node_id += 1

                attrs = dict(data)
                attrs["pts"] = piece
                new_G.add_edge(prev_node, end_node, **attrs)
                prev_node = end_node

    return new_G


# ==========================================================
# 2. GEOMETRIC INTERSECTION CHECKS
# ==========================================================

def segments_intersect(p1, p2, p3, p4):
    """
    Returns True if line segment (p1 -> p2) strictly intersects (p3 -> p4).
    """
    # Quick bounding-box rejection
    if (max(p1[0], p2[0]) < min(p3[0], p4[0]) or
        min(p1[0], p2[0]) > max(p3[0], p4[0]) or
        max(p1[1], p2[1]) < min(p3[1], p4[1]) or
        min(p1[1], p2[1]) > max(p3[1], p4[1])):
        return False

    def ccw(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)

    # Strictly alternating signs indicate interior crossing
    return ((d1 * d2 < -1e-9) and (d3 * d4 < -1e-9))


def count_edge_intersections(p_start, p_end, all_edge_segments, ignore_nodes_pts):
    """
    Counts how many existing graph stroke segments a candidate bridge crosses.
    """
    crossings = 0
    for s_pt, e_pt in all_edge_segments:
        # Ignore segments attached directly to the bridge endpoints
        if (np.linalg.norm(s_pt - p_start) < 1e-4 or np.linalg.norm(e_pt - p_start) < 1e-4 or
            np.linalg.norm(s_pt - p_end) < 1e-4 or np.linalg.norm(e_pt - p_end) < 1e-4):
            continue
        if segments_intersect(p_start, p_end, s_pt, e_pt):
            crossings += 1
    return crossings


# ==========================================================
# 3. DOUBLE MST BRIDGING & PARITY BALANCING (SECTION 2.2.1)
# ==========================================================

def double_mst_bridging(
    G,
    canvas_diag,
    max_edge_len_prop=0.10,     # Edges split if longer than 10% of content diagonal
    even_degree_penalty=1.8,    # Penalise even nodes to favour endpoints
    intersection_penalty=15.0,  # Multiplier per intersected edge
    edge_reuse_penalty=100.0,   # Heavy penalty on duplicate bridges for 2nd pass
    node_reuse_penalty=2.5      # Penalty for re-using the same entry/exit node
):
    """
    Subdivides long edges, then connects disconnected components using a 2-pass MST.
    Prunes redundant bridges so that each component has an even number of bridges (>= 2).
    """
    # Step 1: Subdivide long edges
    max_edge_len = max_edge_len_prop * canvas_diag
    G = split_long_edges(G, max_edge_len)

    # Step 2: Component identification
    components = [list(c) for c in nx.connected_components(G)]
    num_comp = len(components)
    if num_comp <= 1:
        print("Graph is already fully connected. No bridging required.")
        return G

    print(f"Subdivided graph: {G.number_of_nodes()} nodes. Found {num_comp} disconnected components.")

    # Flatten existing edge segments for fast intersection tests
    all_segments = []
    for _, _, d in G.edges(data=True):
        pts = d["pts"]
        for i in range(len(pts) - 1):
            all_segments.append((pts[i], pts[i + 1]))

    # Step 3: Compute candidate bridge costs between all component pairs
    # candidate_bridges[(ci, cj)] = list of candidate dicts
    candidate_bridges = {}

    for i in range(num_comp):
        for j in range(i + 1, num_comp):
            candidates = []
            comp_i = components[i]
            comp_j = components[j]

            for u in comp_i:
                pos_u = G.nodes[u]["pos"]
                deg_u = G.degree(u)

                for v in comp_j:
                    pos_v = G.nodes[v]["pos"]
                    deg_v = G.degree(v)

                    dist = float(np.hypot(pos_u[0] - pos_v[0], pos_u[1] - pos_v[1]))

                    # Intersections with existing graph strokes
                    intersections = count_edge_intersections(pos_u, pos_v, all_segments, [pos_u, pos_v])

                    # Base cost formulation
                    # 1. Base distance (normalised by diagonal)
                    cost = dist

                    # 2. Penalise even-degree nodes (endpoints/odd nodes make cleaner joins)
                    if deg_u % 2 == 0:
                        cost *= even_degree_penalty
                    if deg_v % 2 == 0:
                        cost *= even_degree_penalty

                    # 3. Heavily penalise line intersections
                    if intersections > 0:
                        cost *= (1.0 + intersection_penalty * intersections)

                    candidates.append({
                        "u": u,
                        "v": v,
                        "ci": i,
                        "cj": j,
                        "dist": dist,
                        "base_cost": cost,
                        "intersections": intersections
                    })

            # Sort by lowest base cost and keep the top candidate pool
            candidates.sort(key=lambda c: c["base_cost"])
            candidate_bridges[(i, j)] = candidates[:8]

    # --- PASS 1: FIRST MST ---
    def build_component_mst(candidate_map, cost_calculator):
        H = nx.Graph()
        H.add_nodes_from(range(num_comp))
        
        best_edges = []
        for (ci, cj), cand_list in candidate_map.items():
            best_cand = min(cand_list, key=cost_calculator)
            best_edges.append((ci, cj, cost_calculator(best_cand), best_cand))

        # Sort edges by cost and build MST via Kruskal
        best_edges.sort(key=lambda x: x[2])
        dsu = nx.utils.UnionFind()
        selected_bridges = []

        for ci, cj, cost, cand in best_edges:
            if dsu[ci] != dsu[cj]:
                dsu.union(ci, cj)
                selected_bridges.append(cand)
                if len(selected_bridges) == num_comp - 1:
                    break
        return selected_bridges

    # MST 1 selection
    mst1_bridges = build_component_mst(
        candidate_bridges,
        cost_calculator=lambda c: c["base_cost"]
    )
    print(f"MST Pass 1 selected {len(mst1_bridges)} primary bridges.")

    # Track usage from Pass 1
    used_bridge_pairs = set()
    used_nodes = set()
    for b in mst1_bridges:
        pair = tuple(sorted([b["u"], b["v"]]))
        used_bridge_pairs.add(pair)
        used_nodes.add(b["u"])
        used_nodes.add(b["v"])

    # --- PASS 2: SECOND MST (HEAVILY PENALISE PREVIOUS BRIDGES & NODES) ---
    def pass2_cost(cand):
        cost = cand["base_cost"]
        pair = tuple(sorted([cand["u"], cand["v"]]))
        
        # Heavy penalty for already-used bridge
        if pair in used_bridge_pairs:
            cost *= edge_reuse_penalty

        # Penalty for reusing the same entry/exit node on a component
        if cand["u"] in used_nodes:
            cost *= node_reuse_penalty
        if cand["v"] in used_nodes:
            cost *= node_reuse_penalty

        return cost

    mst2_bridges = build_component_mst(
        candidate_bridges,
        cost_calculator=pass2_cost
    )
    print(f"MST Pass 2 selected {len(mst2_bridges)} return bridges.")

    # Combine both bridge sets
    all_selected_bridges = mst1_bridges + mst2_bridges

    # --- STEP 4: REDUNDANT BRIDGE PRUNING (PARITY BALANCING) ---
    # Build component multigraph H to track bridge degrees
    bridge_graph = nx.MultiGraph()
    bridge_graph.add_nodes_from(range(num_comp))
    for idx, b in enumerate(all_selected_bridges):
        bridge_graph.add_edge(b["ci"], b["cj"], key=idx, bridge=b)

    # Prune redundant bridges to make each component degree even and >= 2
    changed = True
    while changed:
        changed = False
        odd_components = [c for c in bridge_graph.nodes if bridge_graph.degree(c) % 2 == 1]
        if not odd_components:
            break

        # Find candidate bridge to remove that resolves odd degrees without disconnecting
        best_candidate_idx = None
        best_prune_priority = -1

        for u_c, v_c, k, data in bridge_graph.edges(keys=True, data=True):
            # Check if removing keeps bridge_graph connected
            bridge_graph.remove_edge(u_c, v_c, key=k)
            is_connected = nx.is_connected(bridge_graph)
            deg_u = bridge_graph.degree(u_c)
            deg_v = bridge_graph.degree(v_c)
            bridge_graph.add_edge(u_c, v_c, key=k, **data)

            if not is_connected:
                continue

            # Ensure neither component is left stranded (must retain at least 2 bridges)
            if deg_u < 2 or deg_v < 2:
                continue

            # Priority 1: Both incident components are currently odd (removing makes BOTH even!)
            if (u_c in odd_components) and (v_c in odd_components):
                priority = 100.0 + data["bridge"]["base_cost"]
            # Priority 2: One component is odd and has excess bridges (> 2)
            elif (u_c in odd_components or v_c in odd_components):
                priority = 10.0 + data["bridge"]["base_cost"]
            else:
                continue

            if priority > best_prune_priority:
                best_prune_priority = priority
                best_candidate_idx = (u_c, v_c, k)

        if best_candidate_idx is not None:
            u_c, v_c, k = best_candidate_idx
            bridge_graph.remove_edge(u_c, v_c, key=k)
            changed = True

    # Final list of kept bridges
    final_bridges = [data["bridge"] for _, _, _, data in bridge_graph.edges(keys=True, data=True)]
    print(f"After parity pruning: {len(final_bridges)} bridges retained.")

    # Report component degrees
    comp_degrees = [bridge_graph.degree(c) for c in bridge_graph.nodes]
    print(f"Bridge degrees per component (target all even): {comp_degrees}")

    # Step 5: Inject final bridges into the graph
    bridged_G = G.copy()
    for b in final_bridges:
        pos_u = G.nodes[b["u"]]["pos"]
        pos_v = G.nodes[b["v"]]["pos"]
        bridge_pts = np.array([pos_u, pos_v], dtype=float)

        bridged_G.add_edge(
            b["u"],
            b["v"],
            pts=bridge_pts,
            weight=b["dist"],
            bridge=True
        )

    print(f"Final bridged graph connected: {nx.is_connected(bridged_G)}")
    return bridged_G


# ==========================================================
# VISUALISATION HELPER
# ==========================================================

def plot_bridged_graph(G):
    """
    Plots the graph showing original strokes in black and MST bridges in cyan.
    """
    plt.figure(figsize=(9, 8))
    
    # Original vs Bridge edges
    for u, v, k, data in G.edges(keys=True, data=True):
        pts = data["pts"]
        if data.get("bridge", False):
            plt.plot(pts[:, 0], pts[:, 1], "c--", linewidth=2.5, label="MST Bridge")
        else:
            plt.plot(pts[:, 0], pts[:, 1], "k-", linewidth=1.5)

    for n, data in G.nodes(data=True):
        pos = data["pos"]
        plt.scatter(pos[0], pos[1], color="blue", s=25, zorder=5)

    # Avoid duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.title("Double MST Bridged Graph (Cyan = Bridges)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()