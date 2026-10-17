import argparse
import sys
import copy
import networkx as nx
import numpy as np
from svg_to_graph import svg_to_graph, connect_disjoint_components


def make_semi_eulerian(G: nx.MultiGraph):
    """Augments G by duplicating shortest paths strictly along existing edges

    so that exactly two vertices have odd degree (or zero if already Eulerian).
    Returns the augmented graph and the (start_node, end_node) pair.
    """
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected before solving. Run connect_disjoint_components first.")

    odd_nodes = [n for n in G.nodes() if G.degree(n) % 2 != 0]

    # Case 1: Already Eulerian (0 odd nodes)
    if len(odd_nodes) == 0:
        start_node = list(G.nodes())[0]
        return G.copy(), (start_node, start_node)

    # Case 2: Already Semi-Eulerian (exactly 2 odd nodes)
    if len(odd_nodes) == 2:
        return G.copy(), (odd_nodes[0], odd_nodes[1])

    # Case 3: Compute all-pairs shortest paths among odd nodes using existing edges
    odd_dist = {}
    odd_paths = {}
    for u in odd_nodes:
        dist, paths = nx.single_source_dijkstra(G, u, weight="weight")
        odd_dist[u] = dist
        odd_paths[u] = paths

    # Complete auxiliary graph of odd vertices
    aux_G = nx.Graph()
    for i in range(len(odd_nodes)):
        for j in range(i + 1, len(odd_nodes)):
            u, v = odd_nodes[i], odd_nodes[j]
            aux_G.add_edge(u, v, weight=odd_dist[u][v])

    best_cost = float("inf")
    best_matching = []
    best_endpoints = (odd_nodes[0], odd_nodes[1])

    # Find the pair of odd vertices (s, t) to leave un-duplicated that minimizes total retraced length
    for i in range(len(odd_nodes)):
        for j in range(i + 1, len(odd_nodes)):
            s = odd_nodes[i]
            t = odd_nodes[j]

            # Subgraph of remaining |V_odd| - 2 nodes that must be paired up
            remaining_nodes = [n for n in odd_nodes if n != s and n != t]
            sub_G = aux_G.subgraph(remaining_nodes)

            # Minimum weight matching on remaining nodes (maxcardinality is default in NetworkX)
            matching = nx.min_weight_matching(sub_G, weight="weight")
            cost = sum(odd_dist[u][v] for u, v in matching)

            if cost < best_cost:
                best_cost = cost
                best_matching = matching
                best_endpoints = (s, t)

    # Augment G by duplicating existing edges along the matched shortest paths
    G_semi = G.copy()
    for u, v in best_matching:
        path = odd_paths[u][v]
        for idx in range(len(path) - 1):
            n1 = path[idx]
            n2 = path[idx + 1]

            # Select the shortest edge between n1 and n2 to duplicate
            edge_candidates = G[n1][n2]
            best_key = min(edge_candidates.keys(), key=lambda k: edge_candidates[k].get("weight", 1.0))
            dup_data = copy.deepcopy(edge_candidates[best_key])
            dup_data["is_retraced"] = True

            G_semi.add_edge(n1, n2, **dup_data)

    return G_semi, best_endpoints

def simple_eulerian_trail(G_semi: nx.MultiGraph, start_node) -> list:
    """Finds a valid Eulerian path/circuit using standard Hierholzer's algorithm."""
    # Build local adjacency tracking: node -> list of (neighbor, key)
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

        # Find any untraversed edge leaving curr
        unused = [
            (nbr, k) for (nbr, k) in adj[curr]
            if (min(curr, nbr), max(curr, nbr), k) not in used_edges
        ]

        if unused:
            # Pick first available edge (simplest choice)
            nbr, k = unused[0]
            used_edges.add((min(curr, nbr), max(curr, nbr), k))
            stack.append(nbr)
        else:
            trail.append(stack.pop())

    trail.reverse()
    return trail


def reconstruct_full_polyline(G_semi: nx.MultiGraph, node_trail: list) -> list:
    """Stitches node transitions back into the exact curved polyline paths

    without cutting across empty space.
    """
    if len(node_trail) < 2:
        return []

    # Map each edge instance to a tracking count so multiple parallel edges 
    # (or retraced paths) are each traversed exactly as many times as they exist
    edge_tracker = {}
    for u, v, k, data in G_semi.edges(keys=True, data=True):
        edge_id = (min(u, v), max(u, v), k)
        edge_tracker[edge_id] = {
            "path": data.get("path", [u, v]),
            "used": False
        }

    full_coords = [node_trail[0]]

    for i in range(len(node_trail) - 1):
        u = node_trail[i]
        v = node_trail[i + 1]

        # Search for an unused edge instance between u and v
        matched_edge_id = None
        for k in G_semi[u][v]:
            edge_id = (min(u, v), max(u, v), k)
            if not edge_tracker[edge_id]["used"]:
                matched_edge_id = edge_id
                break

        if matched_edge_id is not None:
            edge_tracker[matched_edge_id]["used"] = True
            raw_path = edge_tracker[matched_edge_id]["path"]

            # Compare Euclidean distance to determine orientation
            p_start = np.array(raw_path[0])
            p_end = np.array(raw_path[-1])
            u_arr = np.array(u)

            if np.linalg.norm(p_start - u_arr) <= np.linalg.norm(p_end - u_arr):
                segment = raw_path[1:]
            else:
                segment = list(reversed(raw_path))[1:]

            full_coords.extend(segment)
        else:
            # If for any reason an edge instance was already flagged, take the shortest path geometry
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

def main():
    parser = argparse.ArgumentParser(description="Solve Semi-Eulerian Chinese Postman Problem on an SVG graph.")
    parser.add_argument("svg_file", nargs="?", default="images/output.svg", help="Input SVG file path")
    parser.add_argument("--angle", type=float, default=60.0, help="Corner angle split threshold (deg)")
    parser.add_argument("--len-pct", type=float, default=0.10, help="Max edge length as fraction of image size")
    parser.add_argument("--penalty", type=float, default=40.0, help="Interior node penalty for MST component bridging")

    args = parser.parse_args()

    try:
        # 1. Parse SVG into topological MultiGraph
        G = svg_to_graph(args.svg_file, max_angle_deg=args.angle, length_limit_pct=args.len_pct)
        if nx.number_connected_components(G) > 1:
            G = connect_disjoint_components(G, interior_penalty=args.penalty)

        orig_len = sum(d.get("weight", 0.0) for _, _, d in G.edges(data=True))
        odd_count = sum(1 for n in G.nodes() if G.degree(n) % 2 != 0)
        print(f"Base Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({odd_count} odd-degree vertices).")
        print(f"Base Stroke Length: {orig_len:.2f} px")

        # 2. Augment to Semi-Eulerian
        G_semi, (start_node, end_node) = make_semi_eulerian(G)
        semi_len = sum(d.get("weight", 0.0) for _, _, d in G_semi.edges(data=True))
        retrace_len = semi_len - orig_len
        retrace_pct = (retrace_len / max(1.0, orig_len)) * 100.0

        print(f"\nSemi-Eulerian Graph: {G_semi.number_of_edges()} edges ({G_semi.number_of_edges() - G.number_of_edges()} duplicated).")
        print(f"Traversed Stroke Length: {semi_len:.2f} px (+{retrace_pct:.1f}% retracing).")
        print(f"Start Node: {start_node} -> End Node: {end_node}")

        # 3. Solve Trail using standard Hierholzer
        node_trail = simple_eulerian_trail(G_semi, start_node=start_node)
        print(f"Trail calculated: {len(node_trail)} node steps.")

        # 4. Generate continuous single-line coordinate sequence
        polyline = reconstruct_full_polyline(G_semi, node_trail)
        print(f"Full single-line path: {len(polyline)} continuous (x, y) coordinates.")

    except FileNotFoundError:
        print(f"Error: SVG file not found at '{args.svg_file}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()