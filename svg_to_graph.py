import argparse
import sys
import xml.etree.ElementTree as ET
import numpy as np
import networkx as nx


def parse_svg_paths(svg_path: str):
    """Parses raw polyline paths and canvas bounds from an SVG file[cite: 1].

    Extracts (x, y) coordinate sequences from SVG path strings (M, L, Z commands)[cite: 1].
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Determine canvas bounds from width/height or viewBox[cite: 1]
    width = float(root.attrib.get("width", 0))
    height = float(root.attrib.get("height", 0))
    if "viewBox" in root.attrib:
        vb = [float(v) for v in root.attrib["viewBox"].replace(",", " ").split()]
        width, height = vb[2], vb[3]

    if width == 0 or height == 0:
        width, height = 500.0, 500.0  # Safe fallback if dimensions are omitted

    # Match path elements with or without standard XML namespaces[cite: 1]
    namespaces = {"svg": "http://www.w3.org/2000/svg"}
    path_elements = root.findall(".//svg:path", namespaces) or root.findall(".//path")

    raw_polylines = []
    for elem in path_elements:
        d = elem.attrib.get("d", "")
        tokens = d.replace(",", " ").split()
        if not tokens:
            continue

        current_line = []
        i = 0
        while i < len(tokens):
            cmd = tokens[i]
            if cmd == "M":
                if current_line:
                    raw_polylines.append(current_line)
                    current_line = []
                current_line.append((float(tokens[i + 1]), float(tokens[i + 2])))
                i += 3
            elif cmd == "L":
                current_line.append((float(tokens[i + 1]), float(tokens[i + 2])))
                i += 3
            elif cmd == "Z":
                if current_line and current_line[0] != current_line[-1]:
                    current_line.append(current_line[0])
                raw_polylines.append(current_line)
                current_line = []
                i += 1
            else:
                i += 1

        if current_line:
            raw_polylines.append(current_line)

    return raw_polylines, width, height


def compute_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes the deflection angle between two consecutive segment vectors in degrees[cite: 2]."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def split_polyline_by_angle(points: list, max_angle_deg: float = 60.0):
    """Splits a polyline into sub-polylines at vertices where directional deviation >= max_angle_deg[cite: 2].

    Returns:
        sub_polylines (list of list of tuples): Subdivided line segments.
        corner_points (set of tuples): Coordinates where sharp angle breaks were detected[cite: 2].
    """
    if len(points) <= 2:
        return [points], set()

    sub_polylines = []
    corner_points = set()
    current_segment = [points[0]]

    for i in range(1, len(points) - 1):
        p_prev = np.array(points[i - 1], dtype=np.float64)
        p_curr = np.array(points[i], dtype=np.float64)
        p_next = np.array(points[i + 1], dtype=np.float64)

        v_in = p_curr - p_prev
        v_out = p_next - p_curr

        angle = compute_angle_deg(v_in, v_out)
        current_segment.append(points[i])

        if angle >= max_angle_deg:
            corner_points.add(points[i])
            sub_polylines.append(current_segment)
            current_segment = [points[i]]

    current_segment.append(points[-1])
    sub_polylines.append(current_segment)
    return sub_polylines, corner_points


def polyline_arc_length(points: list) -> float:
    """Calculates cumulative Euclidean arc length along a sequence of coordinate vertices."""
    pts = np.array(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def split_polyline_by_length(points: list, max_length: float) -> list:
    """Subdivides a polyline so no individual edge segment exceeds max_length."""
    total_len = polyline_arc_length(points)
    if total_len <= max_length or len(points) < 2:
        return [points]

    pts = np.array(points, dtype=np.float64)
    seg_dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum_dists = np.insert(np.cumsum(seg_dists), 0, 0.0)

    n_splits = int(np.ceil(total_len / max_length))
    target_dists = np.linspace(0.0, total_len, n_splits + 1)

    sub_polylines = []
    for s in range(n_splits):
        start_d = target_dists[s]
        end_d = target_dists[s + 1]

        sub_pts = []

        # Interpolate start coordinate
        idx_s = np.searchsorted(cum_dists, start_d)
        if idx_s == 0:
            sub_pts.append(tuple(pts[0]))
        else:
            prev_d = cum_dists[idx_s - 1]
            t = (start_d - prev_d) / (cum_dists[idx_s] - prev_d)
            interp = pts[idx_s - 1] + t * (pts[idx_s] - pts[idx_s - 1])
            sub_pts.append((float(interp[0]), float(interp[1])))

        # Preserve interior raw vertices
        for k in range(1, len(points) - 1):
            if start_d < cum_dists[k] < end_d:
                sub_pts.append(points[k])

        # Interpolate end coordinate
        idx_e = np.searchsorted(cum_dists, end_d)
        if idx_e >= len(cum_dists):
            sub_pts.append(tuple(pts[-1]))
        else:
            prev_d = cum_dists[idx_e - 1]
            t = (end_d - prev_d) / (cum_dists[idx_e] - prev_d)
            interp = pts[idx_e - 1] + t * (pts[idx_e] - pts[idx_e - 1])
            sub_pts.append((float(interp[0]), float(interp[1])))

        sub_polylines.append(sub_pts)

    return sub_polylines


def svg_to_graph(
    svg_file: str,
    max_angle_deg: float = 60.0,
    length_limit_pct: float = 0.10,
    decimals: int = 2
) -> nx.MultiGraph:
    """Constructs a NetworkX MultiGraph from an SVG file[cite: 2].

    Applies angle splitting on sharp corners and length-based edge subdivision[cite: 2].

    Parameters:
        svg_file (str): Path to input SVG file.
        max_angle_deg (float): Deflection angle threshold in degrees to split corners[cite: 2].
        length_limit_pct (float): Maximum allowed edge length as a fraction of max canvas dimension.
        decimals (int): Coordinate quantization precision to ensure adjacent edges share exact node IDs.

    Returns:
        nx.MultiGraph: Graph populated with geometric nodes and edge attributes[cite: 2].
    """
    raw_polylines, width, height = parse_svg_paths(svg_file)
    max_edge_len = max(width, height) * length_limit_pct

    G = nx.MultiGraph()
    G.graph["canvas_width"] = width
    G.graph["canvas_height"] = height

    def quantize(pt):
        return (round(float(pt[0]), decimals), round(float(pt[1]), decimals))

    all_corner_nodes = set()

    for poly in raw_polylines:
        # Step A: Split strokes at sharp directional changes[cite: 2]
        angle_splits, corners = split_polyline_by_angle(poly, max_angle_deg=max_angle_deg)
        for c in corners:
            all_corner_nodes.add(quantize(c))

        # Step B: Split long edges exceeding length limit
        final_segments = []
        for seg in angle_splits:
            final_segments.extend(split_polyline_by_length(seg, max_length=max_edge_len))

        # Step C: Register nodes and edges into the MultiGraph[cite: 2]
        for seg in final_segments:
            if len(seg) < 2:
                continue

            u = quantize(seg[0])
            v = quantize(seg[-1])

            length = polyline_arc_length(seg)
            start_tangent = (seg[1][0] - seg[0][0], seg[1][1] - seg[0][1])
            end_tangent = (seg[-1][0] - seg[-2][0], seg[-1][1] - seg[-2][1])

            if u not in G:
                G.add_node(u, pos=u, is_corner=(u in all_corner_nodes))
            if v not in G:
                G.add_node(v, pos=v, is_corner=(v in all_corner_nodes))

            G.add_edge(
                u,
                v,
                path=seg,
                weight=length,
                start_tangent=start_tangent,
                end_tangent=end_tangent,
                is_bridge=False
            )

    # Flag degree-1 nodes as endpoints[cite: 1, 2]
    for n in G.nodes():
        G.nodes[n]["is_endpoint"] = (G.degree(n) == 1)

    return G


def connect_disjoint_components(
    G: nx.MultiGraph,
    interior_penalty: float = 40.0
) -> nx.MultiGraph:
    """Connects all disjoint components using a Minimum Spanning Tree with an interior node penalty[cite: 2].

    Degree-2 interior nodes incur an additional penalty, favoring endpoints or sharp corners[cite: 2].
    """
    components = list(nx.connected_components(G))
    k = len(components)
    if k <= 1:
        return G

    def get_penalty(node):
        if G.degree(node) != 2 or G.nodes[node].get("is_corner", False):
            return 0.0
        return interior_penalty

    # Meta-graph over connected components[cite: 2]
    H = nx.Graph()
    for i in range(k):
        H.add_node(i)

    for i in range(k):
        comp_i = list(components[i])
        for j in range(i + 1, k):
            comp_j = list(components[j])

            best_effective_cost = float("inf")
            best_pair = None
            best_dist = float("inf")

            for u in comp_i:
                u_pos = np.array(G.nodes[u]["pos"])
                pen_u = get_penalty(u)

                for v in comp_j:
                    v_pos = np.array(G.nodes[v]["pos"])
                    pen_v = get_penalty(v)

                    dist = float(np.linalg.norm(u_pos - v_pos))
                    cost = dist + pen_u + pen_v

                    if cost < best_effective_cost:
                        best_effective_cost = cost
                        best_pair = (u, v)
                        best_dist = dist

            H.add_edge(i, j, weight=best_effective_cost, actual_dist=best_dist, bridge_nodes=best_pair)

    # Solve MST to obtain exactly k - 1 bridges[cite: 2]
    mst = nx.minimum_spanning_tree(H, weight="weight")

    for i, j, data in mst.edges(data=True):
        u, v = data["bridge_nodes"]
        dist = data["actual_dist"]
        p1 = G.nodes[u]["pos"]
        p2 = G.nodes[v]["pos"]
        tangent = (p2[0] - p1[0], p2[1] - p1[1])

        G.add_edge(
            u,
            v,
            path=[p1, p2],
            weight=dist,
            start_tangent=tangent,
            end_tangent=tangent,
            is_bridge=True
        )

    return G


def main():
    parser = argparse.ArgumentParser(description="Convert an SVG drawing into a topological NetworkX MultiGraph[cite: 2].")
    parser.add_argument("svg_file", nargs="?", default="images/output.svg", help="Path to input SVG[cite: 1]")
    parser.add_argument("--angle", type=float, default=60.0, help="Max turn angle (deg) before splitting into a corner node[cite: 2]")
    parser.add_argument("--len-pct", type=float, default=0.10, help="Max edge length as fraction of image size (default: 0.10)")
    parser.add_argument("--penalty", type=float, default=40.0, help="Distance penalty for connecting to degree-2 interior nodes[cite: 2]")
    parser.add_argument("--connect", action="store_true", help="Automatically bridge disjoint components via MST[cite: 2]")

    args = parser.parse_args()

    try:
        G = svg_to_graph(args.svg_file, max_angle_deg=args.angle, length_limit_pct=args.len_pct)
        initial_components = nx.number_connected_components(G)
        print(f"Graph constructed:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Connected components: {initial_components}")

        if args.connect and initial_components > 1:
            G = connect_disjoint_components(G, interior_penalty=args.penalty)
            print(f"Disjoint components bridged via MST. Total components now: {nx.number_connected_components(G)}[cite: 2]")

    except FileNotFoundError:
        print(f"Error: SVG file not found at '{args.svg_file}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()