import argparse
import sys
from pathlib import Path
import networkx as nx
import numpy as np

from vectorizer import (
    image_to_binary_mask,
    skeletonize_mask,
    extract_skeleton_topology,
    simplify_all_edges,
    edges_to_svg,
    polyline_to_svg,
)
from graph_solver import (
    build_graph_from_polylines,
    connect_disjoint_components,
    make_semi_eulerian,
    smooth_eulerian_trail,
    reconstruct_full_polyline_from_edges,
)


def image_to_continuous_polyline(
    input_image_path: str,
    output_svg_path: str = None,
    output_coords_path: str = None,
    preview_png_path: str = None,
    sketch_svg_path: str = None,
    mode: str = "primal",
    threshold_val: int = 200,
    min_speck_area: int = None,
    min_edge_pixels: int = None,
    epsilon: float = 1.0,
    max_angle_deg: float = 60.0,
    length_limit_pct: float = 0.10,
    interior_penalty: float = 40.0,
    stroke_width: float = 1.5,
) -> list:
    """End-to-end pipeline: converts any image into a single continuous (x, y) polyline."""
    if min_speck_area is None:
        min_speck_area = 110 if mode == "minimal" else (1 if mode == "drawing" else 12)
    if min_edge_pixels is None:
        min_edge_pixels = 15 if mode == "minimal" else 2

    print(f"[1/6] Extracting binary sketch ({mode} mode): {input_image_path}...")
    binary = image_to_binary_mask(
        input_image_path,
        mode=mode,
        threshold_val=threshold_val,
        min_speck_area=min_speck_area,
        save_preview_path=preview_png_path,
    )
    height, width = binary.shape

    print("[2/6] Skeletonizing and extracting topological edges...")
    skeleton = skeletonize_mask(binary, method="lee")
    endpoints, junctions, raw_edges = extract_skeleton_topology(skeleton)

    if min_edge_pixels > 2:
        raw_edges = [e for e in raw_edges if len(e["path"]) >= min_edge_pixels]

    print(f"      Found {len(endpoints)} endpoints, {len(junctions)} junctions, {len(raw_edges)} edges.")

    print(f"[3/6] Simplifying edges with RDP (epsilon={epsilon})...")
    simplified_edges = simplify_all_edges(raw_edges, epsilon=epsilon)

    if sketch_svg_path:
        edges_to_svg(simplified_edges, width, height, sketch_svg_path, stroke_width=stroke_width)
        print(f"      Saved disjoint primal sketch SVG to: {sketch_svg_path}")

    print("[4/6] Building topological MultiGraph and bridging disjoint sub-graphs...")
    raw_polylines = [e["path"] for e in simplified_edges if len(e["path"]) >= 2]
    G = build_graph_from_polylines(
        raw_polylines,
        width=width,
        height=height,
        max_angle_deg=max_angle_deg,
        length_limit_pct=length_limit_pct,
    )

    num_components = nx.number_connected_components(G)
    if num_components > 1:
        G = connect_disjoint_components(G, interior_penalty=interior_penalty)
        print(f"      Bridged {num_components} disjoint components via penalized MST.")

    print("[5/6] Solving Semi-Eulerian Chinese Postman Problem...")
    orig_len = sum(d.get("weight", 0.0) for _, _, d in G.edges(data=True))
    G_semi, (start_node, end_node) = make_semi_eulerian(G)
    semi_len = sum(d.get("weight", 0.0) for _, _, d in G_semi.edges(data=True))
    retrace_pct = ((semi_len - orig_len) / max(1.0, orig_len)) * 100.0
    print(f"      Duplicated {G_semi.number_of_edges() - G.number_of_edges()} edges (+{retrace_pct:.1f}% retraced length).")

    print("[6/6] Computing Eulerian trail and reconstructing continuous polyline...")
    edge_trail = smooth_eulerian_trail(G_semi, start_node=start_node)
    polyline = reconstruct_full_polyline_from_edges(G_semi, edge_trail)
    print(f"      Generated continuous polyline with {len(polyline)} (x, y) vertices.")
    if output_svg_path:
        polyline_to_svg(polyline, width, height, output_svg_path, stroke_width=stroke_width)
        print(f"      Saved continuous single-line SVG to: {output_svg_path}")

    if output_coords_path:
        coords_arr = np.array(polyline, dtype=np.float64)
        if output_coords_path.endswith(".npy"):
            np.save(output_coords_path, coords_arr)
        else:
            np.savetxt(output_coords_path, coords_arr, delimiter=",", header="x,y", comments="", fmt="%.4f")
        print(f"      Saved (x, y) coordinates to: {output_coords_path}")

    return polyline


def main():
    parser = argparse.ArgumentParser(description="Convert any image into a single continuous polyline.")
    parser.add_argument("input", nargs="?", default="images/orb.png", help="Path to input image")
    parser.add_argument("-o", "--output", default=None, help="Output single-line SVG path")
    parser.add_argument("-c", "--coords", default=None, help="Optional path to save (x, y) coordinates (.csv, .txt, or .npy)")
    parser.add_argument("-m", "--mode", choices=["primal", "minimal", "xdog", "canny", "drawing "], default="drawing")
    parser.add_argument("-e", "--epsilon", type=float, default=1.0, help="RDP simplification tolerance (px)")
    parser.add_argument("--angle", type=float, default=60.0, help="Corner angle split threshold (deg)")
    parser.add_argument("--len-pct", type=float, default=0.10, help="Max edge length as fraction of canvas size")
    parser.add_argument("--penalty", type=float, default=40.0, help="Interior node penalty for MST bridging")
    parser.add_argument("-w", "--stroke-width", type=float, default=1.5, help="SVG stroke width")
    parser.add_argument("-p", "--preview", default=None, help="Optional path to save binary PNG preview")
    parser.add_argument("--sketch-svg", default=None, help="Optional path to save pre-bridged primal sketch SVG")

    args = parser.parse_args()
    input_path = Path(args.input)

    if args.output is None:
        args.output = str(input_path.with_name(f"{input_path.stem}_oneline.svg"))

    try:
        image_to_continuous_polyline(
            input_image_path=args.input,
            output_svg_path=args.output,
            output_coords_path=args.coords,
            preview_png_path=args.preview,
            sketch_svg_path=args.sketch_svg,
            mode=args.mode,
            epsilon=args.epsilon,
            max_angle_deg=args.angle,
            length_limit_pct=args.len_pct,
            interior_penalty=args.penalty,
            stroke_width=args.stroke_width,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()