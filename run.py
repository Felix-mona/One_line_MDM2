"""
run.py

Command-line entrypoint for the MDM2 Group 18 Continuous Line Drawing Pipeline.
Wraps `full_pipeline.run_oneline_pipeline` so any raster sketch can be converted
and evaluated from the terminal without editing source files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure sibling modules resolve regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import networkx as nx
from pipeline.full_pipeline import run_oneline_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a 2D raster sketch into a continuous one-line drawing."
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the input raster image (e.g. PNG, JPG).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="outputs/",
        help="Output directory or file path (default: outputs/).",
    )
    parser.add_argument(
        "-t",
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.04],
        help="Target normalised Hausdorff accuracy threshold(s) in [0, 1] (default: 0.01 0.02 0.04).",
    )
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=10000,
        help="Number of uniform arc-length samples N for DFT (default: 10000).",
    )
    parser.add_argument(
        "--stroke-width",
        type=float,
        default=10.0,
        help="Estimated input stroke width in pixels (default: 10.0).",
    )
    parser.add_argument(
        "--thresh",
        type=int,
        default=128,
        help="Grayscale binarisation threshold T_bin in [0, 255] (default: 128).",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save all intermediate step plots, SVG trajectories, and NumPy arrays.",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also export reconstructed trajectories as standalone SVG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: '{image_path}'")

    results = run_oneline_pipeline(
        image_path=image_path,
        output_path=args.output,
        accuracy_thresholds=args.thresholds,
        n_samples=args.samples,
        input_stroke_width=args.stroke_width,
        binarise_thresh=args.thresh,
        save_all_intermediaries=args.save_all,
        save_intermediate_svg=args.svg,
    )

    # Print summary metrics aligned with Table 1 of the report
    base_graph = results["base_graph"]
    euler_graph = results["euler_graph"]
    canvas_diag = results["canvas_diag"]

    def _polyline_len(pts) -> float:
        import numpy as np
        arr = np.asarray(pts, dtype=float)
        if len(arr) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))

    num_comp = nx.number_connected_components(base_graph)
    l_base = sum(_polyline_len(d["pts"]) for _, _, d in base_graph.edges(data=True))
    l_total = sum(_polyline_len(d["pts"]) for _, _, d in euler_graph.edges(data=True))
    overhead_pct = 100.0 * (l_total - l_base) / max(l_total, 1e-9)

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE — SUMMARY")
    print("=" * 65)
    print(f"Input image      : {image_path}")
    print(f"Components (C)   : {num_comp}")
    print(f"L_total / D      : {l_total / canvas_diag:.2f}")
    print(f"Overhead         : {overhead_pct:.1f}% ((L_total - L_base) / L_total)")
    for rec in results["reconstructions"]:
        print(
            f"  [Acc <= {rec['target_accuracy']}] K = {rec['k']} | "
            f"d_norm = {rec['achieved_accuracy']:.4f} | "
            f"Smoothness S = {rec['smoothness']:.4f} -> {rec['output_image_path']}"
        )


if __name__ == "__main__":
    main()