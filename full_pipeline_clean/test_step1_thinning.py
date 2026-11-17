"""
test_step1_thinning.py

Step 1 Thinning & Graph Extraction Comparison Test (Report Section 2.1 & Appendix A.1).
Generates a synthetic morphological stress-test image (acute 5-pointed star,
parallel 45-degree diagonals, small closed loops, and orthogonal T/X-junctions)
and compares Lee's medial-axis thinning ('lee') against Zhang-Suen thinning
('zhang'), saving borderless skeleton and Rutovitz node plots for both.
"""

from __future__ import annotations

import os
from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw

from graph_formulation import (
    apply_rdp_to_graph,
    compute_rutovitz_crossing_number,
    extract_canonical_nodes,
    load_and_scale_image,
    merge_nearby_junctions,
    plot_graph_snapshot,
    plot_junction_classification,
    prune_whiskers_and_noise,
    run_thinning,
    save_binary_or_skeleton_image,
    save_skeleton_image,
    trace_skeleton_to_graph,
)
from route_mapping import plot_original_graph

# ==============================================================================
# CONFIGURATION & PARAMETERS
# ==============================================================================
BENCHMARK_IMAGE_PATH    = "thinning_benchmark.png"
OUTPUT_DIR              = "test_outputs"
THINNING_METHODS        = ["lee", "zhang"]

INPUT_STROKE_WIDTH      = 10.0
TARGET_STROKE_WIDTH     = 10.0
BINARISE_THRESH         = 128
RDP_EPSILON_PROP        = 0.001
D_EUCLID_PROP           = 0.008
D_EDGE_FACTOR           = 2.5
WHISKER_PROP            = 0.010
MIN_COMP_PROP           = 0.015


def create_thinning_benchmark_image(save_path: str = BENCHMARK_IMAGE_PATH) -> str:
    """Generates a 400x400 morphological stress-test PNG comparing thinning algorithms."""
    img = Image.new("L", (400, 400), color=255)
    draw = ImageDraw.Draw(img)

    # 1. Acute 5-pointed star
    cx, cy, r_out, r_in = 110, 110, 80, 28
    star_pts = []
    for i in range(10):
        ang = np.deg2rad(i * 36 - 90)
        r = r_out if i % 2 == 0 else r_in
        star_pts.append((cx + r * np.cos(ang), cy + r * np.sin(ang)))
    draw.polygon(star_pts, outline=0, width=6)

    # 2. Parallel 45-degree diagonal strokes (staircase test)
    for offset in (0, 35, 70):
        draw.line([(230 + offset, 40), (350, 160 - offset)], fill=0, width=6)

    # 3. Small closed loops (Euler characteristic preservation test)
    draw.ellipse([60, 250, 76, 266], outline=0, width=4)
    draw.ellipse([110, 240, 145, 275], outline=0, width=5)

    # 4. Orthogonal T- and X-junctions (spur test)
    draw.line([(230, 260), (350, 260)], fill=0, width=6)
    draw.line([(290, 210), (290, 330)], fill=0, width=6)

    img.save(save_path)
    return save_path


def _run_step1_for_method(
    binary_mask: np.ndarray,
    scale_factor: float,
    canvas_diag: float,
    stem: str,
    out_dir: Path,
    method: str,
) -> dict:
    """Runs skeletonisation and Step 1 graph extraction for a single thinning method."""
    skeleton = run_thinning(binary_mask, method=method)
    save_skeleton_image(skeleton, str(out_dir / f"{stem}_{method}_02_skeleton.png"))

    d_euclid = D_EUCLID_PROP * canvas_diag
    d_edge_thresh = d_euclid * D_EDGE_FACTOR
    rdp_epsilon = RDP_EPSILON_PROP * canvas_diag
    whisker_thresh = WHISKER_PROP * canvas_diag
    min_comp_len = MIN_COMP_PROP * canvas_diag

    cn = compute_rutovitz_crossing_number(skeleton)
    nodes, node_types, node_grid, node_members = extract_canonical_nodes(
        skeleton, cn, merge_dist_work=d_euclid / scale_factor
    )
    plot_junction_classification(
        skeleton,
        nodes,
        node_types,
        scale_factor,
        str(out_dir / f"{stem}_{method}_03_junction_classification.png"),
    )

    raw_graph = trace_skeleton_to_graph(
        skeleton, nodes, node_types, node_grid, node_members, scale_factor
    )
    plot_graph_snapshot(
        raw_graph,
        str(out_dir / f"{stem}_{method}_04_raw_traced_graph.png"),
        label=f"raw traced graph ({method})",
    )

    rdp_graph = apply_rdp_to_graph(raw_graph, epsilon=rdp_epsilon)
    plot_graph_snapshot(
        rdp_graph,
        str(out_dir / f"{stem}_{method}_05_rdp_simplified.png"),
        label=f"RDP simplified graph ({method})",
    )

    contracted_graph = merge_nearby_junctions(rdp_graph, d_edge_thresh=d_edge_thresh)
    base_graph = prune_whiskers_and_noise(
        contracted_graph, whisker_thresh=whisker_thresh, min_comp_len=min_comp_len
    )
    if base_graph.number_of_nodes() > 0:
        plot_original_graph(
            base_graph,
            save_path=str(out_dir / f"{stem}_{method}_06_refined_base_graph.png"),
        )

    n_endpoints = sum(1 for t in node_types.values() if t == "endpoint")
    n_junctions = sum(1 for t in node_types.values() if t == "junction")
    n_comps = nx.number_connected_components(base_graph) if base_graph.number_of_nodes() > 0 else 0

    return {
        "method": method,
        "skeleton_pixels": int(np.sum(skeleton)),
        "endpoints": n_endpoints,
        "junctions": n_junctions,
        "base_graph": base_graph,
        "num_components": n_comps,
    }


def run_step1_comparison_test(
    image_path: str = BENCHMARK_IMAGE_PATH,
    output_dir: str = OUTPUT_DIR,
) -> dict:
    """Executes Step 1 on `image_path` comparing Lee and Zhang-Suen thinning."""
    image_path_obj = Path(image_path)
    stem = image_path_obj.stem
    out_dir = Path(output_dir) / stem
    os.makedirs(out_dir, exist_ok=True)

    binary_mask, scale_factor, canvas_diag = load_and_scale_image(
        image_path=str(image_path_obj),
        input_stroke_width=INPUT_STROKE_WIDTH,
        target_stroke_width=TARGET_STROKE_WIDTH,
        binarise_thresh=BINARISE_THRESH,
    )
    save_binary_or_skeleton_image(
        binary_mask,
        str(out_dir / f"{stem}_01_binary_mask.png"),
        label="binary mask",
    )

    results = {}
    for method in THINNING_METHODS:
        results[method] = _run_step1_for_method(
            binary_mask=binary_mask,
            scale_factor=scale_factor,
            canvas_diag=canvas_diag,
            stem=stem,
            out_dir=out_dir,
            method=method,
        )

    print("\n" + "=" * 65)
    print("STEP 1 SUMMARY COMPARISON (LEE vs. ZHANG-SUEN)")
    print("=" * 65)
    print(f"{'Metric':<28} | {'Lee':<14} | {'Zhang-Suen':<14}")
    print("-" * 65)
    lee, zhang = results["lee"], results["zhang"]
    print(f"{'Skeleton foreground pixels':<28} | {lee['skeleton_pixels']:<14} | {zhang['skeleton_pixels']:<14}")
    print(f"{'CN Endpoints (X=1)':<28} | {lee['endpoints']:<14} | {zhang['endpoints']:<14}")
    print(f"{'CN Junctions (X>=3)':<28} | {lee['junctions']:<14} | {zhang['junctions']:<14}")
    print(f"{'Refined graph nodes':<28} | {lee['base_graph'].number_of_nodes():<14} | {zhang['base_graph'].number_of_nodes():<14}")
    print(f"{'Refined graph edges':<28} | {lee['base_graph'].number_of_edges():<14} | {zhang['base_graph'].number_of_edges():<14}")
    print(f"{'Connected components':<28} | {lee['num_components']:<14} | {zhang['num_components']:<14}")
    print("=" * 65)
    return results


if __name__ == "__main__":
    create_thinning_benchmark_image(BENCHMARK_IMAGE_PATH)
    run_step1_comparison_test(image_path=BENCHMARK_IMAGE_PATH, output_dir=OUTPUT_DIR)
