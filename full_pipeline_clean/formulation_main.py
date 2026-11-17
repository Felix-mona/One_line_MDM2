"""
formulation_main.py

Standalone Runner for Steps 1-3 (Graph Extraction, Double MST Component
Bridging, and Chinese Postman Eulerian Route Mapping).
"""

from __future__ import annotations

import os
from pathlib import Path

import networkx as nx
import numpy as np

from double_mst import double_mst_bridging, plot_bridged_graph
from graph_formulation import build_graph_from_image
from route_mapping import (
    chinese_postman,
    compute_total_turn_angle,
    plot_eulerised_graph,
    plot_original_graph,
    plot_route,
)

# ==============================================================================
# CONFIGURATION & PARAMETERS (ALIGNED WITH REPORT SECTION 2)
# ==============================================================================
IMAGE_PATH              = "images/Sunflower/Sunflower.png"
OUTPUT_DIR              = "images/Sunflower"
SAVE_POLYLINE           = True

# Step 1: Binarisation, Decimation & Topological Refinement (Section 2.1)
INPUT_STROKE_WIDTH      = 10.0
TARGET_STROKE_WIDTH     = 10.0
BINARISE_THRESH         = 128
RDP_EPSILON_PROP        = 0.001   # epsilon_RDP = 0.001 * D
D_EUCLID_PROP           = 0.008   # d_euclid = 0.008 * D
D_EDGE_FACTOR           = 2.5     # d_edge = 2.5 * d_euclid = 0.020 * D
WHISKER_PROP            = 0.010   # 0.01 * D
MIN_COMP_PROP           = 0.015   # 0.015 * D

# Step 2: Attachment-Vertex Subdivision & Double MST Bridging (Section 2.2)
MAX_EDGE_LEN_PROP       = 0.05    # L_max = 0.05 * D
CORNER_SPLIT_ANGLE_DEG  = 65.0    # theta_i >= 65 deg
P_EVEN                  = 4.0     # p_even = 4.0
P_ODD                   = 0.5     # p_odd = 0.5
P_CORNER                = 1.0     # p_corner = 1.0
DELTA_REUSE             = 0.10    # delta_reuse = 0.10
LAMBDA_INT              = 25.0    # lambda_int = 25.0
GAMMA_EDGE              = 10.0    # gamma_edge = 10.0
GAMMA_COMP              = 1.35    # gamma_comp = 1.35
PRUNE_REDUNDANT_BRIDGES = True
MAX_CANDIDATES_PER_PAIR = 32
TOP_K_CANDIDATES        = 25

# Plot Saving Flags
SAVE_SKELETON_IMAGE     = True
SAVE_ORIGINAL_GRAPH     = True
SAVE_BRIDGED_GRAPH      = True
SAVE_EULERISED_GRAPH    = True
SAVE_EULER_ROUTE        = True


def run_pipeline() -> None:
    """Executes Steps 1-3 and saves borderless diagnostic plots and polyline."""
    stem = Path(IMAGE_PATH).stem
    out_dir = Path(OUTPUT_DIR) / stem
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving outputs to: '{out_dir}/'")

    # Step 1: Pre-Processing & Graph Extraction
    print("\n" + "=" * 65)
    print("STEP 1: SKELETONISATION & GRAPH FORMULATION")
    print("=" * 65)
    skel_path = str(out_dir / f"{stem}_01_skeleton.png") if SAVE_SKELETON_IMAGE else None
    base_graph, _, canvas_diag, _ = build_graph_from_image(
        image_path=IMAGE_PATH,
        input_stroke_width=INPUT_STROKE_WIDTH,
        target_stroke_width=TARGET_STROKE_WIDTH,
        binarise_thresh=BINARISE_THRESH,
        d_euclid_prop=D_EUCLID_PROP,
        d_edge_factor=D_EDGE_FACTOR,
        rdp_epsilon_prop=RDP_EPSILON_PROP,
        whisker_prop=WHISKER_PROP,
        min_comp_prop=MIN_COMP_PROP,
        skeleton_save_path=skel_path,
    )
    print(f"Base graph extracted: {base_graph.number_of_nodes()} nodes, {base_graph.number_of_edges()} edges")
    if base_graph.number_of_nodes() == 0 or base_graph.number_of_edges() == 0:
        raise ValueError("Extracted graph is empty. Verify IMAGE_PATH and BINARISE_THRESH.")

    if SAVE_ORIGINAL_GRAPH:
        plot_original_graph(base_graph, save_path=str(out_dir / f"{stem}_02_original_graph.png"))

    # Step 2: Double MST Component Bridging
    print("\n" + "=" * 65)
    print("STEP 2: DOUBLE MST COMPONENT BRIDGING & PARITY PRUNING")
    print("=" * 65)
    bridged_graph = double_mst_bridging(
        graph=base_graph,
        canvas_diag=canvas_diag,
        max_edge_len_prop=MAX_EDGE_LEN_PROP,
        corner_angle_deg=CORNER_SPLIT_ANGLE_DEG,
        p_even=P_EVEN,
        p_odd=P_ODD,
        p_corner=P_CORNER,
        delta_reuse=DELTA_REUSE,
        lambda_int=LAMBDA_INT,
        gamma_edge=GAMMA_EDGE,
        gamma_comp=GAMMA_COMP,
        prune_redundant=PRUNE_REDUNDANT_BRIDGES,
        max_candidates_per_pair=MAX_CANDIDATES_PER_PAIR,
        top_k_candidates=TOP_K_CANDIDATES,
    )
    if not nx.is_connected(bridged_graph):
        raise RuntimeError("Graph could not be connected via MST bridging.")

    if SAVE_BRIDGED_GRAPH:
        plot_bridged_graph(bridged_graph, save_path=str(out_dir / f"{stem}_03_bridged_graph.png"))

    # Step 3: Chinese Postman & Eulerian Circuit
    print("\n" + "=" * 65)
    print("STEP 3: CHINESE POSTMAN ROUTING & POLYLINES")
    print("=" * 65)
    cpp_results = chinese_postman(bridged_graph, canvas_diag=canvas_diag)
    euler_graph = cpp_results["euler_graph"]
    route = cpp_results["route"]
    polyline = cpp_results["polyline"]
    turning_angle = compute_total_turn_angle(euler_graph, route)

    print(f"\nTraversal complete: {len(route)} traversed edge steps")
    print(f"Stitched continuous polyline: {len(polyline)} coordinate points")
    print(f"Graph is Eulerian: {nx.is_eulerian(euler_graph)}")
    print(f"Total cumulative turning angle: {turning_angle:.2f} degrees")

    if SAVE_POLYLINE:
        poly_path = str(out_dir / f"{stem}_output_polyline.npy")
        np.save(poly_path, polyline)
        print(f"Polyline saved to '{poly_path}'")

    if SAVE_EULERISED_GRAPH:
        plot_eulerised_graph(euler_graph, save_path=str(out_dir / f"{stem}_04_eulerised_graph.png"))

    if SAVE_EULER_ROUTE:
        plot_route(euler_graph, route, save_path=str(out_dir / f"{stem}_05_euler_route.png"))


if __name__ == "__main__":
    run_pipeline()
