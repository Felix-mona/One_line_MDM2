"""
Pipeline runner for graph formulation, Double MST component bridging,
and Chinese Postman Eulerian route mapping.

Adjust the configuration parameters below and run this script directly.
"""

import os
from pathlib import Path
import numpy as np
import networkx as nx

# ----------------------------------------------------------
# MODULE IMPORTS
# ----------------------------------------------------------
from graph_formulation import build_graph_from_image
from double_mst import double_mst_bridging, plot_bridged_graph
from route_mapping import (
    chinese_postman,
    compute_total_turn_angle,
    plot_original_graph,
    plot_eulerised_graph,
    plot_route,
)


# ==========================================================
# CONFIGURATION & PARAMETERS
# ==========================================================

# 1. File Input / Output
IMAGE_PATH              = "bridge.png"  # Path to input sketch or drawing
OUTPUT_DIR              = "outputs"                    # Folder where all plots & polylines are saved
SAVE_POLYLINE           = True                         # Save final continuous polyline (.npy)

# 2. Rescaling & Binarisation (Section 2.1.1)
INPUT_STROKE_WIDTH      = 10                          # Estimated input stroke thickness in pixels
TARGET_STROKE_WIDTH     = 10                          # Normalised stroke thickness for skeletonisation
BINARISE_THRESH         = 128                          # Grayscale binarisation threshold (0-255)

# 3. Decimation & Node Merging (Sections 2.1.3 - 2.1.4)
RDP_EPSILON_PROP        = 0.000001                        # RDP tolerance as a fraction of diagonal D (0.2%)
D_EUCLID_PROP           = 0.008                        # Spatial node merge radius as fraction of D (0.8%)
D_EDGE_FACTOR           = 2.5                          # Multiplier on d_euclid for short junction edge merging

# 4. Double MST Component Bridging (Sections 2.1.4 & 2.2.1)
MAX_EDGE_LEN_PROP       = 0.10                         # Long-edge subdivision threshold as fraction of D (10%)
CORNER_SPLIT_ANGLE_DEG  = 65.0                         # Split sharp corners above this turn angle (degrees)
EVEN_DEGREE_PENALTY     = 1.8                          # Cost penalty on even-degree subdivision nodes
INTERSECTION_PENALTY    = 15.0                         # Multiplier penalty per crossed stroke segment
EDGE_REUSE_PENALTY      = 100.0                        # Heavy penalty on reusing Pass 1 bridges in Pass 2
NODE_REUSE_PENALTY      = 2.5                          # Penalty on reusing the same entry/exit node

# 5. Plot Saving Flags
SAVE_SKELETON_IMAGE     = True                         # Save 01_skeleton.png
SAVE_ORIGINAL_GRAPH     = True                         # Save 02_original_graph.png
SAVE_BRIDGED_GRAPH      = True                         # Save 03_bridged_graph.png
SAVE_EULERISED_GRAPH    = True                         # Save 04_eulerised_graph.png
SAVE_EULER_ROUTE        = True                         # Save 05_euler_route.png


# ==========================================================
# PIPELINE EXECUTION
# ==========================================================

def run_pipeline():
    stem = Path(IMAGE_PATH).stem
    out_dir = Path(OUTPUT_DIR) / stem
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving outputs to: '{out_dir}/'")

    # ------------------------------------------------------
    # STEP 1: Graph Formulation (Sections 2.1.1 - 2.1.4)
    # ------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 1: SKELETONISATION & GRAPH FORMULATION")
    print("=" * 65)

    skel_path = str(out_dir / f"{stem}_01_skeleton.png") if SAVE_SKELETON_IMAGE else None

    base_graph, skeleton, canvas_diag, scale_factor = build_graph_from_image(
        image_path=IMAGE_PATH,
        input_stroke_width=INPUT_STROKE_WIDTH,
        target_stroke_width=TARGET_STROKE_WIDTH,
        binarise_thresh=BINARISE_THRESH,
        d_euclid_prop=D_EUCLID_PROP,
        d_edge_factor=D_EDGE_FACTOR,
        rdp_epsilon_prop=RDP_EPSILON_PROP,
        skeleton_save_path=skel_path,
    )

    print(f"Base graph extracted: {base_graph.number_of_nodes()} nodes, {base_graph.number_of_edges()} edges")

    if base_graph.number_of_nodes() == 0 or base_graph.number_of_edges() == 0:
        raise ValueError("Extracted graph is empty. Verify IMAGE_PATH and BINARISE_THRESH.")

    if SAVE_ORIGINAL_GRAPH:
        plot_original_graph(base_graph, save_path=str(out_dir / f"{stem}_02_original_graph.png"))

    # ------------------------------------------------------
    # STEP 2: Double MST Bridging & Parity (Section 2.2.1)
    # ------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 2: DOUBLE MST COMPONENT BRIDGING & PARITY PRUNING")
    print("=" * 65)

    bridged_graph = double_mst_bridging(
        G=base_graph,
        canvas_diag=canvas_diag,
        max_edge_len_prop=MAX_EDGE_LEN_PROP,
        corner_angle_deg=CORNER_SPLIT_ANGLE_DEG,
        even_degree_penalty=EVEN_DEGREE_PENALTY,
        intersection_penalty=INTERSECTION_PENALTY,
        edge_reuse_penalty=EDGE_REUSE_PENALTY,
        node_reuse_penalty=NODE_REUSE_PENALTY,
    )

    if not nx.is_connected(bridged_graph):
        raise RuntimeError("Graph could not be connected via MST bridging.")

    if SAVE_BRIDGED_GRAPH:
        plot_bridged_graph(bridged_graph, save_path=str(out_dir / f"{stem}_03_bridged_graph.png"))

    # ------------------------------------------------------
    # STEP 3: Chinese Postman & Eulerian Circuit (Sections 2.2.2 - 2.2.3)
    # ------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 3: CHINESE POSTMAN ROUTING & POLYLINES")
    print("=" * 65)

    cpp_results = chinese_postman(bridged_graph)
    euler_graph = cpp_results["euler_graph"]
    route       = cpp_results["route"]
    polyline    = cpp_results["polyline"]

    turning_angle = compute_total_turn_angle(euler_graph, route)
    print(f"\nTraversal complete: {len(route)} traversed edge steps")
    print(f"Stitched continuous polyline: {len(polyline)} coordinate points")
    print(f"Graph is Eulerian: {nx.is_eulerian(euler_graph)}")
    print(f"Total cumulative turning angle: {turning_angle:.2f} degrees")

    # ------------------------------------------------------
    # STEP 4: Save Outputs & Traversal Plots
    # ------------------------------------------------------
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