"""
Pipeline runner for graph formulation, Double MST component bridging,
and Chinese Postman Eulerian route mapping.

Adjust the configuration parameters below and run this script directly.
"""

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
IMAGE_PATH              = "Example_images/Shapes.png"          # Path to input sketch or drawing[cite: 1]
SAVE_POLYLINE           = True                   # Save the final continuous polyline to disk
OUTPUT_POLYLINE_PATH    = "output_polyline.npy"  # Destination for numpy coordinate array

# 2. Rescaling & Binarisation (Section 2.1.1)
INPUT_STROKE_WIDTH      = 8.0                    # Manual input stroke thickness in pixels
TARGET_STROKE_WIDTH     = 4.0                    # Normalised stroke thickness for stable skeletonisation
BINARISE_THRESH         = 128                    # Grayscale binarisation threshold (0-255)

# 3. Decimation & Node Merging (Sections 2.1.3 - 2.1.4)
RDP_EPSILON_PROP        = 0.003                  # RDP tolerance as a fraction of content diagonal D (0.3%)[cite: 5]
D_EUCLID_PROP           = 0.015                  # Euclidean node merge threshold as a fraction of D (1.5%)[cite: 5]
D_EDGE_FACTOR           = 3.0                    # Multiplier on d_euclid for merging along short connecting edges[cite: 5]

# 4. Double MST Component Bridging (Sections 2.1.4 & 2.2.1)
MAX_EDGE_LEN_PROP       = 0.10                   # Long-edge subdivision threshold as fraction of D (10%)[cite: 5]
EVEN_DEGREE_PENALTY     = 1.8                    # Cost penalty on even-degree nodes to favour endpoints[cite: 3, 6]
INTERSECTION_PENALTY    = 15.0                   # Multiplier penalty per stroke segment crossed by a bridge
EDGE_REUSE_PENALTY      = 100.0                  # Heavy penalty on reusing Pass 1 bridge paths in Pass 2[cite: 6]
NODE_REUSE_PENALTY      = 2.5                    # Penalty on reusing the same entry/exit node on a component[cite: 6]

# 5. Visualisation Flags
SHOW_ORIGINAL_GRAPH     = True                   # Plot base graph extracted from skeleton[cite: 11]
SHOW_BRIDGED_GRAPH      = True                   # Plot graph with Double MST bridges highlighted in cyan[cite: 6]
SHOW_EULERISED_GRAPH    = True                   # Plot graph with CPP duplicated edges in red[cite: 11]
SHOW_EULER_ROUTE        = True                   # Plot sequential single-stroke traversal order[cite: 11]


# ==========================================================
# PIPELINE EXECUTION
# ==========================================================

def run_pipeline():
    # ------------------------------------------------------
    # STEP 1: Graph Formulation (Sections 2.1.1 - 2.1.4)
    # ------------------------------------------------------
    print("=" * 65)
    print("STEP 1: SKELETONISATION & GRAPH FORMULATION")
    print("=" * 65)

    base_graph, skeleton, canvas_diag, scale_factor = build_graph_from_image(
        image_path=IMAGE_PATH,
        input_stroke_width=INPUT_STROKE_WIDTH,
        target_stroke_width=TARGET_STROKE_WIDTH,
        binarise_thresh=BINARISE_THRESH,
        d_euclid_prop=D_EUCLID_PROP,
        d_edge_factor=D_EDGE_FACTOR,
        rdp_epsilon_prop=RDP_EPSILON_PROP,
    )

    print(f"Base graph extracted: {base_graph.number_of_nodes()} nodes, {base_graph.number_of_edges()} edges")

    if base_graph.number_of_nodes() == 0 or base_graph.number_of_edges() == 0:
        raise ValueError("Extracted graph is empty. Please verify image path and binarisation threshold.")

    if SHOW_ORIGINAL_GRAPH:
        plot_original_graph(base_graph)

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
        even_degree_penalty=EVEN_DEGREE_PENALTY,
        intersection_penalty=INTERSECTION_PENALTY,
        edge_reuse_penalty=EDGE_REUSE_PENALTY,
        node_reuse_penalty=NODE_REUSE_PENALTY,
    )

    if not nx.is_connected(bridged_graph):
        raise RuntimeError("Graph could not be connected via MST bridging. Check component connectivity.")

    if SHOW_BRIDGED_GRAPH:
        plot_bridged_graph(bridged_graph)

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

    # Quality Metrics
    turning_angle = compute_total_turn_angle(euler_graph, route)
    print(f"\nTraversal complete: {len(route)} traversed edge steps")
    print(f"Stitched continuous polyline: {len(polyline)} coordinate points")
    print(f"Graph is Eulerian: {nx.is_eulerian(euler_graph)}")
    print(f"Total cumulative turning angle: {turning_angle:.2f} degrees")

    # ------------------------------------------------------
    # STEP 4: Output & Traversal Plots
    # ------------------------------------------------------
    if SAVE_POLYLINE:
        np.save(OUTPUT_POLYLINE_PATH, polyline)
        print(f"Polyline successfully saved to '{OUTPUT_POLYLINE_PATH}'")

    if SHOW_EULERISED_GRAPH:
        plot_eulerised_graph(euler_graph)

    if SHOW_EULER_ROUTE:
        plot_route(euler_graph, route)


if __name__ == "__main__":
    run_pipeline()