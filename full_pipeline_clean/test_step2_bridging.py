"""
test_step2_bridging.py

Step 2 Component Bridging Verification Test (Report Section 2.2).
Computes Step 1 in memory from an input sketch, saves the refined base graph
(`06_refined_base_graph.png`), executes Step 2 (sharp-corner/edge subdivision
and two-pass parity-aware Kruskal MST bridging), saves all Step 2 plots
(`07` through `10`), and stops prior to Step 3 (Chinese Postman routing).
"""

from __future__ import annotations

import os
from pathlib import Path

import networkx as nx

from double_mst import double_mst_bridging, plot_bridged_graph
from graph_formulation import build_graph_from_image
from route_mapping import plot_original_graph

# ==============================================================================
# CONFIGURATION & PARAMETERS (ALIGNED WITH REPORT SECTION 2.2)
# ==============================================================================
IMAGE_PATH              = "testing_images/shapes.png"
OUTPUT_DIR              = "step2"

# Step 1 parameters
INPUT_STROKE_WIDTH      = 10.0
TARGET_STROKE_WIDTH     = 10.0
BINARISE_THRESH         = 128
RDP_EPSILON_PROP        = 0.001
D_EUCLID_PROP           = 0.008
D_EDGE_FACTOR           = 2.5
WHISKER_PROP            = 0.010
MIN_COMP_PROP           = 0.015

# Step 2 parameters
MAX_EDGE_LEN_PROP       = 0.05
CORNER_SPLIT_ANGLE_DEG  = 65.0
P_EVEN                  = 4.0
P_ODD                   = 0.5
P_CORNER                = 1.0
DELTA_REUSE             = 0.10
LAMBDA_INT              = 25.0
GAMMA_EDGE              = 10.0
GAMMA_COMP              = 1.35
PRUNE_REDUNDANT_BRIDGES = True
MAX_CANDIDATES_PER_PAIR = 32
TOP_K_CANDIDATES        = 25

# Plot Saving Flags
SAVE_BASE_GRAPH         = True
SAVE_SUBDIVIDED_GRAPH   = True
SAVE_MST1_GRAPH         = True
SAVE_MST2_GRAPH         = True
SAVE_BRIDGED_GRAPH      = True


def run_step2_test(
    image_path: str | Path = IMAGE_PATH,
    output_dir: str | Path = OUTPUT_DIR,
) -> dict:
    """Computes Step 1 in memory and saves Step 2 bridging progression plots."""
    image_path = Path(image_path)
    stem = image_path.stem
    out_dir = Path(output_dir) / stem
    os.makedirs(out_dir, exist_ok=True)

    print(f"Running Step 2 test on: '{image_path}'")
    print(f"Saving outputs to: '{out_dir}/'")

    base_graph, _, canvas_diag, scale_factor = build_graph_from_image(
        image_path=str(image_path),
        input_stroke_width=INPUT_STROKE_WIDTH,
        target_stroke_width=TARGET_STROKE_WIDTH,
        binarise_thresh=BINARISE_THRESH,
        d_euclid_prop=D_EUCLID_PROP,
        d_edge_factor=D_EDGE_FACTOR,
        rdp_epsilon_prop=RDP_EPSILON_PROP,
        whisker_prop=WHISKER_PROP,
        min_comp_prop=MIN_COMP_PROP,
    )

    if base_graph.number_of_nodes() == 0 or base_graph.number_of_edges() == 0:
        raise ValueError("Extracted base graph is empty. Verify IMAGE_PATH and BINARISE_THRESH.")

    if SAVE_BASE_GRAPH:
        plot_original_graph(base_graph, save_path=str(out_dir / f"{stem}_06_refined_base_graph.png"))

    subdiv_path = str(out_dir / f"{stem}_07_subdivided_graph.png") if SAVE_SUBDIVIDED_GRAPH else None
    mst1_path   = str(out_dir / f"{stem}_08_mst1_bridges.png") if SAVE_MST1_GRAPH else None
    mst2_path   = str(out_dir / f"{stem}_09_mst2_unpruned_bridges.png") if SAVE_MST2_GRAPH else None

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
        subdivided_save_path=subdiv_path,
        mst1_save_path=mst1_path,
        mst2_save_path=mst2_path,
    )

    if SAVE_BRIDGED_GRAPH:
        plot_bridged_graph(
            bridged_graph,
            save_path=str(out_dir / f"{stem}_10_pruned_bridged_graph.png"),
        )

    odd_nodes = [n for n in bridged_graph.nodes() if bridged_graph.degree(n) % 2 == 1]
    print(
        f"Step 2 complete -> Bridged graph: {bridged_graph.number_of_nodes()} nodes, "
        f"{bridged_graph.number_of_edges()} edges | "
        f"Connected: {nx.is_connected(bridged_graph)} | "
        f"Remaining odd-degree nodes (for Step 3 CPP): {len(odd_nodes)}"
    )
    return {
        "base_graph": base_graph,
        "bridged_graph": bridged_graph,
        "canvas_diag": canvas_diag,
        "scale_factor": scale_factor,
    }


if __name__ == "__main__":
    run_step2_test()
