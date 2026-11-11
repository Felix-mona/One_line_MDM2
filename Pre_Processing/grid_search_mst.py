"""
Parameter Grid Search for the Complete Graph Formulation & Double MST Pipeline.
Caches base graph formulation runs, sweeps across all formulation and MST bridging
parameters, and generates a CSV leaderboard and side-by-side comparison sheets.
"""

import os
from pathlib import Path
import itertools
import csv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Pipeline imports
from graph_formulation import build_graph_from_image
from double_mst import double_mst_bridging, plot_bridged_graph
from route_mapping import chinese_postman, compute_total_turn_angle, plot_route


# ==========================================================
# 1. FIXED FILE PATHS
# ==========================================================
IMAGE_PATH      = "images/indonesia/Indonesia.png"
BASE_OUTPUT_DIR = "outputs/mst_parameter_sweep"


# ==========================================================
# 2. FULL PARAMETER SEARCH GRID (Add/remove values to sweep)
# ==========================================================
PARAM_GRID = {
    # --- A. Preprocessing & Binarisation ---
    "binarise_thresh": [128],                   # Threshold for active pixels (0-255)
    "input_stroke_width": [10],                 # Estimated line stroke width in px
    "target_stroke_width": [10],                # Normalised stroke width before skeletonisation

    # --- B. Decimation & Junction Merging ---
    "rdp_epsilon_prop": [0.001],                # RDP simplification tolerance (% of diagonal)
    "d_euclid_prop": [0.008],                   # Spatial node merge radius (% of diagonal)
    "d_edge_factor": [2.5],                     # Multiplier on d_euclid for contracting short junction edges

    # --- C. Corner & Edge Splitting ---
    "max_edge_len_prop": [0.10],                # Long-edge subdivision threshold (% of diagonal)
    "corner_split_angle_deg": [65.0],           # Deflection angle threshold to insert a corner node

    # --- D. Additive Node & Reuse Penalties ---
    "even_degree_penalty": [1.5, 2.0, 5.0],     # Cost penalty for flat even subdivision nodes
    "odd_junction_penalty": [0.15, 0.3, 0.5],   # Preferred weight for odd nodes & reused nodes turned odd
    "corner_penalty": [0.20, 0.4, 1.0],         # Preferred weight for sharp corner vertices
    "reused_even_bonus": [0.10, 0.30],          # Bonus discount for reusing an even node over an untouched even node

    # --- E. Bridge Crossing Penalties ---
    "intersection_penalty": [50.0],             # Multiplier penalty per crossed stroke or prior bridge

    # --- F. Pass 2 Duplicate Penalties ---
    "edge_reuse_penalty": [50.0],               # Heavy multiplier penalty preventing duplicate edges in Pass 2
    "comp_reuse_penalty": [1.35],               # Penalty for bridging the same component pair again in Pass 2

    # --- G. Candidate Search Space Limits ---
    "max_candidates_per_node": [2],             # Prevents a single closest node from hogging all candidate slots
    "max_candidates_per_pair": [10],            # Candidate edges kept per component pair

    # --- H. Redundant Bridge Pruning ---
    "prune_redundant": [True],                  # Remove excess cycles/pairs from components with >= 4 bridges
}

# Concise abbreviations for folder naming & labels
PARAM_TAGS = {
    "binarise_thresh": "bin",
    "input_stroke_width": "in_w",
    "target_stroke_width": "tgt_w",
    "rdp_epsilon_prop": "rdp",
    "d_euclid_prop": "d_euc",
    "d_edge_factor": "d_edge",
    "max_edge_len_prop": "len",
    "corner_split_angle_deg": "angle",
    "even_degree_penalty": "even",
    "odd_junction_penalty": "odd",
    "corner_penalty": "cor",
    "reused_even_bonus": "bonus",
    "intersection_penalty": "ix",
    "edge_reuse_penalty": "ereuse",
    "comp_reuse_penalty": "creuse",
    "max_candidates_per_node": "c_node",
    "max_candidates_per_pair": "c_pair",
    "prune_redundant": "prune",
}


# ==========================================================
# 3. HELPER FUNCTIONS
# ==========================================================

def compute_bridge_metrics(G):
    """Calculates total bridge count and cumulative bridge Euclidean distance."""
    bridges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("bridge", False)]
    total_len = sum(d.get("weight", 0.0) for _, _, d in bridges)
    return len(bridges), total_len


def generate_comparison_sheet(image_paths, titles, save_path, cols=4):
    """Stitches individual result plots into a contact sheet."""
    n = len(image_paths)
    if n == 0:
        return
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5.5))
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        if i < n:
            img = plt.imread(image_paths[i])
            axes[i].imshow(img)
            axes[i].set_title(titles[i], fontsize=8.5, pad=4)
            axes[i].axis("off")
        else:
            axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nSaved comparison sheet to: '{save_path}'")


# ==========================================================
# 4. EXECUTION RUNNER
# ==========================================================

def run_grid_search():
    stem = Path(IMAGE_PATH).stem
    sweep_dir = Path(BASE_OUTPUT_DIR) / stem
    os.makedirs(sweep_dir, exist_ok=True)

    keys = list(PARAM_GRID.keys())
    combinations = list(itertools.product(*(PARAM_GRID[k] for k in keys)))

    # Identify which parameters are actively varying in this sweep
    varying_keys = [k for k in keys if len(PARAM_GRID[k]) > 1]

    print("=" * 70)
    print(f"MST GRID SEARCH: {len(combinations)} total configurations queued")
    print(f"Actively varying parameters: {varying_keys if varying_keys else 'None (single run)'}")
    print("=" * 70)

    # Base graph cache to prevent recalculating skeletonisation when only MST parameters change
    base_graph_cache = {}

    results_table = []
    grid_img_paths = []
    grid_titles = []
    csv_path = sweep_dir / "sweep_results.csv"

    for idx, combo in enumerate(combinations, 1):
        p = dict(zip(keys, combo))

        # Check if base graph for these formulation params is already in memory
        formulation_key = (
            p["binarise_thresh"],
            p["input_stroke_width"],
            p["target_stroke_width"],
            p["rdp_epsilon_prop"],
            p["d_euclid_prop"],
            p["d_edge_factor"],
        )

        if formulation_key not in base_graph_cache:
            print(f"\n[Formulation] Building base graph for key: {formulation_key}")
            skel_dir = sweep_dir / "skeletons"
            os.makedirs(skel_dir, exist_ok=True)
            skel_path = str(skel_dir / f"skel_t{p['binarise_thresh']}_w{p['target_stroke_width']}.png")

            bg, _, canvas_diag, _ = build_graph_from_image(
                image_path=IMAGE_PATH,
                input_stroke_width=p["input_stroke_width"],
                target_stroke_width=p["target_stroke_width"],
                binarise_thresh=p["binarise_thresh"],
                d_euclid_prop=p["d_euclid_prop"],
                d_edge_factor=p["d_edge_factor"],
                rdp_epsilon_prop=p["rdp_epsilon_prop"],
                skeleton_save_path=skel_path,
            )

            if bg.number_of_nodes() == 0:
                print(f"Skipping: empty graph produced by formulation settings.")
                continue

            base_graph_cache[formulation_key] = (bg, canvas_diag)
        else:
            bg, canvas_diag = base_graph_cache[formulation_key]

        # Dynamically build folder name using all varying parameters
        if varying_keys:
            var_parts = [f"{PARAM_TAGS.get(k, k)}{p[k]}" for k in varying_keys]
            run_name = f"run_{idx:03d}_" + "_".join(var_parts)
        else:
            run_name = f"run_{idx:03d}_default"

        run_dir = sweep_dir / run_name
        os.makedirs(run_dir, exist_ok=True)

        print(f"[{idx}/{len(combinations)}] Testing {run_name}")

        try:
            # Stage 2: Bridging
            bridged_G = double_mst_bridging(
                G=bg,
                canvas_diag=canvas_diag,
                max_edge_len_prop=p["max_edge_len_prop"],
                corner_angle_deg=p["corner_split_angle_deg"],
                even_degree_penalty=p["even_degree_penalty"],
                odd_junction_penalty=p["odd_junction_penalty"],
                corner_penalty=p["corner_penalty"],
                intersection_penalty=p["intersection_penalty"],
                edge_reuse_penalty=p["edge_reuse_penalty"],
                comp_reuse_penalty=p["comp_reuse_penalty"],
                prune_redundant=p["prune_redundant"],
                max_candidates_per_node=p["max_candidates_per_node"],
                max_candidates_per_pair=p["max_candidates_per_pair"],
                reused_even_bonus=p["reused_even_bonus"],
            )

            if not nx.is_connected(bridged_G):
                print(f"  -> Run {idx} did not connect the graph.")
                continue

            num_bridges, total_bridge_len = compute_bridge_metrics(bridged_G)

            # Stage 3: Chinese Postman & Eulerian Circuit
            cpp_results = chinese_postman(bridged_G)
            euler_g     = cpp_results["euler_graph"]
            route       = cpp_results["route"]
            turn_deg    = compute_total_turn_angle(euler_g, route)

            # Save diagnostic plots
            bridge_plot_path = str(run_dir / "03_bridged_graph.png")
            route_plot_path  = str(run_dir / "05_euler_route.png")
            plot_bridged_graph(bridged_G, save_path=bridge_plot_path)
            plot_route(euler_g, route, save_path=route_plot_path)

            # Collect metrics
            row = {
                "run": run_name,
                "num_bridges": num_bridges,
                "total_bridge_len": round(total_bridge_len, 2),
                "total_turning_deg": round(turn_deg, 2),
                "traversed_steps": len(route),
                **p,
            }
            results_table.append(row)

            # Register for contact sheet
            grid_img_paths.append(bridge_plot_path)
            label_parts = [f"{PARAM_TAGS.get(k, k)}={p[k]}" for k in varying_keys]
            grid_titles.append(
                f"#{idx}: {', '.join(label_parts)}\n"
                f"Bridges: {num_bridges} | Len: {total_bridge_len:.1f}px"
            )

        except Exception as e:
            print(f"  -> Failed in run {idx}: {e}")

    # Write summary CSV
    if results_table:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results_table[0].keys()))
            writer.writeheader()
            writer.writerows(results_table)
        print(f"\nSaved CSV leaderboard to: '{csv_path}'")

    # Generate master comparison sheet
    if grid_img_paths:
        master_img_path = str(sweep_dir / "00_mst_comparison_grid.png")
        generate_comparison_sheet(
            image_paths=grid_img_paths,
            titles=grid_titles,
            save_path=master_img_path,
            cols=4,
        )


if __name__ == "__main__":
    run_grid_search()