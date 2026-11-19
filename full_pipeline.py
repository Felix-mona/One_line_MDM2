"""
full_pipeline.py

End-to-End One-Line Drawing Generation and Evaluation Pipeline (Steps 1-5).
Connects:
  - Step 1 (`graph_formulation.py`): Binarisation, Lee thinning, Rutovitz CN,
    RDP decimation, and artefact refinement.
  - Step 2 (`double_mst.py`): Sharp-corner/edge subdivision and parity-aware
    Double MST component bridging.
  - Step 3 (`route_mapping.py`): Chinese Postman matching and Fleury minimum-angle
    Eulerian circuit extraction.
  - Step 4 (`fourier_reconstruction.py`): Uniform N = 10,000 complex sampling,
    DFT frequency truncation, and inverse DFT reconstruction.
  - Step 5 (`metrics.py`): Normalised Hausdorff distance and discrete smoothness.
All output PNGs share a uniform 10x10 white borderless canvas at 300 DPI.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from pipeline.double_mst import double_mst_bridging, plot_bridged_graph
import pipeline.fourier_reconstruction as fourier
from pipeline.graph_formulation import build_graph_from_image
import pipeline.metrics as met
from pipeline.route_mapping import (
    chinese_postman,
    compute_total_turn_angle,
    plot_eulerised_graph,
    plot_original_graph,
    plot_route,
)


# ==============================================================================
# 1. UNIFORM BORDERLESS PLOTTING HELPERS
# ==============================================================================
def plot_polyline_preview(
    coords: np.ndarray,
    save_path: str,
    color: str = "black",
    line_width: float = 1.8,
) -> None:
    """Saves a clean borderless plot of an unfiltered 2D or complex polyline."""
    pts = (
        np.column_stack((np.real(coords), np.imag(coords)))
        if np.iscomplexobj(coords)
        else np.asarray(coords, dtype=float)
    )
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=line_width, solid_capstyle="round")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print(f"Saved polyline plot to '{save_path}'")


def save_reconstruction_image(
    coords_xy: np.ndarray,
    save_path: str,
    target_accuracy: float,
    k_val: int,
    line_color: str = "black",
    bg_color: str = "white",
    line_width: float = 1.8,
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 300,
) -> None:
    """Saves a clean reconstructed one-line drawing PNG with no axes, title, or legend."""
    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_color)
    ax.plot(
        coords_xy[:, 0],
        coords_xy[:, 1],
        color=line_color,
        linewidth=line_width,
        solid_capstyle="round",
    )
    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1, facecolor=bg_color)
    plt.close(fig)
    print(f"Saved reconstruction (Acc={target_accuracy}, K={k_val}) to '{save_path}'")


def plot_hausdorff_comparison(
    original_z: np.ndarray,
    reconstructed_z: np.ndarray,
    hausdorff_index: tuple[int, int, str],
    save_path: str = "hausdorff_comparison.png",
) -> None:
    """Plots the reconstructed curve over the original trajectory and marks the Hausdorff pair."""
    idx_orig, idx_recon, _ = hausdorff_index
    pt_orig = (np.real(original_z[idx_orig]), np.imag(original_z[idx_orig]))
    pt_recon = (np.real(reconstructed_z[idx_recon]), np.imag(reconstructed_z[idx_recon]))

    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    ax.plot(np.real(original_z), np.imag(original_z), color="lightgray", linewidth=2.0)
    ax.plot(np.real(reconstructed_z), np.imag(reconstructed_z), color="royalblue", linewidth=1.8)
    ax.plot(pt_orig[0], pt_orig[1], "ro", markersize=7)
    ax.plot(pt_recon[0], pt_recon[1], "go", markersize=7)
    ax.plot([pt_orig[0], pt_recon[0]], [pt_orig[1], pt_recon[1]], "k--", linewidth=1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print(f"Saved Hausdorff comparison plot to '{save_path}'")


def plot_fourier_accuracy_curve(
    coeffs: np.ndarray,
    original_z: np.ndarray,
    reconstruction_results: list[dict],
    max_k: int | None = None,
    num_eval_points: int = 60,
    save_path: str = "fourier_accuracy_curve.png",
) -> None:
    """Plots Accuracy and Smoothness trajectories over frequency cutoffs K cleanly."""
    n = len(coeffs)
    max_cutoff = min(int(max_k) if max_k is not None else (n - 1) // 2, (n - 1) // 2)
    k_values = np.unique(np.linspace(1, max(1, max_cutoff), num_eval_points, dtype=int))
    accuracies, smoothnesses = [], []
    for k in k_values:
        filtered = fourier.filter_coeffs(coeffs, int(k))
        recon = fourier.inverse_fourier_descriptors(filtered)
        acc, _ = met.calc_metric(recon, original_z, mtype="Accuracy")
        smo = met.calc_metric(recon, original_z, mtype="Smoothness")
        accuracies.append(acc)
        smoothnesses.append(smo)

    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    ax.plot(k_values, accuracies, "g-", linewidth=2.0)
    ax.plot(k_values, smoothnesses, "b-", linewidth=2.0)
    found_ks = [r["k"] for r in reconstruction_results]
    found_accs = [r["achieved_accuracy"] for r in reconstruction_results]
    ax.scatter(found_ks, found_accs, color="crimson", s=55, zorder=5)
    ax.axis("off")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print(f"Saved Fourier accuracy curve to '{save_path}'")


# ==============================================================================
# 2. MASTER PIPELINE FUNCTION
# ==============================================================================
def run_oneline_pipeline(
    image_path: str | Path,
    output_path: str | Path,
    # --- Step 4 & 5: Fourier Reconstruction & Metric Thresholds ---
    accuracy_thresholds: float | list[float] = 0.01,
    n_samples: int = 10000,
    mirror_path: bool = False,
    max_frequency_cutoff: int | None = None,
    gibbs_window: int = 4,
    # --- Step 1: Pre-Processing & Graph Extraction (Section 2.1) ---
    input_stroke_width: float = 10.0,
    target_stroke_width: float = 10.0,
    binarise_thresh: int = 128,
    rdp_epsilon_prop: float = 0.001,
    d_euclid_prop: float = 0.008,
    d_edge_factor: float = 2.5,
    whisker_prop: float = 0.010,
    min_comp_prop: float = 0.015,
    # --- Step 2: Subdivision & Double MST Bridging (Section 2.2) ---
    max_edge_len_prop: float = 0.05,
    corner_angle_deg: float = 65.0,
    p_even: float = 4.0,
    p_odd: float = 0.5,
    p_corner: float = 1.0,
    delta_reuse: float = 0,
    lambda_int: float = 25.0,
    gamma_edge: float = 10.0,
    gamma_comp: float = 1.35,
    prune_redundant_bridges: bool = True,
    max_candidates_per_pair: int = 32,
    top_k_candidates: int = 25,
    # --- Final Drawing Appearance ---
    line_color: str = "black",
    bg_color: str = "white",
    line_width: float = 1.8,
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 300,
    # --- Intermediary Outputs Control ---
    intermediary_dir: str | Path | None = None,
    save_all_intermediaries: bool = False,
    save_binary_mask: bool = False,
    save_skeleton: bool = False,
    save_junction_classification: bool = False,
    save_raw_graph: bool = False,
    save_rdp_graph: bool = False,
    save_base_graph: bool = False,
    save_subdivided_graph: bool = False,
    save_mst1_graph: bool = False,
    save_mst2_graph: bool = False,
    save_bridged_graph: bool = False,
    save_eulerised_graph: bool = False,
    save_euler_route: bool = False,
    save_raw_polyline_plot: bool = False,
    save_fourier_metrics_plot: bool = False,
    save_hausdorff_overlay: bool = False,
    save_intermediate_npy: bool = False,
    save_intermediate_svg: bool = False,
) -> dict:
    """
    Executes the full image-to-one-line drawing pipeline across Steps 1-5 and
    saves uniformly styled PNG outputs.

    Args:
        image_path: Path to input raster sketch.
        output_path: Output directory or file path template.
        accuracy_thresholds: Target Hausdorff accuracy threshold(s) d_norm.
        n_samples: Uniform resampling count N along the closed tour (default 10000).
        mirror_path: If True, mirrors the polyline before resampling.
        max_frequency_cutoff: Optional upper bound on DFT cutoff K.
        gibbs_window: Local backwards search window for Gibbs ripples.
        input_stroke_width: Estimated input stroke thickness in pixels.
        target_stroke_width: Working stroke thickness prior to thinning.
        binarise_thresh: Grayscale threshold T_bin (default 128).
        rdp_epsilon_prop: RDP tolerance fraction of diagonal D (0.001).
        d_euclid_prop: Vertex merge radius fraction of D (0.008).
        d_edge_factor: Multiplier on d_euclid for junction contraction (2.5 -> 0.02 * D).
        whisker_prop: Minimum degree-1 whisker length fraction of D (0.010).
        min_comp_prop: Minimum component arc length fraction of D (0.015).
        max_edge_len_prop: Maximum unsplit edge length fraction of D (0.05).
        corner_angle_deg: Sharp-corner deflection threshold in degrees (65.0).
        p_even: Penalty for untouched even-degree vertices (4.0).
        p_odd: Penalty for odd-degree vertices (0.5).
        p_corner: Penalty for even-degree sharp-corner vertices (1.0).
        delta_reuse: Discount when reusing an even-degree bridged vertex (0).
        lambda_int: Crossing penalty weight (25.0).
        gamma_edge: Pass 2 exact bridge reuse multiplier (10.0).
        gamma_comp: Pass 2 component-pair reuse multiplier (1.35).
        prune_redundant_bridges: Whether to prune redundant >=4-bridge cycles.
        max_candidates_per_pair: Maximum candidate bridges evaluated per component pair.
        top_k_candidates: Top-K lowest-cost candidates retained per component pair.
        line_color: Stroke colour for reconstructed outputs.
        bg_color: Canvas background colour.
        line_width: Stroke width in points.
        figsize: Figure dimensions in inches.
        dpi: Output resolution in dots per inch.
        intermediary_dir: Optional directory for intermediate plots.
        save_all_intermediaries: If True, enables all intermediate save flags.

    Returns:
        Dictionary containing reconstruction results, intermediate graphs,
        route metrics, and saved file paths.
    """
    image_path = Path(image_path)
    stem = image_path.stem
    output_path = Path(output_path)

    if output_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf", ".svg"):
        out_ext = output_path.suffix
        base_out_dir = output_path.parent
        custom_stem = output_path.stem
    else:
        out_ext = ".png"
        base_out_dir = output_path if output_path.name.lower() == stem.lower() else (output_path / stem)
        custom_stem = stem

    inter_dir = Path(intermediary_dir) if intermediary_dir is not None else base_out_dir
    os.makedirs(base_out_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)

    if save_all_intermediaries:
        save_binary_mask = True
        save_skeleton = True
        save_junction_classification = True
        save_raw_graph = True
        save_rdp_graph = True
        save_base_graph = True
        save_subdivided_graph = True
        save_mst1_graph = True
        save_mst2_graph = True
        save_bridged_graph = True
        save_eulerised_graph = True
        save_euler_route = True
        save_raw_polyline_plot = True
        save_fourier_metrics_plot = True
        save_hausdorff_overlay = True
        save_intermediate_npy = True
        save_intermediate_svg = True

    saved_files: dict = {"final_images": []}

    # ==========================================================================
    # STEP 1: PRE-PROCESSING & GRAPH EXTRACTION (SECTION 2.1)
    # ==========================================================================
    print("\n" + "=" * 65)
    print("STEP 1: PRE-PROCESSING & GRAPH EXTRACTION")
    print("=" * 65)
    bin_path   = str(inter_dir / f"{stem}_01_binary_mask.png") if save_binary_mask else None
    skel_path  = str(inter_dir / f"{stem}_02_skeleton.png") if save_skeleton else None
    junc_path  = str(inter_dir / f"{stem}_03_junction_classification.png") if save_junction_classification else None
    raw_g_path = str(inter_dir / f"{stem}_04_raw_traced_graph.png") if save_raw_graph else None
    rdp_g_path = str(inter_dir / f"{stem}_05_rdp_simplified.png") if save_rdp_graph else None

    base_graph, _, canvas_diag, _ = build_graph_from_image(
        image_path=str(image_path),
        input_stroke_width=input_stroke_width,
        target_stroke_width=target_stroke_width,
        binarise_thresh=binarise_thresh,
        d_euclid_prop=d_euclid_prop,
        d_edge_factor=d_edge_factor,
        rdp_epsilon_prop=rdp_epsilon_prop,
        whisker_prop=whisker_prop,
        min_comp_prop=min_comp_prop,
        binary_save_path=bin_path,
        skeleton_save_path=skel_path,
        junction_save_path=junc_path,
        raw_graph_save_path=raw_g_path,
        rdp_save_path=rdp_g_path,
    )

    if bin_path:
        saved_files["binary_mask"] = bin_path
    if skel_path:
        saved_files["skeleton"] = skel_path
    if junc_path:
        saved_files["junction_classification"] = junc_path
    if raw_g_path:
        saved_files["raw_traced_graph"] = raw_g_path
    if rdp_g_path:
        saved_files["rdp_simplified"] = rdp_g_path

    if base_graph.number_of_nodes() == 0 or base_graph.number_of_edges() == 0:
        raise ValueError("Extracted base graph is empty. Check image_path and binarise_thresh.")

    if save_base_graph:
        base_g_path = str(inter_dir / f"{stem}_06_refined_base_graph.png")
        plot_original_graph(base_graph, save_path=base_g_path)
        saved_files["base_graph"] = base_g_path

    # ==========================================================================
    # STEP 2: SUBDIVISION & DOUBLE MST COMPONENT BRIDGING (SECTION 2.2)
    # ==========================================================================
    print("\n" + "=" * 65)
    print("STEP 2: SUBDIVISION & DOUBLE MST COMPONENT BRIDGING")
    print("=" * 65)
    subdiv_path = str(inter_dir / f"{stem}_07_subdivided_graph.png") if save_subdivided_graph else None
    mst1_path   = str(inter_dir / f"{stem}_08_mst1_bridges.png") if save_mst1_graph else None
    mst2_path   = str(inter_dir / f"{stem}_09_mst2_unpruned_bridges.png") if save_mst2_graph else None

    bridged_graph = double_mst_bridging(
        graph=base_graph,
        canvas_diag=canvas_diag,
        max_edge_len_prop=max_edge_len_prop,
        corner_angle_deg=corner_angle_deg,
        p_even=p_even,
        p_odd=p_odd,
        p_corner=p_corner,
        delta_reuse=delta_reuse,
        lambda_int=lambda_int,
        gamma_edge=gamma_edge,
        gamma_comp=gamma_comp,
        prune_redundant=prune_redundant_bridges,
        max_candidates_per_pair=max_candidates_per_pair,
        top_k_candidates=top_k_candidates,
        subdivided_save_path=subdiv_path,
        mst1_save_path=mst1_path,
        mst2_save_path=mst2_path,
    )

    if subdiv_path:
        saved_files["subdivided_graph"] = subdiv_path
    if mst1_path:
        saved_files["mst1_bridges"] = mst1_path
    if mst2_path:
        saved_files["mst2_unpruned_bridges"] = mst2_path

    if not nx.is_connected(bridged_graph):
        raise RuntimeError("Graph could not be connected via Double MST bridging.")

    if save_bridged_graph:
        bridged_g_path = str(inter_dir / f"{stem}_10_pruned_bridged_graph.png")
        plot_bridged_graph(bridged_graph, save_path=bridged_g_path)
        saved_files["bridged_graph"] = bridged_g_path

    # ==========================================================================
    # STEP 3: CHINESE POSTMAN & EULERIAN ROUTE PATHFINDING (SECTION 2.3)
    # ==========================================================================
    print("\n" + "=" * 65)
    print("STEP 3: CHINESE POSTMAN EULERIAN TOUR & POLYLINE STITCHING")
    print("=" * 65)
    cpp_results = chinese_postman(bridged_graph, canvas_diag=canvas_diag)
    euler_graph = cpp_results["euler_graph"]
    route = cpp_results["route"]
    raw_polyline = cpp_results["polyline"]
    total_turn_angle = compute_total_turn_angle(euler_graph, route)

    print(f"Eulerian circuit steps: {len(route)} | Polyline vertices: {len(raw_polyline)}")
    print(f"Cumulative turning angle: {total_turn_angle:.2f} deg")

    if save_eulerised_graph:
        euler_g_path = str(inter_dir / f"{stem}_11_eulerised_graph.png")
        plot_eulerised_graph(euler_graph, save_path=euler_g_path)
        saved_files["eulerised_graph"] = euler_g_path

    if save_euler_route:
        route_path = str(inter_dir / f"{stem}_12_euler_route.png")
        plot_route(euler_graph, route, save_path=route_path)
        saved_files["euler_route"] = route_path

    if save_raw_polyline_plot:
        raw_poly_img = str(inter_dir / f"{stem}_13_raw_polyline.png")
        plot_polyline_preview(raw_polyline, raw_poly_img)
        saved_files["raw_polyline_plot"] = raw_poly_img

    if save_intermediate_npy:
        raw_npy_path = str(inter_dir / f"{stem}_output_polyline.npy")
        np.save(raw_npy_path, raw_polyline)
        saved_files["raw_polyline_npy"] = raw_npy_path

    if save_intermediate_svg:
        raw_svg_path = str(inter_dir / f"{stem}_output_polyline.svg")
        fourier.save_polyline_svg(raw_polyline, raw_svg_path, stroke=line_color, stroke_width=line_width)
        saved_files["raw_polyline_svg"] = raw_svg_path

    # ==========================================================================
    # STEP 4 & 5: FOURIER SMOOTHING & METRIC EVALUATION (SECTIONS 2.4 - 2.5)
    # ==========================================================================
    print("\n" + "=" * 65)
    print("STEP 4 & 5: FOURIER RECONSTRUCTION (LOWEST K PER ACCURACY THRESHOLD)")
    print("=" * 65)
    original_z = fourier.resample_polyline_uniform(
        raw_polyline, n_samples=n_samples, mirror=mirror_path
    )
    coeffs, _ = fourier.forward_fourier(original_z)
    reconstructions = fourier.reconstruct_for_accuracy_thresholds(
        coeffs=coeffs,
        original_z=original_z,
        accuracy_thresholds=accuracy_thresholds,
        max_k=max_frequency_cutoff,
        gibbs_window=gibbs_window,
    )

    if save_fourier_metrics_plot:
        metrics_img = str(inter_dir / f"{stem}_14_fourier_accuracy_curve.png")
        plot_fourier_accuracy_curve(
            coeffs=coeffs,
            original_z=original_z,
            reconstruction_results=reconstructions,
            max_k=max_frequency_cutoff,
            save_path=metrics_img,
        )
        saved_files["fourier_metrics_plot"] = metrics_img

    for res in reconstructions:
        t_acc = res["target_accuracy"]
        ach_acc = res["achieved_accuracy"]
        k_val = res["k"]
        coords_xy = res["coords_xy"]
        recon_z = res["reconstructed_z"]
        h_idx = res["hausdorff_index"]

        print(
            f"-> Target Accuracy <= {t_acc:<6} | Found Lowest K = {k_val:<4} | "
            f"Achieved Accuracy = {ach_acc:.4f} | Smoothness = {res['smoothness']:.4f}"
        )

        labeled_filename = f"{custom_stem}_A{t_acc}_K{k_val}.png"
        img_out_path = str(base_out_dir / labeled_filename)
        save_reconstruction_image(
            coords_xy=coords_xy,
            save_path=img_out_path,
            target_accuracy=t_acc,
            k_val=k_val,
            line_color=line_color,
            bg_color=bg_color,
            line_width=line_width,
            figsize=figsize,
            dpi=dpi,
        )
        res["output_image_path"] = img_out_path
        saved_files["final_images"].append(img_out_path)

        if save_hausdorff_overlay:
            h_path = str(inter_dir / f"{stem}_15_hausdorff_A{t_acc}_K{k_val}.png")
            plot_hausdorff_comparison(
                original_z=original_z,
                reconstructed_z=recon_z,
                hausdorff_index=h_idx,
                save_path=h_path,
            )

        if save_intermediate_svg or out_ext.lower() == ".svg":
            svg_out = str(inter_dir / f"{stem}_A{t_acc}_K{k_val}.svg")
            fourier.save_polyline_svg(coords_xy, svg_out, stroke=line_color, stroke_width=line_width)

        if save_intermediate_npy:
            npy_out = str(inter_dir / f"{stem}_A{t_acc}_K{k_val}.npy")
            np.save(npy_out, coords_xy)

    return {
        "reconstructions": reconstructions,
        "best_k_list": [r["k"] for r in reconstructions],
        "accuracy_thresholds": [r["target_accuracy"] for r in reconstructions],
        "total_turn_angle_deg": total_turn_angle,
        "canvas_diag": canvas_diag,
        "base_graph": base_graph,
        "bridged_graph": bridged_graph,
        "euler_graph": euler_graph,
        "route": route,
        "raw_polyline": raw_polyline,
        "saved_files": saved_files,
    }


if __name__ == "__main__":
    survey_results = run_oneline_pipeline(
        image_path="set_images/Sunflower/Sunflower.png",
        output_path="sunflower_fig/",
        accuracy_thresholds=[0.005, 0.01, 0.02, 0.04, 0.07],
        n_samples=10000,
        save_all_intermediaries=True,
    )
