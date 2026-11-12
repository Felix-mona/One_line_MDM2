"""
full_pipeline.py
End-to-end pipeline from input raster image to Fourier-reconstructed continuous
one-line drawing(s). Accepts a single accuracy threshold or a list of accuracy
thresholds, finds the lowest frequency cutoff K for each, and outputs PNG images
with both the accuracy threshold and chosen K in the title and filename.
"""

import os
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graph_formulation import build_graph_from_image
from double_mst import double_mst_bridging, plot_bridged_graph
from route_mapping import (
    chinese_postman,
    compute_total_turn_angle,
    plot_original_graph,
    plot_eulerised_graph,
    plot_route,
)
import Fourier_reconsctruction as fourier
import Metric as met


# ==========================================================
# PLOTTING HELPERS
# ==========================================================

def plot_polyline_preview(coords, save_path, title="Continuous Polyline", color="black", lw=1.5):
    """Saves a diagnostic plot of the unfiltered 2D polyline."""
    pts = (
        np.column_stack((np.real(coords), np.imag(coords)))
        if np.iscomplexobj(coords)
        else np.asarray(coords, dtype=float)
    )
    plt.figure(figsize=(9, 8))
    plt.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw)
    plt.title(title)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved polyline plot to '{save_path}'")


def save_labeled_reconstruction_image(
    coords_xy,
    save_path,
    stem,
    target_accuracy,
    achieved_accuracy,
    k_val,
    show_labels_on_image=True,
    line_color="black",
    bg_color="white",
    line_width=1.8,
    figsize=(10, 10),
    dpi=300,
):
    """
    Saves a reconstructed one-line drawing PNG with the Accuracy Threshold
    and chosen K cutoff in the plot title and legend.
    """
    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_color)

    ax.plot(
        coords_xy[:, 0],
        coords_xy[:, 1],
        color=line_color,
        linewidth=line_width,
        solid_capstyle="round",
        label=f"Chosen K = {k_val} (Achieved Acc = {achieved_accuracy:.4f})",
    )
    ax.set_aspect("equal")
    ax.axis("off")

    if show_labels_on_image:
        ax.set_title(
            f"{stem} | Accuracy Threshold: {target_accuracy} | Chosen K = {k_val}",
            fontsize=18,
            fontweight="bold",
            pad=14,
        )
        ax.legend(loc="upper right", fontsize=12, frameon=True)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1, facecolor=bg_color)
    plt.close(fig)
    print(f"Saved labeled reconstruction (Acc={target_accuracy}, K={k_val}) to '{save_path}'")


def plot_hausdorff_comparison(
    original_z,
    reconstructed_z,
    target_accuracy,
    achieved_accuracy,
    hausdorff_index,
    selected_k,
    save_path="hausdorff_comparison.png",
):
    """Plots the reconstructed curve over the original path and marks the Hausdorff distance."""
    idx_orig, idx_recon, _ = hausdorff_index
    pt_orig = (np.real(original_z[idx_orig]), np.imag(original_z[idx_orig]))
    pt_recon = (np.real(reconstructed_z[idx_recon]), np.imag(reconstructed_z[idx_recon]))

    plt.figure(figsize=(9, 8))
    plt.plot(
        np.real(original_z),
        np.imag(original_z),
        color="lightgray",
        linewidth=2.0,
        label="Original Euler Polyline",
    )
    plt.plot(
        np.real(reconstructed_z),
        np.imag(reconstructed_z),
        color="royalblue",
        linewidth=1.6,
        label=f"Reconstructed (Target Acc={target_accuracy}, K={selected_k})",
    )
    plt.plot(pt_orig[0], pt_orig[1], "ro", markersize=7, label="Furthest Point (Original)")
    plt.plot(pt_recon[0], pt_recon[1], "go", markersize=7, label="Furthest Point (Reconstructed)")
    plt.plot(
        [pt_orig[0], pt_recon[0]],
        [pt_orig[1], pt_recon[1]],
        "k--",
        linewidth=1.8,
        label=f"Hausdorff Distance = {achieved_accuracy:.4f}",
    )

    plt.title(f"Hausdorff Overlay | Accuracy Threshold: {target_accuracy} | Chosen K = {selected_k}", fontsize=15)
    plt.axis("equal")
    plt.axis("off")
    plt.legend(loc="upper right")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Hausdorff comparison plot to '{save_path}'")


def plot_fourier_accuracy_curve(
    coeffs,
    original_z,
    reconstruction_results,
    max_k=None,
    num_eval_points=60,
    save_path="fourier_accuracy_curve.png",
):
    """Plots Accuracy and Smoothness over frequency cutoffs K, highlighting the found K values."""
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

    plt.figure(figsize=(9, 6))
    plt.plot(k_values, accuracies, "g-", linewidth=2, label="Accuracy (Hausdorff Error)")
    plt.plot(k_values, smoothnesses, "b-", linewidth=2, label="Smoothness (Angle Roughness)")

    found_ks = [r["k"] for r in reconstruction_results]
    found_accs = [r["achieved_accuracy"] for r in reconstruction_results]
    plt.scatter(found_ks, found_accs, color="crimson", s=55, zorder=5, label="Chosen K per Accuracy Threshold")

    for r in reconstruction_results:
        plt.annotate(
            f"Acc≤{r['target_accuracy']}\n(K={r['k']})",
            (r["k"], r["achieved_accuracy"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    plt.title("Accuracy & Smoothness vs. Frequency Cutoff (K)")
    plt.xlabel("Frequency Cutoff K")
    plt.ylabel("Metric Value [0, 1]")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Fourier accuracy curve to '{save_path}'")


# ==========================================================
# MASTER PIPELINE FUNCTION
# ==========================================================

def run_oneline_pipeline(
    image_path,
    output_path,
    # --- Stage 5: Single Accuracy Threshold or List of Thresholds ---
    accuracy_thresholds=0.01,
    n_samples=1000,
    mirror_path=False,
    max_frequency_cutoff=None,
    gibbs_window=4,
    # --- Stage 1: Rescaling & Binarisation (Section 2.1.1) ---
    input_stroke_width=10.0,
    target_stroke_width=10.0,
    binarise_thresh=128,
    # --- Stage 2: Graph Extraction, RDP & Merging (Sections 2.1.2 - 2.1.4) ---
    rdp_epsilon_prop=0.001,
    d_euclid_prop=0.008,
    d_edge_factor=2.5,
    whisker_prop=0.010,
    min_comp_prop=0.010,
    # --- Stage 3: Double MST Component Bridging (Sections 2.1.4 & 2.2.1) ---
    max_edge_len_prop=0.05,
    corner_angle_deg=65.0,
    even_degree_penalty=5.0,
    odd_junction_penalty=0.5,
    corner_penalty=1.0,
    intersection_penalty=25.0,
    edge_reuse_penalty=10.0,
    comp_reuse_penalty=1.35,
    prune_redundant_bridges=True,
    max_candidates_per_pair=16,
    top_k_candidates=10,
    # --- Final Drawing Appearance & Labeling ---
    show_labels_on_image=True,       # Renders Accuracy Threshold and chosen K in the PNG title
    line_color="black",
    bg_color="white",
    line_width=1.8,
    figsize=(10, 10),
    dpi=300,
    # --- Intermediary Outputs Control ---
    intermediary_dir=None,
    save_all_intermediaries=False,
    save_binary_mask=False,
    save_skeleton=False,
    save_raw_graph=False,
    save_base_graph=False,
    save_subdivided_graph=False,
    save_bridged_graph=False,
    save_eulerised_graph=False,
    save_euler_route=False,
    save_raw_polyline_plot=False,
    save_fourier_metrics_plot=False,
    save_hausdorff_overlay=False,
    save_intermediate_npy=False,
    save_intermediate_svg=False,
):
    """
    Executes the full image-to-one-line drawing pipeline and finds the lowest K
    for a single accuracy threshold or a list of accuracy thresholds.
    """
    image_path = Path(image_path)
    stem = image_path.stem
    output_path = Path(output_path)

    # Determine output folder and file extension without duplicating folder names
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
        save_raw_graph = True
        save_base_graph = True
        save_subdivided_graph = True
        save_bridged_graph = True
        save_eulerised_graph = True
        save_euler_route = True
        save_raw_polyline_plot = True
        save_fourier_metrics_plot = True
        save_hausdorff_overlay = True
        save_intermediate_npy = True
        save_intermediate_svg = True

    saved_files = {"final_images": []}

    # ======================================================
    # STAGE 1 & 2: SKELETONISATION & GRAPH FORMULATION
    # ======================================================
    print("\n" + "=" * 65)
    print("STAGE 1 & 2: SKELETONISATION & GRAPH FORMULATION")
    print("=" * 65)

    bin_path = str(inter_dir / f"{stem}_01_binary_mask.png") if save_binary_mask else None
    skel_path = str(inter_dir / f"{stem}_02_skeleton.png") if save_skeleton else None
    raw_g_path = str(inter_dir / f"{stem}_03_raw_traced_graph.png") if save_raw_graph else None

    base_graph, skeleton, canvas_diag, scale_factor = build_graph_from_image(
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
        raw_graph_save_path=raw_g_path,
    )

    if bin_path:
        saved_files["binary_mask"] = bin_path
    if skel_path:
        saved_files["skeleton"] = skel_path
    if raw_g_path:
        saved_files["raw_traced_graph"] = raw_g_path

    if base_graph.number_of_nodes() == 0 or base_graph.number_of_edges() == 0:
        raise ValueError("Extracted base graph is empty. Check image_path and binarise_thresh.")

    if save_base_graph:
        base_g_path = str(inter_dir / f"{stem}_04_base_graph.png")
        plot_original_graph(base_graph, save_path=base_g_path)
        saved_files["base_graph"] = base_g_path

    # ======================================================
    # STAGE 3: DOUBLE MST COMPONENT BRIDGING
    # ======================================================
    print("\n" + "=" * 65)
    print("STAGE 3: DOUBLE MST COMPONENT BRIDGING & PARITY PRUNING")
    print("=" * 65)

    subdiv_path = str(inter_dir / f"{stem}_05_subdivided_graph.png") if save_subdivided_graph else None

    bridged_graph = double_mst_bridging(
        G=base_graph,
        canvas_diag=canvas_diag,
        max_edge_len_prop=max_edge_len_prop,
        corner_angle_deg=corner_angle_deg,
        even_degree_penalty=even_degree_penalty,
        odd_junction_penalty=odd_junction_penalty,
        corner_penalty=corner_penalty,
        intersection_penalty=intersection_penalty,
        edge_reuse_penalty=edge_reuse_penalty,
        comp_reuse_penalty=comp_reuse_penalty,
        prune_redundant=prune_redundant_bridges,
        max_candidates_per_pair=max_candidates_per_pair,
        top_k_candidates=top_k_candidates,
        subdivided_save_path=subdiv_path,
    )

    if subdiv_path:
        saved_files["subdivided_graph"] = subdiv_path

    if not nx.is_connected(bridged_graph):
        raise RuntimeError("Graph could not be connected via Double MST bridging.")

    if save_bridged_graph:
        bridged_g_path = str(inter_dir / f"{stem}_06_bridged_graph.png")
        plot_bridged_graph(bridged_graph, save_path=bridged_g_path)
        saved_files["bridged_graph"] = bridged_g_path

    # ======================================================
    # STAGE 4: CHINESE POSTMAN EULERIAN ROUTING
    # ======================================================
    print("\n" + "=" * 65)
    print("STAGE 4: CHINESE POSTMAN EULERIAN TOUR & POLYLINE STITCHING")
    print("=" * 65)

    cpp_results = chinese_postman(bridged_graph)
    euler_graph = cpp_results["euler_graph"]
    route = cpp_results["route"]
    raw_polyline = cpp_results["polyline"]
    total_turn_angle = compute_total_turn_angle(euler_graph, route)

    print(f"Eulerian circuit steps: {len(route)} | Polyline vertices: {len(raw_polyline)}")
    print(f"Cumulative turning angle: {total_turn_angle:.2f} deg")

    if save_eulerised_graph:
        euler_g_path = str(inter_dir / f"{stem}_07_eulerised_graph.png")
        plot_eulerised_graph(euler_graph, save_path=euler_g_path)
        saved_files["eulerised_graph"] = euler_g_path

    if save_euler_route:
        route_path = str(inter_dir / f"{stem}_08_euler_route.png")
        plot_route(euler_graph, route, save_path=route_path)
        saved_files["euler_route"] = route_path

    if save_raw_polyline_plot:
        raw_poly_img = str(inter_dir / f"{stem}_09_raw_polyline.png")
        plot_polyline_preview(raw_polyline, raw_poly_img, title=f"{stem} - Unfiltered Euler Polyline")
        saved_files["raw_polyline_plot"] = raw_poly_img

    if save_intermediate_npy:
        raw_npy_path = str(inter_dir / f"{stem}_output_polyline.npy")
        np.save(raw_npy_path, raw_polyline)
        saved_files["raw_polyline_npy"] = raw_npy_path

    if save_intermediate_svg:
        raw_svg_path = str(inter_dir / f"{stem}_output_polyline.svg")
        fourier.save_polyline_svg(raw_polyline, raw_svg_path, stroke=line_color, stroke_width=line_width)
        saved_files["raw_polyline_svg"] = raw_svg_path

    # ======================================================
    # STAGE 5: FOURIER RECONSTRUCTION PER ACCURACY THRESHOLD
    # ======================================================
    print("\n" + "=" * 65)
    print("STAGE 5: FOURIER RECONSTRUCTION (LOWEST K PER ACCURACY THRESHOLD)")
    print("=" * 65)

    original_z = fourier.resample_polyline_uniform(
        raw_polyline, n_samples=n_samples, mirror=mirror_path
    )
    coeffs, freqs = fourier.forward_fourier(original_z)

    reconstructions = fourier.reconstruct_for_accuracy_thresholds(
        coeffs=coeffs,
        original_z=original_z,
        accuracy_thresholds=accuracy_thresholds,
        max_k=max_frequency_cutoff,
        gibbs_window=gibbs_window,
    )

    if save_fourier_metrics_plot:
        metrics_img = str(inter_dir / f"{stem}_10_fourier_accuracy_curve.png")
        plot_fourier_accuracy_curve(
            coeffs=coeffs,
            original_z=original_z,
            reconstruction_results=reconstructions,
            max_k=max_frequency_cutoff,
            save_path=metrics_img,
        )
        saved_files["fourier_metrics_plot"] = metrics_img

    # Save each reconstructed PNG with Accuracy Threshold and K in the title and filename
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

        save_labeled_reconstruction_image(
            coords_xy=coords_xy,
            save_path=img_out_path,
            stem=stem,
            target_accuracy=t_acc,
            achieved_accuracy=ach_acc,
            k_val=k_val,
            show_labels_on_image=show_labels_on_image,
            line_color=line_color,
            bg_color=bg_color,
            line_width=line_width,
            figsize=figsize,
            dpi=dpi,
        )

        res["output_image_path"] = img_out_path
        saved_files["final_images"].append(img_out_path)

        if save_hausdorff_overlay:
            h_path = str(inter_dir / f"{stem}_11_hausdorff_A{t_acc}_K{k_val}.png")
            plot_hausdorff_comparison(
                original_z=original_z,
                reconstructed_z=recon_z,
                target_accuracy=t_acc,
                achieved_accuracy=ach_acc,
                hausdorff_index=h_idx,
                selected_k=k_val,
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


# ==========================================================
# EXAMPLE USAGE
# ==========================================================

if __name__ == "__main__":
    survey_results = run_oneline_pipeline(
        image_path="images/Sunflower/Sunflower.png",
        output_path="outputs3/Sunflower",
        accuracy_thresholds=[0.01, 0.034, 0.058, 0.082, 0.106],
        n_samples=10000,
        show_labels_on_image=True,
        save_all_intermediaries=True,
    )