"""
metrics.py

Step 5: Quantitative Image Evaluation Metrics (Report Section 2.5).
Implements:
  1. Normalised two-way Hausdorff distance d_norm = d / diag(A U B) in [0, 1]
     via O(N log N) cKDTree queries (Section 2.5.1).
  2. Discrete interior turning-angle smoothness penalty S = 1 - theta_bar
     in [0, 1] using the cosine rule across consecutive sampled points
     z_{i-1}, z_i, z_{i+1} (Section 2.5.2).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

import pipeline.fourier_reconstruction as fourier


def compute_weighted_loss(
    accuracy: float,
    smoothness: float,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """Computes a convex linear combination alpha * accuracy + beta * smoothness in [0, 1]."""
    if not np.isclose(alpha + beta, 1.0, atol=1e-5):
        raise ValueError(f"alpha ({alpha}) + beta ({beta}) must equal 1.0")
    return float(alpha * accuracy + beta * smoothness)


def gather_lowest_metric(
    coefficients: np.ndarray,
    max_frequency_cutoff: int,
    samples: int,
    alpha_beta_pairs: list[tuple[float, float]],
    original_points: np.ndarray | None = None,
) -> list[list]:
    """Evaluates (k, alpha, beta) combinations across `samples` frequency cutoffs."""
    if original_points is None:
        original_points = fourier.inverse_fourier_descriptors(coefficients)
    frequencies = np.unique(np.linspace(1, max_frequency_cutoff, int(samples), dtype=int))
    combination_per_freq = []
    for k in frequencies:
        k = int(k)
        min_loss = float("inf")
        minimum_combination: list = []
        kfiltered = fourier.filter_coeffs(coefficients, k)
        temp_ifd = fourier.inverse_fourier_descriptors(kfiltered)
        temp_accuracy, _ = calc_metric(temp_ifd, original_points, mtype="Accuracy")
        temp_smoothness = calc_metric(temp_ifd, original_points, mtype="Smoothness")
        for alpha, beta in alpha_beta_pairs:
            temp_loss = compute_weighted_loss(temp_accuracy, temp_smoothness, alpha=alpha, beta=beta)
            if temp_loss < min_loss:
                min_loss = temp_loss
                minimum_combination = [k, alpha, beta]
        combination_per_freq.append(minimum_combination)
    return combination_per_freq


def calculate_distances(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """Computes row-wise Euclidean distances between two (N, 2) coordinate arrays."""
    return np.linalg.norm(coords1 - coords2, axis=1)


def calc_metric(
    new_image: np.ndarray,
    old_image: np.ndarray,
    mtype: str | None = None,
    tree_old: cKDTree | None = None,
    stride: int = 1,
) -> float | tuple[float, tuple[int, int, str]]:
    """
    Computes normalised geometric fidelity ('Accuracy') or discrete path roughness
    ('Smoothness') in [0, 1], where 0 is optimal and 1 is worst (Report Section 2.5):

    - 'Accuracy' (Section 2.5.1):
        d = max(h(A, B), h(B, A))
        d_norm = d / diag(A U B), where diag(A U B) = sqrt((x_max - x_min)^2 + (y_max - y_min)^2)
        taken over all points in A U B.

    - 'Smoothness' (Section 2.5.2):
        c_i = |z_i - z_{i-1}|, b_i = |z_{i+1} - z_i|, a_i = |z_{i+1} - z_{i-1}|
        theta_i = arccos((b_i^2 + c_i^2 - a_i^2) / (2 * b_i * c_i))
        S = 1 - (1 / N) * sum(theta_i / pi)

    Args:
        new_image: Complex 1D array B of reconstructed sample points.
        old_image: Complex 1D array A of original sample points.
        mtype: Either 'Accuracy' or 'Smoothness' (case-insensitive).
        tree_old: Optional pre-built cKDTree on `old_image` for fast repeated queries.
        stride: Neighbour step offset for smoothness (default 1, matching z_{i-1}, z_i, z_{i+1}).

    Returns:
        For 'Accuracy': tuple `(d_norm, (idx_orig, idx_recon, direction))`.
        For 'Smoothness': float penalty `S` in [0, 1].
    """
    if mtype in ("accuracy", "Accuracy"):
        pts_a = np.column_stack((np.real(old_image), np.imag(old_image)))
        pts_b = np.column_stack((np.real(new_image), np.imag(new_image)))
        if tree_old is None:
            tree_old = cKDTree(pts_a)
        tree_new = cKDTree(pts_b)

        # Directed Hausdorff h(A, B)
        dists_a_to_b, nearest_b_for_a = tree_new.query(pts_a, k=1)
        i1 = int(np.argmax(dists_a_to_b))
        d1 = float(dists_a_to_b[i1])
        j1 = int(nearest_b_for_a[i1])

        # Directed Hausdorff h(B, A)
        dists_b_to_a, nearest_a_for_b = tree_old.query(pts_b, k=1)
        i2 = int(np.argmax(dists_b_to_a))
        d2 = float(dists_b_to_a[i2])
        j2 = int(nearest_a_for_b[i2])

        if d1 >= d2:
            d = d1
            point1, point2 = i1, j1
            direction = "original_to_reconstructed"
        else:
            d = d2
            point1, point2 = j2, i2
            direction = "reconstructed_to_original"

        # Normalise by bounding-box diagonal diag(A U B)
        union_pts = np.vstack((pts_a, pts_b))
        span_x = float(np.ptp(union_pts[:, 0]))
        span_y = float(np.ptp(union_pts[:, 1]))
        diag_union = float(np.hypot(span_x, span_y))
        accuracy = float(d / diag_union) if diag_union > 0 else 0.0
        index = (point1, point2, direction)
        return accuracy, index

    if mtype in ("smoothness", "Smoothness"):
        pts_z = np.column_stack((np.real(new_image), np.imag(new_image)))
        step = max(1, int(stride))
        pts_prev = np.roll(pts_z, step, axis=0)
        pts_next = np.roll(pts_z, -step, axis=0)

        c = calculate_distances(pts_z, pts_prev)
        b = calculate_distances(pts_z, pts_next)
        a = calculate_distances(pts_prev, pts_next)

        denom = np.maximum(2.0 * b * c, 1e-12)
        cos_angle = np.clip((b ** 2 + c ** 2 - a ** 2) / denom, -1.0, 1.0)
        theta = np.arccos(cos_angle)
        theta_bar = float(np.mean(theta / np.pi))
        return float(1.0 - theta_bar)

    raise ValueError(f"Unsupported mtype '{mtype}'. Expected 'Accuracy' or 'Smoothness'.")
