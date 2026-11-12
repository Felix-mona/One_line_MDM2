"""
Metric.py
Fast O(N log N) cKDTree Hausdorff Accuracy and scale-invariant Turning-Angle
Smoothness metrics for high-resolution (N >= 10,000) Fourier reconstructions.
"""

import numpy as np
from scipy.spatial import cKDTree
import Fourier_reconsctruction as fourier


def Loss(accuracy, smoothness, alpha=0.5, beta=0.5):
    """Computes weighted linear combination of accuracy and smoothness in [0, 1]."""
    if not np.isclose(alpha + beta, 1.0, atol=1e-5):
        raise ValueError(f"alpha ({alpha}) + beta ({beta}) must equal 1.0")
    return float(alpha * accuracy + beta * smoothness)


def gather_lowest_metric(coefficients, max_frequency_cutoff, samples, alpha_beta_pairs, original_points=None):
    """Evaluates (k, alpha, beta) combinations across `samples` frequency cutoffs."""
    if original_points is None:
        original_points = fourier.inverse_fourier_descriptors(coefficients)

    frequencies = np.unique(np.linspace(1, max_frequency_cutoff, int(samples), dtype=int))
    combination_per_freq = []

    for k in frequencies:
        k = int(k)
        min_loss = float("inf")
        minimum_combination = []
        kfiltered = fourier.filter_coeffs(coefficients, k)
        temp_ifd = fourier.inverse_fourier_descriptors(kfiltered)

        temp_accuracy, _ = calc_metric(temp_ifd, original_points, "accuracy")
        temp_smoothness = calc_metric(temp_ifd, original_points, "smoothness")

        for alpha, beta in alpha_beta_pairs:
            temp_loss = Loss(temp_accuracy, temp_smoothness, alpha=alpha, beta=beta)
            if temp_loss < min_loss:
                min_loss = temp_loss
                minimum_combination = [k, alpha, beta]

        combination_per_freq.append(minimum_combination)

    return combination_per_freq


def calculate_distances(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """Calculates Euclidean distances between row-wise pairs of (x, y) coordinates."""
    return np.linalg.norm(coords1 - coords2, axis=1)


def calc_metric(new_image, old_image, mtype=None, tree_old=None):
    """
    Computes 'Accuracy' (exact two-way Hausdorff distance via O(N log N) cKDTree)
    or 'Smoothness' (stride-normalised mean exterior turning angle).
    """
    if mtype in ("accuracy", "Accuracy"):
        A = np.column_stack((np.real(old_image), np.imag(old_image)))
        B = np.column_stack((np.real(new_image), np.imag(new_image)))

        if tree_old is None:
            tree_old = cKDTree(A)
        tree_new = cKDTree(B)

        # Directed Hausdorff A -> B (for each point in A, distance to nearest in B)
        dists_a_to_b, nearest_b_for_a = tree_new.query(A, k=1)
        i1 = int(np.argmax(dists_a_to_b))
        d1 = float(dists_a_to_b[i1])
        j1 = int(nearest_b_for_a[i1])

        # Directed Hausdorff B -> A (for each point in B, distance to nearest in A)
        dists_b_to_a, nearest_a_for_b = tree_old.query(B, k=1)
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

        both_real_ptp = max(np.ptp(A[:, 0]), np.ptp(B[:, 0]))
        both_imag_ptp = max(np.ptp(A[:, 1]), np.ptp(B[:, 1]))
        scale = float(np.hypot(both_real_ptp, both_imag_ptp))

        accuracy = float(d / scale) if scale > 0 else 0.0
        index = (point1, point2, direction)
        return accuracy, index

    elif mtype in ("smoothness", "Smoothness"):
        A_coords = np.column_stack((np.real(new_image), np.imag(new_image)))
        # Normalise step stride to ~1000 effective segments so Smoothness is invariant to N_SAMPLES
        stride = max(1, len(A_coords) // 1000)

        B_coords = np.roll(A_coords, stride, axis=0)
        C_coords = np.roll(A_coords, -stride, axis=0)

        c = calculate_distances(A_coords, B_coords)
        b = calculate_distances(A_coords, C_coords)
        a = calculate_distances(B_coords, C_coords)

        denom = np.maximum(2.0 * b * c, 1e-12)
        cos_angle = np.clip((b**2 + c**2 - a**2) / denom, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return float(1.0 - np.mean(angle / np.pi))

    else:
        raise ValueError(f"Unsupported mtype '{mtype}'. Expected 'Accuracy' or 'Smoothness'.")