"""
fourier_reconstruction.py

Step 4: Post-Processing using the Discrete Fourier Transform (Report Section 2.4).
Resamples the closed Eulerian polyline uniformly by arc length into N = 10,000
complex coordinates z_n = x_n + i y_n, computes forward DFT descriptors c_k,
truncates high-frequency bands via low-pass symmetric retention, reconstructs
smooth closed trajectories via inverse DFT, exports vector SVGs, and performs
warm-started binary search for target Hausdorff accuracy thresholds.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import scipy.fft as fft
from scipy.spatial import cKDTree

import metrics as met


def resample_polyline_uniform(
    polyline: np.ndarray,
    n_samples: int = 10000,
    mirror: bool = False,
) -> np.ndarray:
    """
    Resamples a 2D polyline uniformly by arc length into `n_samples` complex
    coordinates z_n = x_n + i * y_n (Section 2.4.1).

    Args:
        polyline: Coordinate array of shape (M, 2) or complex array of shape (M,).
        n_samples: Number of uniformly spaced sample points N (default 10000).
        mirror: If True, concatenates the reversed path to enforce open-end closure.

    Returns:
        Complex 1D NumPy array of shape `(n_samples,)`.
    """
    pts = np.asarray(polyline, dtype=float)
    if np.iscomplexobj(pts):
        pts = np.column_stack((np.real(pts), np.imag(pts)))
    if mirror:
        pts = np.vstack((pts, pts[::-1]))
    if len(pts) < 2:
        raise ValueError("Polyline must contain at least 2 points for resampling.")

    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    keep = np.concatenate(([True], seg_lens > 1e-12))
    pts = pts[keep]

    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total_len = cum_len[-1]

    if total_len <= 1e-12:
        return np.full(n_samples, pts[0, 0] + 1j * pts[0, 1], dtype=complex)

    sample_s = np.linspace(0.0, total_len, int(n_samples), endpoint=False)
    x_interp = np.interp(sample_s, cum_len, pts[:, 0])
    y_interp = np.interp(sample_s, cum_len, pts[:, 1])
    return x_interp + 1j * y_interp


def save_polyline_svg(
    coords: np.ndarray,
    svg_path: str,
    stroke: str = "black",
    stroke_width: float = 1.5,
    padding: float = 20.0,
) -> None:
    """Exports a 2D (N, 2) or complex (N,) trajectory to a standalone vector SVG file."""
    pts = (
        np.column_stack((np.real(coords), np.imag(coords)))
        if np.iscomplexobj(coords)
        else np.asarray(coords, dtype=float)
    )
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    shifted = pts - min_xy + padding
    shifted[:, 1] = (span[1] + 2 * padding) - shifted[:, 1]
    width = float(span[0] + 2 * padding)
    height = float(span[1] + 2 * padding)

    points_attr = " ".join(f"{float(x):.3f},{float(y):.3f}" for x, y in shifted)
    svg_xml = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.1f}px" height="{height:.1f}px" '
        f'viewBox="0 0 {width:.1f} {height:.1f}">\n'
        f'  <polyline points="{points_attr}" '
        f'stroke="{stroke}" fill="none" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round" />\n'
        f"</svg>\n"
    )
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_xml)
    print(f"Saved SVG polyline to '{svg_path}'")


def forward_fourier(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes forward normalised Discrete Fourier Transform descriptors c_k
    and corresponding integer frequency channels (Report Equation 5).

    Args:
        z: Complex coordinate array of shape (N,).

    Returns:
        Tuple of (coefficients, freq_channels).
    """
    n = len(z)
    coefficients = fft.fft(z, norm="forward")
    freq_channels = fft.fftfreq(n, d=1 / n).astype(int)
    return coefficients, freq_channels


def filter_coeffs(coefficients: np.ndarray, cutoff: int | None = None) -> np.ndarray | None:
    """
    Applies symmetric low-pass frequency truncation (Report Section 2.4.2, Eq. 7):
    retains the `cutoff` lowest positive anti-clockwise frequencies and `cutoff`
    lowest negative clockwise frequencies from both ends of `coefficients`,
    zeroing out all high-frequency descriptors in `[cutoff + 1 : N - cutoff]`.

    Args:
        coefficients: Complex FFT descriptor array of shape (N,).
        cutoff: Frequency half-bandwidth cutoff k (retains 2k + 1 low modes).

    Returns:
        Filtered complex descriptor array of shape (N,).
    """
    coefficients = np.asarray(coefficients)
    if cutoff is None:
        return coefficients
    if cutoff == -1:
        return None
    n = len(coefficients)
    cutoff = min(int(cutoff), (n - 1) // 2)
    fourier_filtered = coefficients.copy()
    fourier_filtered[cutoff + 1 : n - cutoff] = 0
    return fourier_filtered


def inverse_fourier_descriptors(coefficients: np.ndarray | None) -> np.ndarray | None:
    """
    Reconstructs smooth complex spatial coordinates z_hat_n from truncated
    Fourier descriptors via the inverse DFT (Report Equations 6 & 7).
    """
    if coefficients is None:
        print("Error: no coefficients given")
        return None
    return fft.ifft(coefficients, norm="forward")


def find_lowest_k_for_accuracy(
    coeffs: np.ndarray,
    original_z: np.ndarray,
    target_accuracy: float,
    min_k: int = 1,
    max_k: int | None = None,
    gibbs_window: int = 4,
    eval_cache: dict | None = None,
) -> tuple[int, np.ndarray, float, tuple[int, int, str], bool]:
    """
    Finds the lowest frequency cutoff K achieving normalised Hausdorff accuracy
    d_norm <= `target_accuracy` using binary search with a pre-built cKDTree
    and a local backwards Gibbs-ripple check.

    Args:
        coeffs: Forward FFT descriptors of `original_z`.
        original_z: Original complex trajectory of shape (N,).
        target_accuracy: Upper bound threshold for d_norm in [0, 1].
        min_k: Lower search bound on cutoff K.
        max_k: Upper search bound on cutoff K.
        gibbs_window: Number of lower K values checked locally for Gibbs ripples.
        eval_cache: Optional shared cache across multiple thresholds.

    Returns:
        Tuple of (best_k, reconstructed_z, achieved_accuracy, hausdorff_index, reached).
    """
    n = len(coeffs)
    if max_k is None:
        max_k = (n - 1) // 2
    max_k = max(1, min(int(max_k), (n - 1) // 2))
    low = max(1, int(min_k))
    high = max(low, max_k)
    if eval_cache is None:
        eval_cache = {}

    if "_tree_old" not in eval_cache:
        orig_xy = np.column_stack((np.real(original_z), np.imag(original_z)))
        eval_cache["_tree_old"] = cKDTree(orig_xy)
    tree_old = eval_cache["_tree_old"]

    def evaluate_k(k: int) -> tuple[np.ndarray, float, tuple[int, int, str]]:
        k = int(k)
        if k not in eval_cache:
            filtered = filter_coeffs(coeffs, k)
            recon_z = inverse_fourier_descriptors(filtered)
            acc, idx = met.calc_metric(recon_z, original_z, mtype="Accuracy", tree_old=tree_old)
            eval_cache[k] = (recon_z, acc, idx)
        return eval_cache[k]

    recon_low, acc_low, idx_low = evaluate_k(low)
    if acc_low <= target_accuracy:
        return low, recon_low, acc_low, idx_low, True

    recon_high, acc_high, idx_high = evaluate_k(high)
    if acc_high > target_accuracy:
        print(
            f"Warning: target_accuracy={target_accuracy} not reached by max_k={high} "
            f"(achieved accuracy={acc_high:.4f}). Using K={high}."
        )
        return high, recon_high, acc_high, idx_high, False

    best_k = high
    while low <= high:
        mid = (low + high) // 2
        _, acc_mid, _ = evaluate_k(mid)
        if acc_mid <= target_accuracy:
            best_k = mid
            high = mid - 1
        else:
            low = mid + 1

    check_start = max(int(min_k), best_k - gibbs_window)
    for k_cand in range(check_start, best_k):
        _, acc_cand, _ = evaluate_k(k_cand)
        if acc_cand <= target_accuracy:
            best_k = k_cand
            break

    recon_best, acc_best, idx_best = evaluate_k(best_k)
    return best_k, recon_best, acc_best, idx_best, True


def reconstruct_for_accuracy_thresholds(
    coeffs: np.ndarray,
    original_z: np.ndarray,
    accuracy_thresholds: float | Iterable[float] = 0.034,
    max_k: int | None = None,
    gibbs_window: int = 4,
) -> list[dict]:
    """
    Finds the lowest frequency cutoff K for a single accuracy threshold or an
    iterable of thresholds using warm-started binary search, returning results
    in the exact requested order.

    Args:
        coeffs: Forward FFT descriptors of `original_z`.
        original_z: Original complex trajectory of shape (N,).
        accuracy_thresholds: Target threshold(s) for normalised Hausdorff distance.
        max_k: Optional maximum cutoff K.
        gibbs_window: Local backwards Gibbs check window.

    Returns:
        List of reconstruction result dictionaries.
    """
    if isinstance(accuracy_thresholds, (int, float, np.number)):
        thresh_list = [float(accuracy_thresholds)]
    elif isinstance(accuracy_thresholds, Iterable):
        thresh_list = [float(t) for t in accuracy_thresholds]
    else:
        raise TypeError("accuracy_thresholds must be a float or an iterable of floats.")

    if len(thresh_list) == 0:
        raise ValueError("accuracy_thresholds cannot be empty.")

    eval_cache: dict = {}
    sorted_unique = sorted(set(thresh_list), reverse=True)
    computed_by_thresh: dict[float, dict] = {}
    current_min_k = 1

    for thresh in sorted_unique:
        best_k, recon_z, achieved_acc, h_idx, reached = find_lowest_k_for_accuracy(
            coeffs=coeffs,
            original_z=original_z,
            target_accuracy=thresh,
            min_k=current_min_k,
            max_k=max_k,
            gibbs_window=gibbs_window,
            eval_cache=eval_cache,
        )
        smoothness = met.calc_metric(recon_z, original_z, mtype="Smoothness")
        coords_xy = np.column_stack((np.real(recon_z), np.imag(recon_z)))
        computed_by_thresh[thresh] = {
            "target_accuracy": thresh,
            "achieved_accuracy": achieved_acc,
            "k": best_k,
            "reached_threshold": reached,
            "reconstructed_z": recon_z,
            "coords_xy": coords_xy,
            "smoothness": smoothness,
            "hausdorff_index": h_idx,
        }
        current_min_k = best_k

    return [computed_by_thresh[t] for t in thresh_list]
