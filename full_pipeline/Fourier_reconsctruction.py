"""
Fourier_reconsctruction.py
Handles uniform arc-length resampling of polylines into complex coordinates,
forward/inverse Discrete Fourier Transforms, frequency filtering, SVG export,
and fast binary-search K selection for single or multiple accuracy thresholds.
"""

from collections.abc import Iterable
import numpy as np
import scipy.fft as fft
import svgwrite
import Metric as met


def resample_polyline_uniform(polyline, n_samples=2000, mirror=False):
    """
    Resamples a 2D polyline of shape (M, 2) uniformly by arc length s into
    n_samples complex coordinates z(s) = x(s) + i*y(s).
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


def save_polyline_svg(coords, svg_path, stroke="black", stroke_width=1.5, padding=20.0):
    """Exports a 2D (N, 2) or complex (N,) polyline directly to an SVG file."""
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

    dwg = svgwrite.Drawing(str(svg_path), size=(f"{width:.1f}px", f"{height:.1f}px"))
    point_tuples = [(float(x), float(y)) for x, y in shifted]
    dwg.add(
        dwg.polyline(
            points=point_tuples,
            stroke=stroke,
            fill="none",
            stroke_width=str(stroke_width),
            stroke_linecap="round",
            stroke_linejoin="round",
        )
    )
    dwg.save()
    print(f"Saved SVG polyline to '{svg_path}'")


def forward_fourier(z):
    """Computes forward normalised FFT coefficients and integer frequency channels."""
    n = len(z)
    coefficients = fft.fft(z, norm="forward")
    freq_channels = fft.fftfreq(n, d=1 / n).astype(int)
    return coefficients, freq_channels


def filter_coeffs(coefficients, cutoff=None):
    """Zeroes out frequency bands strictly above +cutoff and below -cutoff."""
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


def inverse_fourier_descriptors(coefficients):
    """Reconstructs complex spatial coordinates from filtered Fourier coefficients."""
    if coefficients is None:
        print("Error: no coefficients given")
        return None
    return fft.ifft(coefficients, norm="forward")


def find_lowest_k_for_accuracy(
    coeffs,
    original_z,
    target_accuracy,
    min_k=1,
    max_k=None,
    gibbs_window=4,
    eval_cache=None,
):
    """
    Finds the lowest frequency cutoff K achieving Hausdorff Accuracy <= target_accuracy
    using Binary Search + a local backwards Gibbs scan.

    Returns:
        (best_k, reconstructed_z, achieved_accuracy, hausdorff_index, reached_threshold)
    """
    n = len(coeffs)
    if max_k is None:
        max_k = (n - 1) // 2
    max_k = max(1, min(int(max_k), (n - 1) // 2))
    low = max(1, int(min_k))
    high = max(low, max_k)

    if eval_cache is None:
        eval_cache = {}

    def evaluate_k(k):
        k = int(k)
        if k not in eval_cache:
            filtered = filter_coeffs(coeffs, k)
            recon_z = inverse_fourier_descriptors(filtered)
            acc, idx = met.calc_metric(recon_z, original_z, mtype="Accuracy")
            eval_cache[k] = (recon_z, acc, idx)
        return eval_cache[k]

    # 1. Check lower bound
    recon_low, acc_low, idx_low = evaluate_k(low)
    if acc_low <= target_accuracy:
        return low, recon_low, acc_low, idx_low, True

    # 2. Check upper bound
    recon_high, acc_high, idx_high = evaluate_k(high)
    if acc_high > target_accuracy:
        print(
            f"Warning: target_accuracy={target_accuracy} not reached by max_k={high} "
            f"(achieved accuracy={acc_high:.4f}). Using K={high}."
        )
        return high, recon_high, acc_high, idx_high, False

    # 3. Binary search in [low, high]
    best_k = high
    while low <= high:
        mid = (low + high) // 2
        _, acc_mid, _ = evaluate_k(mid)
        if acc_mid <= target_accuracy:
            best_k = mid
            high = mid - 1
        else:
            low = mid + 1

    # 4. Local backwards check in case of minor Gibbs ringing ripples
    check_start = max(int(min_k), best_k - gibbs_window)
    for k_cand in range(check_start, best_k):
        _, acc_cand, _ = evaluate_k(k_cand)
        if acc_cand <= target_accuracy:
            best_k = k_cand
            break

    recon_best, acc_best, idx_best = evaluate_k(best_k)
    return best_k, recon_best, acc_best, idx_best, True


def reconstruct_for_accuracy_thresholds(
    coeffs,
    original_z,
    accuracy_thresholds=0.034,
    max_k=None,
    gibbs_window=4,
):
    """
    Accepts either a single accuracy threshold (float) or a list/iterable of
    accuracy thresholds, finds the lowest K for each using warm-started binary
    search, and returns a list of reconstruction dictionaries in the exact order
    requested.
    """
    if isinstance(accuracy_thresholds, (int, float, np.number)):
        thresh_list = [float(accuracy_thresholds)]
    elif isinstance(accuracy_thresholds, Iterable):
        thresh_list = [float(t) for t in accuracy_thresholds]
    else:
        raise TypeError("accuracy_thresholds must be a float or an iterable of floats.")

    if len(thresh_list) == 0:
        raise ValueError("accuracy_thresholds cannot be empty.")

    # Shared evaluation cache across all thresholds for this image
    eval_cache = {}

    # Sort descending (loosest accuracy threshold first -> smallest K first)
    sorted_unique = sorted(set(thresh_list), reverse=True)
    computed_by_thresh = {}
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
        # Warm-start the next (stricter) threshold from this K
        current_min_k = best_k

    return [computed_by_thresh[t] for t in thresh_list]