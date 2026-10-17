"""
batch_test.py

Batch testing suite for the continuous single-line and Fourier epicycle pipeline.
Processes multiple images across all non-drawing modes ('primal', 'minimal', 'xdog', 'canny')
and evaluates Fourier reconstructions across varying harmonic counts k = [25, 50, 100, 250, 500].

Output Directory Structure:
    outputs/
    └── <image_stem>/
        ├── <mode>_preview.png
        ├── <mode>_oneline.svg
        ├── <mode>_k025.png
        ├── <mode>_k050.png
        ├── <mode>_k100.png
        ├── <mode>_k250.png
        ├── <mode>_k500.png
        └── <image_stem>_comparison_grid.png
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
# Use non-interactive Agg backend for fast, headless batch rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from image_to_oneline import image_to_continuous_polyline
from animate_polyline import (
    resample_polyline_uniform,
    compute_fourier_coefficients,
    animate_fourier_drawing,
)

DEFAULT_MODES = ["primal", "minimal", "xdog", "canny"]
DEFAULT_K_VALUES = [25, 50, 100, 250, 500]
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}


def collect_image_paths(inputs: list[str]) -> list[Path]:
    """Resolves a list of file or directory paths into unique image files."""
    image_paths = []
    for item in inputs:
        p = Path(item)
        if not p.exists():
            print(f"[Warning] Input path does not exist, skipping: {p}", file=sys.stderr)
            continue
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    image_paths.append(child)
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(p)
        else:
            print(f"[Warning] Unsupported file type, skipping: {p}", file=sys.stderr)

    # Deduplicate while preserving order
    seen = set()
    unique_paths = []
    for p in image_paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(p)

    return unique_paths


def save_fourier_png(
    epicycles: list,
    resampled_points: np.ndarray,
    png_path: Path,
    total_frames: int = 1500,
    trail_color: str = "black",
) -> None:
    """
    Directly evaluates the Fourier series over [0, 2*pi] and exports the static
    reconstructed drawing to a PNG file without spinning up FuncAnimation.
    """
    min_x, max_x = np.min(resampled_points[:, 0]), np.max(resampled_points[:, 0])
    min_y, max_y = np.min(resampled_points[:, 1]), np.max(resampled_points[:, 1])

    pad_x = (max_x - min_x) * 0.10 or 20.0
    pad_y = (max_y - min_y) * 0.10 or 20.0

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(max_y + pad_y, min_y - pad_y)  # Invert Y to match image coordinates
    ax.set_aspect("equal")
    ax.axis("off")

    t_vals = np.linspace(0, 2 * np.pi, total_frames, endpoint=True)
    final_x = np.zeros_like(t_vals)
    final_y = np.zeros_like(t_vals)

    for epi in epicycles:
        freq = epi["freq"]
        angle = freq * t_vals + epi["phase"]
        final_x += epi["radius"] * np.cos(angle)
        final_y += epi["radius"] * np.sin(angle)

    ax.plot(final_x, final_y, color=trail_color, lw=1.4, solid_capstyle="round")
    fig.savefig(str(png_path), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def generate_comparison_grid(
    image_name: str,
    image_out_dir: Path,
    modes: list[str],
    k_values: list[int],
) -> None:
    """Renders a visual summary matrix (rows = modes, columns = k values) for quick inspection."""
    nrows = len(modes)
    ncols = len(k_values)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), dpi=150)

    # Normalize axes array to 2D for single-row or single-column edge cases
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(f"Fourier Reconstruction Grid: {image_name}", fontsize=16, fontweight="bold", y=0.98)

    for r, mode in enumerate(modes):
        for c, k in enumerate(k_values):
            ax = axes[r, c]
            img_file = image_out_dir / f"{mode}_k{k:03d}.png"

            if img_file.exists():
                img_data = mpimg.imread(str(img_file))
                ax.imshow(img_data)
            else:
                ax.text(0.5, 0.5, "Failed / Empty", ha="center", va="center", fontsize=10, color="red")

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#dddddd")
                spine.set_linewidth(0.8)

            if r == 0:
                ax.set_title(f"k = {k}", fontsize=12, fontweight="bold", pad=8)
            if c == 0:
                ax.set_ylabel(mode.upper(), fontsize=12, fontweight="bold", labelpad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    grid_path = image_out_dir / f"{image_name}_comparison_grid.png"
    fig.savefig(str(grid_path), bbox_inches="tight")
    plt.close(fig)
    print(f"      [+] Saved comparison grid: {grid_path}")


def run_batch(
    image_paths: list[Path],
    output_root: Path,
    modes: list[str],
    k_values: list[int],
    num_samples: int = 1500,
    save_animations: bool = False,
    anim_ext: str = ".gif",
    fps: int = 30,
    speed: float = 1.5,
) -> None:
    """Runs the full test matrix across all images, modes, and harmonic k-values."""
    output_root.mkdir(parents=True, exist_ok=True)
    total_images = len(image_paths)
    max_k = max(k_values)

    # Ensure num_samples is at least as large as max_k so we have enough frequencies
    effective_samples = max(num_samples, max_k + 100)

    print("=" * 72)
    print(f"Starting Batch Test")
    print(f"  Images ({total_images}) : {[p.name for p in image_paths]}")
    print(f"  Modes  ({len(modes)}) : {modes}")
    print(f"  k-vals ({len(k_values)}) : {k_values}")
    print(f"  Samples      : {effective_samples}")
    print(f"  Output Root  : {output_root.resolve()}")
    print("=" * 72)

    batch_start = time.time()

    for idx, img_path in enumerate(image_paths, start=1):
        img_stem = img_path.stem
        img_out_dir = output_root / img_stem
        img_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx}/{total_images}] Processing Image: {img_path.name} -> {img_out_dir}/")

        for mode in modes:
            mode_start = time.time()
            print(f"\n  --> Mode: '{mode}'")

            preview_png = img_out_dir / f"{mode}_preview.png"
            oneline_svg = img_out_dir / f"{mode}_oneline.svg"
            coords_csv = img_out_dir / f"{mode}_coords.csv"

            try:
                # 1. Convert image to continuous 1-line polyline for this mode
                polyline = image_to_continuous_polyline(
                    input_image_path=str(img_path),
                    output_svg_path=str(oneline_svg),
                    output_coords_path=str(coords_csv),
                    preview_png_path=str(preview_png),
                    mode=mode,
                )

                coords = np.array(polyline, dtype=np.float64)
                if len(coords) < 3:
                    raise ValueError(f"Insufficient vertices ({len(coords)}) generated in mode '{mode}'.")

                # 2. Resample uniformly and compute all Fourier coefficients once per mode
                resampled = resample_polyline_uniform(coords, num_samples=effective_samples)
                all_epicycles, _ = compute_fourier_coefficients(resampled, num_harmonics=max_k)

                # 3. Sweep through all requested k values
                for k in k_values:
                    k_epicycles = all_epicycles[:k]
                    png_out = img_out_dir / f"{mode}_k{k:03d}.png"

                    save_fourier_png(
                        epicycles=k_epicycles,
                        resampled_points=resampled,
                        png_path=png_out,
                        total_frames=effective_samples,
                    )
                    print(f"      [k={k:3d}] Saved Fourier reconstruction: {png_out.name}")

                    if save_animations:
                        anim_out = img_out_dir / f"{mode}_k{k:03d}{anim_ext}"
                        animate_fourier_drawing(
                            epicycles=k_epicycles,
                            resampled_points=resampled,
                            output_anim_path=str(anim_out),
                            total_frames=effective_samples,
                            fps=fps,
                            speed=speed,
                            save_png=False,
                            show_window=False,
                        )

                elapsed = time.time() - mode_start
                print(f"      Done with mode '{mode}' in {elapsed:.2f}s.")

            except Exception as e:
                print(f"  [!] Error processing '{img_path.name}' in mode '{mode}': {e}", file=sys.stderr)

        # Build a side-by-side comparison grid for this image
        generate_comparison_grid(img_stem, img_out_dir, modes, k_values)

    total_elapsed = time.time() - batch_start
    print("\n" + "=" * 72)
    print(f"Batch processing complete in {total_elapsed:.2f}s. Outputs saved to: {output_root.resolve()}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Batch test multiple images across all non-drawing modes and varying Fourier harmonic k values."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["images"],
        help="Input image file paths and/or directories containing images (default: 'images' folder).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="outputs",
        help="Root output directory where per-image subfolders will be created (default: 'outputs').",
    )
    parser.add_argument(
        "-m", "--modes",
        nargs="+",
        choices=["primal", "minimal", "xdog", "canny"],
        default=DEFAULT_MODES,
        help="Modes to evaluate (default: primal minimal xdog canny).",
    )
    parser.add_argument(
        "-k", "--k-values",
        nargs="+",
        type=int,
        default=DEFAULT_K_VALUES,
        help="List of Fourier harmonic counts (k) to test (default: 25 50 100 250 500).",
    )
    parser.add_argument(
        "-n", "--samples",
        type=int,
        default=1500,
        help="Number of uniform spatial samples along the polyline (default: 1500).",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Also export full epicycle animations (.gif or .mp4) for every (mode, k) combination.",
    )
    parser.add_argument(
        "--anim-ext",
        choices=[".gif", ".mp4"],
        default=".gif",
        help="Animation file format when --animate is enabled (default: .gif).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Animation frames per second when --animate is enabled (default: 30).",
    )
    parser.add_argument(
        "-s", "--speed",
        type=float,
        default=2.0,
        help="Animation playback speed multiplier when --animate is enabled (default: 2.0).",
    )

    args = parser.parse_args()

    image_paths = collect_image_paths(args.inputs)
    if not image_paths:
        print("Error: No valid images found in the specified input paths.", file=sys.stderr)
        sys.exit(1)

    run_batch(
        image_paths=image_paths,
        output_root=Path(args.output_dir),
        modes=args.modes,
        k_values=sorted(set(args.k_values)),
        num_samples=args.samples,
        save_animations=args.animate,
        anim_ext=args.anim_ext,
        fps=args.fps,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()