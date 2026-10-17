"""
fourier_draw_polyline.py

Decomposes a 2D continuous polyline into discrete Fourier series epicycles
(complex Fourier coefficients) and renders an animated visualization of rotating
circles and vectors drawing the continuous line.
"""

import argparse
import json
import os
import re
import sys
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET


def load_polyline_coordinates(file_path: str) -> np.ndarray:
    """Loads ordered 2D coordinates from a JSON polyline file, standard CSV, or SVG file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Polyline file not found: {file_path}")

    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        coords = np.array(data["coordinates"], dtype=np.float64)
    elif file_path.endswith(".csv"):
        coords = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=(1, 2), dtype=np.float64)
    elif file_path.endswith(".svg"):
        tree = ET.parse(file_path)
        root = tree.getroot()
        coords_list = []
        
        for elem in root.iter():
            tag = elem.tag.split("}")[-1]
            
            # 1. Handle polyline and polygon points
            if tag in ("polyline", "polygon"):
                points_str = elem.get("points", "")
                numbers = [float(n) for n in re.findall(r"[-+]?[0-9]*\.?[0-9]+", points_str)]
                for i in range(0, len(numbers) - 1, 2):
                    coords_list.append([numbers[i], numbers[i + 1]])
            
            # 2. Handle SVG paths (extract all coordinate pairs)
            elif tag == "path":
                d_str = elem.get("d", "")
                numbers = [float(n) for n in re.findall(r"[-+]?[0-9]*\.?[0-9]+", d_str)]
                for i in range(0, len(numbers) - 1, 2):
                    coords_list.append([numbers[i], numbers[i + 1]])
                    
            # 3. Handle simple lines
            elif tag == "line":
                try:
                    x1 = float(elem.get("x1", 0))
                    y1 = float(elem.get("y1", 0))
                    x2 = float(elem.get("x2", 0))
                    y2 = float(elem.get("y2", 0))
                    coords_list.append([x1, y1])
                    coords_list.append([x2, y2])
                except ValueError:
                    pass

        coords = np.array(coords_list, dtype=np.float64)
    else:
        raise ValueError("Unsupported file format. Provide a .json, .svg or .csv file.")

    if len(coords) < 3:
        raise ValueError(f"Polyline must contain at least 3 points for Fourier analysis (found {len(coords)}).")

    return coords


def resample_polyline_uniform(coords: np.ndarray, num_samples: int = 1024) -> np.ndarray:
    """
    Resamples the continuous polyline at equal arc-length intervals.
    Uniform spatial sampling ensures steady drawing speed and prevents
    high-frequency distortion across dense vertex clusters.
    """
    diffs = np.diff(coords, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_dist = np.insert(np.cumsum(seg_lengths), 0, 0.0)
    total_len = cum_dist[-1]

    if total_len == 0:
        raise ValueError("Polyline has 0 total length.")

    # Target equally spaced distance values along path
    target_dists = np.linspace(0.0, total_len, num_samples, endpoint=False)

    x_interp = np.interp(target_dists, cum_dist, coords[:, 0])
    y_interp = np.interp(target_dists, cum_dist, coords[:, 1])

    return np.column_stack([x_interp, y_interp])


def compute_fourier_coefficients(points: np.ndarray, num_harmonics: int = 150):
    """
    Treats 2D points as complex values z = x + 1j * y and computes the Discrete
    Fourier Transform (DFT). Returns sorted epicycles ordered by radius.
    """
    N = len(points)
    # Convert 2D coordinates into complex plane values
    z = points[:, 0] + 1j * points[:, 1]

    # Compute FFT
    coeffs = np.fft.fft(z) / N
    freqs = np.fft.fftfreq(N, d=1.0) * N  # Integer frequency indices (-N/2 to N/2-1)

    epicycles = []
    for freq, coeff in zip(freqs, coeffs):
        int_freq = int(round(freq))
        radius = float(np.abs(coeff))
        phase = float(np.angle(coeff))
        epicycles.append({
            "freq": int_freq,
            "coeff": coeff,
            "radius": radius,
            "phase": phase
        })

    # Sort epicycles by descending radius for smoother visual epicycle nesting
    epicycles.sort(key=lambda item: item["radius"], reverse=True)

    # Restrict to specified number of largest harmonics (keeping DC component)
    if num_harmonics is not None and num_harmonics < len(epicycles):
        epicycles = epicycles[:num_harmonics]

    return epicycles, N


def animate_fourier_drawing(
    epicycles: list,
    resampled_points: np.ndarray,
    output_anim_path: str = None,
    total_frames: int = 500,
    fps: int = 30,
    trail_color: str = "black",
    epicycle_color: str = "#888888",
    show_window: bool = True
):
    """Animates epicycle circles, linkage vectors, and drawing tip creating the line."""
    min_x, max_x = np.min(resampled_points[:, 0]), np.max(resampled_points[:, 0])
    min_y, max_y = np.min(resampled_points[:, 1]), np.max(resampled_points[:, 1])

    pad_x = (max_x - min_x) * 0.15 or 20.0
    pad_y = (max_y - min_y) * 0.15 or 20.0

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(max_y + pad_y, min_y - pad_y)  # Inverted Y for image/SVG coordinates
    ax.set_aspect("equal")
    plt.axis("off")

    # Animation artist primitives
    trail_line, = ax.plot([], [], color=trail_color, lw=1.6, solid_capstyle="round")
    vectors_line, = ax.plot([], [], color="#444444", lw=0.9, alpha=0.7)
    draw_point, = ax.plot([], [], marker="o", markersize=4, color="crimson")

    # Render epicycle circles dynamically via Circle patches
    max_circles_to_draw = min(60, len(epicycles))
    circle_patches = [
        plt.Circle((0, 0), 1.0, color=epicycle_color, fill=False, lw=0.6, alpha=0.35)
        for _ in range(max_circles_to_draw)
    ]
    for cp in circle_patches:
        ax.add_patch(cp)

    drawn_x, drawn_y = [], []
    t_vals = np.linspace(0, 2 * np.pi, total_frames, endpoint=False)

    def init():
        trail_line.set_data([], [])
        vectors_line.set_data([], [])
        draw_point.set_data([], [])
        for cp in circle_patches:
            cp.set_center((0, 0))
            cp.set_radius(0)
        return [trail_line, vectors_line, draw_point] + circle_patches

    def update(frame):
        t = t_vals[frame]

        # Epicycle coordinate chaining
        curr_x = 0.0
        curr_y = 0.0
        vector_xs = [curr_x]
        vector_ys = [curr_y]

        for i, epi in enumerate(epicycles):
            prev_x, prev_y = curr_x, curr_y
            freq = epi["freq"]
            angle = freq * t + epi["phase"]

            dx = epi["radius"] * np.cos(angle)
            dy = epi["radius"] * np.sin(angle)

            curr_x += dx
            curr_y += dy

            vector_xs.append(curr_x)
            vector_ys.append(curr_y)

            # Update visible circles
            if i < max_circles_to_draw:
                circle_patches[i].set_center((prev_x, prev_y))
                circle_patches[i].set_radius(epi["radius"])

        # Track pen coordinates
        drawn_x.append(curr_x)
        drawn_y.append(curr_y)

        trail_line.set_data(drawn_x, drawn_y)
        vectors_line.set_data(vector_xs, vector_ys)
        draw_point.set_data([curr_x], [curr_y])

        return [trail_line, vectors_line, draw_point] + circle_patches

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        init_func=init,
        blit=True,
        interval=1000.0 / fps
    )

    if output_anim_path:
        print(f"Exporting animation to: {output_anim_path}...")
        if output_anim_path.endswith(".mp4"):
            writer = animation.FFMpegWriter(
                fps=fps, 
                bitrate=3000, 
                codec="libx264", 
                extra_args=["-pix_fmt", "yuv420p"]
            )
            ani.save(output_anim_path, writer=writer)
        else:
            # Default to GIF with explicit fps
            writer = animation.PillowWriter(fps=fps)
            ani.save(output_anim_path, writer=writer, fps=fps)
        print(f"Animation saved: {output_anim_path}")

    if show_window:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compute Fourier epicycles from a continuous polyline and animate the reconstruction."
    )
    parser.add_argument(
        "polyline_file",
        nargs="?",
        default="output/drawing_polyline.json",
        help="Path to polyline coordinate file (.json, .svg, or .csv)"
    )
    parser.add_argument(
        "-n", "--samples",
        type=int,
        default=1200,
        help="Number of uniform spatial samples along polyline (default: 1200)"
    )
    parser.add_argument(
        "-k", "--harmonics",
        type=int,
        default=180,
        help="Number of dominant Fourier frequency circles/harmonics to retain (default: 180)"
    )
    parser.add_argument(
        "-f", "--frames",
        type=int,
        default=None,
        help="Total frames for one complete drawing loop (defaults to match sample count)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Animation frames per second (default: 30)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional output animation filename (.gif or .mp4)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive display window"
    )

    args = parser.parse_args()

    try:
        print(f"[1/4] Loading polyline from: {args.polyline_file}...")
        coords = load_polyline_coordinates(args.polyline_file)
        print(f"      Loaded {len(coords)} vertices.")

        print(f"[2/4] Resampling polyline to {args.samples} uniform spatial samples...")
        resampled = resample_polyline_uniform(coords, num_samples=args.samples)

        # Automatically ensure enough frames to cover the full length/samples if not explicitly provided
        total_frames = args.frames if args.frames is not None else max(args.samples, 1200)

        print(f"[3/4] Computing Discrete Fourier Transform (DFT)...")
        epicycles, N = compute_fourier_coefficients(resampled, num_harmonics=args.harmonics)
        print(f"      Generated {len(epicycles)} epicycle components sorted by radius.")

        print(f"[4/4] Starting animation engine ({total_frames} frames at {args.fps} FPS)...")
        animate_fourier_drawing(
            epicycles=epicycles,
            resampled_points=resampled,
            output_anim_path=args.output,
            total_frames=total_frames,
            fps=args.fps,
            show_window=not args.no_show
        )

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()