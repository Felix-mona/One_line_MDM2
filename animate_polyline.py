"""
animate_polyline.py

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
                numbers = [float(n) for n in re.findall(r"[-+]?[0-9]*\.?[0-9]+", numbers_str if 'numbers_str' in locals() else d_str)]
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
    z = points[:, 0] + 1j * points[:, 1]

    coeffs = np.fft.fft(z) / N
    freqs = np.fft.fftfreq(N, d=1.0) * N

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

    epicycles.sort(key=lambda item: item["radius"], reverse=True)

    if num_harmonics is not None and num_harmonics < len(epicycles):
        epicycles = epicycles[:num_harmonics]

    return epicycles, N


def animate_fourier_drawing(
    epicycles: list,
    resampled_points: np.ndarray,
    output_anim_path: str = None,
    total_frames: int = 500,
    fps: int = 30,
    speed: float = 1.0,
    save_png: bool = False,
    png_path: str = None,
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

    trail_line, = ax.plot([], [], color=trail_color, lw=1.6, solid_capstyle="round")
    vectors_line, = ax.plot([], [], color="#444444", lw=0.9, alpha=0.7)
    draw_point, = ax.plot([], [], marker="o", markersize=4, color="crimson")

    max_circles_to_draw = min(60, len(epicycles))
    circle_patches = [
        plt.Circle((0, 0), 1.0, color=epicycle_color, fill=False, lw=0.6, alpha=0.35)
        for _ in range(max_circles_to_draw)
    ]
    for cp in circle_patches:
        ax.add_patch(cp)

    drawn_x, drawn_y = [], []
    
    # Adjust total frames based on the speed multiplier so the animation loops faster
    actual_frames = max(int(total_frames / speed), 1)
    t_vals = np.linspace(0, 2 * np.pi, actual_frames, endpoint=False)

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

            if i < max_circles_to_draw:
                circle_patches[i].set_center((prev_x, prev_y))
                circle_patches[i].set_radius(epi["radius"])

        drawn_x.append(curr_x)
        drawn_y.append(curr_y)

        trail_line.set_data(drawn_x, drawn_y)
        vectors_line.set_data(vector_xs, vector_ys)
        draw_point.set_data([curr_x], [curr_y])

        return [trail_line, vectors_line, draw_point] + circle_patches

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=actual_frames,
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
            # Fixed: do not pass fps here since it's already in the writer instance
            ani.save(output_anim_path, writer=writer)
        else:
            writer = animation.PillowWriter(fps=fps)
            # Fixed: removed fps=fps from save()
            ani.save(output_anim_path, writer=writer)
        print(f"Animation saved: {output_anim_path}")

    # Render and save the completed line as PNG if requested
    if save_png and png_path:
        print(f"Exporting final frame to: {png_path}...")
        png_fig, png_ax = plt.subplots(figsize=(8, 8), dpi=200)
        png_fig.patch.set_facecolor("#ffffff")
        png_ax.set_facecolor("#ffffff")
        png_ax.set_xlim(min_x - pad_x, max_x + pad_x)
        png_ax.set_ylim(max_y + pad_y, min_y - pad_y)
        png_ax.set_aspect("equal")
        png_ax.axis("off")

        # Compute all final points of the loop
        t_final = np.linspace(0, 2 * np.pi, total_frames, endpoint=True)
        final_x = np.zeros_like(t_final)
        final_y = np.zeros_like(t_final)

        for epi in epicycles:
            freq = epi["freq"]
            angle = freq * t_final + epi["phase"]
            final_x += epi["radius"] * np.cos(angle)
            final_y += epi["radius"] * np.sin(angle)

        png_ax.plot(final_x, final_y, color=trail_color, lw=1.6, solid_capstyle="round")
        png_fig.savefig(png_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(png_fig)
        print(f"Final drawing saved: {png_path}")

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
        "-s", "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier for animation playback (e.g. 2.0 is 2x faster, 0.5 is half speed, default: 1.0)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional output animation filename (.gif or .mp4)"
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Save the fully reconstructed static drawing as a PNG image using the same base name"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive display window"
    )

    args = parser.parse_args()

    # Determine PNG output path
    png_path = None
    if args.save_png:
        base_target = args.output if args.output else args.polyline_file
        base_name, _ = os.path.splitext(base_target)
        png_path = f"{base_name}.png"

    try:
        print(f"[1/4] Loading polyline from: {args.polyline_file}...")
        coords = load_polyline_coordinates(args.polyline_file)
        print(f"      Loaded {len(coords)} vertices.")

        print(f"[2/4] Resampling polyline to {args.samples} uniform spatial samples...")
        resampled = resample_polyline_uniform(coords, num_samples=args.samples)

        total_frames = args.frames if args.frames is not None else max(args.samples, 1200)

        print(f"[3/4] Computing Discrete Fourier Transform (DFT)...")
        epicycles, N = compute_fourier_coefficients(resampled, num_harmonics=args.harmonics)
        print(f"      Generated {len(epicycles)} epicycle components sorted by radius.")

        actual_frames = max(int(total_frames / args.speed), 1)
        print(f"[4/4] Starting animation engine ({actual_frames} frames at {args.fps} FPS, {args.speed}x speed)...")
        animate_fourier_drawing(
            epicycles=epicycles,
            resampled_points=resampled,
            output_anim_path=args.output,
            total_frames=total_frames,
            fps=args.fps,
            speed=args.speed,
            save_png=args.save_png,
            png_path=png_path,
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