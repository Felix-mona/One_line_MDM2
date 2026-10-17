import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import numpy as np
import networkx as nx

from svg_to_graph import svg_to_graph, connect_disjoint_components
from solve_ccp import make_semi_eulerian, simple_eulerian_trail, reconstruct_full_polyline


def plot_static_gradient(
    polyline: list,
    width: float,
    height: float,
    output_path: str = "images/ccp_gradient_trail.png",
    dpi: int = 300
):
    """Renders the continuous stroke with a color gradient from start to end."""
    pts = np.array(polyline, dtype=np.float64)
    if len(pts) < 2:
        print("Error: Polyline contains fewer than 2 points.", file=sys.stderr)
        return

    # Set up canvas proportional to SVG dimensions[cite: 1]
    fig_w = 10.0
    fig_h = max(4.0, fig_w * (height / width))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#18181b")
    ax.set_facecolor("#18181b")

    # Split into consecutive line segments for gradient coloring
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    norm_progress = np.linspace(0.0, 1.0, len(segments))

    # Line collection with smooth viridis/plasma gradient
    lc = LineCollection(segments, cmap="plasma", linewidth=2.0, alpha=0.9, zorder=2)
    lc.set_array(norm_progress)
    ax.add_collection(lc)

    # Highlight Start (green) and End (red) nodes
    ax.scatter(pts[0, 0], pts[0, 1], c="#00f5d4", s=90, edgecolors="#ffffff", linewidths=1.2, zorder=4, label=f"Start {tuple(pts[0])}")
    ax.scatter(pts[-1, 0], pts[-1, 1], c="#ff0055", s=90, edgecolors="#ffffff", linewidths=1.2, zorder=4, label=f"End {tuple(pts[-1])}")

    # SVG coordinate system: (0, 0) is top-left[cite: 1]
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")

    # Styling and colorbar
    for spine in ax.spines.values():
        spine.set_color("#333338")
    ax.tick_params(colors="#71717a", labelsize=8)
    ax.grid(True, color="#27272a", linestyle=":", linewidth=0.6)

    cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Drawing Progress (0% Start → 100% Finish)", color="#ffffff", fontsize=9)
    cbar.ax.tick_params(labelsize=8, colors="#a1a1aa")

    ax.set_title(f"Semi-Eulerian Continuous Single-Line Path ({len(pts)} Points)", color="#ffffff", fontsize=12, pad=12)
    ax.legend(loc="upper right", facecolor="#27272a", edgecolor="#3f3f46", labelcolor="#ffffff", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Static gradient image saved to: {output_path}")


def animate_drawing(
    polyline: list,
    width: float,
    height: float,
    output_path: str = "images/drawing_animation.gif",
    fps: int = 30,
    total_duration_sec: float = 6.0
):
    """Generates an animated GIF/MP4 showing the continuous line being drawn."""
    pts = np.array(polyline, dtype=np.float64)
    total_frames = int(fps * total_duration_sec)
    step_indices = np.linspace(1, len(pts), total_frames).astype(int)

    fig, ax = plt.subplots(figsize=(8, 8 * (height / width)), facecolor="#18181b")
    ax.set_facecolor("#18181b")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Dim background trace of entire drawing
    ax.plot(pts[:, 0], pts[:, 1], color="#27272a", linewidth=1.0, zorder=1)

    # Active line and moving pen tip
    line, = ax.plot([], [], color="#00f5d4", linewidth=2.0, zorder=2)
    pen_tip, = ax.plot([], [], marker="o", color="#ff0055", markersize=6, zorder=3)

    def init():
        line.set_data([], [])
        pen_tip.set_data([], [])
        return line, pen_tip

    def update(frame):
        idx = step_indices[frame]
        line.set_data(pts[:idx, 0], pts[:idx, 1])
        pen_tip.set_data([pts[idx - 1, 0]], [pts[idx - 1, 1]])
        return line, pen_tip

    ani = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True)
    ani.save(output_path, fps=fps, writer="pillow")
    plt.close()
    print(f"Drawing animation saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize the continuous CCP Eulerian single-line trail.")
    parser.add_argument("svg_file", nargs="?", default="images/output.svg", help="Path to input SVG[cite: 1]")
    parser.add_argument("-o", "--output", default="images/ccp_gradient_trail.png", help="Path for gradient image")
    parser.add_argument("--animate", action="store_true", help="Generate an animated GIF of the stroke path")
    parser.add_argument("--gif-output", default="images/drawing_animation.gif", help="Path for animated GIF")
    parser.add_argument("--angle", type=float, default=60.0, help="Corner angle split threshold (deg)[cite: 2]")
    parser.add_argument("--len-pct", type=float, default=0.10, help="Max edge length as fraction of image size")
    parser.add_argument("--penalty", type=float, default=40.0, help="Interior node bridge penalty[cite: 2]")

    args = parser.parse_args()

    try:
        # 1. Build and bridge MultiGraph[cite: 2]
        G = svg_to_graph(args.svg_file, max_angle_deg=args.angle, length_limit_pct=args.len_pct)
        if nx.number_connected_components(G) > 1:
            G = connect_disjoint_components(G, interior_penalty=args.penalty)

        w = G.graph.get("canvas_width", 500.0)
        h = G.graph.get("canvas_height", 500.0)

        # 2. Solve Semi-Eulerian trail
        G_semi, (start_node, _) = make_semi_eulerian(G)
        node_trail = simple_eulerian_trail(G_semi, start_node=start_node)
        polyline = reconstruct_full_polyline(G_semi, node_trail)

        # 3. Output visualization
        plot_static_gradient(polyline, width=w, height=h, output_path=args.output)

        if args.animate:
            print("Rendering animation (this may take a few seconds)...")
            animate_drawing(polyline, width=w, height=h, output_path=args.gif_output)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()