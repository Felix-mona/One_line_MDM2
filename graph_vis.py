import argparse
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from svg_to_graph import svg_to_graph, connect_disjoint_components


def plot_svg_graph(
    G: nx.MultiGraph,
    output_image_path: str = "images/graph_visualization.png",
    dpi: int = 300,
    show_labels: bool = False
):
    """Renders the graph topology with color-coded nodes and edge types.

    Parameters:
        G (nx.MultiGraph): Graph produced by svg_to_graph and connect_disjoint_components.
        output_image_path (str): Filepath to save the rendered PNG/PDF.
        dpi (int): Resolution of exported image.
        show_labels (bool): Whether to annotate node coordinates.
    """
    width = G.graph.get("canvas_width", 500.0)
    height = G.graph.get("canvas_height", 500.0)

    # Set up figure dimensions proportional to SVG viewBox
    fig_w = 10.0
    fig_h = max(4.0, fig_w * (height / width))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#1e1e24")
    ax.set_facecolor("#1e1e24")

    # 1. Draw Edges
    original_drawn = False
    bridge_drawn = False

    for u, v, k, data in G.edges(keys=True, data=True):
        pts = np.array(data.get("path", [G.nodes[u]["pos"], G.nodes[v]["pos"]]))
        is_bridge = data.get("is_bridge", False)

        if is_bridge:
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color="#ff0055",          # Bright neon pink / magenta for bridge edges
                linestyle="--",
                linewidth=2.2,
                alpha=0.95,
                label="Bridge Edge (New MST Connection)" if not bridge_drawn else None,
                zorder=2
            )
            bridge_drawn = True
        else:
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color="#4cc9f0",          # Bright sky blue for original strokes
                linestyle="-",
                linewidth=1.6,
                alpha=0.8,
                label="Original Edge" if not original_drawn else None,
                zorder=1
            )
            original_drawn = True

    # 2. Categorize and Draw Nodes
    node_groups = {
        "endpoint": {"pts": [], "color": "#f72585", "size": 35, "marker": "o", "label": "Endpoint (deg 1)"},
        "junction": {"pts": [], "color": "#fee440", "size": 45, "marker": "s", "label": "Junction (deg >= 3)"},
        "corner":   {"pts": [], "color": "#7209b7", "size": 30, "marker": "^", "label": "Corner Split"},
        "interior": {"pts": [], "color": "#ffffff", "size": 8,  "marker": "o", "label": "Interior Node"}
    }

    for node, data in G.nodes(data=True):
        pos = data["pos"]
        deg = G.degree(node)

        # Prioritize categorization
        if deg == 1:
            cat = "endpoint"
        elif deg >= 3:
            cat = "junction"
        elif data.get("is_corner", False):
            cat = "corner"
        else:
            cat = "interior"

        node_groups[cat]["pts"].append(pos)

        if show_labels:
            ax.text(pos[0] + 1.5, pos[1] - 1.5, f"{node}", color="#ffffff", fontsize=5, alpha=0.6)

    for cat, conf in node_groups.items():
        if conf["pts"]:
            arr = np.array(conf["pts"])
            ax.scatter(
                arr[:, 0],
                arr[:, 1],
                c=conf["color"],
                s=conf["size"],
                marker=conf["marker"],
                label=conf["label"],
                edgecolors="#000000",
                linewidths=0.5,
                zorder=3
            )

    # Invert Y axis to match SVG screen coordinate conventions (0, 0 at top-left)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")

    # Clean border styling and legend
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.tick_params(colors="#888888", labelsize=8)
    ax.grid(True, color="#333333", linestyle=":", linewidth=0.5, alpha=0.6)

    ax.set_title(
        f"Graph Representation: {G.number_of_nodes()} Nodes | {G.number_of_edges()} Edges",
        color="#ffffff",
        fontsize=12,
        pad=12
    )
    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        facecolor="#2b2b36",
        edgecolor="#444444",
        labelcolor="#ffffff",
        fontsize=8
    )
    legend.set_zorder(10)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=dpi)
    plt.close()
    print(f"Graph visualization saved to: {output_image_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SVG topological graph and bridge connections.")
    parser.add_argument("svg_file", nargs="?", default="images/output.svg", help="Path to input SVG file[cite: 1]")
    parser.add_argument("-o", "--output", default="images/graph_visualization.png", help="Path to save output image")
    parser.add_argument("--angle", type=float, default=60.0, help="Max turn angle (deg) before splitting into a corner node[cite: 2]")
    parser.add_argument("--len-pct", type=float, default=0.10, help="Max edge length as fraction of image dimension")
    parser.add_argument("--penalty", type=float, default=40.0, help="Distance penalty for connecting to degree-2 interior nodes[cite: 2]")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI resolution")
    parser.add_argument("--labels", action="store_true", help="Annotate node coordinates on the plot")

    args = parser.parse_args()

    try:
        # Step 1: Build graph from SVG
        G = svg_to_graph(args.svg_file, max_angle_deg=args.angle, length_limit_pct=args.len_pct)
        initial_components = nx.number_connected_components(G)

        # Step 2: Bridge disjoint components
        if initial_components > 1:
            G = connect_disjoint_components(G, interior_penalty=args.penalty)

        # Step 3: Plot and save
        plot_svg_graph(G, output_image_path=args.output, dpi=args.dpi, show_labels=args.labels)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()