import cv2
import numpy as np
import argparse
from pathlib import Path
from drawing_to_svg import (
    skeletonize_mask,
    detect_nodes,
    walk_edges,
    simplify_all_edges,
    edges_to_svg,
)


def remove_small_specks(binary_mask: np.ndarray, min_area: int = 12) -> np.ndarray:
    """Removes isolated connected components smaller than min_area pixels."""
    if min_area <= 1:
        return binary_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def any_image_to_binary_sketch(
    image_path: str,
    mode: str = "primal",
    min_speck_area: int = 12,
    save_preview_path: str = None
) -> np.ndarray:
    """Converts ANY image into a binary Primal Sketch (255 = ink, 0 = background).

    Modes:
        - 'primal'  : Detailed centerline sketch; keeps small text & fine linework.
        - 'minimal' : Ultra-stripped skeletal sketch; removes hatching, small text,
                      and secondary textures, keeping only dominant structural lines.
        - 'xdog'    : Extended Difference of Gaussians for natural photographs.
        - 'canny'   : Gradient-based Canny edge detector.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")

    # 1. Handle Alpha channel if PNG has transparency
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        alpha_f = a.astype(np.float32) / 255.0
        bgr = cv2.merge([b, g, r]).astype(np.float32)
        white_bg = np.full_like(bgr, 255.0)
        img = (bgr * alpha_f[..., None] + white_bg * (1.0 - alpha_f[..., None])).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    if mode == "primal":
        smooth = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)
        adaptive_ink = cv2.adaptiveThreshold(
            smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=15, C=9
        )
        dist = cv2.distanceTransform(adaptive_ink, cv2.DIST_L2, 5)
        thick_interior = (dist > 4.5).astype(np.uint8) * 255
        binary_ink = cv2.subtract(adaptive_ink, thick_interior)

    elif mode == "minimal":
        # 1. Two passes of strong bilateral filtering to wipe out fine hatching,
        # small text, and surface textures while preserving major edges
        smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=100, sigmaSpace=100)
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=100, sigmaSpace=100)

        # 2. High-threshold adaptive ink extraction (C=16 ignores faint/thin lines)
        adaptive_ink = cv2.adaptiveThreshold(
            smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=25, C=16
        )

        # 3. Morphological opening to erase thin stray whiskers before hollowing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        adaptive_ink = cv2.morphologyEx(adaptive_ink, cv2.MORPH_OPEN, kernel)

        # 4. Hollow out thick solid shapes so only their outer silhouette remains
        dist = cv2.distanceTransform(adaptive_ink, cv2.DIST_L2, 5)
        thick_interior = (dist > 4.0).astype(np.uint8) * 255
        binary_ink = cv2.subtract(adaptive_ink, thick_interior)

    elif mode == "xdog":
        smooth = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)
        g1 = cv2.GaussianBlur(smooth, (3, 3), 0.8)
        g2 = cv2.GaussianBlur(smooth, (7, 7), 2.0)
        dog = cv2.subtract(g2, g1)
        _, binary_ink = cv2.threshold(dog, 7, 255, cv2.THRESH_BINARY)

    elif mode == "canny":
        smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=60, sigmaSpace=60)
        v = np.median(smooth)
        lower = int(max(0, 0.67 * v))
        upper = int(min(255, 1.33 * v))
        binary_ink = cv2.Canny(smooth, lower, upper)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'primal', 'minimal', 'xdog', or 'canny'.")

    # 3. Prune isolated components smaller than min_speck_area
    cleaned_ink = remove_small_specks(binary_ink, min_area=min_speck_area)

    # 4. Save preview image (Black ink on White background)
    if save_preview_path:
        drawn_image_white_bg = 255 - cleaned_ink
        cv2.imwrite(save_preview_path, drawn_image_white_bg)
        print(f"Saved intermediate drawn image to: {save_preview_path}")

    return cleaned_ink


def run_universal_pipeline(
    input_image_path: str,
    output_svg_path: str,
    mode: str = "primal",
    epsilon: float = 0.8,
    stroke_width: float = 1.5,
    min_speck_area: int = None,
    min_edge_pixels: int = None,
    preview_png_path: str = None
) -> dict:
    """Runs the Primal Sketch to SVG pipeline with automatic presets per mode."""
    # Automatically scale pruning aggressiveness if user didn't override via CLI
    if min_speck_area is None:
        min_speck_area = 110 if mode == "minimal" else 12
    if min_edge_pixels is None:
        min_edge_pixels = 15 if mode == "minimal" else 2

    print(f"[1/6] Converting image to Primal Sketch ({mode} mode, min_area={min_speck_area}): {input_image_path}...")
    binary = any_image_to_binary_sketch(
        input_image_path,
        mode=mode,
        min_speck_area=min_speck_area,
        save_preview_path=preview_png_path
    )
    height, width = binary.shape

    print(f"[2/6] Thinning strokes via skeletonization...")
    skeleton = skeletonize_mask(binary, method='lee')

    print(f"[3/6] Detecting endpoints and junction nodes...")
    endpoints, junctions, _ = detect_nodes(skeleton)
    print(f"      Found {len(endpoints)} endpoints, {len(junctions)} junctions.")

    print(f"[4/6] Tracing edges between nodes...")
    raw_edges = walk_edges(skeleton, endpoints, junctions)

    # Filter out tiny skeleton twigs/spurs when in minimal mode
    if min_edge_pixels > 2:
        before_count = len(raw_edges)
        raw_edges = [e for e in raw_edges if len(e["path"]) >= min_edge_pixels]
        print(f"      Pruned {before_count - len(raw_edges)} short skeleton spurs (< {min_edge_pixels} px).")

    raw_point_count = sum(len(e["path"]) for e in raw_edges)
    print(f"      Retained {len(raw_edges)} edge paths ({raw_point_count} total pixels).")

    print(f"[5/6] Simplifying paths using RDP (epsilon={epsilon})...")
    simplified_edges = simplify_all_edges(raw_edges, epsilon=epsilon)

    print(f"[6/6] Serializing paths to SVG: {output_svg_path}...")
    edges_to_svg(simplified_edges, width, height, output_svg_path, stroke_width=stroke_width)

    return {
        "endpoints": endpoints,
        "junctions": junctions,
        "edges": simplified_edges
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ANY photo or image into a Primal Sketch SVG.")
    parser.add_argument(
        "input",
        nargs="?",
        default="images/orb.png",
        help="Path to any input photo/image (default: images/orb.png)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output SVG path (default: <input_name>_output.svg)"
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["primal", "minimal", "xdog", "canny"],
        default="primal",
        help="Sketch extraction method: 'primal' (detailed) or 'minimal' (stripped-down skeletal)"
    )
    parser.add_argument(
        "-s", "--min-size",
        type=int,
        default=None,
        help="Min pixel area for noise pruning (auto: 12 for primal, 110 for minimal)"
    )
    parser.add_argument(
        "-l", "--min-length",
        type=int,
        default=None,
        help="Min pixel length for traced skeleton edges (auto: 2 for primal, 15 for minimal)"
    )
    parser.add_argument(
        "-e", "--epsilon",
        type=float,
        default=1.0,
        help="RDP curve simplification tolerance (default: 1.0)"
    )
    parser.add_argument(
        "-p", "--preview",
        default=None,
        help="Save intermediate drawn PNG (default: <input_name>_preview.png)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output is None:
        args.output = str(input_path.with_name(f"{input_path.stem}_output.svg"))

    if args.preview is None:
        args.preview = str(input_path.with_name(f"{input_path.stem}_preview.png"))

    run_universal_pipeline(
        input_image_path=args.input,
        output_svg_path=args.output,
        mode=args.mode,
        epsilon=args.epsilon,
        min_speck_area=args.min_size,
        min_edge_pixels=args.min_length,
        preview_png_path=args.preview
    )