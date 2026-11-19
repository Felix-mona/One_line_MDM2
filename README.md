# Continuous Line Drawing via Graph Routing, Fourier Descriptors, and Aesthetic Optimisation

**MDM2 Group Project 1 — Group 18**

Theo Coates, Felix Monaghan, Matteo Lisi, Louis Norcutt, and Finlay Stevens

An automated pipeline that converts any 2D raster sketch into a smooth, unbroken single-line trajectory using medial-axis vectorisation, parity-aware Double Minimum Spanning Tree (MST) bridging, Chinese Postman Eulerian routing, and symmetric Discrete Fourier Transform (DFT) frequency truncation.

## Repository Structure

```
.
├── README.md                          # Project documentation & parameter reference
├── run.py                             # Command-line entrypoint for running the pipeline
├── full_pipeline.py                   # End-to-end orchestrator connecting Steps 1–5
├── Set_Images/                        # Benchmark test sketches (Sunflower, Hedgehog, Indonesia, etc.)
├── archive/                           # Archived exploratory scripts and earlier prototypes
└── pipeline/                          # Core one-line generation & evaluation sub-modules
    ├── graph_formulation.py           # Step 1: Binarisation, Lee thinning, Rutovitz CN, RDP & refinement
    ├── double_mst.py                  # Step 2: Sharp-corner/edge subdivision & Double MST bridging
    ├── route_mapping.py               # Step 3: Chinese Postman matching & minimum-angle Fleury walk
    ├── fourier_reconstruction.py      # Step 4: Uniform resampling (N = 10,000), FFT & low-pass filtering
    ├── metrics.py                     # Step 5: Normalised Hausdorff distance & discrete smoothness
    └── formulation_main.py            # Standalone runner for Steps 1–3 only
```

| File | Report Section | Description | 
| ----- | ----- | ----- | 
| `run.py` | — | Command-line entrypoint at the project root for running the full pipeline on any image. | 
| `full_pipeline.py` | Sections 2.1–2.5 | End-to-end orchestrator at the project root connecting Steps 1–5 and exporting 300 DPI borderless plots. | 
| `pipeline/graph_formulation.py` | Section 2.1 | Binarisation, Lee medial-axis thinning, Rutovitz crossing numbers, RDP decimation, and graph refinement. | 
| `pipeline/double_mst.py` | Section 2.2 | Sharp-corner ($\ge 65^\circ$) & long-edge subdivision and two-pass Kruskal MST bridging with T-join parity pruning. | 
| `pipeline/route_mapping.py` | Section 2.3 | Chinese Postman Problem (Dijkstra + minimum-weight matching) and minimum-deflection Fleury Eulerian walk. | 
| `pipeline/fourier_reconstruction.py` | Section 2.4 | Uniform arc-length complex resampling ($N = 10{,}000$), forward FFT, low-pass descriptor truncation, and SVG export. | 
| `pipeline/metrics.py` | Section 2.5 | Normalised two-way Hausdorff distance ($d_{\text{norm}}$) and discrete interior turning-angle smoothness penalty ($S$). | 
| `pipeline/formulation_main.py` | Sections 2.1–2.3 | Standalone runner for graph extraction, bridging, and Eulerian routing (Steps 1–3 only). | 

## Installation

Requires **Python 3.10+** and the following scientific Python packages:

```bash
pip install numpy scipy networkx matplotlib pillow scikit-image
```

## Quick Start

### 1. Run via Command Line (`run.py`)

Convert any sketch from the repository root using default accuracy thresholds (`0.01`, `0.02`, `0.04`):

```bash
python run.py Set_Images/Sunflower/Sunflower.png -o outputs/
```

Specify custom Hausdorff accuracy thresholds, export vector SVGs, and save all intermediate diagnostic figures (skeleton, junctions, MST passes, Eulerian tour, and metric curves):

```bash
python run.py Set_Images/Sunflower/Sunflower.png -o outputs/ -t 0.01 0.02 0.05 --svg --save-all
```

#### CLI Flags (`run.py`)

| Flag | Default | Description | 
| ----- | ----- | ----- | 
| `image` | *(Required)* | Path to the input raster sketch (`.png`, `.jpg`, etc.). | 
| `-o, --output` | `outputs/` | Output directory or file path template. | 
| `-t, --thresholds` | `0.01 0.02 0.04` | One or more target normalised Hausdorff distance thresholds $d_{\text{norm}} \in [0, 1]$. | 
| `-n, --samples` | `10000` | Number of uniform arc-length sample points $N$ around the closed Eulerian circuit. | 
| `--stroke-width` | `10.0` | Estimated input stroke thickness in pixels (used to rescale very thick strokes before thinning). | 
| `--thresh` | `128` | Grayscale binarisation threshold $T_{\text{bin}} \in [0, 255]$. | 
| `--svg` | `False` | Export reconstructed one-line trajectories as standalone vector `.svg` files. | 
| `--save-all` | `False` | Save all intermediate step PNGs, `.npy` polylines, and `.svg` files. | 

### 2. Run Programmatically (`full_pipeline.py`)

Import `run_oneline_pipeline` directly from `full_pipeline` at the repository root into Python scripts or notebooks for full control over every stage parameter:

```python
from full_pipeline import run_oneline_pipeline

results = run_oneline_pipeline(
    image_path="Set_Images/Sunflower/Sunflower.png",
    output_path="outputs/sunflower/",
    accuracy_thresholds=[0.005, 0.01, 0.02, 0.04, 0.07],
    n_samples=10000,
    save_all_intermediaries=True,
)

for rec in results["reconstructions"]:
    print(f"Target <= {rec['target_accuracy']}: K={rec['k']}, S={rec['smoothness']:.4f}")
```

### 3. Run Graph Routing Only (`pipeline/formulation_main.py`)

To inspect Steps 1–3 (skeletonisation, Double MST component bridging, and Chinese Postman Eulerian circuit construction) without running Fourier smoothing:

1. Edit `IMAGE_PATH` and `OUTPUT_DIR` at the top of `pipeline/formulation_main.py`.
2. Run from the repository root:

   ```bash
   python pipeline/formulation_main.py
   ```

## Complete Parameter Reference (`run_oneline_pipeline`)

All spatial tolerances are defined proportionally to the cropped foreground bounding-box diagonal $D = \sqrt{W^2 + H^2}$ so that behaviour remains scale-invariant across input image resolutions.

### 1. Input & Output Paths

| Parameter | Type | Default | Description | 
| ----- | ----- | ----- | ----- | 
| `image_path` | `str \| Path` | *(Required)* | Path to the input raster image. Alpha channels are automatically composited onto a white background. | 
| `output_path` | `str \| Path` | *(Required)* | Output directory (creates `<output_path>/<stem>/`) or explicit filename template. | 
| `intermediary_dir` | `str \| Path \| None` | `None` | Custom directory for intermediate plots, `.npy`, and `.svg` files. Defaults to the main output directory. | 

### 2. Step 1: Pre-Processing & Graph Extraction (`pipeline/graph_formulation.py`)

| Parameter | Type | Default | Report Symbol / Eq. | Effect & Tuning Guidance | 
| ----- | ----- | ----- | ----- | ----- | 
| `binarise_thresh` | `int` | `128` | $T_{\text{bin}} = 128$ | Grayscale cutoff in $[0, 255]$. Pixels darker than $T_{\text{bin}}$ become foreground strokes. Increase for faint pencil sketches; decrease if background noise appears. | 
| `input_stroke_width` | `float` | `10.0` | — | Estimated stroke thickness (in pixels) of the input image. If `input_stroke_width / target_stroke_width > 1.05`, the mask is downsampled prior to thinning to avoid medial-axis branching artifacts on thick lines. | 
| `target_stroke_width` | `float` | `10.0` | — | Working stroke thickness (in pixels) before running Lee's medial-axis thinning. | 
| `rdp_epsilon_prop` | `float` | `0.001` | $\varepsilon_{\text{RDP}} = 0.001D$ | Ramer–Douglas–Peucker perpendicular tolerance as a fraction of diagonal $D$. Larger values produce coarser polylines with fewer vertices; smaller values preserve pixel-level staircasing. | 
| `d_euclid_prop` | `float` | `0.008` | $d_{\text{euclid}} = 0.008D$ | Spatial KD-Tree merge radius for collapsing nearby Rutovitz endpoints/junctions into a single geometric medoid vertex. | 
| `d_edge_factor` | `float` | `2.5` | $d_{\text{edge}} = 0.020D$ | Multiplier on `d_euclid` defining the close-junction contraction threshold ($2.5 \times 0.008D = 0.020D$). Contracts short internal edges between junctions caused by crossing strokes. | 
| `whisker_prop` | `float` | `0.010` | $0.010D$ | Minimum arc length for degree-1 terminal edges; shorter spurs ("whiskers") are pruned. | 
| `min_comp_prop` | `float` | `0.015` | $0.015D$ | Minimum total arc length for isolated connected components; smaller specks or noise dots are removed. | 

### 3. Step 2: Attachment-Vertex Subdivision & Double MST Bridging (`pipeline/double_mst.py`)

| Parameter | Type | Default | Report Symbol / Eq. | Effect & Tuning Guidance | 
| ----- | ----- | ----- | ----- | ----- | 
| `corner_angle_deg` | `float` | `65.0` | $\theta_i \ge 65^\circ$ | Exterior deflection angle threshold at which interior polyline vertices are split into explicit `corner` vertices for natural bridge attachment. | 
| `max_edge_len_prop` | `float` | `0.05` | $L_{\max} = 0.05D$ | Maximum unsplit edge arc length; longer edges are subdivided into equal-length segments to provide intermediate attachment nodes along smooth curves. | 
| `p_even` | `float` | `4.0` | $p_{\text{even}}$ (Eq. 3) | Additive node penalty for untouched even-degree non-corner vertices ($d_{\text{eff}}(n)$ even). High values discourage attaching bridges mid-stroke where they create odd-degree junctions. | 
| `p_odd` | `float` | `0.5` | $p_{\text{odd}}$ (Eq. 3) | Additive node penalty for odd-degree vertices ($d_{\text{eff}}(n) \ge 3$ odd). Low values attract bridges to odd junctions to convert them into even-degree nodes. | 
| `p_corner` | `float` | `1.0` | $p_{\text{corner}}$ (Eq. 3) | Additive node penalty for even-degree sharp-corner vertices, making corners preferred entry/exit points on closed loops. | 
| `delta_reuse` | `float` | `0` | $\delta_{\text{reuse}}$ | Discount subtracted from `p_even` when reusing an even-degree vertex that already has incident bridges ($b(n) > 0$). | 
| `lambda_int` | `float` | `25.0` | $\lambda_{\text{int}}$ (Eq. 2) | Multiplicative crossing penalty per existing stroke or bridge intersected ($N_{\text{cross}}(u,v)$). High values prevent bridges from cutting across existing ink lines. | 
| `gamma_edge` | `float` | `10.0` | $\gamma_{\text{edge}}$ (Eq. 4) | Pass 2 weight multiplier penalising exact node-to-node duplication of a Pass 1 bridge, encouraging distinct entry and exit bridges. | 
| `gamma_comp` | `float` | `1.35` | $\gamma_{\text{comp}}$ (Eq. 4) | Pass 2 weight multiplier encouraging bridges between component pairs not already directly linked in Pass 1. | 
| `prune_redundant_bridges` | `bool` | `True` | — | Removes redundant parallel bridge pairs and simple bridge cycles among components with $\ge 4$ bridges while preserving graph connectivity and even bridge parity. | 
| `max_candidates_per_pair` | `int` | `32` | — | Maximum candidate vertex-to-vertex bridges evaluated with exact planar intersection checks per pair of disconnected components. | 
| `top_k_candidates` | `int` | `25` | — | Top-$K$ lowest-cost candidate bridges retained per component pair during Kruskal MST construction. | 

### 4. Step 4 & 5: Fourier Reconstruction & Metric Thresholds (`pipeline/fourier_reconstruction.py`, `pipeline/metrics.py`)

| Parameter | Type | Default | Report Symbol / Eq. | Effect & Tuning Guidance | 
| ----- | ----- | ----- | ----- | ----- | 
| `accuracy_thresholds` | `float \| list[float]` | `0.01` | $d_{\text{norm}} \in [0, 1]$ | Target upper bound(s) on normalised two-way Hausdorff distance (Eq. 11). Warm-started binary search finds the lowest frequency cutoff $K$ satisfying $d_{\text{norm}} \le \text{threshold}$. Smaller values retain sharper detail (higher $K$); larger values produce smoother curves (lower $K$). | 
| `n_samples` | `int` | `10000` | $N = 10{,}000$ | Number of uniformly spaced arc-length points $z_n = x_n + \mathrm{i}\,y_n$ sampled along the closed Eulerian circuit before computing the FFT (Eqs. 5–6). Keeping $N$ fixed ensures scale-consistent discrete smoothness scores $S$ (Eq. 16). | 
| `mirror_path` | `bool` | `False` | — | If `True`, appends the reversed trajectory onto itself before resampling to enforce open-endpoint continuity. Not required for closed Eulerian circuits. | 
| `max_frequency_cutoff` | `int \| None` | `None` | $K_{\max}$ | Optional upper bound on frequency cutoff $K$ during binary search. Defaults to $(N - 1) // 2 = 4999$. | 
| `gibbs_window` | `int` | `4` | — | Number of frequency indices below the binary-search solution checked locally to catch non-monotonic Gibbs-ripple threshold crossings. | 

### 5. Canvas Styling & Diagnostic Export Flags

| Parameter | Type | Default | Description | 
| ----- | ----- | ----- | ----- | 
| `line_color` | `str` | `"black"` | Stroke colour used when rendering final reconstructed one-line PNGs and SVGs. | 
| `bg_color` | `str` | `"white"` | Canvas background colour for exported figures. | 
| `line_width` | `float` | `1.8` | Stroke line width in points. | 
| `figsize` | `tuple[float, float]` | `(10, 10)` | Figure dimensions in inches (borderless square canvas). | 
| `dpi` | `int` | `300` | Export resolution in dots per inch. | 
| `save_all_intermediaries` | `bool` | `False` | Master toggle enabling all 17 intermediate diagnostic exports listed below. | 
| `save_binary_mask` | `bool` | `False` | Saves `*_01_binary_mask.png` (cropped foreground mask). | 
| `save_skeleton` | `bool` | `False` | Saves `*_02_skeleton.png` (Lee's single-pixel medial-axis skeleton). | 
| `save_junction_classification` | `bool` | `False` | Saves `*_03_junction_classification.png` (endpoints in green, junctions in red). | 
| `save_raw_graph` | `bool` | `False` | Saves `*_04_raw_traced_graph.png` (unsimplified 8-connected multigraph). | 
| `save_rdp_graph` | `bool` | `False` | Saves `*_05_rdp_simplified.png` (post-RDP decimation multigraph). | 
| `save_base_graph` | `bool` | `False` | Saves `*_06_refined_base_graph.png` (after close-junction contraction & whisker pruning). | 
| `save_subdivided_graph` | `bool` | `False` | Saves `*_07_subdivided_graph.png` (after sharp-corner and long-edge splitting). | 
| `save_mst1_graph` | `bool` | `False` | Saves `*_08_mst1_bridges.png` (Pass 1 Kruskal MST bridges in cyan). | 
| `save_mst2_graph` | `bool` | `False` | Saves `*_09_mst2_unpruned_bridges.png` (Pass 1 cyan + unpruned Pass 2 magenta bridges). | 
| `save_bridged_graph` | `bool` | `False` | Saves `*_10_pruned_bridged_graph.png` (final connected multigraph after parity pruning). | 
| `save_eulerised_graph` | `bool` | `False` | Saves `*_11_eulerised_graph.png` (CPP duplicated edges highlighted in crimson). | 
| `save_euler_route` | `bool` | `False` | Saves `*_12_euler_route.png` (directed Eulerian walk colour-coded by traversal order). | 
| `save_raw_polyline_plot` | `bool` | `False` | Saves `*_13_raw_polyline.png` (unsmoothed closed Eulerian trajectory). | 
| `save_fourier_metrics_plot` | `bool` | `False` | Saves `*_14_fourier_accuracy_curve.png` ($d_{\text{norm}}$ and $S$ plotted across $K$). | 
| `save_hausdorff_overlay` | `bool` | `False` | Saves `*_15_hausdorff_A<thresh>_K<k>.png` (reconstructed curve overlaid on original with max Hausdorff pair). | 
| `save_intermediate_npy` | `bool` | `False` | Exports raw and reconstructed `(N, 2)` coordinate arrays as `.npy` files. | 
| `save_intermediate_svg` | `bool` | `False` | Exports raw and reconstructed trajectories as standalone scalable `.svg` vector graphics. | 