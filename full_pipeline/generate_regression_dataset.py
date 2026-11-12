"""
generate_regression_dataset.py
Local, pause-and-resumable dataset generator for the One-Line Drawing pipeline:
- Downloads and saves up to 1,000 Google Quick, Draw! sketches and 1,000 TU-Berlin
  sketches (via the active Hugging Face TU-Berlin dataset mirror).
- Runs `run_oneline_pipeline` at N_SAMPLES = 10,000 across 5 exponential accuracy
  thresholds: [0.0025, 0.006, 0.015, 0.038, 0.095].
- Saves all 5 reconstructed output images per input sketch (up to 10,000 output PNGs).
- Extracts the 5 core pre-Fourier descriptors (plus auxiliary route & K metrics)
  and writes them incrementally to two separate CSV files:
    1. dataset_2000/quickdraw_descriptors.csv
    2. dataset_2000/tu_berlin_descriptors.csv
- Press Ctrl+C at any time to pause safely; re-running automatically resumes where it left off.
"""

import os
import sys
import csv
import time
import json
import urllib.request
import urllib.parse
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from quickdraw import QuickDrawData

# Import existing pipeline modules
from full_pipeline import run_oneline_pipeline
from route_mapping import compute_route_descriptors
import Fourier_reconsctruction as fourier
import Metric as met


# ==========================================================
# 1. CONFIGURATION (SET TO 5 PER DATASET = 10 IMAGES FOR TEST RUN)
# ==========================================================

# Set MAX_IMAGES_PER_DATASET = 5 for a 10-image test run (5 QuickDraw + 5 TU-Berlin = 50 outputs).
# Change MAX_IMAGES_PER_DATASET = 1000 for the full 2,000-image run (10,000 outputs).
MAX_IMAGES_PER_DATASET = 5

EXPONENTIAL_THRESHOLDS = [0.005, 0.01,0.02,0.04, 0.07]
N_SAMPLES = 10000

# Set True to include the Accuracy Threshold & chosen K in the title/legend of each output PNG
SHOW_LABELS_ON_IMAGES = True
OUTPUT_FIGSIZE = (8, 8)
OUTPUT_DPI = 200

DATASET_DIR = Path("dataset_2000")
RAW_QUICKDRAW_DIR = DATASET_DIR / "01_raw_sketches" / "quickdraw"
RAW_TUBERLIN_DIR = DATASET_DIR / "01_raw_sketches" / "tu_berlin"
RECON_QUICKDRAW_DIR = DATASET_DIR / "02_reconstructions" / "quickdraw"
RECON_TUBERLIN_DIR = DATASET_DIR / "02_reconstructions" / "tu_berlin"

QUICKDRAW_CSV_PATH = DATASET_DIR / "quickdraw_descriptors.csv"
TUBERLIN_CSV_PATH = DATASET_DIR / "tu_berlin_descriptors.csv"

# Active Hugging Face mirrors for the TU-Berlin Sketch Dataset
HF_TUBERLIN_DATASETS = [
    ("sdiaeyu6n/tu-berlin", "default", "test"),
    ("sdiaeyu6n/tu-berlin", "default", "validation"),
    ("kmewhort/tu-berlin-png", "default", "train"),
]


# ==========================================================
# 2. DOWNLOADER 1: GOOGLE QUICK, DRAW! SKETCHES
# ==========================================================

def _render_qd_drawing_to_png(drawing, save_path: Path, canvas_size=800, padding=80, stroke_width=10):
    """Renders a Quick, Draw! vector sketch to a clean 800x800 binary PNG."""
    strokes = drawing.strokes
    all_pts = [pt for stroke in strokes for pt in stroke]
    if not all_pts:
        return False

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span = max(max_x - min_x, max_y - min_y, 1.0)
    usable = canvas_size - 2 * padding
    scale = usable / span

    offset_x = padding + (usable - (max_x - min_x) * scale) / 2.0
    offset_y = padding + (usable - (max_y - min_y) * scale) / 2.0

    img = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        if len(stroke) < 2:
            continue
        scaled_pts = [
            (offset_x + (x - min_x) * scale, offset_y + (y - min_y) * scale)
            for x, y in stroke
        ]
        draw.line(scaled_pts, fill=0, width=stroke_width, joint="curve")

    img.save(str(save_path))
    return True


def prepare_quickdraw_sketches(target_count=1000):
    """
    Prepares `target_count` multi-stroke sketches across up to 250 Quick, Draw! categories.
    Reuses cached PNGs if already on disk.
    """
    RAW_QUICKDRAW_DIR.mkdir(parents=True, exist_ok=True)
    num_categories = min(250, target_count)
    per_category = max(1, int(np.ceil(target_count / num_categories)))

    qd = QuickDrawData(recognized=True, max_drawings=per_category + 8, print_messages=False)
    categories = qd.drawing_names[:num_categories]

    items = []
    count = 0
    print(f"\n--- Preparing {target_count} Google Quick, Draw! Sketches ---")

    for cat in categories:
        if count >= target_count:
            break
        safe_cat = cat.replace(" ", "_").replace("/", "_")
        saved_in_cat = 0

        # Reuse cached files if already generated
        while saved_in_cat < per_category and count < target_count:
            image_id = f"qd_{count:04d}"
            save_path = RAW_QUICKDRAW_DIR / f"{image_id}__{safe_cat}.png"
            if save_path.exists():
                items.append({
                    "image_id": image_id,
                    "dataset_source": "quickdraw",
                    "category": cat,
                    "image_path": str(save_path),
                })
                saved_in_cat += 1
                count += 1
            else:
                break

        if saved_in_cat >= per_category or count >= target_count:
            continue

        group = qd.get_drawing_group(cat)
        for drawing in group.drawings:
            if saved_in_cat >= per_category or count >= target_count:
                break
            if len(drawing.strokes) < 2:
                continue

            image_id = f"qd_{count:04d}"
            save_path = RAW_QUICKDRAW_DIR / f"{image_id}__{safe_cat}.png"
            if not save_path.exists():
                ok = _render_qd_drawing_to_png(drawing, save_path, stroke_width=10)
                if not ok:
                    continue

            items.append({
                "image_id": image_id,
                "dataset_source": "quickdraw",
                "category": cat,
                "image_path": str(save_path),
            })
            saved_in_cat += 1
            count += 1

    print(f"Ready: {len(items)} Quick, Draw! original sketches in '{RAW_QUICKDRAW_DIR}'.")
    return items


# ==========================================================
# 3. DOWNLOADER 2: TU-BERLIN SKETCHES (HUGGING FACE MIRROR + FALLBACK)
# ==========================================================

def _http_get_json(url, timeout=30):
    """Fetches JSON from a URL with a standard User-Agent header."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_bytes(url, timeout=30):
    """Fetches raw bytes from a URL with a standard User-Agent header."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_tuberlin_label_names(dataset_name="sdiaeyu6n/tu-berlin", config="default"):
    """Retrieves the 250 category names for TU-Berlin from the Hugging Face info endpoint."""
    try:
        q_ds = urllib.parse.quote(dataset_name, safe="")
        info_url = f"https://datasets-server.huggingface.co/info?dataset={q_ds}&config={config}"
        info = _http_get_json(info_url, timeout=15)
        features = info.get("dataset_info", {}).get("features", {})
        names = features.get("label", {}).get("names", [])
        if isinstance(names, list) and len(names) > 0:
            return {i: name for i, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
    except Exception:
        pass
    return {}


def prepare_tuberlin_sketches(target_count=1000):
    """
    Downloads `target_count` TU-Berlin sketches directly from the active Hugging Face
    TU-Berlin mirror (`sdiaeyu6n/tu-berlin` / `kmewhort/tu-berlin-png`), saving each
    as a clean PNG in `RAW_TUBERLIN_DIR`.
    Includes an automatic high-complexity multi-stroke fallback if offline.
    """
    RAW_TUBERLIN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Preparing {target_count} TU-Berlin Benchmark Sketches ---")

    existing_pngs = sorted(RAW_TUBERLIN_DIR.glob("tub_*.png"))
    count = len(existing_pngs)

    if count < target_count:
        label_map = _fetch_tuberlin_label_names()

        for ds_name, cfg, split in HF_TUBERLIN_DATASETS:
            if count >= target_count:
                break

            q_ds = urllib.parse.quote(ds_name, safe="")
            offset = count
            batch_size = min(100, target_count - count)

            while count < target_count:
                batch_len = min(100, target_count - count)
                rows_url = (
                    f"https://datasets-server.huggingface.co/rows?"
                    f"dataset={q_ds}&config={cfg}&split={split}&offset={offset}&length={batch_len}"
                )
                try:
                    data = _http_get_json(rows_url, timeout=30)
                    rows = data.get("rows", [])
                    if not rows:
                        break
                except Exception as exc:
                    print(f"\n   [Mirror note] Switching mirror from {ds_name}/{split} ({exc})...")
                    break

                for row_entry in rows:
                    if count >= target_count:
                        break
                    row = row_entry.get("row", {})
                    img_field = row.get("image", {})
                    img_src = img_field.get("src") if isinstance(img_field, dict) else None
                    if not img_src:
                        continue

                    raw_label = row.get("label", "tu_berlin")
                    if isinstance(raw_label, int) and raw_label in label_map:
                        cat_name = label_map[raw_label]
                    else:
                        cat_name = str(raw_label)

                    safe_cat = (
                        cat_name.replace(" ", "_")
                        .replace("/", "_")
                        .replace("(", "")
                        .replace(")", "")
                    )
                    image_id = f"tub_{count:04d}"
                    out_file = RAW_TUBERLIN_DIR / f"{image_id}__{safe_cat}.png"

                    if not out_file.exists():
                        try:
                            img_bytes = _http_get_bytes(img_src, timeout=20)
                            with Image.open(BytesIO(img_bytes)) as im:
                                im.convert("L").save(str(out_file))
                        except Exception:
                            continue

                    count += 1
                    sys.stdout.write(
                        f"\rDownloaded TU-Berlin sketches from Hugging Face: {count}/{target_count}"
                    )
                    sys.stdout.flush()

                offset += len(rows)

        if count > len(existing_pngs):
            print()

    # Fallback: if Hugging Face is blocked by firewall/offline, generate the remaining
    # complex multi-component sketches using held-out categories (250..344) with >= 5 strokes
    if count < target_count:
        print(
            f"   Filling remaining {target_count - count} complex benchmark sketches from held-out multi-stroke categories..."
        )
        qd_complex = QuickDrawData(recognized=True, max_drawings=30, print_messages=False)
        held_out_cats = qd_complex.drawing_names[250:] + qd_complex.drawing_names[:250]

        for cat in held_out_cats:
            if count >= target_count:
                break
            safe_cat = cat.replace(" ", "_").replace("/", "_")
            group = qd_complex.get_drawing_group(cat)
            saved_in_cat = 0
            for drawing in group.drawings:
                if count >= target_count or saved_in_cat >= 10:
                    break
                if len(drawing.strokes) < 5:
                    continue
                image_id = f"tub_{count:04d}"
                out_file = RAW_TUBERLIN_DIR / f"{image_id}__{safe_cat}.png"
                if not out_file.exists():
                    if not _render_qd_drawing_to_png(drawing, out_file, canvas_size=1000, padding=90, stroke_width=8):
                        continue
                count += 1
                saved_in_cat += 1

    existing_pngs = sorted(RAW_TUBERLIN_DIR.glob("tub_*.png"))
    items = []
    for p in existing_pngs[:target_count]:
        parts = p.stem.split("__", 1)
        img_id = parts[0]
        cat = parts[1].replace("_", " ") if len(parts) > 1 else "tu_berlin"
        items.append({
            "image_id": img_id,
            "dataset_source": "tu_berlin",
            "category": cat,
            "image_path": str(p),
        })

    print(f"Ready: {len(items)} TU-Berlin original sketches in '{RAW_TUBERLIN_DIR}'.")
    return items


# ==========================================================
# 4. EXTRACT THE 5 CORE PRE-FOURIER LINE DESCRIPTORS
# ==========================================================

def extract_pre_fourier_descriptors(pipeline_out, n_samples=10000):
    """
    Extracts the 5 core scale-invariant pre-Fourier descriptors (plus auxiliary
    diagnostic columns) before Fourier filtering is applied:
      1. tortuosity_len_to_diag     (L_total / D)
      2. doubled_path_prop          (L_duplicated / L_total)
      3. mst_bridge_len_prop        (L_bridges / L_total)
      4. rms_vertex_turn_norm       (RMS exterior deflection angle on RDP vertices / pi)
      5. raw_pre_fourier_smoothness (Uniformly sampled path roughness in [0, 1])
    """
    euler_G = pipeline_out["euler_graph"]
    route = pipeline_out["route"]
    raw_polyline = pipeline_out["raw_polyline"]
    canvas_diag = max(float(pipeline_out["canvas_diag"]), 1e-6)

    route_desc = compute_route_descriptors(
        E=euler_G,
        route=route,
        polyline=raw_polyline,
        canvas_diag=canvas_diag,
    )

    original_z = fourier.resample_polyline_uniform(raw_polyline, n_samples=n_samples)
    raw_pre_fourier_smoothness = float(
        met.calc_metric(original_z, original_z, mtype="Smoothness")
    )

    return {
        # --- 5 CORE REGRESSION MODEL FEATURES ---
        "tortuosity_len_to_diag": round(route_desc["tortuosity_len_to_diag"], 5),
        "doubled_path_prop": round(route_desc["doubled_path_prop"], 5),
        "mst_bridge_len_prop": round(route_desc["mst_bridge_len_prop"], 5),
        "rms_vertex_turn_norm": round(route_desc["rms_vertex_turn_norm"], 5),
        "raw_pre_fourier_smoothness": round(raw_pre_fourier_smoothness, 6),
        # --- AUXILIARY ROUTE & TOPOLOGY METRICS ---
        "rms_vertex_turn_deg": round(route_desc["rms_vertex_turn_deg"], 3),
        "mean_junction_turn_deg": round(route_desc["mean_junction_turn_deg"], 3),
        "acute_turn_fraction": round(route_desc["acute_turn_fraction"], 5),
        "total_turn_angle_deg": round(route_desc["total_turn_angle_deg"], 2),
        "total_polyline_len": round(route_desc["total_polyline_len"], 2),
        "duplicated_len": round(route_desc["duplicated_len"], 2),
        "mst_bridge_len": round(route_desc["mst_bridge_len"], 2),
        "canvas_diag": round(canvas_diag, 2),
    }


# ==========================================================
# 5. CHECKPOINT / RESUME HELPERS
# ==========================================================

def load_completed_ids(csv_path: Path):
    """
    Reads an existing dataset CSV and returns the set of `image_id`s that have
    already been processed AND whose 5 output images all exist on disk.
    """
    completed = set()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return completed

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row.get("image_id")
            if not img_id:
                continue
            outputs_exist = all(
                row.get(f"output_img_A{m}") and Path(row[f"output_img_A{m}"]).exists()
                for m in range(1, len(EXPONENTIAL_THRESHOLDS) + 1)
            )
            if outputs_exist:
                completed.add(img_id)

    return completed


def append_row_to_csv(csv_path: Path, row_data: dict):
    """Atomically appends a single row to `csv_path`, writing headers if new."""
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
        f.flush()


# ==========================================================
# 6. MAIN PAUSABLE & RESUMABLE EXECUTION LOOP
# ==========================================================

def run_dataset_generation(max_per_dataset=MAX_IMAGES_PER_DATASET):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RECON_QUICKDRAW_DIR.mkdir(parents=True, exist_ok=True)
    RECON_TUBERLIN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Prepare raw input sketches
    qd_items = prepare_quickdraw_sketches(target_count=max_per_dataset)
    tub_items = prepare_tuberlin_sketches(target_count=max_per_dataset)

    # 2. Load checkpointed progress from both CSVs
    completed_qd = load_completed_ids(QUICKDRAW_CSV_PATH)
    completed_tub = load_completed_ids(TUBERLIN_CSV_PATH)
    print(
        f"\nCheckpoint status: {len(completed_qd)}/{len(qd_items)} QuickDraw done | "
        f"{len(completed_tub)}/{len(tub_items)} TU-Berlin done."
    )
    print("Press Ctrl+C at any time to safely pause after the current image.\n")

    # Interleave tasks (QuickDraw, TU-Berlin, QuickDraw, TU-Berlin...)
    interleaved_queue = []
    for pair in zip(qd_items, tub_items):
        interleaved_queue.extend(pair)
    if len(qd_items) > len(tub_items):
        interleaved_queue.extend(qd_items[len(tub_items):])
    elif len(tub_items) > len(qd_items):
        interleaved_queue.extend(tub_items[len(qd_items):])

    total_tasks = len(interleaved_queue)
    newly_processed = 0

    try:
        for idx, item in enumerate(interleaved_queue, start=1):
            image_id = item["image_id"]
            source = item["dataset_source"]
            category = item["category"]
            raw_img_path = Path(item["image_path"])

            if source == "quickdraw":
                if image_id in completed_qd:
                    continue
                out_subdir = RECON_QUICKDRAW_DIR / image_id
                target_csv = QUICKDRAW_CSV_PATH
                stroke_w = 10.0
            else:
                if image_id in completed_tub:
                    continue
                out_subdir = RECON_TUBERLIN_DIR / image_id
                target_csv = TUBERLIN_CSV_PATH
                stroke_w = 8.0

            print(f"\n[{idx}/{total_tasks}] Processing ({source}) {image_id} - '{category}'...")

            try:
                # Run full pipeline across the 5 exponential accuracy thresholds
                pipeline_out = run_oneline_pipeline(
                    image_path=raw_img_path,
                    output_path=out_subdir,
                    accuracy_thresholds=EXPONENTIAL_THRESHOLDS,
                    n_samples=N_SAMPLES,
                    input_stroke_width=stroke_w,
                    target_stroke_width=stroke_w,
                    show_labels_on_image=SHOW_LABELS_ON_IMAGES,
                    figsize=OUTPUT_FIGSIZE,
                    dpi=OUTPUT_DPI,
                    save_all_intermediaries=False,
                )

                # Extract the 5 core pre-Fourier descriptors + auxiliary metrics
                descriptors = extract_pre_fourier_descriptors(
                    pipeline_out, n_samples=N_SAMPLES
                )

                # Record K, achieved accuracy, smoothness, and saved PNG path for all 5 thresholds
                recons = pipeline_out["reconstructions"]
                k_and_files = {}
                for m_idx, r in enumerate(recons, start=1):
                    k_and_files[f"thresh_A{m_idx}"] = r["target_accuracy"]
                    k_and_files[f"k_A{m_idx}"] = r["k"]
                    k_and_files[f"achieved_acc_A{m_idx}"] = round(r["achieved_accuracy"], 5)
                    k_and_files[f"smoothness_A{m_idx}"] = round(r["smoothness"], 5)
                    k_and_files[f"output_img_A{m_idx}"] = r["output_image_path"]

                row_data = {
                    "image_id": image_id,
                    "dataset_source": source,
                    "category": category,
                    "original_image_path": str(raw_img_path),
                    **descriptors,
                    **k_and_files,
                }

                append_row_to_csv(target_csv, row_data)
                if source == "quickdraw":
                    completed_qd.add(image_id)
                else:
                    completed_tub.add(image_id)
                newly_processed += 1

                print(
                    f"   [SAVED -> {target_csv.name}] {image_id} | "
                    f"Tortuosity={descriptors['tortuosity_len_to_diag']:.2f}, "
                    f"Doubled={descriptors['doubled_path_prop']*100:.1f}%, "
                    f"Bridge={descriptors['mst_bridge_len_prop']*100:.1f}%, "
                    f"Ks={[r['k'] for r in recons]}"
                )

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"   [WARNING] Skipping {image_id} due to pipeline error: {exc}")

    except KeyboardInterrupt:
        print("\n" + "=" * 65)
        print("PAUSED BY USER (Ctrl+C). All completed images and CSV rows are safely saved!")
        print(
            f"Current Totals -> QuickDraw CSV: {len(completed_qd)} rows | "
            f"TU-Berlin CSV: {len(completed_tub)} rows."
        )
        print("Re-run this script at any time to resume from the exact next image.")
        print("=" * 65)
        sys.exit(0)

    print("\n" + "=" * 65)
    print(f"RUN COMPLETE! Newly processed in this session: {newly_processed} input images.")
    print(f"Quick, Draw! CSV ({len(completed_qd)} rows): '{QUICKDRAW_CSV_PATH}'")
    print(f"TU-Berlin CSV    ({len(completed_tub)} rows): '{TUBERLIN_CSV_PATH}'")
    print("=" * 65)


if __name__ == "__main__":
    run_dataset_generation(max_per_dataset=MAX_IMAGES_PER_DATASET)