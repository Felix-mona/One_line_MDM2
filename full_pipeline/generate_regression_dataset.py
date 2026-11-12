"""
generate_regression_dataset.py
End-to-end training data generator for the Fourier Cutoff Regression Model:
1. Generates 1,000 diverse primitive black-on-white line sketches via AI (with resume support).
2. Runs `run_oneline_pipeline` across 5 exponential accuracy thresholds:
   [0.0025, 0.006, 0.015, 0.038, 0.095].
3. Extracts 18 scale-invariant pre-Fourier line descriptors from the graph & polyline stages.
4. Uses a Multimodal Vision AI Judge to score & rank the 5 output drawings (1-10)
   and compute both the winning threshold and a continuous weighted target threshold.
5. Saves all features, K values, and judge scores incrementally to a CSV file.
"""

import os
import csv
import time
import json
import itertools
from pathlib import Path
import numpy as np
import networkx as nx
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Import existing pipeline modules
from full_pipeline import run_oneline_pipeline
from route_mapping import compute_polyline_length, compute_total_turn_angle
import Fourier_reconsctruction as fourier
import Metric as met


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

NUM_IMAGES = 1000
EXPONENTIAL_THRESHOLDS = [0.0025, 0.006, 0.015, 0.038, 0.095]
N_SAMPLES = 10000

DATASET_DIR = Path("dataset_1000")
RAW_SKETCHES_DIR = DATASET_DIR / "01_raw_sketches"
RECONSTRUCTIONS_DIR = DATASET_DIR / "02_reconstructions"
CSV_OUTPUT_PATH = DATASET_DIR / "regression_training_data.csv"

# Model identifiers (Google GenAI SDK)
IMAGE_GEN_MODEL = "imagen-3.0-generate-002"
VISION_JUDGE_MODEL = "gemini-2.5-flash"


# ==========================================================
# 2. 1,000 DIVERSE PRIMITIVE SKETCH PROMPT GENERATOR
# ==========================================================

SUBJECT_CATEGORIES = {
    "animals": [
        "owl", "fox", "elephant", "seahorse", "tortoise", "hummingbird", "stag",
        "octopus", "penguin", "chameleon", "wolf", "butterfly", "flamingo", "crab",
        "hedgehog", "swan", "dolphin", "koala", "camel", "dragonfly", "squirrel",
        "toucan", "jellyfish", "rabbit", "lion",
    ],
    "botanical": [
        "sunflower", "monstera leaf", "bonsai tree", "lotus flower", "pinecone",
        "cactus in pot", "oak leaf", "rose blossom", "fern frond", "mushroom cluster",
        "tulip", "palm tree", "acorn", "dandelion", "hibiscus", "bamboo stalk",
        "wheat ear", "maple leaf", "orchid", "succulent rosette",
    ],
    "objects_and_tools": [
        "vintage key", "hourglass", "pocket watch", "teacup and saucer", "teapot",
        "pair of scissors", "lightbulb", "table lamp", "umbrella", "eyeglasses",
        "watering can", "padlock", "compass rose", "feather quill", "candle in holder",
        "coffee mug", "wine glass", "magnifying glass", "lantern", "camera",
    ],
    "vehicles_and_architecture": [
        "sailboat", "hot air balloon", "vintage bicycle", "lighthouse", "windmill",
        "steam locomotive", "paper airplane", "rocket ship", "biplane", "castle tower",
        "suspension bridge", "pagoda", "igloo", "canoe with paddle", "tractor",
        "submarine", "scooter", "helicopter", "camping tent", "barn",
    ],
    "music_and_symbols": [
        "acoustic guitar", "violin", "grand piano", "saxophone", "harp",
        "trumpet", "treble clef", "french horn", "drum kit", "banjo",
        "anchor", "crown", "origami crane", "chess knight", "infinity knot",
        "celtic triangle", "crescent moon and star", "lightning bolt", "heart emblem", "snowflake",
    ],
}

STYLE_MODIFIERS = [
    "minimalist icon with bold clean curves and 1 disconnected inner detail",
    "geometric angular sketch with sharp corners and straight segments",
    "organic flowing contour drawing with smooth curves and 2 internal loops",
    "structured line drawing with 3 distinct disconnected sub-parts",
    "detailed primitive sketch combining sharp geometric edges and curved petals/contours",
    "simple continuous-style silhouette with minimal internal junctions",
    "symmetrical emblem sketch with concentric inner shapes",
    "asymmetrical dynamic sketch with fine terminal branches",
    "blocky technical-style outline with crisp 90-degree and 60-degree corners",
    "loose expressive doodle with multiple intersecting loops",
]


def build_1000_prompts(limit=1000):
    """Generates 1,000 unique (subject, category, style_modifier, full_prompt) tuples."""
    all_subjects = [
        (cat, subj)
        for cat, subjs in SUBJECT_CATEGORIES.items()
        for subj in subjs
    ]  # 100 unique subjects
    combinations = list(itertools.product(all_subjects, STYLE_MODIFIERS))  # 100 * 10 = 1,000

    prompts = []
    for idx, ((category, subject), style) in enumerate(combinations[:limit]):
        full_prompt = (
            f"A flat 2D primitive black line sketch of a {subject} on a pure solid white background. "
            f"Style: {style}. Uniform black marker stroke thickness, high contrast, "
            f"no shading, no gradients, no greyscale fill, no text, centered with white margin."
        )
        prompts.append({
            "image_id": f"sketch_{idx:04d}",
            "category": category,
            "subject": subject,
            "style_modifier": style,
            "prompt": full_prompt,
        })
    return prompts


# ==========================================================
# 3. IMAGE GENERATION (WITH PROCEDURAL FALLBACK)
# ==========================================================

def generate_primitive_sketch(client, prompt_text, save_path: Path):
    """
    Calls the Imagen model to generate a binary primitive line sketch and saves it.
    Skips generation if `save_path` already exists on disk.
    """
    if save_path.exists():
        return True

    save_path.parent.mkdir(parents=True, exist_ok=True)
    response = client.models.generate_images(
        model=IMAGE_GEN_MODEL,
        prompt=prompt_text,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            output_mime_type="image/png",
            aspect_ratio="1:1",
            person_generation="DONT_ALLOW",
        ),
    )

    for generated_image in response.generated_images:
        generated_image.image.save(str(save_path))
        return True

    return False


# ==========================================================
# 4. PRE-FOURIER LINE DESCRIPTOR EXTRACTION
# ==========================================================

def extract_pre_fourier_descriptors(pipeline_out, n_samples=10000):
    """
    Extracts 18 scale-invariant topological and geometric features of the line
    BEFORE Fourier filtering is applied.
    """
    base_G = pipeline_out["base_graph"]
    bridged_G = pipeline_out["bridged_graph"]
    euler_G = pipeline_out["euler_graph"]
    route = pipeline_out["route"]
    raw_polyline = pipeline_out["raw_polyline"]
    canvas_diag = max(float(pipeline_out["canvas_diag"]), 1e-6)
    total_turn_deg = float(pipeline_out["total_turn_angle_deg"])

    # 1. Base & Bridged Graph Topology
    num_base_nodes = base_G.number_of_nodes()
    num_base_edges = base_G.number_of_edges()
    num_components = nx.number_connected_components(base_G)

    num_bridged_nodes = max(1, bridged_G.number_of_nodes())
    num_bridged_edges = max(1, bridged_G.number_of_edges())

    # Node type ratios (corners vs junctions vs endpoints)
    num_corners = sum(1 for _, d in bridged_G.nodes(data=True) if d.get("kind") == "corner")
    num_junctions = sum(1 for _, d in bridged_G.nodes(data=True) if d.get("kind") == "junction")
    corner_node_ratio = num_corners / num_bridged_nodes
    junction_node_ratio = num_junctions / num_bridged_nodes

    # 2. MST Bridge & CPP Duplication Overhead
    bridge_edges = [d for _, _, d in bridged_G.edges(data=True) if d.get("bridge", False)]
    num_mst_bridges = len(bridge_edges)
    mst_bridge_edge_ratio = num_mst_bridges / num_bridged_edges
    total_bridge_len = sum(compute_polyline_length(d["pts"]) for d in bridge_edges)

    num_euler_edges = max(1, euler_G.number_of_edges())
    num_duplicated_edges = sum(1 for _, _, _, d in euler_G.edges(keys=True, data=True) if d.get("duplicated", False))
    cpp_duplication_ratio = num_duplicated_edges / num_euler_edges

    # 3. Continuous Polyline Geometry & Angular Complexity
    total_len = max(compute_polyline_length(raw_polyline), 1e-6)
    tortuosity_len_to_diag = total_len / canvas_diag
    mst_bridge_len_ratio = total_bridge_len / total_len

    turn_density_deg_per_px = total_turn_deg / total_len
    turn_density_norm = total_turn_deg / (total_len / canvas_diag)  # Scale-invariant angular density
    mean_turn_per_step = total_turn_deg / max(1, len(route))

    # Segment length statistics
    diffs = np.diff(raw_polyline, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    seg_len_mean_norm = float(np.mean(seg_lens)) / canvas_diag if len(seg_lens) > 0 else 0.0
    seg_len_std_norm = float(np.std(seg_lens)) / canvas_diag if len(seg_lens) > 0 else 0.0
    seg_len_cv = float(np.std(seg_lens) / max(np.mean(seg_lens), 1e-9)) if len(seg_lens) > 0 else 0.0

    # 4. Pre-Fourier Resampled Roughness & Multi-Band Spectral Energy Priors (N = 10,000)
    original_z = fourier.resample_polyline_uniform(raw_polyline, n_samples=n_samples)
    raw_pre_fourier_smoothness = float(met.calc_metric(original_z, original_z, mtype="Smoothness"))

    coeffs, _ = fourier.forward_fourier(original_z)
    power = np.abs(coeffs) ** 2
    power[0] = 0.0  # Ignore DC translation offset
    total_power = max(float(np.sum(power)), 1e-12)

    def band_energy_ratio(k_band):
        k_b = min(int(k_band), (n_samples - 1) // 2)
        band_power = float(np.sum(power[1 : k_b + 1]) + np.sum(power[-k_b:]))
        return band_power / total_power

    # Fixed harmonic band ratios give the regression model a direct readout of
    # how much energy lives in coarse (K<=25), medium (K<=100), and fine (K<=300) frequencies
    spectral_energy_k25 = band_energy_ratio(25)
    spectral_energy_k100 = band_energy_ratio(100)
    spectral_energy_k300 = band_energy_ratio(300)



    return {
        "canvas_diag": round(canvas_diag, 3),
        "num_base_nodes": num_base_nodes,
        "num_base_edges": num_base_edges,
        "num_components": num_components,
        "corner_node_ratio": round(corner_node_ratio, 5),
        "junction_node_ratio": round(junction_node_ratio, 5),
        "num_mst_bridges": num_mst_bridges,
        "mst_bridge_edge_ratio": round(mst_bridge_edge_ratio, 5),
        "mst_bridge_len_ratio": round(mst_bridge_len_ratio, 5),
        "cpp_duplication_ratio": round(cpp_duplication_ratio, 5),
        "total_polyline_len": round(total_len, 3),
        "tortuosity_len_to_diag": round(tortuosity_len_to_diag, 5),
        "total_turn_angle_deg": round(total_turn_deg, 3),
        "turn_density_deg_per_px": round(turn_density_deg_per_px, 5),
        "turn_density_norm": round(turn_density_norm, 3),
        "mean_turn_per_step_deg": round(mean_turn_per_step, 3),
        "seg_len_mean_norm": round(seg_len_mean_norm, 6),
        "seg_len_cv": round(seg_len_cv, 5),
        "raw_pre_fourier_smoothness": round(raw_pre_fourier_smoothness, 6),
        "spectral_energy_k25": round(spectral_energy_k25, 6),
        "spectral_energy_k100": round(spectral_energy_k100, 6),
         "spectral_energy_k300": round(spectral_energy_k300, 6),
    }


# ==========================================================
# 5. MULTIMODAL VISION AI JUDGE (STRUCTURED JSON OUTPUT)
# ==========================================================

class JudgeEvaluation(BaseModel):
    score_1: float = Field(description="Score from 1.0 to 10.0 for Candidate Image 1")
    score_2: float = Field(description="Score from 1.0 to 10.0 for Candidate Image 2")
    score_3: float = Field(description="Score from 1.0 to 10.0 for Candidate Image 3")
    score_4: float = Field(description="Score from 1.0 to 10.0 for Candidate Image 4")
    score_5: float = Field(description="Score from 1.0 to 10.0 for Candidate Image 5")
    best_candidate_index: int = Field(description="Integer 1 to 5 indicating the single best candidate")
    reasoning: str = Field(description="Brief 1-sentence explanation of why the winner balances accuracy and smoothness best")


def judge_reconstructions_with_ai(client, original_img_path, reconstructions):
    """
    Sends the original sketch and the 5 Fourier reconstructions (Candidate 1..5)
    to Gemini Vision to rate each on a 1-10 aesthetic scale and pick the winner.
    """
    contents = [
        "You are an expert evaluator of algorithmic continuous single-line drawings. "
        "The first image is the original input sketch. Following it are 5 single-line Fourier "
        "reconstructions (Candidate 1 through Candidate 5) ordered from highest frequency detail "
        "(Candidate 1) to strongest smoothing (Candidate 5).\n\n"
        "Rate each Candidate from 1.0 to 10.0 based on the optimal aesthetic trade-off between:\n"
        "1. Recognisability & geometric fidelity to the original sketch (penalise over-smoothed blobs).\n"
        "2. Fluid, organic single-line continuity (penalise harsh straight-line bridge jumps or jagged pixel noise).\n"
        "3. Parsimony (if two candidates look equally recognisable and clean, prefer the smoother one with fewer frequencies).\n\n"
        "Original Input Sketch:",
        Image.open(original_img_path),
    ]

    for idx, res in enumerate(reconstructions, start=1):
        contents.append(f"Candidate {idx}:")
        contents.append(Image.open(res["output_image_path"]))

    response = client.models.generate_content(
        model=VISION_JUDGE_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JudgeEvaluation,
            temperature=0.1,
        ),
    )

    eval_data = json.loads(response.text)
    scores = [
        float(eval_data["score_1"]),
        float(eval_data["score_2"]),
        float(eval_data["score_3"]),
        float(eval_data["score_4"]),
        float(eval_data["score_5"]),
    ]
    thresholds = [float(r["target_accuracy"]) for r in reconstructions]
    ks = [int(r["k"]) for r in reconstructions]

    # Discrete winner
    best_idx = int(np.argmax(scores))
    best_thresh = thresholds[best_idx]
    best_k = ks[best_idx]

    # Softmax-weighted continuous optimal threshold (in log-space so exponential steps interpolate cleanly)
    scores_arr = np.asarray(scores, dtype=float)
    exp_weights = np.exp((scores_arr - np.max(scores_arr)) * 1.5)
    weights = exp_weights / np.sum(exp_weights)
    weighted_log_thresh = np.sum(weights * np.log(thresholds))
    continuous_optimal_thresh = float(np.exp(weighted_log_thresh))

    return {
        "ai_score_A1": scores[0],
        "ai_score_A2": scores[1],
        "ai_score_A3": scores[2],
        "ai_score_A4": scores[3],
        "ai_score_A5": scores[4],
        "best_candidate_idx": best_idx + 1,
        "best_accuracy_thresh": best_thresh,
        "continuous_optimal_thresh": round(continuous_optimal_thresh, 6),
        "optimal_k": best_k,
        "ai_reasoning": eval_data.get("reasoning", ""),
    }


# ==========================================================
# 6. MAIN DATASET GENERATION LOOP
# ==========================================================

def build_regression_dataset(num_images=NUM_IMAGES):
    RAW_SKETCHES_DIR.mkdir(parents=True, exist_ok=True)
    RECONSTRUCTIONS_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Please set the GEMINI_API_KEY environment variable before running.")

    client = genai.Client(api_key=api_key)
    prompts = build_1000_prompts(limit=num_images)

    # Check already processed image_ids so we can resume cleanly
    completed_ids = set()
    if CSV_OUTPUT_PATH.exists():
        with open(CSV_OUTPUT_PATH, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_ids.add(row["image_id"])
        print(f"Resuming dataset generation: {len(completed_ids)} images already recorded in '{CSV_OUTPUT_PATH}'.")

    csv_file_exists = CSV_OUTPUT_PATH.exists() and CSV_OUTPUT_PATH.stat().st_size > 0

    for idx, item in enumerate(prompts, start=1):
        image_id = item["image_id"]
        if image_id in completed_ids:
            continue

        print(f"\n[{idx}/{num_images}] Processing {image_id} ({item['category']} -> {item['subject']})...")
        raw_img_path = RAW_SKETCHES_DIR / f"{image_id}.png"
        out_subdir = RECONSTRUCTIONS_DIR / image_id

        try:
            # Step A: Generate Primitive Sketch (or load if already on disk)
            generate_primitive_sketch(client, item["prompt"], raw_img_path)

            # Step B: Run the One-Line Pipeline across the 5 Exponential Thresholds
            # Note: show_labels_on_image=False prevents the Vision AI from reading K/threshold numbers on the canvas
            pipeline_out = run_oneline_pipeline(
                image_path=raw_img_path,
                output_path=out_subdir,
                accuracy_thresholds=EXPONENTIAL_THRESHOLDS,
                n_samples=N_SAMPLES,
                show_labels_on_image=False,
                save_all_intermediaries=False,
            )

            # Step C: Extract Pre-Fourier Line Descriptors
            descriptors = extract_pre_fourier_descriptors(pipeline_out, n_samples=N_SAMPLES)

            # Record the 5 K-values and smoothness scores found for the 5 thresholds
            recons = pipeline_out["reconstructions"]
            k_and_metrics = {}
            for m_idx, r in enumerate(recons, start=1):
                k_and_metrics[f"thresh_A{m_idx}"] = r["target_accuracy"]
                k_and_metrics[f"k_A{m_idx}"] = r["k"]
                k_and_metrics[f"achieved_acc_A{m_idx}"] = round(r["achieved_accuracy"], 5)
                k_and_metrics[f"smoothness_A{m_idx}"] = round(r["smoothness"], 5)

            # Step D: Ask Multimodal Vision AI to Judge the 5 Reconstructions
            judge_results = judge_reconstructions_with_ai(client, raw_img_path, recons)

            # Combine all columns into a single flat row
            row_data = {
                "image_id": image_id,
                "category": item["category"],
                "subject": item["subject"],
                "style_modifier": item["style_modifier"],
                **descriptors,
                **k_and_metrics,
                **judge_results,
            }

            # Step E: Append immediately to CSV
            with open(CSV_OUTPUT_PATH, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
                if not csv_file_exists:
                    writer.writeheader()
                    csv_file_exists = True
                writer.writerow(row_data)

            print(
                f"   Saved row for {image_id} | Winner: Candidate {judge_results['best_candidate_idx']} "
                f"(Acc={judge_results['best_accuracy_thresh']}, K={judge_results['optimal_k']}, "
                f"Continuous Target={judge_results['continuous_optimal_thresh']})"
            )

        except Exception as exc:
            print(f"   [ERROR] Skipping {image_id} due to error: {exc}")
            time.sleep(2.0)

    print(f"\nDataset generation complete! Training CSV saved at: '{CSV_OUTPUT_PATH}'")


if __name__ == "__main__":
    build_regression_dataset(num_images=NUM_IMAGES)