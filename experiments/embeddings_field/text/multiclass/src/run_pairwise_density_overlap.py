from __future__ import annotations

import argparse
import json
from itertools import combinations
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PAIRWISE_ROOT_DEFAULT = EXPERIMENT_ROOT / "pairwise"
PLOTS_CONFIG = PAIRWISE_ROOT_DEFAULT / "mplconfig"
PLOTS_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOTS_CONFIG))
VIDIQ_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DATA_ROOT = VIDIQ_ROOT / "experiments" / "text" / "multiclass" / "dair-ai-emotion"
DEFAULT_EMBEDDINGS_NAME = "dair_ai_emotion_train_bge-base-en-v1-5_raw.npy"
DEFAULT_LABELS_REL = Path("data") / "processed" / "train" / "labels.npy"

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_embeddings(embeddings_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    return embeddings, labels


def distance_stats(embeddings: np.ndarray, labels: np.ndarray, class_a: int, class_b: int) -> dict[str, object]:
    mask_a = labels == class_a
    mask_b = labels == class_b
    emb_a = embeddings[mask_a]
    emb_b = embeddings[mask_b]
    centroid_a = emb_a.mean(axis=0)
    centroid_b = emb_b.mean(axis=0)

    dist_a = np.linalg.norm(emb_a - centroid_a, axis=1)
    dist_b = np.linalg.norm(emb_b - centroid_b, axis=1)
    cross_ab = np.linalg.norm(emb_a - centroid_b, axis=1)
    cross_ba = np.linalg.norm(emb_b - centroid_a, axis=1)
    return {
        "centroids": {"a": centroid_a.tolist(), "b": centroid_b.tolist()},
        "distances": {
            "a": dist_a.tolist(),
            "b": dist_b.tolist(),
            "a_to_b": cross_ab.tolist(),
            "b_to_a": cross_ba.tolist(),
        },
    }


def build_bins(distances: np.ndarray, opposite: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    edges = np.percentile(distances, np.linspace(0, 100, n_bins + 1))
    bins = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (distances >= start) & (distances < end if end > start else distances >= end)
        opp_mask = (opposite >= start) & (opposite < end if end > start else opposite >= end)
        width = max(end - start, 1e-8)
        density = int(mask.sum())
        opp_count = int(opp_mask.sum())
        bins.append(
            {
                "start": float(start),
                "end": float(end),
                "midpoint": float((start + end) / 2),
                "density": density,
                "density_per_unit": density / width,
                "overlap_count": opp_count,
                "overlap_ratio": float(opp_count / (density + 1e-12)) if density else 0.0,
                "bin_width": float(width),
            }
        )
    return bins


def plot_density_decay(stats: dict[str, object], output_dir: Path, label_a: str, label_b: str) -> None:
    plt.figure(figsize=(9, 5))
    midpoints = [b["midpoint"] for b in stats["a_bins"]]
    plt.plot(midpoints, [b["density_per_unit"] for b in stats["a_bins"]], label=f"{label_a} density")
    plt.plot(midpoints, [b["density_per_unit"] for b in stats["b_bins"]], label=f"{label_b} density")
    plt.xlabel("Distance from centroid")
    plt.ylabel("Density per unit")
    plt.title(f"{label_a} vs {label_b} density decay")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "density-decay.png", dpi=160)
    plt.close()


def plot_overlap_volume(stats: dict[str, object], output_dir: Path, label_a: str, label_b: str) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    midpoints = [b["midpoint"] for b in stats["a_bins"]]
    ax1.plot(midpoints, [b["overlap_ratio"] for b in stats["a_bins"]], label=f"{label_a} overlap", color="#246eb5")
    ax1.plot(midpoints, [b["overlap_ratio"] for b in stats["b_bins"]], label=f"{label_b} overlap", color="#f79d65", linestyle="--")
    cum_a = np.cumsum([b["overlap_count"] for b in stats["a_bins"]])
    cum_b = np.cumsum([b["overlap_count"] for b in stats["b_bins"]])
    ax2.plot(midpoints, cum_a, label=f"{label_a} cum overlap", color="#246eb5", alpha=0.5)
    ax2.plot(midpoints, cum_b, label=f"{label_b} cum overlap", color="#f79d65", alpha=0.5, linestyle=":")
    ax1.set_xlabel("Distance from centroid")
    ax1.set_ylabel("Overlap ratio")
    ax2.set_ylabel("Cumulative overlap")
    ax1.set_title(f"{label_a} vs {label_b} overlap volume")
    ax1.grid(alpha=0.3)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize="small")
    fig.tight_layout()
    plt.savefig(output_dir / "overlap-volume.png", dpi=160)
    plt.close()


def run_pair(
    split: str,
    a: int,
    b: int,
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_root: Path,
    max_bins: int = 10,
) -> None:
    label_a = LABELS[a]
    label_b = LABELS[b]
    pair_dir = ensure_dir(output_root / f"{label_a}_vs_{label_b}")
    stats_path = pair_dir / "metrics.json"

    mask = (labels == a) | (labels == b)
    sel_embeddings = embeddings[mask]
    sel_labels = labels[mask]

    centroid_info = distance_stats(sel_embeddings, sel_labels, a, b)
    dist_a = np.array(centroid_info["distances"]["a"])
    dist_b = np.array(centroid_info["distances"]["b"])
    cross_ab = np.array(centroid_info["distances"]["a_to_b"])
    cross_ba = np.array(centroid_info["distances"]["b_to_a"])

    stats = {
        "labels": [label_a, label_b],
        "counts": {"a": int(np.sum(sel_labels == a)), "b": int(np.sum(sel_labels == b))},
        "a_bins": build_bins(dist_a, cross_ab, n_bins=max_bins),
        "b_bins": build_bins(dist_b, cross_ba, n_bins=max_bins),
    }

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    plot_density_decay(stats, pair_dir, label_a, label_b)
    plot_overlap_volume(stats, pair_dir, label_a, label_b)
    print(f"Pair {label_a} vs {label_b} complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pairwise density/overlap on dair-ai/emotion.")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--max-bins", type=int, default=10)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--embeddings-name", default=DEFAULT_EMBEDDINGS_NAME)
    parser.add_argument("--labels-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PAIRWISE_ROOT_DEFAULT)
    args = parser.parse_args()

    embeddings_path = args.data_root / "artifacts" / "embeddings" / args.embeddings_name
    labels_path = args.labels_path or args.data_root / DEFAULT_LABELS_REL
    embeddings, labels = load_embeddings(embeddings_path, labels_path)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    for a, b in combinations(range(len(LABELS)), 2):
        run_pair(args.split, a, b, embeddings, labels, output_root, max_bins=args.max_bins)


if __name__ == "__main__":
    main()
