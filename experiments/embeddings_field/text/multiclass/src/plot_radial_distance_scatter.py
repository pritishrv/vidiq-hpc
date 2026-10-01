from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = EXPERIMENT_ROOT / "balanced_dataset"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "all_class" / "radial_distance"


def load_embeddings(embeddings_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    return embeddings, labels


def compute_stats(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    centroids = np.stack([embeddings[labels == idx].mean(axis=0) for idx in range(len(LABELS))])
    rng = np.random.default_rng(42)
    for idx, label in enumerate(LABELS):
        mask = labels == idx
        distances = np.linalg.norm(embeddings[mask] - centroids[idx], axis=1)
        bins = np.linspace(distances.min(), distances.max(), 8)
        stats[label] = {
            "count": int(mask.sum()),
            "min": float(distances.min()),
            "max": float(distances.max()),
            "percentiles": {
                "10": float(np.percentile(distances, 10)),
                "50": float(np.median(distances)),
                "90": float(np.percentile(distances, 90)),
            },
            "histogram": [int(np.sum((distances >= a) & (distances < b))) for a, b in zip(bins[:-1], bins[1:])],
        }
    return stats


def scatter_radial(embeddings: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    centroids = np.stack([embeddings[labels == idx].mean(axis=0) for idx in range(len(LABELS))])
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, label in enumerate(LABELS):
        mask = labels == idx
        distances = np.linalg.norm(embeddings[mask] - centroids[idx], axis=1)
        jitter = rng.normal(0, 0.08, size=len(distances))
        ax.scatter(np.full_like(distances, idx) + jitter, distances, s=12, alpha=0.4, label=label)
        ax.hlines(np.median(distances), idx - 0.3, idx + 0.3, colors="k", linewidth=1.5)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Distance to class centroid")
    ax.set_title("Radial distance distribution per emotion (balanced dataset)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "radial-distance-scatter.png", dpi=180)
    plt.close(fig)


def save_stats(stats: dict[str, dict], output_dir: Path) -> None:
    with (output_dir / "radial-distance-stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot radial distance scatter for balanced emotion embeddings.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--embeddings-name", default="dair_ai_emotion_train_balanced_bge-base-en-v1-5_raw.npy")
    parser.add_argument("--labels-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    embeddings_path = args.data_root / "artifacts" / "embeddings" / args.embeddings_name
    labels_path = args.labels_path or args.data_root / "data" / "processed" / "train" / "labels.npy"
    embeddings, labels = load_embeddings(embeddings_path, labels_path)
    stats = compute_stats(embeddings, labels)
    scatter_radial(embeddings, labels, args.output_root)
    save_stats(stats, args.output_root)
    print(f"Radial scatter saved to {args.output_root}")


if __name__ == "__main__":
    main()
