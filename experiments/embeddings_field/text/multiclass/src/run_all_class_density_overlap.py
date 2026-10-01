from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ALL_CLASS_ROOT = EXPERIMENT_ROOT / "all_class"
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMBEDDING_DIM = 768
LOG_SURFACE_CONST = math.log(2.0) + (EMBEDDING_DIM / 2.0) * math.log(math.pi) - math.lgamma(EMBEDDING_DIM / 2.0)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_embeddings(embeddings_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    return embeddings, labels


def mean_centroids(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    centroids = []
    for label in range(len(LABELS)):
        centroids.append(np.mean(embeddings[labels == label], axis=0))
    return np.vstack(centroids)


def surface_log_volume(midpoint: float, width: float) -> float:
    r = max(midpoint, 1e-6)
    width = max(width, 1e-9)
    log_r_power = (EMBEDDING_DIM - 1) * math.log(r)
    return LOG_SURFACE_CONST + log_r_power + math.log(width)


def build_bins(distances: np.ndarray, opp_distances: np.ndarray, n_bins: int = 12):
    edges = np.percentile(distances, np.linspace(0, 100, n_bins + 1))
    bins = []
    for start, end in zip(edges[:-1], edges[1:]):
        width = max(end - start, 1e-8)
        mask = (distances >= start) & (distances < end if end > start else distances >= end)
        opp_mask = (opp_distances >= start) & (opp_distances < end if end > start else opp_distances >= end)
        midpoint = (start + end) / 2
        log_vol = surface_log_volume(midpoint, width)
        count = int(mask.sum())
        if count > 0:
            surface_log_density = math.log(count) - log_vol
        else:
            surface_log_density = float("-inf")
        bins.append(
            {
                "start": float(start),
                "end": float(end),
                "midpoint": float(midpoint),
                "density": count,
                "density_per_unit": float(count / width),
                "overlap_count": int(opp_mask.sum()),
                "overlap_ratio": float(opp_mask.sum() / (count + 1e-12)),
                "surface_log_density": surface_log_density,
            }
        )
    return bins


def compute_class_stats(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    max_bins: int,
) -> dict[str, dict]:
    stats = {}
    for label_idx in range(len(LABELS)):
        mask = labels == label_idx
        own = embeddings[mask]
        others = embeddings[~mask]
        distances_self = np.linalg.norm(own - centroids[label_idx], axis=1)
        distances_others = np.linalg.norm(others - centroids[label_idx], axis=1)
        stats[label_idx] = {
            "count": int(len(own)),
            "bins": build_bins(distances_self, distances_others, n_bins=max_bins),
            "centroid": centroids[label_idx].tolist(),
        }
    return stats


def plot_density_all(stats: dict[str, dict], output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for idx, label in enumerate(LABELS):
        bins = stats[idx]["bins"]
        plt.plot(
            [b["midpoint"] for b in bins],
            [b["density_per_unit"] for b in bins],
            label=label,
            linewidth=2,
        )
    plt.xlabel("Distance from centroid")
    plt.ylabel("Density per unit")
    plt.title("All-class density decay")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "all-class-density-decay.png", dpi=160)
    plt.close()


def plot_surface_density_all(stats: dict[str, dict], output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for idx, label in enumerate(LABELS):
        bins = stats[idx]["bins"]
        values = [b["surface_log_density"] for b in bins]
        values = [v if np.isfinite(v) else np.nan for v in values]
        plt.plot(
            [b["midpoint"] for b in bins],
            values,
            label=label,
            linewidth=2,
        )
    plt.xlabel("Distance from centroid")
    plt.ylabel("Log density per hyperspace volume")
    plt.title("All-class log density decay (768D)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "all-class-surface-density-decay.png", dpi=160)
    plt.close()


def plot_overlap_all(stats: dict[str, dict], output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for idx, label in enumerate(LABELS):
        bins = stats[idx]["bins"]
        plt.plot(
            [b["midpoint"] for b in bins],
            [b["overlap_ratio"] for b in bins],
            label=label,
            linewidth=2,
        )
    plt.xlabel("Distance from centroid")
    plt.ylabel("Overlap ratio")
    plt.title("All-class overlap ratios")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "all-class-overlap-ratio.png", dpi=160)
    plt.close()


def save_metrics(stats: dict[str, dict], output_dir: Path) -> None:
    metrics = {LABELS[idx]: {"count": stats[idx]["count"], "bins": stats[idx]["bins"]} for idx in range(len(LABELS))}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all-class density/overlap overview.")
    parser.add_argument("--max-bins", type=int, default=12)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[5] / "experiments" / "text" / "multiclass" / "dair-ai-emotion")
    parser.add_argument("--embeddings-name", default="dair_ai_emotion_train_bge-base-en-v1-5_raw.npy")
    parser.add_argument("--labels-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=ALL_CLASS_ROOT)
    args = parser.parse_args()

    embeddings_path = args.data_root / "artifacts" / "embeddings" / args.embeddings_name
    labels_path = args.labels_path or args.data_root / "data" / "processed" / "train" / "labels.npy"
    embeddings, labels = load_embeddings(embeddings_path, labels_path)
    centroids = mean_centroids(embeddings, labels)

    stats = compute_class_stats(embeddings, labels, centroids, max_bins=args.max_bins)
    output_root = ensure_dir(args.output_root)
    plot_density_all(stats, output_root)
    plot_surface_density_all(stats, output_root)
    plot_overlap_all(stats, output_root)
    save_metrics(stats, output_root)
    print(f"All-class overview saved in {output_root}")


if __name__ == "__main__":
    main()
