from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EMBEDDING_DIM = 768 
LOG_SURFACE_CONST = math.log(2.0) + (EMBEDDING_DIM / 2.0) * math.log(math.pi) - math.lgamma(EMBEDDING_DIM / 2.0)

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_embeddings(embeddings_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    return embeddings, labels

def mean_centroids(embeddings: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    centroids = []
    for label in range(num_classes):
        class_embs = embeddings[labels == label]
        if len(class_embs) > 0:
            centroids.append(np.mean(class_embs, axis=0))
        else:
            centroids.append(np.zeros(embeddings.shape[1]))
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
    label_names: list[str],
    max_bins: int,
) -> dict[int, dict]:
    stats = {}
    for label_idx in range(len(label_names)):
        mask = labels == label_idx
        own = embeddings[mask]
        others = embeddings[~mask]
        if len(own) == 0:
            continue
        distances_self = np.linalg.norm(own - centroids[label_idx], axis=1)
        distances_others = np.linalg.norm(others - centroids[label_idx], axis=1)
        stats[label_idx] = {
            "count": int(len(own)),
            "bins": build_bins(distances_self, distances_others, n_bins=max_bins),
            "centroid": centroids[label_idx].tolist(),
        }
    return stats

def load_metadata(metadata_path: Path) -> dict:
    with open(metadata_path, "r") as f:
        return json.load(f)

def plot_density_all(stats: dict[int, dict], label_names: list[str], output_dir: Path) -> None:
    plt.figure(figsize=(12, 8))
    # Too many classes for a clear legend, so we'll only label a few or use colors
    cmap = matplotlib.colormaps['turbo']
    colors = cmap(np.linspace(0, 1, len(label_names)))
    for idx, label in enumerate(label_names):
        if idx not in stats: continue
        bins = stats[idx]["bins"]
        plt.plot(
            [b["midpoint"] for b in bins],
            [b["density_per_unit"] for b in bins],
            color=colors[idx],
            alpha=0.6,
            linewidth=1,
        )
    plt.xlabel("Distance from centroid")
    plt.ylabel("Density per unit")
    plt.title("All-class density decay (Animal Dataset)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "all-class-density-decay.png", dpi=160)
    plt.close()

def plot_overlap_all(stats: dict[int, dict], label_names: list[str], output_dir: Path) -> None:
    plt.figure(figsize=(12, 8))
    cmap = matplotlib.colormaps['turbo']
    colors = cmap(np.linspace(0, 1, len(label_names)))
    for idx, label in enumerate(label_names):
        if idx not in stats: continue
        bins = stats[idx]["bins"]
        plt.plot(
            [b["midpoint"] for b in bins],
            [b["overlap_ratio"] for b in bins],
            color=colors[idx],
            alpha=0.6,
            linewidth=1,
        )
    plt.xlabel("Distance from centroid")
    plt.ylabel("Overlap ratio")
    plt.title("All-class overlap ratios (Animal Dataset)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "all-class-overlap-ratio.png", dpi=160)
    plt.close()

def save_metrics(stats: dict[int, dict], label_names: list[str], output_dir: Path) -> None:
    metrics = {label_names[idx]: {"count": stats[idx]["count"], "bins": stats[idx]["bins"]} for idx in stats}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

def main() -> None:
    global EMBEDDING_DIM, LOG_SURFACE_CONST
    
    parser = argparse.ArgumentParser(description="Run all-class density/overlap overview for animal embeddings.")
    parser.add_argument("--max-bins", type=int, default=12)
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments/embeddings_field/image/animals"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    embeddings_path = args.experiment_root / "artifacts" / "embeddings" / "full_raw.npy"
    labels_path = args.experiment_root / "data" / "processed" / "full" / "labels.npy"
    metadata_path = args.experiment_root / "data" / "processed" / "metadata.json"
    
    embeddings, labels = load_embeddings(embeddings_path, labels_path)
    
    EMBEDDING_DIM = embeddings.shape[1]
    LOG_SURFACE_CONST = math.log(2.0) + (EMBEDDING_DIM / 2.0) * math.log(math.pi) - math.lgamma(EMBEDDING_DIM / 2.0)
    
    metadata = load_metadata(metadata_path)
    label_names = metadata["classes"]
    
    centroids = mean_centroids(embeddings, labels, len(label_names))

    stats = compute_class_stats(embeddings, labels, centroids, label_names, max_bins=args.max_bins)
    
    if args.output_root:
        output_root = args.output_root
    else:
        output_root = args.experiment_root / "reports" / "all_class"
    
    ensure_dir(output_root)
    plot_density_all(stats, label_names, output_root)
    plot_overlap_all(stats, label_names, output_root)
    save_metrics(stats, label_names, output_root)
    print(f"All-class overview saved in {output_root}")

if __name__ == "__main__":
    main()
