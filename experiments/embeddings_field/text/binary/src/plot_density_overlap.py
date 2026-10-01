from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Literal

BINARY_DIR = Path(__file__).resolve().parents[1]
PLOTS_CONFIG_DIR = BINARY_DIR / "plots" / "mplconfig"
PLOTS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOTS_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
BINARY_SST2_DIR = REPO_ROOT / "experiments" / "text" / "binary" / "sst2"
METRICS_DIR_DEFAULT = BINARY_DIR / "metrics"
PLOTS_DIR_DEFAULT = BINARY_DIR / "plots"

CLASS_COLORS = {0: "#d1495b", 1: "#00798c"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _format_value(value: object) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_stats(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_centroid(vectors: np.ndarray) -> np.ndarray:
    return vectors.mean(axis=0)


def compute_distances(
    vectors: np.ndarray,
    centroid: np.ndarray,
    metric: Literal["euclidean", "cosine"],
) -> np.ndarray:
    if metric == "euclidean":
        return np.linalg.norm(vectors - centroid, axis=1)
    centroid_norm = np.linalg.norm(centroid)
    vec_norms = np.linalg.norm(vectors, axis=1)
    denom = vec_norms * centroid_norm
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    cos_sim = (vectors @ centroid) / denom
    return 1.0 - np.clip(cos_sim, -1.0, 1.0)


def load_embeddings(split: str, variant: str, model_slug: str) -> np.ndarray:
    path = (
        BINARY_SST2_DIR
        / "artifacts"
        / "embeddings"
        / f"sst2_{split}_{model_slug}_{variant}.npy"
    )
    return np.load(path)


def load_labels(split: str) -> np.ndarray:
    return np.load(BINARY_SST2_DIR / "data" / "processed" / split / "labels.npy")


def sample_indices(total: int, limit: int) -> np.ndarray:
    if total <= limit:
        return np.arange(total)
    rng = np.random.default_rng(42)
    return rng.choice(total, size=limit, replace=False)


def plot_density_curve(stats: dict[str, object], variant: str, output_dir: Path) -> None:
    pos_bins = stats["pos_bins"]
    neg_bins = stats["neg_bins"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        [bin["midpoint"] for bin in pos_bins],
        [bin["density"] for bin in pos_bins],
        label="positive density",
        color=CLASS_COLORS[1],
    )
    ax.plot(
        [bin["midpoint"] for bin in neg_bins],
        [bin["density"] for bin in neg_bins],
        label="negative density",
        color=CLASS_COLORS[0],
    )
    ax.plot(
        [bin["midpoint"] for bin in pos_bins],
        [bin["opposite_count"] for bin in pos_bins],
        linestyle="--",
        color=CLASS_COLORS[1],
        label="positive opposite count",
    )
    ax.plot(
        [bin["midpoint"] for bin in neg_bins],
        [bin["opposite_count"] for bin in neg_bins],
        linestyle="--",
        color=CLASS_COLORS[0],
        label="negative opposite count",
    )
    ax.set_title(f"{variant} density and overlap")
    ax.set_xlabel("Distance to own centroid")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{variant}-density-overlap.png", dpi=160)
    plt.close(fig)


def plot_overlap_histogram(stats: dict[str, object], variant: str, output_dir: Path) -> None:
    bins = np.arange(1, len(stats["pos_bins"]) + 1)
    pos_density = np.array([bin["density"] for bin in stats["pos_bins"]])
    neg_density = np.array([bin["density"] for bin in stats["neg_bins"]])
    pos_overlap = np.array([bin["opposite_count"] for bin in stats["pos_bins"]])
    neg_overlap = np.array([bin["opposite_count"] for bin in stats["neg_bins"]])
    width = 0.3
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        bins - width / 2,
        pos_density,
        width,
        label="positive density",
        color=CLASS_COLORS[1],
        alpha=0.8,
    )
    ax.bar(
        bins + width / 2,
        neg_density,
        width,
        label="negative density",
        color=CLASS_COLORS[0],
        alpha=0.8,
    )
    ax.plot(bins, pos_overlap, label="positive opposite", linestyle="--", color=CLASS_COLORS[1])
    ax.plot(bins, neg_overlap, label="negative opposite", linestyle="--", color=CLASS_COLORS[0])
    ax.set_title(f"{variant} overlap histogram")
    ax.set_xlabel("Distance bin")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{variant}-overlap-hist.png", dpi=160)
    plt.close(fig)


def plot_distance_histogram(
    pos_dist: np.ndarray, neg_dist: np.ndarray, pos_to_neg: np.ndarray, neg_to_pos: np.ndarray, variant: str, output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pos_dist, bins=40, alpha=0.6, label="positive to own", density=True)
    ax.hist(pos_to_neg, bins=40, alpha=0.3, label="positive to negative", density=True)
    ax.hist(neg_dist, bins=40, alpha=0.4, label="negative to own", density=True)
    ax.hist(neg_to_pos, bins=40, alpha=0.2, label="negative to positive", density=True)
    ax.set_title(f"{variant} distance distributions")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{variant}-distance-hist.png", dpi=160)
    plt.close(fig)


def plot_overlap_heatmap(
    primary: np.ndarray,
    secondary: np.ndarray,
    label: str,
    variant_dir: Path,
    variant: str,
) -> None:
    heat, xedges, yedges = np.histogram2d(primary, secondary, bins=30)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        np.log1p(heat.T),
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    ax.set_title(f"{variant} {label} distance overlap heatmap")
    ax.set_xlabel("Distance to own centroid")
    ax.set_ylabel("Distance to opposite centroid")
    fig.colorbar(im, label="log count")
    fig.tight_layout()
    fig.savefig(variant_dir / f"{variant}-{label}-heatmap.png", dpi=160)
    plt.close(fig)


def project_pca(embeddings: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, str]:
    centered = embeddings - embeddings.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components].T
    coords = centered @ components
    return coords, "PCA"


def plot_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    variant_dir: Path,
    variant: str,
    label: str,
    stats: dict[str, object],
    index_map: dict[int, int],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.plasma
    norm = plt.Normalize(distances.min(), distances.max())
    for class_id, name in [(0, "negative"), (1, "positive")]:
        mask = labels == class_id
        scatter = ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=distances[mask],
            cmap=cmap,
            norm=norm,
            s=24,
            alpha=0.7,
            edgecolor="w",
            linewidth=0.15,
            label=name,
        )
    ax.legend()
    title = f"{variant} {label} projection"
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="Distance to own centroid")

    for class_id, info in [("positive", stats["pos_furthest"]), ("negative", stats["neg_furthest"])]:
        idx = info["index"]
        if idx is None:
            continue
        slot = index_map.get(idx)
        if slot is None:
            continue
        loc = coords[slot]
        ax.scatter(
            loc[0],
            loc[1],
            marker="X",
            color="k",
            s=90,
            edgecolor="yellow",
            linewidth=1.5,
            label=f"{class_id} furthest",
        )
        ax.annotate(
            f"{class_id} furthest",
            (loc[0], loc[1]),
            textcoords="offset points",
            xytext=(5, -8),
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(variant_dir / f"{variant}-{label}-scatter.png", dpi=160)
    plt.close(fig)


def write_summary(stats: dict[str, object], variant: str, output_dir: Path) -> None:
    summary_path = output_dir / f"{variant}-summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Variant: {variant}\n")
        f.write(f"Split: {stats['split']}\n")
        f.write(f"Metric: {stats['metric']}\n")
        f.write(f"Positive overlap ratio: {stats['pos_overlap_ratio']:.4f}\n")
        f.write(f"Negative overlap ratio: {stats['neg_overlap_ratio']:.4f}\n")
        f.write(
            f"Positive furthest dist: {_format_value(stats['pos_furthest']['distance'])} idx={_format_value(stats['pos_furthest']['index'])}\n"
        )
        f.write(
            f"Negative furthest dist: {_format_value(stats['neg_furthest']['distance'])} idx={_format_value(stats['neg_furthest']['index'])}\n"
        )
        f.write(f"Bins: {len(stats['pos_bins'])}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize density/overlap results.")
    parser.add_argument(
        "--variants", nargs="+", default=["raw"]
    )
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--model-slug", default="bge-base-en-v1-5")
    parser.add_argument(
        "--metrics-dir", type=Path, default=METRICS_DIR_DEFAULT
    )
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR_DEFAULT)
    parser.add_argument("--max-projection-points", type=int, default=4000)
    args = parser.parse_args()

    for variant in args.variants:
        stats_path = args.metrics_dir / f"density_overlap_{args.split}_{variant}.json"
        stats = load_stats(stats_path)
        variant_dir = ensure_dir(args.plots_dir / args.split / variant)
        plot_density_curve(stats, variant, variant_dir)
        plot_overlap_histogram(stats, variant, variant_dir)

        embeddings = load_embeddings(args.split, variant, args.model_slug)
        labels = load_labels(args.split)
        positive_indices = np.where(labels == 1)[0]
        negative_indices = np.where(labels == 0)[0]
        positives = embeddings[positive_indices]
        negatives = embeddings[negative_indices]

        pos_centroid = compute_centroid(positives)
        neg_centroid = compute_centroid(negatives)

        pos_dist = compute_distances(positives, pos_centroid, stats["metric"])  # type: ignore[arg-type]
        neg_dist = compute_distances(negatives, neg_centroid, stats["metric"])  # type: ignore[arg-type]
        pos_to_neg = compute_distances(positives, neg_centroid, stats["metric"])  # type: ignore[arg-type]
        neg_to_pos = compute_distances(negatives, pos_centroid, stats["metric"])  # type: ignore[arg-type]

        plot_distance_histogram(pos_dist, neg_dist, pos_to_neg, neg_to_pos, variant, variant_dir)
        plot_overlap_heatmap(pos_dist, pos_to_neg, "positive", variant_dir, variant)
        plot_overlap_heatmap(neg_dist, neg_to_pos, "negative", variant_dir, variant)

        # projection scatter
        total = embeddings.shape[0]
        subset_idx = sample_indices(total, args.max_projection_points)
        sampled_embeddings = embeddings[subset_idx]
        sampled_labels = labels[subset_idx]
        own_distances = np.empty(labels.shape[0], dtype=float)
        own_distances[positive_indices] = pos_dist
        own_distances[negative_indices] = neg_dist
        sampled_own_distances = own_distances[subset_idx]
        coords, method = project_pca(sampled_embeddings)
        index_map = {int(global_idx): int(pos) for pos, global_idx in enumerate(subset_idx)}
        plot_scatter(
            coords,
            sampled_labels,
            sampled_own_distances,
            variant_dir,
            variant,
            method,
            stats,
            index_map,
        )

        write_summary(stats, variant, variant_dir)


if __name__ == "__main__":
    main()
