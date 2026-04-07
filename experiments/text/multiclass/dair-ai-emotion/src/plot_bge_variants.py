from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from paths import ARTIFACTS_DIR, PLOTS_DIR

os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS_DIR / "logs" / "mplconfig"))

import matplotlib.pyplot as plt

from io_utils import ensure_dir, read_json, write_json


VARIANT_RUNS = {
    "raw": "run-101-bge-base-en-v1-5-raw-meanpool",
    "l2": "run-102-bge-base-en-v1-5-l2-meanpool",
    "centered_l2": "run-103-bge-base-en-v1-5-centered-l2-meanpool",
}

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
PALETTE = ["#355070", "#6d597a", "#b56576", "#e56b6f", "#eaac8b", "#94d2bd"]


def _load_labels(experiment_root: Path) -> np.ndarray:
    return np.load(experiment_root / "data" / "processed" / "validation" / "labels.npy")


def _load_variant_paths(experiment_root: Path, run_name: str) -> tuple[np.ndarray, dict]:
    artifacts = read_json(experiment_root / "runs" / run_name / "artifacts.json")
    variant_key = artifacts["embedding_variant"]
    vectors = np.load(artifacts["validation_embeddings"][variant_key])
    return vectors, artifacts


def _pca_projection(vectors: np.ndarray) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vectors)
    return coords, pca


def _nonlinear_projection(vectors: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=25,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(vectors), "umap"
    except Exception:
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
        return reducer.fit_transform(vectors), "tsne"


def _scatter_plot(coords: np.ndarray, labels: np.ndarray, title: str, output_path: Path, xlab: str, ylab: str) -> None:
    plt.figure(figsize=(8, 6))
    for label, name, color in zip(range(len(LABEL_NAMES)), LABEL_NAMES, PALETTE):
        mask = labels == label
        plt.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.65, c=color, label=name)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _centroid_heatmap(matrix: np.ndarray, title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(LABEL_NAMES)), LABEL_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualization-only plots for multiclass BGE variants.")
    parser.add_argument("--experiment-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    experiment_root = args.experiment_root
    labels = _load_labels(experiment_root)
    plot_root = ensure_dir(PLOTS_DIR / "bge-variant-visuals")
    summary: dict[str, dict] = {}

    for variant_name, run_name in VARIANT_RUNS.items():
        vectors, artifacts = _load_variant_paths(experiment_root, run_name)
        metrics_summary = read_json(experiment_root / "runs" / run_name / "metrics" / "summary.json")
        variant_dir = ensure_dir(plot_root / variant_name)

        pca_coords, pca = _pca_projection(vectors)
        nonlinear_coords, nonlinear_method = _nonlinear_projection(vectors)

        _scatter_plot(
            pca_coords,
            labels,
            f"dair-ai/emotion {variant_name} PCA (validation only)",
            variant_dir / "pca-2d.png",
            "PC1",
            "PC2",
        )
        _scatter_plot(
            nonlinear_coords,
            labels,
            f"dair-ai/emotion {variant_name} {nonlinear_method.upper()} (validation only)",
            variant_dir / f"{nonlinear_method}-2d.png",
            f"{nonlinear_method.upper()}-1",
            f"{nonlinear_method.upper()}-2",
        )
        _centroid_heatmap(
            np.array(metrics_summary["geometry"]["centroid_cosine_distance_matrix"]),
            f"dair-ai/emotion {variant_name} centroid cosine distance",
            variant_dir / "centroid-cosine-heatmap.png",
        )

        class_centroids = {}
        for label, name in enumerate(LABEL_NAMES):
            class_centroids[name] = pca_coords[labels == label].mean(axis=0).tolist()

        summary[variant_name] = {
            "run_name": run_name,
            "embedding_variant": artifacts["embedding_variant"],
            "nonlinear_projection_method": nonlinear_method,
            "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
            "pca_class_centroids": class_centroids,
        }

    write_json(plot_root / "projection-summary.json", summary)


if __name__ == "__main__":
    main()
