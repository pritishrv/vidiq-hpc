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

PALETTE = ["#355070", "#6d597a", "#b56576", "#e56b6f", "#eaac8b", "#94d2bd"]


def _resolve_bridge_artifact_path(bridge_run_dir: Path, rel_path: str) -> Path:
    return (bridge_run_dir / rel_path).resolve()


def _source_run_metadata(experiment_root: Path, bridge_run: str) -> dict:
    bridge_dir = experiment_root / "runs" / bridge_run
    bridge_artifacts = read_json(bridge_dir / "artifacts.json")
    return read_json(_resolve_bridge_artifact_path(bridge_dir, bridge_artifacts["run_metadata"]))


def _variant_runs(config: dict) -> dict[str, str]:
    start_index = int(config["run_start_index"])
    model_slug = config["model"]["slug"]
    runs: dict[str, str] = {}
    for offset, variant_cfg in enumerate(config["variants"]):
        variant_name = str(variant_cfg["name"])
        variant_key = str(variant_cfg["embedding_key"])
        runs[variant_key] = f"run-{start_index + offset:03d}-{model_slug}-bge-parity-{variant_name}"
    return runs


def _load_eval_labels(experiment_root: Path, config: dict) -> np.ndarray:
    metadata_path = experiment_root / "artifacts" / "embeddings" / f"{config['embedding_export']['base_slug']}_metadata.json"
    metadata = read_json(metadata_path)
    counts = metadata["split_provenance"]["eval_label_counts"]
    if not any(counts):
        raise ValueError("Missing eval label counts in Qwen parity metadata.")
    source_metadata = _source_run_metadata(experiment_root, config["bridge_run"])
    label_to_id = {str(label): int(idx) for label, idx in source_metadata["dataset"]["label_to_id"].items()}
    csv_path = experiment_root / "data" / "raw" / "balanced_emotions_6classes.csv"
    raw_labels: list[int] = []
    import csv
    from sklearn.model_selection import StratifiedShuffleSplit

    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            text = str(row.get("cleaned_text", "") or row.get("sentence", "") or row.get("text", "")).strip()
            label = str(row.get("emotion", "")).strip()
            if not text or not label:
                continue
            raw_labels.append(label_to_id[label])
    labels = np.array(raw_labels, dtype=np.int64)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(source_metadata["args"]["test_size"]),
        random_state=int(source_metadata["args"]["seed"]),
    )
    _, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return labels[test_idx]


def _load_variant_vectors(experiment_root: Path, run_name: str) -> tuple[np.ndarray, dict]:
    artifacts = read_json(experiment_root / "runs" / run_name / "artifacts.json")
    variant_key = artifacts["embedding_variant"]
    vectors = np.load(artifacts["eval_embeddings"][variant_key])
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


def _scatter_plot(coords: np.ndarray, labels: np.ndarray, label_names: list[str], title: str, output_path: Path, xlab: str, ylab: str) -> None:
    plt.figure(figsize=(8, 6))
    for label, name, color in zip(range(len(label_names)), label_names, PALETTE):
        mask = labels == label
        plt.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.65, c=color, label=name)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _centroid_heatmap(matrix: np.ndarray, label_names: list[str], title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(label_names)), label_names, rotation=45, ha="right")
    plt.yticks(range(len(label_names)), label_names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualization-only plots for Qwen parity embedding variants.")
    parser.add_argument("--experiment-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "qwen-bge-parity.json",
    )
    args = parser.parse_args()

    experiment_root = args.experiment_root
    config = read_json(args.config)
    labels = _load_eval_labels(experiment_root, config)
    source_metadata = _source_run_metadata(experiment_root, config["bridge_run"])
    label_to_id = source_metadata["dataset"]["label_to_id"]
    label_names = [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]
    variant_runs = _variant_runs(config)

    plot_root = ensure_dir(PLOTS_DIR / config["embedding_export"]["output_bundle_slug"])
    summary: dict[str, dict] = {}

    for variant_name, run_name in variant_runs.items():
        vectors, artifacts = _load_variant_vectors(experiment_root, run_name)
        metrics_summary = read_json(experiment_root / "runs" / run_name / "metrics" / "summary.json")
        variant_dir = ensure_dir(plot_root / variant_name)

        pca_coords, pca = _pca_projection(vectors)
        nonlinear_coords, nonlinear_method = _nonlinear_projection(vectors)

        _scatter_plot(
            pca_coords,
            labels,
            label_names,
            f"Qwen parity {variant_name} PCA (heldout eval)",
            variant_dir / "pca-2d.png",
            "PC1",
            "PC2",
        )
        _scatter_plot(
            nonlinear_coords,
            labels,
            label_names,
            f"Qwen parity {variant_name} {nonlinear_method.upper()} (heldout eval)",
            variant_dir / f"{nonlinear_method}-2d.png",
            f"{nonlinear_method.upper()}-1",
            f"{nonlinear_method.upper()}-2",
        )
        _centroid_heatmap(
            np.array(metrics_summary["geometry"]["centroid_cosine_distance_matrix"]),
            label_names,
            f"Qwen parity {variant_name} centroid cosine distance",
            variant_dir / "centroid-cosine-heatmap.png",
        )

        class_centroids = {}
        for label, name in enumerate(label_names):
            class_centroids[name] = pca_coords[labels == label].mean(axis=0).tolist()

        summary[variant_name] = {
            "run_name": run_name,
            "embedding_variant": artifacts["embedding_variant"],
            "nonlinear_projection_method": nonlinear_method,
            "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
            "pca_class_centroids": class_centroids,
            "comparison_note": "These plots are derived from held-out Qwen eval embeddings only and are not train/validation embedding-generation outputs like the earlier BGE stage.",
        }

    write_json(plot_root / "projection-summary.json", summary)


if __name__ == "__main__":
    main()
