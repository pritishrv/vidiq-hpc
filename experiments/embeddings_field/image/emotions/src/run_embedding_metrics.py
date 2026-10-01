from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from metrics import (
    _safe_davies_bouldin,
    _safe_silhouette,
    available_embedding_splits,
    centroid_summary,
    ensure_dir,
    load_metadata,
    load_split_embeddings,
    load_split_labels,
    pca_metrics,
    write_json,
)


def write_centroid_markdown(summary: dict, output_path: Path) -> None:
    labels = summary["label_names"]
    matrix = summary["euclidean_matrix"]
    lines = ["# Centroid Distance Matrix", "", "| Class | " + " | ".join(labels) + " |", "|" + " --- |" * (len(labels) + 1)]
    for row_label, row in zip(labels, matrix):
        values = [f"{value:.4f}" for value in row]
        lines.append(f"| {row_label} | " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_split(experiment_root: Path, split: str) -> dict:
    vectors = load_split_embeddings(experiment_root, split)
    labels = load_split_labels(experiment_root, split)
    metadata = load_metadata(experiment_root)
    label_names = metadata["classes"]

    cluster = {
        "silhouette": _safe_silhouette(vectors, labels),
        "davies_bouldin": _safe_davies_bouldin(vectors, labels),
        "num_examples": int(vectors.shape[0]),
        "embedding_dimensionality": int(vectors.shape[1]),
        "num_classes": int(len(np.unique(labels))),
    }
    centroids = centroid_summary(vectors, labels, label_names)
    pca = pca_metrics(vectors)
    return {
        "split": split,
        "cluster": cluster,
        "centroids": centroids,
        "pca": pca,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute image embedding metrics for FER2013 CLIP embeddings.")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("experiments/embeddings_field/image/emotions/fer2013_balanced_no_disgust_clip_vitl14"),
    )
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    output_root = args.output_root or experiment_root / "reports" / "embedding_metrics"
    requested_splits = args.splits or available_embedding_splits(experiment_root)
    if not requested_splits:
        raise FileNotFoundError(f"No embedding splits found under {experiment_root / 'artifacts' / 'embeddings'}")

    for split in requested_splits:
        split_root = ensure_dir(output_root / split)
        summary = run_split(experiment_root, split)
        write_json(split_root / "summary.json", summary)
        write_centroid_markdown(summary["centroids"], split_root / "centroid-distance-matrix.md")
        print(
            f"{split}: silhouette={summary['cluster']['silhouette']}, "
            f"davies_bouldin={summary['cluster']['davies_bouldin']}"
        )


if __name__ == "__main__":
    main()
