from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    return np.load(embeddings_path)


def load_labels(labels_path: Path) -> np.ndarray:
    return np.load(labels_path)


def compute_centroids(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    centroids = []
    for label in range(len(LABELS)):
        centroids.append(np.mean(embeddings[labels == label], axis=0))
    return np.vstack(centroids)


def reduce_dim(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    reducer = PCA(n_components=n_components, random_state=42)
    return reducer.fit_transform(embeddings)


def plot_scatter(coords: np.ndarray, labels: np.ndarray, centroids: np.ndarray, output_root: Path) -> None:
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(10, 8))
    for idx, label in enumerate(LABELS):
        mask = labels == idx
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.6,
            color=colors[idx % len(colors)],
            label=label,
        )
    centroid_coords = reduce_dim(np.vstack([centroids, centroids]), n_components=2)[: len(centroids)]
    plt.scatter(
        centroid_coords[:, 0],
        centroid_coords[:, 1],
        s=220,
        marker="*",
        edgecolors="black",
        linewidth=1.6,
        color="yellow",
        label="centroids",
    )
    for idx, label in enumerate(LABELS):
        plt.text(
            centroid_coords[idx, 0] + 0.03,
            centroid_coords[idx, 1] + 0.03,
            label,
            fontsize=9,
            fontweight="bold",
            color=colors[idx % len(colors)],
        )
    plt.title("All-class projection with centroids")
    plt.xlabel("PCA dim 1")
    plt.ylabel("PCA dim 2")
    plt.legend(fontsize="small", markerscale=1.5)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_root / "all-class-cluster-projection.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all-class projection + centroids for dairy-ai/emotion embeddings.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[5] / "experiments" / "text" / "multiclass" / "dair-ai-emotion")
    parser.add_argument("--embeddings-name", default="dair_ai_emotion_train_bge-base-en-v1-5_raw.npy")
    parser.add_argument("--labels-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "all_class")
    args = parser.parse_args()

    embeddings_path = args.data_root / "artifacts" / "embeddings" / args.embeddings_name
    labels_path = args.labels_path or args.data_root / "data" / "processed" / "train" / "labels.npy"

    embeddings = load_embeddings(embeddings_path)
    labels = load_labels(labels_path)
    centroids = compute_centroids(embeddings, labels)
    coords = reduce_dim(embeddings, n_components=2)

    output_root = ensure_dir(args.output_root)
    plot_scatter(coords, labels, centroids, output_root)
    metadata = {
        "embeddings": str(embeddings_path),
        "labels": str(labels_path),
        "projection": "PCA-2D",
        "classes": LABELS,
    }
    (output_root / "projection-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Cluster projection saved to {output_root}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    main()
