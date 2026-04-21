from __future__ import annotations

import argparse
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


def build_output_dir(base: Path) -> Path:
    out = base / "cluster_snapshots"
    out.mkdir(parents=True, exist_ok=True)
    return out


def reduce_dim(embeddings: np.ndarray) -> np.ndarray:
    reducer = PCA(n_components=2, random_state=42)
    return reducer.fit_transform(embeddings)


def plot_class_snapshot(
    coords: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    class_idx: int,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    mask = labels == class_idx
    others = ~mask
    plt.scatter(
        coords[others, 0],
        coords[others, 1],
        s=18,
        color="lightgrey",
        alpha=0.4,
        label="other classes",
    )
    plt.scatter(
        coords[mask, 0],
        coords[mask, 1],
        s=25,
        color="#e63946",
        alpha=0.8,
        label=LABELS[class_idx],
    )
    centroid = centroids[class_idx]
    plt.scatter(
        centroid[0],
        centroid[1],
        color="gold",
        marker="*",
        s=260,
        edgecolor="black",
        linewidth=1.5,
        label=f"{LABELS[class_idx]} centroid",
    )
    plt.text(
        centroid[0] + 0.02,
        centroid[1] + 0.02,
        LABELS[class_idx],
        fontsize=10,
        fontweight="bold",
    )
    plt.title(f"{LABELS[class_idx]} cluster snapshot")
    plt.xlabel("PCA dim 1")
    plt.ylabel("PCA dim 2")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"cluster_{LABELS[class_idx]}.png", dpi=160)
    plt.close()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-class cluster snapshots.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[5] / "experiments" / "text" / "multiclass" / "dair-ai-emotion")
    parser.add_argument("--embeddings-name", default="dair_ai_emotion_train_bge-base-en-v1-5_raw.npy")
    parser.add_argument("--labels-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "all_class")
    args = parser.parse_args()

    embeddings_path = args.data_root / "artifacts" / "embeddings" / args.embeddings_name
    labels_path = args.labels_path or args.data_root / "data" / "processed" / "train" / "labels.npy"
    embeddings = load_embeddings(embeddings_path)
    labels = load_labels(labels_path)
    coords = reduce_dim(embeddings)
    centroids = []
    for idx in range(len(LABELS)):
        centroids.append(np.mean(coords[labels == idx], axis=0))
    snapshot_dir = build_output_dir(args.output_root)
    for idx, label in enumerate(LABELS[:5]):
        plot_class_snapshot(coords, labels, np.vstack(centroids), idx, snapshot_dir)
    print(f"Generated class snapshots in {snapshot_dir}")


if __name__ == "__main__":
    main()
