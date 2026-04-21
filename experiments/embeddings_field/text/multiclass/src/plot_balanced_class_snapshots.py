from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from sklearn.decomposition import PCA


LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    return np.load(embeddings_path)


def load_labels(labels_path: Path) -> np.ndarray:
    return np.load(labels_path)


def build_output_dir(base: Path) -> Path:
    out = base / "all_class" / "cluster_snapshots"
    out.mkdir(parents=True, exist_ok=True)
    return out


def reduce_dim(embeddings: np.ndarray) -> np.ndarray:
    reducer = PCA(n_components=2, random_state=42)
    return reducer.fit_transform(embeddings)


def radius_stats(distances: np.ndarray) -> tuple[float, float]:
    if distances.size == 0:
        return 0.0, 0.0
    max_dist = distances.max()
    edges = np.linspace(0, max_dist * 1.05 + 1e-8, 100)
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2
    counts, _ = np.histogram(distances, bins=edges)
    density = counts / widths
    peak_idx = int(np.nanargmax(density))
    peak_radius = centers[peak_idx]
    threshold = density[peak_idx] * 0.1
    drop_idx = peak_idx
    for i in range(peak_idx + 1, len(density)):
        if density[i] <= threshold:
            drop_idx = i
            break
    drop_radius = centers[drop_idx]
    return peak_radius, drop_radius


def plot_snapshot(coords: np.ndarray, labels: np.ndarray, centroids: np.ndarray, idx: int, output_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    mask = labels == idx
    others = ~mask
    plt.scatter(coords[others, 0], coords[others, 1], s=14, color="lightgrey", alpha=0.3, label="others")
    plt.scatter(coords[mask, 0], coords[mask, 1], s=22, color="#00b4d8", alpha=0.85, label=LABELS[idx])
    centroid = centroids[idx]
    distances = np.linalg.norm(coords[mask] - centroid, axis=1)
    radius_peak, radius_drop = radius_stats(distances)
    plt.scatter(
        centroid[0],
        centroid[1],
        marker="*",
        color="gold",
        edgecolors="black",
        linewidth=1.5,
        s=250,
        label=f"{LABELS[idx]} centroid",
    )
    plt.text(
        centroid[0] + 0.03,
        centroid[1] + 0.02,
        LABELS[idx],
        fontsize=10,
        fontweight="bold",
    )
    circle_peak = Circle(centroid, radius_peak, edgecolor="#252525", facecolor="none", linestyle="--", linewidth=1.4)
    circle_drop = Circle(centroid, radius_drop, edgecolor="#b90000", facecolor="none", linestyle="-.", linewidth=1.2)
    plt.gca().add_patch(circle_peak)
    plt.gca().add_patch(circle_drop)
    plt.title(f"{LABELS[idx]} snapshot (balanced)")
    plt.xlabel("PCA dim 1")
    plt.ylabel("PCA dim 2")
    plt.grid(alpha=0.2)
    circle_peak = Circle(centroid, radius_peak, edgecolor="#252525", facecolor="none", linestyle="--", linewidth=1.4)
    circle_drop = Circle(centroid, radius_drop, edgecolor="#b90000", facecolor="none", linestyle="-.", linewidth=1.2)
    plt.gca().add_patch(circle_peak)
    plt.gca().add_patch(circle_drop)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"balanced_cluster_{LABELS[idx]}.png", dpi=160)
    plt.close()


def plot_combined(coords: np.ndarray, labels: np.ndarray, centroids: np.ndarray, output_dir: Path) -> None:
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10.colors
    for idx, label in enumerate(LABELS):
        mask = labels == idx
        plt.scatter(coords[mask, 0], coords[mask, 1], s=20, alpha=0.7, color=colors[idx % len(colors)], label=label)
        centroid = centroids[idx]
        plt.scatter(
            centroid[0],
            centroid[1],
            marker="*",
            color="gold",
            edgecolors="black",
            linewidth=1.5,
            s=260,
        )
        plt.text(
            centroid[0] + 0.03,
            centroid[1] + 0.02,
            label,
            fontsize=10,
            fontweight="bold",
            color=colors[idx % len(colors)],
        )
    plt.title("Balanced dataset: all clusters")
    plt.xlabel("PCA dim 1")
    plt.ylabel("PCA dim 2")
    plt.grid(alpha=0.2)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(output_dir / "balanced_all_clusters.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot balanced dataset class snapshots.")
    base = Path(__file__).resolve().parents[1] / "balanced_dataset"
    parser.add_argument(
        "--embedding-file",
        type=Path,
        default=base / "artifacts" / "embeddings" / "dair_ai_emotion_train_balanced_bge-base-en-v1-5_raw.npy",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=base / "data" / "processed" / "train" / "labels.npy",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=base,
    )
    args = parser.parse_args()

    embeddings = load_embeddings(args.embedding_file)
    labels = load_labels(args.labels_file)
    coords = reduce_dim(embeddings)
    centroids = []
    for idx in range(len(LABELS)):
        centroids.append(np.mean(coords[labels == idx], axis=0))
    output_dir = build_output_dir(args.output_root)
    stacked_centroids = np.vstack(centroids)
    for idx in range(len(LABELS)):
        plot_snapshot(coords, labels, stacked_centroids, idx, output_dir)
    plot_combined(coords, labels, stacked_centroids, output_dir)
    print(f"Balanced cluster snapshots saved in {output_dir}")


if __name__ == "__main__":
    main()
