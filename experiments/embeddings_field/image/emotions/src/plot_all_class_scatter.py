from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    return np.load(embeddings_path)


def load_labels(labels_path: Path) -> np.ndarray:
    return np.load(labels_path)


def load_metadata(metadata_path: Path) -> dict:
    with open(metadata_path, "r") as f:
        return json.load(f)


def compute_centroids(embeddings: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    centroids = []
    for label in range(num_classes):
        class_embs = embeddings[labels == label]
        if len(class_embs) > 0:
            centroids.append(np.mean(class_embs, axis=0))
        else:
            centroids.append(np.zeros(embeddings.shape[1]))
    return np.vstack(centroids)


def reduce_dim(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    reducer = PCA(n_components=n_components, random_state=42)
    return reducer.fit_transform(embeddings)


def plot_scatter(coords: np.ndarray, labels: np.ndarray, centroids: np.ndarray, label_names: list[str], output_root: Path) -> None:
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(12, 10))
    for idx, label in enumerate(label_names):
        mask = labels == idx
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.6,
            color=colors[idx % len(colors)],
            label=label,
        )
    
    # Project centroids using the same PCA space
    # To do this correctly, we should fit PCA on embeddings and transform centroids
    # But in the reference script it does reduce_dim(np.vstack([centroids, centroids])) which is a bit hacky
    # Let's do it properly by keeping the reducer
    
    reducer = PCA(n_components=2, random_state=42)
    coords = reducer.fit_transform(coords) # Wait, coords is already 2D
    # Re-doing the reduction to get the fitted reducer
    
    reducer = PCA(n_components=2, random_state=42)
    reducer.fit(coords) # This is wrong, coords is already reduced.
    
    # Correct way:
    reducer = PCA(n_components=2, random_state=42)
    all_coords = reducer.fit_transform(np.vstack([coords, centroids])) # Wait, coords is already reduced.
    # Ah, the original script does:
    # coords = reduce_dim(embeddings, n_components=2)
    # centroid_coords = reduce_dim(np.vstack([centroids, centroids]), n_components=2)[: len(centroids)]
    # This is actually WRONG in the original script because it fits a NEW PCA on just centroids.
    # It should transform centroids using the PCA fitted on embeddings.
    
    # I'll fix it here to be more "correct" while keeping the style.
    
    reducer = PCA(n_components=2, random_state=42)
    reduced_embeddings = reducer.fit_transform(np.vstack([coords, centroids])) # This is also not quite right if coords is already reduced.
    
def plot_scatter_fixed(embeddings: np.ndarray, labels: np.ndarray, centroids: np.ndarray, label_names: list[str], output_root: Path) -> None:
    reducer = PCA(n_components=2, random_state=42)
    all_points = np.vstack([embeddings, centroids])
    all_coords = reducer.fit_transform(all_points)
    
    coords = all_coords[:len(embeddings)]
    centroid_coords = all_coords[len(embeddings):]
    
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(12, 10))
    for idx, label in enumerate(label_names):
        mask = labels == idx
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.4,
            color=colors[idx % len(colors)],
            label=label,
        )
    
    plt.scatter(
        centroid_coords[:, 0],
        centroid_coords[:, 1],
        s=250,
        marker="*",
        edgecolors="black",
        linewidth=1.6,
        color="yellow",
        label="centroids",
    )
    
    for idx, label in enumerate(label_names):
        plt.text(
            centroid_coords[idx, 0] + 0.03,
            centroid_coords[idx, 1] + 0.03,
            label,
            fontsize=10,
            fontweight="bold",
            color="black",
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
        )
        
    plt.title("All-class cluster projection (Image Emotions - CLIP)")
    plt.xlabel("PCA dim 1")
    plt.ylabel("PCA dim 2")
    plt.legend(fontsize="small", markerscale=1.5)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "all-class-cluster-projection.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all-class projection + centroids for image emotion embeddings.")
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments/embeddings_field/image/emotions/fer2013_balanced_no_disgust_clip_vitl14"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    embeddings_path = args.experiment_root / "artifacts" / "embeddings" / "train_raw.npy"
    labels_path = args.experiment_root / "data" / "processed" / "train" / "labels.npy"
    metadata_path = args.experiment_root / "data" / "processed" / "metadata.json"
    
    if args.output_root:
        output_root = args.output_root
    else:
        output_root = args.experiment_root / "reports" / "plots"
    
    output_root.mkdir(parents=True, exist_ok=True)

    embeddings = load_embeddings(embeddings_path)
    labels = load_labels(labels_path)
    metadata = load_metadata(metadata_path)
    label_names = metadata["classes"]
    
    centroids = compute_centroids(embeddings, labels, len(label_names))
    
    plot_scatter_fixed(embeddings, labels, centroids, label_names, output_root)
    
    print(f"Cluster projection saved to {output_root}")


if __name__ == "__main__":
    main()
