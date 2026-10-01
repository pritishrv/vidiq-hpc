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


def plot_scatter_fixed(embeddings: np.ndarray, labels: np.ndarray, centroids: np.ndarray, label_names: list[str], output_root: Path) -> None:
    reducer = PCA(n_components=2, random_state=42)
    all_points = np.vstack([embeddings, centroids])
    all_coords = reducer.fit_transform(all_points)
    
    coords = all_coords[:len(embeddings)]
    centroid_coords = all_coords[len(embeddings):]
    
    # Use a colormap that can handle 90 classes better
    cmap = matplotlib.colormaps['turbo']
    colors = cmap(np.linspace(0, 1, len(label_names)))
    
    plt.figure(figsize=(16, 14))
    for idx, label in enumerate(label_names):
        mask = labels == idx
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=8,
            alpha=0.4,
            color=colors[idx],
            label=None, # Too many for legend
        )
    
    plt.scatter(
        centroid_coords[:, 0],
        centroid_coords[:, 1],
        s=100,
        marker="*",
        edgecolors="black",
        linewidth=0.8,
        color="yellow",
        label="centroids",
    )
    
    # Only label some centroids or use very small text if too many
    for idx, label in enumerate(label_names):
        if idx % 2 == 0: # Label every 2nd animal to avoid too much clutter
            plt.text(
                centroid_coords[idx, 0] + 0.01,
                centroid_coords[idx, 1] + 0.01,
                label,
                fontsize=7,
                color="black",
                alpha=0.8
            )
        
    plt.title("All-class cluster projection (Animal Dataset - CLIP)")
    plt.xlabel("PCA dim 1")
    plt.ylabel("PCA dim 2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_root / "all-class-cluster-projection.png", dpi=250)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all-class projection + centroids for animal embeddings.")
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments/embeddings_field/image/animals"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    embeddings_path = args.experiment_root / "artifacts" / "embeddings" / "full_raw.npy"
    labels_path = args.experiment_root / "data" / "processed" / "full" / "labels.npy"
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
