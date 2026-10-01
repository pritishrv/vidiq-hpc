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

def build_output_dir(base: Path) -> Path:
    out = base / "cluster_snapshots"
    out.mkdir(parents=True, exist_ok=True)
    return out

def plot_class_snapshot(
    coords: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    class_idx: int,
    label_names: list[str],
    output_dir: Path,
) -> None:
    plt.figure(figsize=(10, 8))
    mask = labels == class_idx
    others = ~mask
    
    # Plot background points
    plt.scatter(
        coords[others, 0],
        coords[others, 1],
        s=10,
        color="lightgrey",
        alpha=0.2,
        label="other animals",
        edgecolors="none"
    )
    
    # Plot target class points
    plt.scatter(
        coords[mask, 0],
        coords[mask, 1],
        s=30,
        color="#e63946",
        alpha=0.8,
        label=label_names[class_idx],
        edgecolors="none"
    )
    
    centroid = centroids[class_idx]
    plt.scatter(
        centroid[0],
        centroid[1],
        color="gold",
        marker="*",
        s=300,
        edgecolor="black",
        linewidth=1.2,
        label=f"{label_names[class_idx]} centroid",
    )
    
    plt.title(f"{label_names[class_idx]} Cluster Snapshot (Animal Dataset)")
    plt.xlabel("PCA dim 1")
    plt.ylabel("PCA dim 2")
    plt.grid(alpha=0.1)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / f"cluster_{label_names[class_idx]}.png", dpi=180)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-class cluster snapshots for animal embeddings.")
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments/embeddings_field/image/animals"))
    parser.add_argument("--num-snapshots", type=int, default=10, help="Number of class snapshots to generate.")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    embeddings_path = args.experiment_root / "artifacts" / "embeddings" / "full_raw.npy"
    labels_path = args.experiment_root / "data" / "processed" / "full" / "labels.npy"
    metadata_path = args.experiment_root / "data" / "processed" / "metadata.json"
    
    if args.output_root:
        output_root = args.output_root
    else:
        output_root = args.experiment_root / "reports" / "plots"
    
    embeddings = load_embeddings(embeddings_path)
    labels = load_labels(labels_path)
    metadata = load_metadata(metadata_path)
    label_names = metadata["classes"]
    
    # Fit PCA once
    reducer = PCA(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)
    
    centroids = []
    for idx in range(len(label_names)):
        class_coords = coords[labels == idx]
        if len(class_coords) > 0:
            centroids.append(np.mean(class_coords, axis=0))
        else:
            centroids.append(np.zeros(2))
    
    snapshot_dir = build_output_dir(output_root)
    # Generate for the first N classes
    for idx in range(min(args.num_snapshots, len(label_names))):
        plot_class_snapshot(coords, labels, np.vstack(centroids), idx, label_names, snapshot_dir)
        
    print(f"Generated {min(args.num_snapshots, len(label_names))} class snapshots in {snapshot_dir}")

if __name__ == "__main__":
    main()
