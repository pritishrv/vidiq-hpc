from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "bge-ablation-stage.json"
PLOTS_ROOT = BASE_DIR / "artifacts" / "plots" / "pairwise-emotions"
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)


def load_label_names() -> list[str]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("dataset", {}).get("label_names", [])


def project_pca(points: np.ndarray, components: int = 2) -> np.ndarray:
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:components].T


def sample_indices(mask: np.ndarray, max_per_class: int) -> np.ndarray:
    if max_per_class is None or max_per_class >= mask.sum():
        return np.where(mask)[0]
    rng = np.random.default_rng(42)
    return np.sort(rng.choice(np.where(mask)[0], size=max_per_class, replace=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scatter plot pairwise emotion embeddings.")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--max-per-class", type=int, default=1500)
    parser.add_argument("--output-dir", type=Path, default=PLOTS_ROOT)
    args = parser.parse_args()

    embeddings_path = BASE_DIR / "artifacts" / "embeddings" / f"dair_ai_emotion_{args.split}_bge-base-en-v1-5_raw.npy"
    labels_path = BASE_DIR / "data" / "processed" / args.split / "labels.npy"
    labels = np.load(labels_path)
    embeddings = np.load(embeddings_path)
    label_names = load_label_names()
    unique_labels = sorted(set(labels))
    color_map = plt.cm.get_cmap("tab10")

    for idx_a, idx_b in combinations(unique_labels, 2):
        sel_mask = (labels == idx_a) | (labels == idx_b)
        sel_indices = np.concatenate(
            [
                sample_indices(labels == idx_a, args.max_per_class),
                sample_indices(labels == idx_b, args.max_per_class),
            ]
        )
        sel_embeddings = embeddings[sel_indices]
        sel_labels = labels[sel_indices]
        coords = project_pca(sel_embeddings, 2)

        fig, ax = plt.subplots(figsize=(8, 6))
        for label in (idx_a, idx_b):
            mask = sel_labels == label
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=18,
                alpha=0.7,
                label=label_names[label] if label < len(label_names) else f"class_{label}",
                color=color_map(label % 10),
                edgecolor="none",
            )
        ax.set_title(f"{label_names[idx_a]} vs {label_names[idx_b]} ({args.split})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
        ax.legend()
        output_file = args.output_dir / f"{label_names[idx_a]}_vs_{label_names[idx_b]}_{args.split}.png"
        fig.tight_layout()
        fig.savefig(output_file, dpi=160)
        plt.close(fig)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
