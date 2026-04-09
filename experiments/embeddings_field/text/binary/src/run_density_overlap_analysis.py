from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)


def compute_centroid(vectors: np.ndarray) -> np.ndarray:
    return vectors.mean(axis=0)


def compute_distances(vectors: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vectors - centroid, axis=1)


def build_bins(distances: np.ndarray, opposite_distances: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    edges = np.percentile(distances, np.linspace(0, 100, n_bins + 1))
    bins = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (distances >= start) & (distances < end if end > start else distances >= end)
        opposite_mask = (opposite_distances >= start) & (opposite_distances < end if end > start else opposite_distances >= end)
        bins.append(
            {
                "start": float(start),
                "end": float(end),
                "density": int(mask.sum()),
                "opposite_count": int(opposite_mask.sum()),
                "midpoint": float((start + end) / 2),
            }
        )
    return bins


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute density and overlap for SST-2 embeddings.")
    parser.add_argument("--variant", choices=["raw", "centered_l2"], required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/embeddings_field/text/binary/metrics"))
    args = parser.parse_args()

    base = f"sst2_train_bge-base-en-v1-5_{args.variant}"
    embeddings_path = Path(f"experiments/text/binary/sst2/artifacts/embeddings/{base}.npy")
    metadata_path = Path(f"experiments/text/binary/sst2/artifacts/embeddings/{base}_metadata.json")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"{embeddings_path} missing")
    embeddings = load_embeddings(embeddings_path)

    labels_path = Path("experiments/text/binary/sst2/data/processed/train/labels.npy")
    labels = np.load(labels_path)

    positives = embeddings[labels == 1]
    negatives = embeddings[labels == 0]

    pos_centroid = compute_centroid(positives)
    neg_centroid = compute_centroid(negatives)

    pos_dist = compute_distances(positives, pos_centroid)
    neg_dist = compute_distances(negatives, neg_centroid)
    pos_to_neg = compute_distances(positives, neg_centroid)
    neg_to_pos = compute_distances(negatives, pos_centroid)

    pos_overlap = (pos_to_neg < pos_dist).mean()
    neg_overlap = (neg_to_pos < neg_dist).mean()

    pos_bins = build_bins(pos_dist, neg_dist)
    neg_bins = build_bins(neg_dist, pos_dist)

    stats = {
        "variant": args.variant,
        "pos_centroid_distance_avg": float(pos_dist.mean()),
        "neg_centroid_distance_avg": float(neg_dist.mean()),
        "pos_overlap_ratio": float(pos_overlap),
        "neg_overlap_ratio": float(neg_overlap),
        "pos_bins": pos_bins,
        "neg_bins": neg_bins,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"density_overlap_{args.variant}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved density/overlap metrics to {output_path}")


if __name__ == "__main__":
    main()
