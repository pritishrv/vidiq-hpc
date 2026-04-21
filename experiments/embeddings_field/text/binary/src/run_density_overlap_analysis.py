from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
BINARY_SST2_DIR = REPO_ROOT / "experiments" / "text" / "binary" / "sst2"
EMBEDDINGS_DIR = BINARY_SST2_DIR / "artifacts" / "embeddings"
PROCESSED_DIR = BINARY_SST2_DIR / "data" / "processed"


def load_texts(split_dir: Path) -> list[str]:
    text_path = split_dir / "texts.jsonl"
    texts: list[str] = []
    with text_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                texts.append(str(data.get("text", "")).strip())
            except json.JSONDecodeError:
                texts.append(line.strip())
    return texts


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)


def compute_centroid(vectors: np.ndarray) -> np.ndarray:
    return vectors.mean(axis=0)


def compute_distances(
    vectors: np.ndarray,
    centroid: np.ndarray,
    metric: Literal["euclidean", "cosine"],
) -> np.ndarray:
    if metric == "euclidean":
        return np.linalg.norm(vectors - centroid, axis=1)
    centroid_norm = np.linalg.norm(centroid)
    vec_norms = np.linalg.norm(vectors, axis=1)
    denom = vec_norms * centroid_norm
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    cos_sim = (vectors @ centroid) / denom
    return 1.0 - np.clip(cos_sim, -1.0, 1.0)


def build_bins(
    distances: np.ndarray,
    opposite_distances: np.ndarray,
    n_bins: int = 10,
) -> tuple[list[dict[str, float]], list[float]]:
    edges = np.percentile(distances, np.linspace(0, 100, n_bins + 1))
    bins: list[dict[str, float]] = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (distances >= start) & (
            distances < end if end > start else distances >= end
        )
        opposite_mask = (opposite_distances >= start) & (
            opposite_distances < end if end > start else opposite_distances >= end
        )
        width = max(end - start, 1e-8)
        density = int(mask.sum())
        opposite_count = int(opposite_mask.sum())
        bins.append(
            {
                "start": float(start),
                "end": float(end),
                "midpoint": float((start + end) / 2),
                "density": density,
                "density_per_unit": float(density / width),
                "opposite_count": opposite_count,
                "overlap_ratio": float(
                    opposite_count / (density + opposite_count + 1e-12)
                ),
                "bin_width": float(width),
            }
        )
    return bins, edges.tolist()


def get_text_subset(
    texts: list[str], indices: np.ndarray, split_indices: np.ndarray
) -> list[str]:
    return [texts[int(split_indices[i])] for i in indices]


def furthest_info(
    distances: np.ndarray, indices: np.ndarray, texts: list[str]
) -> dict[str, float | str | int]:
    if distances.size == 0:
        return {"index": None, "distance": None, "text": ""}
    idx = int(np.argmax(distances))
    return {
        "index": int(indices[idx]),
        "distance": float(distances[idx]),
        "text": texts[idx] if idx < len(texts) else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute centroid density+overlap stats.")
    parser.add_argument("--variant", choices=["raw"], default="raw")
    parser.add_argument(
        "--split", choices=["train", "validation"], default="train"
    )
    parser.add_argument(
        "--metric", choices=["euclidean", "cosine"], default="euclidean"
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--model-slug", default="bge-base-en-v1-5", help="Used to build the embedding filename."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "metrics",
    )
    args = parser.parse_args()

    metadata_path = EMBEDDINGS_DIR / f"sst2_{args.split}_{args.model_slug}_{args.variant}_metadata.json"
    embeddings_path = EMBEDDINGS_DIR / f"sst2_{args.split}_{args.model_slug}_{args.variant}.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"{embeddings_path} missing")

    embeddings = load_embeddings(embeddings_path)
    labels_path = PROCESSED_DIR / args.split / "labels.npy"
    texts_path = PROCESSED_DIR / args.split / "texts.jsonl"
    labels = np.load(labels_path)
    full_texts = load_texts(PROCESSED_DIR / args.split)

    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]
    positives = embeddings[positive_indices]
    negatives = embeddings[negative_indices]

    pos_centroid = compute_centroid(positives)
    neg_centroid = compute_centroid(negatives)

    pos_dist = compute_distances(positives, pos_centroid, args.metric)
    neg_dist = compute_distances(negatives, neg_centroid, args.metric)
    pos_to_neg = compute_distances(positives, neg_centroid, args.metric)
    neg_to_pos = compute_distances(negatives, pos_centroid, args.metric)

    pos_overlap = float((pos_to_neg < pos_dist).mean())
    neg_overlap = float((neg_to_pos < neg_dist).mean())

    pos_bins, pos_edges = build_bins(pos_dist, neg_dist, args.bins)
    neg_bins, neg_edges = build_bins(neg_dist, pos_dist, args.bins)

    pos_texts = get_text_subset(full_texts, np.arange(len(positives)), positive_indices)
    neg_texts = get_text_subset(full_texts, np.arange(len(negatives)), negative_indices)
    pos_furthest = furthest_info(pos_dist, positive_indices, pos_texts)
    neg_furthest = furthest_info(neg_dist, negative_indices, neg_texts)

    stats = {
        "variant": args.variant,
        "split": args.split,
        "metric": args.metric,
        "model_slug": args.model_slug,
        "num_examples": int(len(labels)),
        "num_positive": int(len(positives)),
        "num_negative": int(len(negatives)),
        "pos_centroid_distance_avg": float(pos_dist.mean()),
        "neg_centroid_distance_avg": float(neg_dist.mean()),
        "pos_overlap_ratio": pos_overlap,
        "neg_overlap_ratio": neg_overlap,
        "pos_centroid_norm": float(np.linalg.norm(pos_centroid)),
        "neg_centroid_norm": float(np.linalg.norm(neg_centroid)),
        "pos_furthest": pos_furthest,
        "neg_furthest": neg_furthest,
        "pos_bins": pos_bins,
        "neg_bins": neg_bins,
        "pos_bin_edges": pos_edges,
        "neg_bin_edges": neg_edges,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"density_overlap_{args.split}_{args.variant}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved density/overlap metrics to {output_path}")


if __name__ == "__main__":
    main()
