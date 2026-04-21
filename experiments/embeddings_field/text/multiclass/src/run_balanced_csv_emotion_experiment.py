from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_CSV = Path(
    "/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/data/Text_Datasets/20-emotions/"
    "balanced_emotions_6classes.csv"
)
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1] / "balanced_emotions_6classes"
)
DEFAULT_LABELS = ["sadness", "happiness", "love", "anger", "fear", "surprise"]
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_SLUG = "bge-base-en-v1-5"
EMBEDDING_DIM = 768
LOG_SURFACE_CONST = (
    math.log(2.0)
    + (EMBEDDING_DIM / 2.0) * math.log(math.pi)
    - math.lgamma(EMBEDDING_DIM / 2.0)
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_dataset(
    csv_path: Path,
    text_column: str,
    label_column: str,
    label_names: list[str],
) -> tuple[list[str], np.ndarray]:
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    texts: list[str] = []
    labels: list[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {text_column, label_column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        for row in reader:
            text = (row[text_column] or "").strip()
            label = (row[label_column] or "").strip()
            if not text:
                continue
            if label not in label_to_id:
                raise ValueError(f"Unknown label {label!r}; expected {label_names}")
            texts.append(text)
            labels.append(label_to_id[label])
    return texts, np.array(labels, dtype=np.int64)


def persist_dataset(
    output_root: Path,
    csv_path: Path,
    text_column: str,
    label_column: str,
    label_names: list[str],
    texts: list[str],
    labels: np.ndarray,
) -> None:
    data_dir = ensure_dir(output_root / "data" / "processed" / "train")
    with (data_dir / "texts.jsonl").open("w", encoding="utf-8") as fh:
        for text, label in zip(texts, labels):
            fh.write(
                json.dumps(
                    {"text": text, "label": int(label), "label_name": label_names[int(label)]},
                    ensure_ascii=True,
                )
                + "\n"
            )
    np.save(data_dir / "labels.npy", labels)
    counts = Counter(int(label) for label in labels)
    token_lengths = [len(text.split()) for text in texts]
    metadata = {
        "source_csv": str(csv_path),
        "text_column": text_column,
        "label_column": label_column,
        "label_names": label_names,
        "num_examples": len(texts),
        "class_counts": {label_names[idx]: int(counts[idx]) for idx in range(len(label_names))},
        "avg_whitespace_token_length": float(np.mean(token_lengths)) if token_lengths else 0.0,
        "median_whitespace_token_length": float(np.median(token_lengths)) if token_lengths else 0.0,
    }
    write_json(data_dir / "metadata.json", metadata)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(texts: list[str], max_length: int, batch_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    run_device = device()
    model.to(run_device)
    model.eval()

    outputs = []
    truncation_count = 0
    token_lengths: list[int] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), total=total_batches, unit="batch"):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_length=True,
            )
            lengths = encoded.pop("length").tolist()
            token_lengths.extend(int(length) for length in lengths)
            truncation_count += sum(1 for length in lengths if int(length) >= max_length)
            encoded = {key: value.to(run_device) for key, value in encoded.items()}
            model_output = model(**encoded)
            pooled = mean_pool(model_output.last_hidden_state, encoded["attention_mask"])
            outputs.append(pooled.cpu().numpy())
    metadata = {
        "model_name": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "device": run_device,
        "max_length": max_length,
        "batch_size": batch_size,
        "avg_tokenized_length": float(np.mean(token_lengths)) if token_lengths else 0.0,
        "median_tokenized_length": float(np.median(token_lengths)) if token_lengths else 0.0,
        "truncation_count": int(truncation_count),
        "truncation_rate": float(truncation_count / len(texts)) if texts else 0.0,
    }
    return np.vstack(outputs), metadata


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def save_embeddings(
    output_root: Path,
    dataset_slug: str,
    texts: list[str],
    max_length: int,
    batch_size: int,
    force: bool,
) -> tuple[np.ndarray, Path]:
    embeddings_dir = ensure_dir(output_root / "artifacts" / "embeddings")
    raw_path = embeddings_dir / f"{dataset_slug}_train_{MODEL_SLUG}_raw.npy"
    metadata_path = embeddings_dir / f"{dataset_slug}_train_{MODEL_SLUG}_metadata.json"
    if raw_path.exists() and not force:
        return np.load(raw_path), raw_path

    raw, metadata = embed_texts(texts, max_length=max_length, batch_size=batch_size)
    np.save(raw_path, raw)
    np.save(embeddings_dir / f"{dataset_slug}_train_{MODEL_SLUG}_l2.npy", l2_normalize(raw))
    np.save(
        embeddings_dir / f"{dataset_slug}_train_{MODEL_SLUG}_centered_l2.npy",
        l2_normalize(raw - raw.mean(axis=0, keepdims=True)),
    )
    write_json(metadata_path, metadata)
    return raw, raw_path


def centroids(embeddings: np.ndarray, labels: np.ndarray, label_names: list[str]) -> np.ndarray:
    return np.vstack([embeddings[labels == idx].mean(axis=0) for idx in range(len(label_names))])


def plot_all_class_scatter(
    output_root: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
) -> None:
    out = ensure_dir(output_root / "all_class")
    reducer = PCA(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)
    centroid_coords = reducer.transform(centroids(embeddings, labels, label_names))
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, label in enumerate(label_names):
        mask = labels == idx
        ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.6, color=colors[idx], label=label)
        ax.scatter(
            centroid_coords[idx, 0],
            centroid_coords[idx, 1],
            s=220,
            marker="*",
            edgecolors="black",
            linewidth=1.6,
            color="yellow",
        )
        ax.text(centroid_coords[idx, 0] + 0.03, centroid_coords[idx, 1] + 0.03, label, fontsize=9)
    ax.set_title("All-class projection with centroids")
    ax.set_xlabel("PCA dim 1")
    ax.set_ylabel("PCA dim 2")
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out / "all-class-cluster-projection.png", dpi=180)
    plt.close(fig)


def radius_stats(distances: np.ndarray) -> tuple[float, float]:
    if distances.size == 0:
        return 0.0, 0.0
    edges = np.linspace(0, distances.max() * 1.05 + 1e-8, 100)
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2
    counts, _ = np.histogram(distances, bins=edges)
    density = counts / widths
    peak_idx = int(np.nanargmax(density))
    threshold = density[peak_idx] * 0.1
    drop_idx = peak_idx
    for idx in range(peak_idx + 1, len(density)):
        if density[idx] <= threshold:
            drop_idx = idx
            break
    return float(centers[peak_idx]), float(centers[drop_idx])


def plot_class_snapshots(
    output_root: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
) -> None:
    out = ensure_dir(output_root / "all_class" / "cluster_snapshots")
    reducer = PCA(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)
    centroid_coords = np.vstack([coords[labels == idx].mean(axis=0) for idx in range(len(label_names))])
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, label in enumerate(label_names):
        mask = labels == idx
        ax.scatter(coords[mask, 0], coords[mask, 1], s=20, alpha=0.7, color=colors[idx], label=label)
        ax.scatter(centroid_coords[idx, 0], centroid_coords[idx, 1], marker="*", color="gold", edgecolors="black", s=260)
        ax.text(centroid_coords[idx, 0] + 0.03, centroid_coords[idx, 1] + 0.02, label, fontsize=10)
    ax.set_title("Balanced CSV dataset: all clusters")
    ax.set_xlabel("PCA dim 1")
    ax.set_ylabel("PCA dim 2")
    ax.grid(alpha=0.2)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out / "balanced_all_clusters.png", dpi=180)
    plt.close(fig)

    for idx, label in enumerate(label_names):
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = labels == idx
        ax.scatter(coords[~mask, 0], coords[~mask, 1], s=14, color="lightgrey", alpha=0.3, label="others")
        ax.scatter(coords[mask, 0], coords[mask, 1], s=22, color="#00b4d8", alpha=0.85, label=label)
        centroid = centroid_coords[idx]
        distances = np.linalg.norm(coords[mask] - centroid, axis=1)
        peak_radius, drop_radius = radius_stats(distances)
        ax.scatter(centroid[0], centroid[1], marker="*", color="gold", edgecolors="black", s=250, label=f"{label} centroid")
        ax.add_patch(Circle(centroid, peak_radius, edgecolor="#252525", facecolor="none", linestyle="--", linewidth=1.4))
        ax.add_patch(Circle(centroid, drop_radius, edgecolor="#b90000", facecolor="none", linestyle="-.", linewidth=1.2))
        ax.set_title(f"{label} snapshot (balanced CSV)")
        ax.set_xlabel("PCA dim 1")
        ax.set_ylabel("PCA dim 2")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"balanced_cluster_{label}.png", dpi=160)
        plt.close(fig)


def surface_log_volume(midpoint: float, width: float) -> float:
    r = max(midpoint, 1e-6)
    return LOG_SURFACE_CONST + ((EMBEDDING_DIM - 1) * math.log(r)) + math.log(max(width, 1e-9))


def build_bins(distances: np.ndarray, opposite: np.ndarray, n_bins: int) -> list[dict[str, float]]:
    edges = np.percentile(distances, np.linspace(0, 100, n_bins + 1))
    bins = []
    for start, end in zip(edges[:-1], edges[1:]):
        width = max(float(end - start), 1e-8)
        mask = (distances >= start) & (distances < end if end > start else distances >= end)
        opp_mask = (opposite >= start) & (opposite < end if end > start else opposite >= end)
        count = int(mask.sum())
        opp_count = int(opp_mask.sum())
        midpoint = float((start + end) / 2)
        log_vol = surface_log_volume(midpoint, width)
        bins.append(
            {
                "start": float(start),
                "end": float(end),
                "midpoint": midpoint,
                "density": count,
                "density_per_unit": float(count / width),
                "overlap_count": opp_count,
                "overlap_ratio": float(opp_count / (count + 1e-12)) if count else 0.0,
                "bin_width": width,
                "surface_log_density": float(math.log(count) - log_vol) if count else float("-inf"),
            }
        )
    return bins


def run_all_class_density(
    output_root: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    n_bins: int,
) -> None:
    out = ensure_dir(output_root / "all_class")
    class_centroids = centroids(embeddings, labels, label_names)
    stats: dict[str, dict[str, Any]] = {}
    for idx, label in enumerate(label_names):
        own = embeddings[labels == idx]
        others = embeddings[labels != idx]
        distances = np.linalg.norm(own - class_centroids[idx], axis=1)
        opp_distances = np.linalg.norm(others - class_centroids[idx], axis=1)
        stats[label] = {
            "count": int(len(own)),
            "centroid": class_centroids[idx].tolist(),
            "bins": build_bins(distances, opp_distances, n_bins),
        }

    plot_line(
        out / "all-class-density-decay.png",
        stats,
        "density_per_unit",
        "Distance from centroid",
        "Density per unit",
        "All-class density decay",
    )
    plot_line(
        out / "all-class-surface-density-decay.png",
        stats,
        "surface_log_density",
        "Distance from centroid",
        "Log density per hyperspace volume",
        "All-class log density decay (768D)",
        nan_inf=True,
    )
    plot_line(
        out / "all-class-overlap-ratio.png",
        stats,
        "overlap_ratio",
        "Distance from centroid",
        "Overlap ratio",
        "All-class overlap ratios",
    )
    write_json(out / "metrics.json", stats)


def plot_line(
    path: Path,
    stats: dict[str, dict[str, Any]],
    key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    nan_inf: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, entry in stats.items():
        values = [bin_entry[key] for bin_entry in entry["bins"]]
        if nan_inf:
            values = [value if np.isfinite(value) else np.nan for value in values]
        ax.plot([bin_entry["midpoint"] for bin_entry in entry["bins"]], values, label=label, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_radial_distance(
    output_root: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
) -> None:
    out = ensure_dir(output_root / "all_class" / "radial_distance")
    class_centroids = centroids(embeddings, labels, label_names)
    rng = np.random.default_rng(0)
    stats: dict[str, dict[str, Any]] = {}
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, label in enumerate(label_names):
        distances = np.linalg.norm(embeddings[labels == idx] - class_centroids[idx], axis=1)
        bins = np.linspace(distances.min(), distances.max(), 8)
        stats[label] = {
            "count": int(len(distances)),
            "min": float(distances.min()),
            "max": float(distances.max()),
            "percentiles": {
                "10": float(np.percentile(distances, 10)),
                "50": float(np.median(distances)),
                "90": float(np.percentile(distances, 90)),
            },
            "histogram": [int(np.sum((distances >= start) & (distances < end))) for start, end in zip(bins[:-1], bins[1:])],
        }
        jitter = rng.normal(0, 0.08, size=len(distances))
        ax.scatter(np.full_like(distances, idx) + jitter, distances, s=12, alpha=0.4, label=label)
        ax.hlines(np.median(distances), idx - 0.3, idx + 0.3, colors="k", linewidth=1.5)
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names)
    ax.set_ylabel("Distance to class centroid")
    ax.set_title("Radial distance distribution per emotion (balanced CSV dataset)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "radial-distance-scatter.png", dpi=180)
    plt.close(fig)
    write_json(out / "radial-distance-stats.json", stats)


def run_pairwise(
    output_root: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    n_bins: int,
) -> None:
    out = ensure_dir(output_root / "pairwise")
    for a, b in combinations(range(len(label_names)), 2):
        pair_dir = ensure_dir(out / f"{label_names[a]}_vs_{label_names[b]}")
        mask = (labels == a) | (labels == b)
        subset_embeddings = embeddings[mask]
        subset_labels = labels[mask]
        a_points = embeddings[labels == a]
        b_points = embeddings[labels == b]
        centroid_a = a_points.mean(axis=0)
        centroid_b = b_points.mean(axis=0)
        dist_a = np.linalg.norm(a_points - centroid_a, axis=1)
        dist_b = np.linalg.norm(b_points - centroid_b, axis=1)
        cross_ab = np.linalg.norm(a_points - centroid_b, axis=1)
        cross_ba = np.linalg.norm(b_points - centroid_a, axis=1)
        stats = {
            "labels": [label_names[a], label_names[b]],
            "counts": {"a": int(len(a_points)), "b": int(len(b_points))},
            "a_bins": build_bins(dist_a, cross_ab, n_bins),
            "b_bins": build_bins(dist_b, cross_ba, n_bins),
        }
        write_json(pair_dir / "metrics.json", stats)
        plot_pair_density(pair_dir, stats)
        plot_pair_overlap(pair_dir, stats)
        plot_pair_overlap_bar(pair_dir, stats)
        plot_pair_scatter(pair_dir, subset_embeddings, subset_labels, embeddings, labels, label_names, a, b)


def plot_pair_density(pair_dir: Path, stats: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([b["midpoint"] for b in stats["a_bins"]], [b["density_per_unit"] for b in stats["a_bins"]], label=f"{stats['labels'][0]} density")
    ax.plot([b["midpoint"] for b in stats["b_bins"]], [b["density_per_unit"] for b in stats["b_bins"]], label=f"{stats['labels'][1]} density")
    ax.set_xlabel("Distance from centroid")
    ax.set_ylabel("Density per unit")
    ax.set_title(f"{stats['labels'][0]} vs {stats['labels'][1]} density decay")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pair_dir / "density-decay.png", dpi=160)
    plt.close(fig)


def plot_pair_overlap(pair_dir: Path, stats: dict[str, Any]) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    midpoints = [b["midpoint"] for b in stats["a_bins"]]
    ax1.plot(midpoints, [b["overlap_ratio"] for b in stats["a_bins"]], label=f"{stats['labels'][0]} overlap", color="#246eb5")
    ax1.plot(midpoints, [b["overlap_ratio"] for b in stats["b_bins"]], label=f"{stats['labels'][1]} overlap", color="#f79d65", linestyle="--")
    ax2.plot(midpoints, np.cumsum([b["overlap_count"] for b in stats["a_bins"]]), label=f"{stats['labels'][0]} cum overlap", color="#246eb5", alpha=0.5)
    ax2.plot(midpoints, np.cumsum([b["overlap_count"] for b in stats["b_bins"]]), label=f"{stats['labels'][1]} cum overlap", color="#f79d65", alpha=0.5, linestyle=":")
    ax1.set_xlabel("Distance from centroid")
    ax1.set_ylabel("Overlap ratio")
    ax2.set_ylabel("Cumulative overlap")
    ax1.set_title(f"{stats['labels'][0]} vs {stats['labels'][1]} overlap volume")
    ax1.grid(alpha=0.3)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize="small")
    fig.tight_layout()
    fig.savefig(pair_dir / "overlap-volume.png", dpi=160)
    plt.close(fig)


def plot_pair_overlap_bar(pair_dir: Path, stats: dict[str, Any]) -> None:
    centers = np.array([b["midpoint"] for b in stats["a_bins"]])
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(centers - width / 2, [b["overlap_count"] for b in stats["a_bins"]], width, label=f"{stats['labels'][0]} overlap", color="#246eb5")
    ax.bar(centers + width / 2, [b["overlap_count"] for b in stats["b_bins"]], width, label=f"{stats['labels'][1]} overlap", color="#f79d65")
    ax.set_title(f"{stats['labels'][0]} vs {stats['labels'][1]} bin overlap counts")
    ax.set_xlabel("Distance bin center")
    ax.set_ylabel("Overlap count")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pair_dir / "overlap-count-bar.png", dpi=160)
    plt.close(fig)


def plot_pair_scatter(
    pair_dir: Path,
    subset_embeddings: np.ndarray,
    subset_labels: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    a: int,
    b: int,
) -> None:
    reducer = PCA(n_components=2, random_state=42)
    coords = reducer.fit_transform(subset_embeddings)
    centroid_coords = reducer.transform(np.vstack([embeddings[labels == a].mean(axis=0), embeddings[labels == b].mean(axis=0)]))
    a_mask = subset_labels == a
    b_mask = subset_labels == b
    a_points = subset_embeddings[a_mask]
    b_points = subset_embeddings[b_mask]
    centroid_a = embeddings[labels == a].mean(axis=0)
    centroid_b = embeddings[labels == b].mean(axis=0)
    overlap_mask = np.zeros(len(subset_labels), dtype=bool)
    overlap_mask[a_mask] = np.linalg.norm(a_points - centroid_b, axis=1) < np.linalg.norm(a_points - centroid_a, axis=1)
    overlap_mask[b_mask] = np.linalg.norm(b_points - centroid_a, axis=1) < np.linalg.norm(b_points - centroid_b, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#246eb5", "#f79d65"]
    for idx, cls in enumerate((a, b)):
        cls_mask = subset_labels == cls
        ax.scatter(coords[cls_mask, 0], coords[cls_mask, 1], s=28, alpha=0.55, c=colors[idx], label=label_names[cls], edgecolors="none")
    ax.scatter(coords[overlap_mask, 0], coords[overlap_mask, 1], facecolors="none", edgecolors="k", linewidth=0.8, s=60, label="overlap point")
    ax.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker="X", s=120, c=colors, edgecolors="k", linewidth=1.2)
    ax.set_title(f"{label_names[a]} vs {label_names[b]} scatter")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(pair_dir / "scatter-centroids.png", dpi=160)
    plt.close(fig)


def parse_labels(raw: str) -> list[str]:
    return [label.strip() for label in raw.split(",") if label.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run six-emotion embedding geometry experiments from a balanced CSV.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-slug", default="balanced_emotions_6classes")
    parser.add_argument("--text-column", default="cleaned_text")
    parser.add_argument("--label-column", default="emotion")
    parser.add_argument("--label-names", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--force-embeddings", action="store_true")
    args = parser.parse_args()

    label_names = parse_labels(args.label_names)
    output_root = ensure_dir(args.output_root)
    texts, labels = read_csv_dataset(args.csv_path, args.text_column, args.label_column, label_names)
    persist_dataset(output_root, args.csv_path, args.text_column, args.label_column, label_names, texts, labels)
    embeddings, raw_path = save_embeddings(
        output_root,
        args.dataset_slug,
        texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        force=args.force_embeddings,
    )
    if len(embeddings) != len(labels):
        raise ValueError(f"Embedding count {len(embeddings)} does not match label count {len(labels)}")

    run_all_class_density(output_root, embeddings, labels, label_names, args.bins)
    plot_all_class_scatter(output_root, embeddings, labels, label_names)
    plot_class_snapshots(output_root, embeddings, labels, label_names)
    run_radial_distance(output_root, embeddings, labels, label_names)
    run_pairwise(output_root, embeddings, labels, label_names, args.bins)
    write_json(
        output_root / "run-summary.json",
        {
            "csv_path": str(args.csv_path),
            "raw_embeddings_path": str(raw_path),
            "label_names": label_names,
            "num_examples": int(len(labels)),
            "outputs": {
                "all_class": str(output_root / "all_class"),
                "pairwise": str(output_root / "pairwise"),
            },
        },
    )
    print(f"CSV emotion geometry experiment complete: {output_root}")


if __name__ == "__main__":
    main()
