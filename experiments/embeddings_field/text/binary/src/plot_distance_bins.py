from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
PLOTS_CONFIG = BASE_DIR / "plots" / "mplconfig"
PLOTS_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOTS_CONFIG))

EXP_TEXT_BINARY_DIR = BASE_DIR.parents[2] / "text" / "binary" / "sst2"


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_edges(edges: str | None, bin_width: float, min_d: float, max_d: float) -> np.ndarray:
    if edges:
        values = sorted(float(x) for x in edges.split(","))
        if values[0] > min_d:
            values.insert(0, min_d)
        if values[-1] < max_d:
            values.append(max_d)
        return np.asarray(values)
    edges = np.arange(min_d, max_d + bin_width, bin_width)
    if edges[-1] < max_d:
        edges = np.append(edges, max_d)
    return edges


def bin_metrics(
    distances: np.ndarray,
    opposite_distances: np.ndarray,
    overlap_mask: np.ndarray,
    edges: np.ndarray,
) -> list[dict[str, float]]:
    bins = []
    cumulative_overlap = 0.0
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (distances >= start) & (distances < end if end < edges[-1] else distances <= end)
        overlap = mask & overlap_mask
        count = int(mask.sum())
        overlap_count = int(overlap.sum())
        cumulative_overlap += overlap_count
        ratio = float(overlap_count / (count + 1e-12)) if count else 0.0
        bins.append(
            {
                "start": float(start),
                "end": float(end),
                "count": count,
                "overlap_count": overlap_count,
                "overlap_ratio": ratio,
                "cumulative_overlap": float(cumulative_overlap),
            }
        )
    return bins


def _format_bin(bin_info: dict[str, float], label: str) -> str:
    return (
        f"{label} bin [{bin_info['start']:.2f}, {bin_info['end']:.2f}]: "
        f"count={int(bin_info['count'])}, overlap={int(bin_info['overlap_count'])}, "
        f"ratio={bin_info['overlap_ratio']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bin distances and chart overlap by range.")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--bin-width", type=float, default=0.2)
    parser.add_argument("--edges", help="Comma-separated bin edges (forces fixed bins)")
    parser.add_argument(
        "--unit-bins",
        action="store_true",
        help="Force bins starting at 0 with width=bin-width (e.g. 0-1,1-2,...).",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=BASE_DIR / "metrics",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=BASE_DIR / "plots",
    )
    parser.add_argument("--title", default="Distance bin overlap growth")
    args = parser.parse_args()

    stats_path = args.metrics_dir / f"density_overlap_{args.split}_raw.json"
    stats = load_json(stats_path)
    plot_root = args.plots_dir / args.split
    plot_root.mkdir(parents=True, exist_ok=True)


    min_start = min(b["start"] for b in stats["pos_bins"])
    max_end = max(b["end"] for b in stats["pos_bins"])
    if args.unit_bins:
        edges = np.arange(0.0, max_end + args.bin_width, args.bin_width)
    else:
        edges = parse_edges(
            args.edges,
            args.bin_width,
            min_start,
            max_end,
        )
    embeddings = np.load(EXP_TEXT_BINARY_DIR / "artifacts" / "embeddings" / f"sst2_{args.split}_bge-base-en-v1-5_raw.npy")
    labels = np.load(EXP_TEXT_BINARY_DIR / "data" / "processed" / args.split / "labels.npy")

    positives = embeddings[labels == 1]
    negatives = embeddings[labels == 0]
    pos_centroid = positives.mean(axis=0)
    neg_centroid = negatives.mean(axis=0)
    pos_dist = np.linalg.norm(positives - pos_centroid, axis=1)
    neg_dist = np.linalg.norm(negatives - neg_centroid, axis=1)
    pos_to_neg = np.linalg.norm(positives - neg_centroid, axis=1)
    neg_to_pos = np.linalg.norm(negatives - pos_centroid, axis=1)
    pos_overlap_mask = pos_to_neg < pos_dist
    neg_overlap_mask = neg_to_pos < neg_dist

    pos_bins = bin_metrics(pos_dist, pos_to_neg, pos_overlap_mask, edges)
    neg_bins = bin_metrics(neg_dist, neg_to_pos, neg_overlap_mask, edges)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    centers = [(b["start"] + b["end"]) / 2 for b in pos_bins]
    ax1.plot(
        centers,
        [b["overlap_ratio"] for b in pos_bins],
        label="positive overlap ratio",
        color="#246eb5",
    )
    ax1.plot(
        centers,
        [b["overlap_ratio"] for b in neg_bins],
        label="negative overlap ratio",
        color="#f79d65",
        linestyle="--",
    )
    ax2.plot(
        centers,
        [b["cumulative_overlap"] for b in pos_bins],
        label="positive cumulative overlap",
        color="#246eb5",
        alpha=0.6,
    )
    ax2.plot(
        centers,
        [b["cumulative_overlap"] for b in neg_bins],
        label="negative cumulative overlap",
        color="#f79d65",
        linestyle=":",
        alpha=0.6,
    )
    ax1.set_xlabel("Distance bin center")
    ax1.set_ylabel("Overlap ratio per bin")
    ax2.set_ylabel("Cumulative overlap count")
    ax1.set_title(args.title)
    ax1.grid(alpha=0.2)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize="small")
    fig.tight_layout()
    output_path = plot_root / "distance-bin-overlap.png"
    plt.savefig(output_path, dpi=160)

    result = {
        "centers": centers,
        "pos_bins": pos_bins,
        "neg_bins": neg_bins,
    }
    result_path = plot_root / "distance-bin-overlap.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved bin overlap plot to {output_path}")
    print(f"Saved bin overlap stats to {result_path}")
    print("Per-bin overlap count (positive/negative):")
    for idx, (p_bin, n_bin) in enumerate(zip(pos_bins, neg_bins), start=1):
        print(f"Bin {idx}: {_format_bin(p_bin, 'pos')}; {_format_bin(n_bin, 'neg')}")


if __name__ == "__main__":
    main()
