from __future__ import annotations

import json
import os
from pathlib import Path

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
PAIRWISE_ROOT_DEFAULT = BASE_DIR / "pairwise"
PLOTS_CONFIG = PAIRWISE_ROOT_DEFAULT / "mplconfig"
PLOTS_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOTS_CONFIG))


def gather_pairs(root: Path) -> list[Path]:
    return [p for p in root.iterdir() if p.is_dir()]


def load_stats(pair_dir: Path) -> dict:
    metrics_path = pair_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def plot_bar(pair_dir: Path, stats: dict) -> None:
    centers = [(b["start"] + b["end"]) / 2 for b in stats["a_bins"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    ax.bar(
        np.array(centers) - width / 2,
        [b["overlap_count"] for b in stats["a_bins"]],
        width,
        label=f"{stats['labels'][0]} overlap",
        color="#246eb5",
    )
    ax.bar(
        np.array(centers) + width / 2,
        [b["overlap_count"] for b in stats["b_bins"]],
        width,
        label=f"{stats['labels'][1]} overlap",
        color="#f79d65",
    )
    ax.set_title(f"{stats['labels'][0]} vs {stats['labels'][1]} bin overlap counts")
    ax.set_xlabel("Distance bin center")
    ax.set_ylabel("Overlap count")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = pair_dir / "overlap-count-bar.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pairwise overlap bars.")
    parser.add_argument("--root", type=Path, default=PAIRWISE_ROOT_DEFAULT)
    args = parser.parse_args()
    for pair_dir in gather_pairs(args.root):
        try:
            stats = load_stats(pair_dir)
        except FileNotFoundError:
            continue
        plot_bar(pair_dir, stats)


if __name__ == "__main__":
    main()
