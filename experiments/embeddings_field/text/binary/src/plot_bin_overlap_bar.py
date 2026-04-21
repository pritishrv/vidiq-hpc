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


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-bin overlap counts.")
    parser.add_argument(
        "--json-path",
        type=Path,
        default=BASE_DIR / "plots" / "train" / "distance-bin-overlap.json",
    )
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "plots" / "train")
    args = parser.parse_args()

    data = load_json(args.json_path)
    pos_bins = data["pos_bins"]
    neg_bins = data["neg_bins"]
    centers = [(b["start"] + b["end"]) / 2 for b in pos_bins]

    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        np.array(centers) - width / 2,
        [b["overlap_count"] for b in pos_bins],
        width,
        label="positive overlap",
        color="#246eb5",
    )
    ax.bar(
        np.array(centers) + width / 2,
        [b["overlap_count"] for b in neg_bins],
        width,
        label="negative overlap",
        color="#f79d65",
    )
    ax.set_title("Overlap counts per distance bin")
    ax.set_xlabel("Distance bin center")
    ax.set_ylabel("Overlap count")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output_path = args.output_dir / "distance-bin-overlap-counts.png"
    fig.savefig(output_path, dpi=160)
    print(f"Saved overlap count bar plot to {output_path}")


if __name__ == "__main__":
    main()
