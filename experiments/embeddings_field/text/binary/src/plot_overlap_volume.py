from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
PLOTS_CONFIG = BASE_DIR / "plots" / "mplconfig"
PLOTS_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOTS_CONFIG))


def load_stats(metrics_path: Path) -> dict[str, object]:
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_overlap_volume(bins: list[dict[str, float]]) -> tuple[list[float], list[float]]:
    cumulative = []
    running = 0.0
    for b in bins:
        overlap = float(b["opposite_count"])
        width = float(b["bin_width"])
        volume = overlap * width
        running += volume
        cumulative.append(running)
    return cumulative, [float(b["overlap_ratio"]) for b in bins]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot overlap volume and ratios.")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--variants", nargs="+", default=["raw"])
    parser.add_argument("--metrics-dir", type=Path, default=BASE_DIR / "metrics")
    parser.add_argument("--plots-dir", type=Path, default=BASE_DIR / "plots")
    args = parser.parse_args()

    plot_root = args.plots_dir / args.split
    plot_root.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    colors = {"raw": "#246eb5", "centered_l2": "#f79d65"}

    for variant in args.variants:
        path = args.metrics_dir / f"density_overlap_{args.split}_{variant}.json"
        stats = load_stats(path)
        midpoints = [bin["midpoint"] for bin in stats["pos_bins"]]
        neg_midpoints = [bin["midpoint"] for bin in stats["neg_bins"]]

        pos_cumulative, pos_ratios = compute_overlap_volume(stats["pos_bins"])
        neg_cumulative, neg_ratios = compute_overlap_volume(stats["neg_bins"])

        ax1.plot(
            midpoints,
            pos_ratios,
            label=f"{variant} positive ratio",
            color=colors[variant],
            linestyle="-",
        )
        ax1.plot(
            neg_midpoints,
            neg_ratios,
            label=f"{variant} negative ratio",
            color=colors[variant],
            linestyle="--",
        )

        ax2.plot(
            midpoints,
            pos_cumulative,
            label=f"{variant} positive volume",
            color=colors[variant],
            alpha=0.5,
        )
        ax2.plot(
            neg_midpoints,
            neg_cumulative,
            label=f"{variant} negative volume",
            color=colors[variant],
            linestyle=":",
            alpha=0.5,
        )

    ax1.set_xlabel("Distance from centroid")
    ax1.set_ylabel("Overlap ratio")
    ax2.set_ylabel("Cumulative overlap volume")
    ax1.set_title("Overlap ratio and volume per distance bin")
    ax1.grid(alpha=0.2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize="small", ncol=2)
    fig.tight_layout()
    output_path = plot_root / "overlap-volume.png"
    plt.savefig(output_path, dpi=160)
    print(f"Saved overlap-volume plot to {output_path}")


if __name__ == "__main__":
    main()
