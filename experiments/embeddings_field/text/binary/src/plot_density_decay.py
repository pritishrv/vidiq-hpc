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


def build_curve(stats: dict[str, object]) -> tuple[list[float], list[float], list[float]]:
    midpoints = [bin["midpoint"] for bin in stats["pos_bins"]]
    pos_density = [bin["density_per_unit"] for bin in stats["pos_bins"]]
    neg_density = [bin["density_per_unit"] for bin in stats["neg_bins"]]
    return midpoints, pos_density, neg_density


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot density decay from centroids.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["raw"],
    )
    parser.add_argument("--split", choices=["train", "validation"], default="train")
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
    parser.add_argument("--title", default="SST-2 density decay by centroid")
    args = parser.parse_args()

    plot_root = args.plots_dir / args.split
    plot_root.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    for variant in args.variants:
        metrics_path = args.metrics_dir / f"density_overlap_{args.split}_{variant}.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        stats = load_stats(metrics_path)
        midpoints, pos_curve, neg_curve = build_curve(stats)
        plt.plot(midpoints, pos_curve, marker="o", label=f"{variant} positive", linewidth=2)
        plt.plot(midpoints, neg_curve, marker="s", label=f"{variant} negative", linewidth=2)

    plt.title(args.title)
    plt.xlabel("Distance from own centroid")
    plt.ylabel("Density per unit distance")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = plot_root / "density-decay-combined.png"
    plt.savefig(output_path, dpi=160)
    print(f"Saved density decay plot to {output_path}")


if __name__ == "__main__":
    main()
