from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from metrics import ensure_dir, read_json, upper_triangle_values, write_json


def compute_rsa_matrix(rdm_payload: dict[str, dict], metric_key: str) -> tuple[list[str], np.ndarray]:
    names = sorted(rdm_payload.keys(), key=lambda value: int(value))
    values = [upper_triangle_values(rdm_payload[name][metric_key]) for name in names]
    rsa = np.zeros((len(names), len(names)), dtype=np.float64)
    for i, left in enumerate(values):
        for j, right in enumerate(values):
            rsa[i, j] = float(spearmanr(left, right).statistic)
    return names, rsa


def plot_heatmap(names: list[str], rsa: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(rsa, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_title("Image RSA Across Erasure Checkpoints")
    for row in range(rsa.shape[0]):
        for col in range(rsa.shape[1]):
            ax.text(col, row, f"{rsa[row, col]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RSA across image centroid-distance checkpoints.")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("experiments/embeddings_field/image/emotions/fer2013_balanced_no_disgust_clip_vitl14"),
    )
    parser.add_argument("--rdm-path", type=Path, default=None)
    parser.add_argument("--metric", choices=["euclidean_norm_matrix", "cosine_norm_matrix"], default="euclidean_norm_matrix")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    rdm_path = args.rdm_path or experiment_root / "reports" / "phase3" / "rdm_checkpoints.json"
    output_root = ensure_dir(args.output_root or experiment_root / "reports" / "phase5")

    payload = read_json(rdm_path)
    names, rsa = compute_rsa_matrix(payload, args.metric)
    plot_heatmap(names, rsa, output_root / "rsa_correlation_matrix.png")
    write_json(
        output_root / "rsa_results.json",
        {
            "metric": args.metric,
            "checkpoints": names,
            "matrix": rsa.tolist(),
        },
    )
    print(f"Saved RSA analysis to {output_root}")


if __name__ == "__main__":
    main()
