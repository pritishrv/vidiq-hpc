from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PAIRWISE_ROOT_DEFAULT = BASE_DIR / "pairwise"
PLOTS_CONFIG = PAIRWISE_ROOT_DEFAULT / "mplconfig"
PLOTS_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOTS_CONFIG))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VIDIQ_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DATA_ROOT = VIDIQ_ROOT / "experiments" / "text" / "multiclass" / "dair-ai-emotion"
DEFAULT_EMBEDDINGS_NAME = "dair_ai_emotion_train_bge-base-en-v1-5_raw.npy"
DEFAULT_LABELS_REL = Path("data") / "processed" / "train" / "labels.npy"
CONFIG_PATH = DEFAULT_DATA_ROOT / "configs" / "bge-ablation-stage.json"
DEFAULT_LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def load_label_names() -> list[str]:
    if not CONFIG_PATH.exists():
        return DEFAULT_LABEL_NAMES
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("dataset", {}).get("label_names", [])


def load_data(embeddings_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    return embeddings, labels


def project_pca(points: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components].T
    coords = centered @ components
    return coords, centroid, components


def density_belt_radii(distances: np.ndarray, bins: int = 80, drop_ratio: float = 0.1) -> tuple[float, float]:
    if len(distances) == 0:
        return 0.0, 0.0
    hist, edges = np.histogram(distances, bins=bins)
    widths = np.diff(edges)
    widths = np.where(widths <= 0, 1e-6, widths)
    density_per_unit = hist / widths
    peak_idx = int(np.argmax(density_per_unit)) if density_per_unit.size else 0
    peak_radius = float((edges[peak_idx] + edges[peak_idx + 1]) / 2)
    threshold = density_per_unit[peak_idx] * drop_ratio
    drop_idx = peak_idx
    for idx in range(peak_idx, len(density_per_unit)):
        if density_per_unit[idx] <= threshold:
            drop_idx = idx
            break
    drop_radius = float((edges[drop_idx] + edges[drop_idx + 1]) / 2)
    return peak_radius, drop_radius


def sample_overlap_angles(coords: np.ndarray, centroid: np.ndarray, max_angles: int = 6) -> np.ndarray:
    if len(coords) == 0:
        return np.array([])
    vectors = coords - centroid
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles = np.sort(angles)
    if len(angles) <= max_angles:
        return angles
    indices = np.linspace(0, len(angles) - 1, max_angles, dtype=int)
    return angles[indices]


def plot_overlap_arcs(
    ax: matplotlib.axes.Axes,
    centroid: np.ndarray,
    radius: float,
    angles: np.ndarray,
    color: str,
    linestyle: str,
    label: str | None = None,
    arc_width: float = np.pi / 18,
) -> None:
    if radius <= 0 or len(angles) == 0:
        return
    for idx, angle in enumerate(angles):
        theta = np.linspace(angle - arc_width, angle + arc_width, 64)
        x = centroid[0] + radius * np.cos(theta)
        y = centroid[1] + radius * np.sin(theta)
        plot_kwargs = dict(color=color, linestyle=linestyle, linewidth=1.4)
        if label and idx == 0:
            plot_kwargs["label"] = label
        else:
            plot_kwargs["alpha"] = 0.8
        ax.plot(x, y, **plot_kwargs)


def compute_plot_scale(coords: np.ndarray, centroid: np.ndarray, distances: np.ndarray) -> float:
    if len(coords) == 0 or len(distances) == 0:
        return 1.0
    plot_dists = np.linalg.norm(coords - centroid, axis=1)
    auto = np.where(distances <= 0, 1e-6, distances)
    ratios = plot_dists / auto
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return 1.0
    return float(np.median(ratios))


def ensure_pair_dirs(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def scatter_pair(
    pair_dir: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    a: int,
    b: int,
):
    mask = (labels == a) | (labels == b)
    subset_embeddings = embeddings[mask]
    subset_labels = labels[mask]

    coords, centroid, components = project_pca(subset_embeddings)

    # compute centroids of each class before PCA projection
    class_centroids = {}
    for cls in (a, b):
        cls_points = embeddings[labels == cls]
        centroid_vec = cls_points.mean(axis=0)
        class_centroids[cls] = (centroid_vec - centroid) @ components

    other_centroid = class_centroids[b if subset_labels[0] == a else a]  # not used

    # determine overlap per point
    a_centroid = class_centroids[a]
    b_centroid = class_centroids[b]
    a_mask = subset_labels == a
    b_mask = subset_labels == b
    distances_a = np.linalg.norm(subset_embeddings[a_mask] - embeddings[labels == a].mean(axis=0), axis=1)
    distances_a_to_b = np.linalg.norm(subset_embeddings[a_mask] - embeddings[labels == b].mean(axis=0), axis=1)
    distances_b = np.linalg.norm(subset_embeddings[b_mask] - embeddings[labels == b].mean(axis=0), axis=1)
    distances_b_to_a = np.linalg.norm(subset_embeddings[b_mask] - embeddings[labels == a].mean(axis=0), axis=1)

    overlap_mask = np.zeros(len(subset_labels), dtype=bool)
    overlap_mask[subset_labels == a] = distances_a_to_b < distances_a
    overlap_mask[subset_labels == b] = distances_b_to_a < distances_b

    a_peak_radius, a_drop_radius = density_belt_radii(distances_a)
    b_peak_radius, b_drop_radius = density_belt_radii(distances_b)
    a_coords = coords[a_mask]
    b_coords = coords[b_mask]
    a_scale = compute_plot_scale(a_coords, a_centroid, distances_a)
    b_scale = compute_plot_scale(b_coords, b_centroid, distances_b)
    a_peak_plot = a_peak_radius * a_scale
    a_drop_plot = a_drop_radius * a_scale
    b_peak_plot = b_peak_radius * b_scale
    b_drop_plot = b_drop_radius * b_scale
    a_overlap_angles = sample_overlap_angles(coords[a_mask & overlap_mask], a_centroid)
    if len(a_overlap_angles) == 0:
        a_overlap_angles = sample_overlap_angles(a_coords, a_centroid)
    b_overlap_angles = sample_overlap_angles(coords[b_mask & overlap_mask], b_centroid)
    if len(b_overlap_angles) == 0:
        b_overlap_angles = sample_overlap_angles(b_coords, b_centroid)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#246eb5", "#f79d65"]
    for idx, cls in enumerate((a, b)):
        mask_cls = subset_labels == cls
        ax.scatter(
            coords[mask_cls, 0],
            coords[mask_cls, 1],
            s=28,
            alpha=0.55,
            c=colors[idx],
            label=label_names[cls] if cls < len(label_names) else f"class_{cls}",
            edgecolors="none",
        )

    # highlight overlaps
    ax.scatter(
        coords[overlap_mask, 0],
        coords[overlap_mask, 1],
        facecolors="none",
        edgecolors="k",
        linewidth=0.8,
        s=60,
        label="overlap point",
    )

    ax.scatter(
        a_centroid[0],
        a_centroid[1],
        marker="X",
        s=120,
        c=colors[0],
        edgecolors="k",
        linewidth=1.2,
        label=f"{label_names[a]} centroid",
    )
    ax.scatter(
        b_centroid[0],
        b_centroid[1],
        marker="X",
        s=120,
        c=colors[1],
        edgecolors="k",
        linewidth=1.2,
        label=f"{label_names[b]} centroid",
    )

    plot_overlap_arcs(
        ax,
        a_centroid,
        a_peak_plot,
        a_overlap_angles,
        color=colors[0],
        linestyle="-",
        label=f"{label_names[a]} belt peak",
    )
    plot_overlap_arcs(
        ax,
        a_centroid,
        a_drop_plot,
        a_overlap_angles,
        color=colors[0],
        linestyle="--",
        label=None,
    )
    plot_overlap_arcs(
        ax,
        b_centroid,
        b_peak_plot,
        b_overlap_angles,
        color=colors[1],
        linestyle="-",
        label=f"{label_names[b]} belt peak",
    )
    plot_overlap_arcs(
        ax,
        b_centroid,
        b_drop_plot,
        b_overlap_angles,
        color=colors[1],
        linestyle="--",
        label=None,
    )

    ax.set_title(f"{label_names[a]} vs {label_names[b]} scatter")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path = pair_dir / "scatter-centroids.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved scatter {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scatter plot per emotion pair showing centroids and overlap."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--embeddings-name", default=DEFAULT_EMBEDDINGS_NAME)
    parser.add_argument("--labels-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PAIRWISE_ROOT_DEFAULT)
    args = parser.parse_args()

    embeddings_path = args.data_root / "artifacts" / "embeddings" / args.embeddings_name
    labels_path = args.labels_path or args.data_root / DEFAULT_LABELS_REL
    embeddings, labels = load_data(embeddings_path, labels_path)
    label_names = load_label_names()
    for a, b in combinations(range(len(label_names)), 2):
        pair_dir = ensure_pair_dirs(args.output_root / f"{label_names[a]}_vs_{label_names[b]}")
        scatter_pair(pair_dir, embeddings, labels, label_names, a, b)


if __name__ == "__main__":
    main()
