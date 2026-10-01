from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_metadata(experiment_root: Path) -> dict[str, Any]:
    return read_json(experiment_root / "data" / "processed" / "metadata.json")


def load_split_embeddings(experiment_root: Path, split: str) -> np.ndarray:
    return np.load(experiment_root / "artifacts" / "embeddings" / f"{split}_raw.npy")


def load_split_labels(experiment_root: Path, split: str) -> np.ndarray:
    return np.load(experiment_root / "data" / "processed" / split / "labels.npy")


def available_embedding_splits(experiment_root: Path) -> list[str]:
    splits: list[str] = []
    for split in ("train", "test", "validation"):
        if (experiment_root / "artifacts" / "embeddings" / f"{split}_raw.npy").exists():
            splits.append(split)
    return splits


def compute_centroids(vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    classes = sorted(int(x) for x in np.unique(labels))
    return np.vstack([vectors[labels == cls].mean(axis=0) for cls in classes])


def normalize_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    max_value = float(np.max(matrix)) if matrix.size else 0.0
    if max_value <= 0.0:
        return np.zeros_like(matrix, dtype=np.float64)
    return matrix / max_value


def _safe_silhouette(vectors: np.ndarray, labels: np.ndarray) -> float | None:
    try:
        return float(silhouette_score(vectors, labels))
    except Exception:
        return None


def _safe_davies_bouldin(vectors: np.ndarray, labels: np.ndarray) -> float | None:
    try:
        return float(davies_bouldin_score(vectors, labels))
    except Exception:
        return None


def pca_metrics(vectors: np.ndarray, n_components: int = 5) -> dict[str, Any]:
    n_components = min(n_components, vectors.shape[0], vectors.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(vectors)
    ratios = [float(x) for x in pca.explained_variance_ratio_]
    return {
        "explained_variance_ratio": ratios,
        "top_component_ratio": ratios[0] if ratios else None,
    }


def centroid_summary(vectors: np.ndarray, labels: np.ndarray, label_names: list[str]) -> dict[str, Any]:
    classes = sorted(int(x) for x in np.unique(labels))
    centroids = compute_centroids(vectors, labels)
    euclidean = euclidean_distances(centroids, centroids)
    cosine = cosine_distances(centroids, centroids)

    return {
        "labels": classes,
        "label_names": [label_names[idx] for idx in classes],
        "centroids": centroids.tolist(),
        "euclidean_matrix": euclidean.tolist(),
        "euclidean_norm_matrix": normalize_distance_matrix(euclidean).tolist(),
        "cosine_matrix": cosine.tolist(),
        "cosine_norm_matrix": normalize_distance_matrix(cosine).tolist(),
    }


def upper_triangle_values(matrix: list[list[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    return arr[np.triu_indices(arr.shape[0], k=1)]
