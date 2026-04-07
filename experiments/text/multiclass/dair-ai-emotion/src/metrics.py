from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.neighbors import KNeighborsClassifier


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


def logistic_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    max_iter: int,
) -> tuple[dict[str, Any], np.ndarray]:
    clf = LogisticRegression(max_iter=max_iter, random_state=42)
    clf.fit(train_x, train_y)
    preds = clf.predict(val_x)
    return {
        "accuracy": float(accuracy_score(val_y, preds)),
        "macro_f1": float(f1_score(val_y, preds, average="macro")),
    }, preds


def knn_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    k_values: list[int],
) -> dict[str, Any]:
    metrics = {}
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(train_x, train_y)
        preds = clf.predict(val_x)
        metrics[str(k)] = {
            "accuracy": float(accuracy_score(val_y, preds)),
            "macro_f1": float(f1_score(val_y, preds, average="macro")),
        }
    return metrics


def centroid_metrics(vectors: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    classes = sorted(int(x) for x in np.unique(labels))
    centroids = []
    for cls in classes:
        centroids.append(vectors[labels == cls].mean(axis=0))
    centroids_arr = np.vstack(centroids)

    centroid_cos = cosine_distances(centroids_arr, centroids_arr)
    centroid_euc = euclidean_distances(centroids_arr, centroids_arr)

    same_distances = []
    cross_distances = []
    for idx in range(len(vectors)):
        same = vectors[labels == labels[idx]]
        cross = vectors[labels != labels[idx]]
        if len(same) > 1:
            same_centroid = same.mean(axis=0)
            same_distances.append(float(np.linalg.norm(vectors[idx] - same_centroid)))
        if len(cross) > 0:
            cross_centroid = cross.mean(axis=0)
            cross_distances.append(float(np.linalg.norm(vectors[idx] - cross_centroid)))

    avg_same = float(np.mean(same_distances)) if same_distances else None
    avg_cross = float(np.mean(cross_distances)) if cross_distances else None
    ratio = float(avg_same / avg_cross) if avg_same is not None and avg_cross not in (None, 0.0) else None

    return {
        "labels": classes,
        "centroid_cosine_distance_matrix": centroid_cos.tolist(),
        "centroid_euclidean_distance_matrix": centroid_euc.tolist(),
        "avg_same_class_distance": avg_same,
        "avg_cross_class_distance": avg_cross,
        "same_to_cross_distance_ratio": ratio,
    }


def pca_metrics(vectors: np.ndarray, n_components: int = 5) -> dict[str, Any]:
    n_components = min(n_components, vectors.shape[0], vectors.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(vectors)
    values = [float(x) for x in pca.explained_variance_ratio_]
    return {
        "explained_variance_ratio": values,
        "top_component_ratio": values[0] if values else None,
    }


def confusion_matrix_payload(
    labels: list[int],
    val_y: np.ndarray,
    preds: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    return {
        "labels": labels,
        "label_names": label_names,
        "matrix": confusion_matrix(val_y, preds, labels=labels).tolist(),
    }
