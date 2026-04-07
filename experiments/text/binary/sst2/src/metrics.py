from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, silhouette_score
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier


def _safe_silhouette(vectors: np.ndarray, labels: np.ndarray) -> float | None:
    try:
        return float(silhouette_score(vectors, labels))
    except Exception:
        return None


def logistic_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    max_iter: int,
) -> dict[str, Any]:
    clf = LogisticRegression(max_iter=max_iter, random_state=42)
    clf.fit(train_x, train_y)
    preds = clf.predict(val_x)
    return {
        "accuracy": float(accuracy_score(val_y, preds)),
        "macro_f1": float(f1_score(val_y, preds, average="macro")),
    }


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
    pos = vectors[labels == 1]
    neg = vectors[labels == 0]
    mu_pos = pos.mean(axis=0, keepdims=True)
    mu_neg = neg.mean(axis=0, keepdims=True)

    centroid_cos = float(cosine_distances(mu_pos, mu_neg)[0, 0])
    centroid_euc = float(euclidean_distances(mu_pos, mu_neg)[0, 0])

    intra = []
    inter = []
    for idx in range(len(vectors)):
        same = vectors[labels == labels[idx]]
        diff = vectors[labels != labels[idx]]
        intra.append(float(np.linalg.norm(vectors[idx] - same.mean(axis=0))))
        inter.append(float(np.linalg.norm(vectors[idx] - diff.mean(axis=0))))
    intra_mean = float(np.mean(intra))
    inter_mean = float(np.mean(inter))

    return {
        "centroid_cosine_distance": centroid_cos,
        "centroid_euclidean_distance": centroid_euc,
        "avg_intra_class_distance": intra_mean,
        "avg_inter_class_distance": inter_mean,
        "intra_to_inter_ratio": float(intra_mean / inter_mean) if inter_mean else None,
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


def evaluate_model_selection(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    k_values: list[int],
    logistic_max_iter: int,
) -> dict[str, Any]:
    return {
        "logistic_regression": logistic_probe(train_x, train_y, val_x, val_y, logistic_max_iter),
        "knn": knn_probe(train_x, train_y, val_x, val_y, k_values),
        "geometry": centroid_metrics(val_x, val_y),
        "cluster": {"silhouette": _safe_silhouette(val_x, val_y)},
        "pca": pca_metrics(val_x),
    }


def confusion_matrix_payload(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    max_iter: int,
) -> dict[str, Any]:
    clf = LogisticRegression(max_iter=max_iter, random_state=42)
    clf.fit(train_x, train_y)
    preds = clf.predict(val_x)
    return {
        "labels": [0, 1],
        "matrix": confusion_matrix(val_y, preds, labels=[0, 1]).tolist(),
    }
