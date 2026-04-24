from __future__ import annotations

import numpy as np
import torch


def calculate_centroids(embeddings: torch.Tensor, labels: torch.Tensor) -> dict[int, torch.Tensor]:
    centroids = {}
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        class_embeddings = embeddings[labels == label]
        centroids[int(label)] = class_embeddings.mean(dim=0)
    return centroids


def calculate_radial_density(embeddings: torch.Tensor, centroid: torch.Tensor, bins: int = 50) -> dict:
    distances = torch.norm(embeddings - centroid, dim=1)
    hist, bin_edges = np.histogram(distances.numpy(), bins=bins)
    
    # Normalize by shell volume/area if required for true density
    # For now, following the "number of data points per radial distance" pattern
    return {
        "hist": hist.tolist(),
        "bin_edges": bin_edges.tolist()
    }


def calculate_overlap(embeddings_a: torch.Tensor, centroid_a: torch.Tensor, 
                      embeddings_b: torch.Tensor, centroid_b: torch.Tensor) -> float:
    # Simplified overlap: ratio of points closer to other centroid than their own
    dist_to_a = torch.norm(embeddings_a - centroid_a, dim=1)
    dist_to_b = torch.norm(embeddings_a - centroid_b, dim=1)
    
    overlap_count = (dist_to_b < dist_to_a).sum().item()
    return overlap_count / len(embeddings_a)
