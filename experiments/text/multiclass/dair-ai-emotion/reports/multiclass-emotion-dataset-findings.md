# dair-ai/emotion Multiclass Findings

## Experiment Essentials

- dataset: `dair-ai/emotion`
- final model: `BAAI/bge-base-en-v1.5`
- embeddings: raw, L2-normalized, centered+L2 mean pooled vectors on the full train and validation splits
- validation labels: six emotions (sadness, joy, love, anger, fear, surprise)

## Quantitative Outcome

- raw mean pool:
  - logistic reg macro F1: `0.697`
  - kNN@5 macro F1: `0.647`
  - silhouette: `0.0038`
  - Davies-Bouldin: `7.91`
- L2 mean pool:
  - logistic reg macro F1: `0.655`
  - kNN@5 macro F1: `0.650`
- centered+L2 mean pool:
  - logistic reg macro F1: `0.676`
  - kNN@5 macro F1: `0.636`
- centroids remain near each other (same/cross distance ratio ≈ `0.953`)
- the best metric balance is still the raw embeddings, so they remain the default

## Confusion Insights

- `joy` vs `love`, and the `anger`/`fear`/`sadness` trio show the most overlap (see `metrics/confusion-matrix.json`)
- `surprise` is the weakest class, frequently confused with multiple others
- these patterns match the dataset narrative and explain the tiny silhouette scores

## Geometry Interpretation

- PCA plots show a dominant axis but no six-way clean clustering
- t-SNE fallback outlines partially overlapping blobs, which is expected when classes share affective meaning
- centroid heatmaps confirm the pairwise distances listed in the summary
- the overlaps are real (not just visualization artifacts), as the metrics were computed in the native 768D space

## Decision

- keep `BAAI/bge-base-en-v1.5` raw mean pooling as the multiclass default
- reference L2 and centered-L2 variants for stress tests or data shifts
- the next additions should be:
  1. a TF-IDF + logistic regression lexical baseline
  2. optional CLS pooling or `max_length=128` check if truncation changes behaviour
