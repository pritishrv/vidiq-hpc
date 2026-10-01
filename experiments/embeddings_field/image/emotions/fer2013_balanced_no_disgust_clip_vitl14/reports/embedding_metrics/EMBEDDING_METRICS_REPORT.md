# FER2013 CLIP Embedding Metrics Report

Date: 2026-09-01

## Scope

This report summarizes the image-side embedding diagnostics for:

- dataset: balanced FER2013 without `disgust`
- split analyzed: `train`
- encoder: `openai/clip-vit-large-patch14`
- embedding dimensionality: `768`

Source metrics:

- `train/summary.json`
- `train/centroid-distance-matrix.md`

## Cluster Diagnostics

- silhouette score: `-0.0008713421411812305`
- Davies-Bouldin score: `8.081093244483114`
- examples: `19026`
- classes: `6`

Interpretation:

- The silhouette score is effectively zero and slightly negative, which indicates strong overlap between class clouds in the raw CLIP embedding space.
- This is closer to the earlier pretrained-text pattern than the fine-tuned Qwen pattern.

## Centroid Geometry

Nearest centroid pairs in Euclidean space:

- `fear` ↔ `sad`: `2.0257`
- `angry` ↔ `fear`: `2.2576`
- `happy` ↔ `neutral`: `2.7556`

Most separated centroid pairs:

- `neutral` ↔ `surprise`: `4.9933`
- `happy` ↔ `surprise`: `4.8659`
- `sad` ↔ `surprise`: `4.5044`

Interpretation:

- Negative-affect facial classes (`fear`, `sad`, `angry`) are the tightest centroid family.
- `surprise` sits furthest from most other class centroids in this CLIP space.
- `happy` and `neutral` are notably closer than `happy` is to the negative-affect classes.

## Outputs

- `train/summary.json`
- `train/centroid-distance-matrix.md`

