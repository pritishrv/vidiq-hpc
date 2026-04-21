# Multiclass Pairwise Embedding Report

This effort documents the work to understand how the raw BGE-Base embeddings from `dair-ai/emotion` organize six emotion classes in embedding space, especially when viewed pairwise. The goal is to keep the findings, diagnostics, and newly created balanced dataset inside `experiments/embeddings_field` so they can be re-used for downstream prompt context.

## What Was Done

- Created `balanced_dataset` by subsampling 572 examples from each emotion label (the smallest class size) and saved the trimmed texts, labels, and embeddings under `experiments/embeddings_field/text/multiclass/balanced_dataset/`.
- Added automation under `experiments/embeddings_field/text/multiclass/src/run_pairwise_density_overlap.py` to iterate over all 15 emotion pairs, compute centroid-informed distances, per-bin densities/overlaps, and save JSON metrics plus `density-decay.png` and `overlap-volume.png` in `pairwise/<labelA>_vs_<labelB>/`.
- Built plotting utilities that reuse those metrics to deliver:
  - `overlap-count-bar.png` (bin-wise bar charts of overlapping counts) via `plot_pairwise_overlap_bar.py`.
  - `scatter-centroids.png` (PCA scatter of the two classes with centroids and overlap points highlighted) via `plot_pairwise_scatter_with_centroids.py`.
- Ran the same pipeline on the balanced dataset and stored the outputs in `experiments/embeddings_field/text/multiclass/balanced_pairwise`.
- Maintained the original unbalanced outputs under `experiments/embeddings_field/text/multiclass/pairwise` for comparison.

## Findings

- Each emotion class builds up a “belt centre”—a radius where the density curve peaks—even though the centroid sits at the origin. The overlap metrics spike immediately after this belt, so the shared region is a narrow annulus rather than a point.
- Semantically distant pairs (e.g., sadness vs. fear) show thin belts and short overlap sections; semantically similar pairs (joy/love, anger/fear/sadness) have overlapping belts that span wider radii and far more points, evident in the bar charts and scatter overlays.
- After balancing, these structures persist, indicating they are intrinsic to the embedding geometry rather than artifacts of class imbalance.

## General Project Direction

The broader ambition remains to attack cross-modal trust/truth questions by examining embedding geometry across text, image, and eventually video. These binary and multiclass text pipelines establish a reproducible workflow: prep the data, compute multiple normalization/centering variants, collect rich geometry/overlap metrics, and visualize the resulting belts/overlap pockets. Future steps will reuse the same scripts in the `embeddings_field` hierarchy for image/video embeddings and compare their behavior to the text baselines created here.
