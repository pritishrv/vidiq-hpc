# Multiclass Pairwise Embedding Experiment Plan

## Objective

Replicate the density/overlap experiments from the binary pipeline for every possible pair of labels in `dair-ai/emotion`. The goal is to characterize how any two emotions occupy the same embedding space slice (centroid density, overlap ratios, cumulative overlap volume), and to preserve those diagnostics under `embeddings_field/text/multiclass` for future prompt context.

## Dataset Summary

- Source: `dair-ai/emotion` (train split) using the raw BGE-Base embeddings already extracted in `experiments/text/multiclass/dair-ai-emotion/artifacts/embeddings`.
- Labels: `["sadness", "joy", "love", "anger", "fear", "surprise"]`.
- Embedding dimension: 768 (raw mean-pooled BGE vectors).

## Steps

1. **Structure**: create `experiments/embeddings_field/text/multiclass` with `metrics/`, `plots/`, and `pair-comparisons/` accomodations. Each pair of labels gets its own subdirectory containing diagnostics.
2. **Pair loops**: iterate over all combinations of two emotions (15 total). For each pair:
   - Filter embeddings + labels to the two classes.
   - Compute centroid distances, per-class distance bins, bin overlap counts, and cumulative overlap.
   - Save the stats JSON & text summary so future prompts can parse counts/ratios for each bin.
   - Produce the same **density-decay** and **overlap-volume** plots as in the binary experiment.
3. **Visualization**:
   - Provide combined density-decay lines (positive vs. negative bins) so the shape of each class’s mass becomes visible.
   - Show overlap-volume curves to highlight where shared space accumulates.
4. **Reporting**:
   - Document any notable overlaps (e.g., joy/love with low ratio vs. anger/fear with higher overlap).
   - Store metrics and plots within `embeddings_field/text/multiclass/pair-comparisons/<label_a>_vs_<label_b>/`.
5. **Automation**: wrap the above in a script meant to be rerun if embeddings change. The plan also includes rerunning the per-pair two-plot generator (density-decay, overlap-volume) every time we add new embeddings or label combinations.

## Success Criteria

- Each emotion pair has JSON metrics + density/overlap plots saved in `embeddings_field/text/multiclass`.
- The visual style matches the binary pipeline so plots are immediately comparable across experiments.
- We can reuse the generated metadata + visuals as prompt context for downstream analysis.
