## Binary Embedding Density Analysis Plan

### Aim
Analyze `BAAI/bge-base-en-v1.5` embeddings for SST-2 in the new `embeddings_field/text/binary` experiment area. We will compare the **raw** pooled vectors and the **mean-centered + L2 normalized** variant to understand centroid structure, density falloff, outliers, and inter-class overlap.

### Step 1: Data & Embedding Loading
1. Load SST-2 train/validation text, labels, and two embedding variants: raw and centered_l2 from `experiments/text/binary/sst2/artifacts/embeddings/`.
2. Optionally limit to the official validation split or a stratified subset for rapid prototyping, but keep the full 768D vectors for metric calculations.
3. Record tokenizer metadata (max length, truncation rate) for reference.

### Step 2: Centroid computation
1. For each class label (positive/negative), compute the centroid (mean) vector for each variant.
2. Store centroids plus counts of points per class; log them for later comparison.

### Step 3: Distance & Density profiling
1. For every point, compute distance to its own centroid and to the opposite centroid (cosine/euclidean).
2. Bin distances into quantiles (e.g., quartiles) or fixed-radius shells; compute density = points_per_bin / shell_volume proxy (difference of spheres) and record the fraction of opposite-class points landing in the same bin (overlap volume). The script `run_density_overlap_analysis.py` will output both.
3. Produce a density curve per class showing how density decreases as distance increases.
4. Compare density curves between raw and centered variants to see if centering tightens mass near the centroid.

### Step 4: Outlier and overlap metrics
1. Identify the furthest point (max distance) per class and inspect its text sample for semantic drift.
2. Count overlap cases where cross-centroid distance < own-centroid distance (per variant) and treat those as the overlap metric. Report overlap ratio, distribution, and relate them back to density bins.
3. Compute ratio of mean distances for own vs opposite centroid; use that to flag starting points for visualization.

### Step 5: Visualizations
For each variant:
1. **Radial density plot**: Show density vs distance (line chart). Plot two lines (positive, negative) on single axes.
2. **Projected scatter**: Use PCA/UMAP (with t-SNE fallback) on validation set, color by distance bin; overlay centroids and highlight farthest outliers.
3. **Overlap heatmap**: Show confusion-style heatmap of cross-centroid distances (gradient across classes).
4. **Distance histogram**: Separate histograms for own and opposite centroid distances, emphasizing overlap area.

Visualizations should live under `experiments/embeddings_field/text/binary/plots/` and include metadata JSON identifying the variant/distance thresholds.

### Step 6: Reporting
1. Save numeric summaries (centroid positions, densities, overlap ratios) under `metrics/`.
2. Draft a plan/report document in this folder summarizing methodology, key findings, and next actions (e.g., extend to embedded images/videos later).

### Visual Questions / Deliverables
- How much density remains within the first quartile of each centroid?
- Do the two variants differ in how quickly their density decays?
- Which points are the most distant? Do they align with misclassified examples or lexically ambiguous cases?
- How often does a point sit closer to the opposite centroid?
- Can we triangulate these behaviors with PCA/UMAP plots showing distance bins?

This plan keeps the analysis in the multidimensional space, and yes—each step has a corresponding visualization so you can explore the answers. Let me know if you want me to extend the plan for image/video embeddings next.
