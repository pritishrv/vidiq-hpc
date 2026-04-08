# Meeting Minutes — LLM Embedding Geometry: Sentiment Experiments & NeurIPS Paper

**Date:** 7 April 2026, 14:55  
**Attendees:** Daniel Sikar, Pritish Ranjan (PG-Verma)  
**Transcript:** `2026-04-07_transcript_llm-embedding-geometry_sentiment-experiments_neurips.docx`  
**NeurIPS submission deadline:** 4 May 2026

---

## Experiments Completed

### Datasets
- **Binary:** SST-2 (Stanford Sentiment Treebank) — positive/negative sentiment. Some data points are intentionally ambiguous; this is noted as a feature of the dataset, not a bug.
- **Multi-class:** Six-emotion dataset — joy, love, anger, sadness, fear, surprise.

### Model Selection
- Two candidate models evaluated on 10% of training data + 100% of validation data.
- Rationale: model selection is a lighter task than embedding validation; compute is better spent on pooling and metric selection. Chosen model carries forward to all subsequent experiments. (This justification should appear explicitly in the paper — reviewers will probe it.)

### Embedding Extraction — Three Methods Compared
| Method | Description |
|--------|-------------|
| Raw pooling | Retains magnitude and direction |
| L2 normalisation | Normalises magnitude; preserves direction. Max pairwise distance = √2 in the normalised space |
| Log normalisation | Intermediate compression |

**Binary result:** Raw pooling performed best; L2 close but not superior.  
**Multi-class:** Same pooling/model selection carries over.  
Reason: for semantic classification the direction of the embedding vector is the primary signal; magnitude adds noise rather than information.

### Dimensionality and Clustering
- Embeddings are 768-dimensional (chosen model).
- **All clustering metrics are computed in the native 768D space.** PCA to 2D is used only for visualisation; information loss is acknowledged.
- Binary clusters: visible separation between positive and negative.
- Multi-class clusters:
  - Joy + love overlap strongly.
  - Anger + sadness + fear partially overlap.
  - **Surprise is the weakest/most dispersed class** — confirmed both visually and via metrics. Interpretation: *surprise is a reaction rather than an emotion*, hence it is geometrically mixed with other states.

### Sequence Truncation
- Max sequence length: 128 tokens.
- Binary: ~0.2% of data truncated — acceptable.
- Multi-class: ~6% of data truncated — flagged as a risk; justification needed or experiment repeated with longer max length.

---

## Core Hypothesis

> There exists a mathematical formulation — expressible in terms of **centroid distance** and **cluster density** — that describes the geometric relationships between semantic classes in LLM embedding space, and this formulation holds across modalities (text, single image, video).

This is the "law of gravity in LLM embedding space": a universal geometric law, analogous to Newtonian gravity, that may be revised by future work but serves as the founding abstraction of the paper.

Key properties of the formulation:
- Operates in the native N-dimensional space (768 for text; larger for images/video).
- Reducible to a scalar radius (Euclidean distance) regardless of dimensionality.
- Density is the bridge concept: for a given density threshold (e.g. capturing 80% of class members), each class requires a different radius. This radius-per-density relationship is the geometric signature of a class.

---

## Mathematical Abstractions to Develop

1. **Centroid per class** — mean embedding vector in 768D.
2. **Inter-centroid distances** — pairwise distance matrix across all classes.
3. **Density-radius relationship** — for a given density threshold D, compute the radius r such that D% of class members lie within r of the centroid. Compare across classes. Hypothesis: surprise requires the largest r for any fixed D; tightly-packed classes (e.g. anger) require the smallest.
4. **Overlap volume** — intersection of class "spheres" defined by their density-radius. Formalise as a scalar overlap score.
5. **Cross-modality invariant** — test whether the rank ordering of inter-centroid distances and density-radius values is preserved when the same pipeline is applied to single images (and later video).

The aspiration: a small set of axioms that compress all of the above into a single law.

---

## Visualisation Plan

- **Per-class density plots**: show tight vs. diffuse clusters side by side (e.g. anger vs. surprise).
- **Pairwise overlap plots**: 2-class comparisons for the most significant pairs. Target a 2×2 or 2×4 panel for the paper.
- **Single images**: once the image pipeline is running, produce equivalent plots to enable direct visual comparison with text.
- All 2D plots use PCA; all numbers quoted in the paper use 768D metrics.

---

## Next Steps Towards Single Images

- The text pipeline (embedding extraction → clustering → centroid/density analysis) is the template.
- Single-image embeddings will have larger vectors than 768; the formulation must remain modality-agnostic.
- Goal: run one image experiment before the paper deadline to include at least a preliminary cross-modality result.

---

## Action Points

### Pritish

| # | Action | Deadline |
|---|--------|----------|
| P1 | Compute centroids for all six emotion classes in native 768D space; produce centroid coordinate table and inter-centroid distance matrix | 11 Apr |
| P2 | Pairwise overlap plots: select the 4–6 most significant class pairs and produce 2D PCA visualisations with both classes overlaid | 13 Apr |
| P3 | Density-radius experiment: for density thresholds of 80% and 99%, compute the required radius for each class; tabulate and plot | 16 Apr |
| P4 | Address the 6% truncation risk in multi-class: either justify (document that removing 6% does not shift class distributions) or re-run with longer max sequence length | 16 Apr |
| P5 | Set up single-image embedding pipeline mirroring the text pipeline (raw / L2 / log pooling, same metrics) | 21 Apr |
| P6 | Run single-image experiment; produce initial centroid and density-radius results for at least two contrasting image classes | 25 Apr |
| P7 | Ensure all experiments have complete prompts, reports, and work-diary entries (required for paper methodology section) | ongoing |

### Daniel

| # | Action | Deadline |
|---|--------|----------|
| D1 | Write the methodology justification paragraphs: model selection on 10% data; raw pooling choice; truncation decision | 12 Apr |
| D2 | Draft the mathematical formulation section: centroid, density-radius, overlap volume definitions; sketch the cross-modality invariance claim | 16 Apr |
| D3 | Produce paper outline (sections, target figures, target tables) and share with Pritish | 18 Apr |
| D4 | Draft introduction and related-work sections; incorporate literature survey already completed | 23 Apr |
| D5 | Integrate Pritish's single-image results into the cross-modality argument | 27 Apr |
| D6 | Full paper draft for joint review | 29 Apr |
| D7 | Final revisions and submission | 3 May |

---

## Submission Timeline

```
Apr 11  Centroids + inter-centroid distances (P1)
Apr 13  Pairwise overlap visualisations (P2)
Apr 16  Density-radius results + truncation resolved (P3, P4)
Apr 16  Mathematical formulation draft (D2)
Apr 18  Paper outline agreed (D3)
Apr 21  Image pipeline ready (P5)
Apr 23  Related work + intro drafted (D4)
Apr 25  Image results in (P6)
Apr 27  Cross-modality section drafted (D5)
Apr 29  Full draft complete (D6)
May 3   Final submission (D7)
May 4   NeurIPS deadline
```

---

## Also Discussed (see transcript for detail)

MacBook Mini purchase for charity/NHS hosting projects; Pritish's PhD options and funding; hackathon at Richmond American University; AI workshop for 16-year-old schoolgirls (July, Narella's event); music charity instrument-rental inventory app; Codex token budget exhausted — Gemini in use as temporary substitute.
