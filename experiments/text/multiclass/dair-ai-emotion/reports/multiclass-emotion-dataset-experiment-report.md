# Multiclass Emotion Dataset Experiment Report

## Purpose

This document records the current recommendation for the multiclass follow-up experiment in `vidiq-hpc`.

It covers:

- which dataset to use
- which embedding models to use
- how to construct embeddings
- how to validate multiclass embedding geometry
- which risks and confounds to track

It is intended to be reused as prompt context for later implementation, analysis, and reporting work.

## Scope

This recommendation is based on:

- the existing dataset surveys in this repository
- the local literature sweep over `vidiq/lit-survey/gemini`
- the extracted title / abstract / conclusion report in `vidiq/reports/title-abstract-conclusion.md`
- the current binary experiment guidance already recorded for SST-2

The immediate target use case is a short-text multiclass emotion dataset that extends the binary SST-2 setup into richer categorical structure.

## Dataset-Specific Decision

The recommended next multiclass dataset is:

- `dair-ai/emotion`

Reason:

- short-text format similar to SST-2
- clean, interpretable multiclass labels
- widely used and easy to access
- good bridge from a binary polarity experiment to multiclass affect geometry

The primary alternative is:

- `ISEAR`

But `ISEAR` is less suitable as the immediate next step because:

- it is smaller
- it is more subjective
- several classes are semantically close in ways that can make first-pass geometry harder to interpret

## dair-ai/emotion Dataset Profile

`dair-ai/emotion` is a six-class emotion classification dataset.

The common label set is:

- `sadness`
- `joy`
- `love`
- `anger`
- `fear`
- `surprise`

Important practical properties:

- short text inputs
- multiclass rather than binary
- natural follow-up to SST-2 because the inputs remain compact and sentence-like
- good for studying categorical clusters rather than only a single positive / negative axis

Important expected difficulties:

- class imbalance may be present
- nearby classes may overlap semantically
- `joy` and `love` may cluster close together
- `anger`, `fear`, and `sadness` may partially overlap
- `surprise` may behave as the least stable or least semantically coherent class

## Executive Decision

For the multiclass experiment, keep the same primary model family as the binary setup:

- contrastive sentence encoder

Primary model:

- `BAAI/bge-base-en-v1.5`

Strong alternate:

- `sentence-transformers/all-mpnet-base-v2`

Optional later comparison:

- a SimCSE-family model
- one decoder-LLM hidden-state baseline

Do not switch to raw decoder hidden states as the main model for the first multiclass pass.

## Why This Is the Current Recommendation

The literature sweep supports the same high-level logic as the binary setup:

- sentence embeddings work best when the encoder has been trained for sentence-level similarity or contrastive objectives
- embedding spaces can show anisotropy, drift, and dominant common directions
- a single distance metric is not enough
- multiclass structure requires stronger per-class evaluation than binary polarity work

The dataset reports add one more important point:

- `dair-ai/emotion` is the cleanest multiclass extension after SST-2 for emotion geometry

This makes it the best candidate for testing whether embeddings form:

- categorical clusters
- local class neighborhoods
- hierarchical or near-hierarchical relations between emotion classes

## Recommended Model Stack

### Tier 1: Primary Model

Use:

- `BAAI/bge-base-en-v1.5`

Reason:

- strong sentence embedding baseline
- practical model size
- stable for clustering and similarity analysis
- good first reference for multiclass geometry

### Tier 2: Practical Alternate

Use:

- `sentence-transformers/all-mpnet-base-v2`

Reason:

- strong sentence-transformers baseline
- good robustness check against BGE

### Tier 3: Later Comparison Models

Add only after the primary multiclass pipeline is stable:

- a SimCSE-family encoder
- one decoder-LLM hidden-state baseline

Reason:

- useful for comparison
- not required for the first multiclass geometry pass

## Embedding Construction

### Unit of Embedding

Create one embedding per text example.

For `dair-ai/emotion`, the canonical unit is:

- one text example row

Do not begin with token-level embeddings as the primary object of analysis.

### Pooling Rule

Use masked mean pooling over the final hidden states.

Default rule:

1. tokenize the text
2. run the encoder
3. take the final hidden state for all non-padding tokens
4. compute the attention-mask-weighted mean

Do not default to:

- raw `CLS` only
- first-token pooling unless the model explicitly expects it
- unpooled token sets as the main representation

Reason:

- short-text emotion examples benefit from stable snippet-level aggregation
- multiclass clustering is better studied on one vector per example than on token clouds

### Sequence Length Rule

Default starting point:

- `max_length = 64`

Escalate only if truncation is non-trivial.

Fallback:

- `max_length = 128`

Reason:

- inputs are short
- small max length improves throughput
- long-sequence settings are unlikely to help much on this dataset

### Normalization Rule

Always save at least three versions:

1. pooled embedding
2. L2-normalized pooled embedding
3. mean-centered, then L2-normalized embedding

Reason:

- centering may reduce shared-direction artifacts
- normalized vectors support cleaner distance analysis
- saving multiple variants simplifies ablation

### Suggested Saved Outputs

For each split, save:

- `embeddings_raw.npy`
- `embeddings_l2.npy`
- `embeddings_centered_l2.npy`
- `labels.npy`
- `texts.jsonl`
- metadata describing:
  - model name
  - tokenizer
  - pooling rule
  - normalization steps
  - dataset name
  - split name
  - timestamp

## Recommended Experimental Workflow

### Phase 1: Dataset Preparation

- load the dataset
- remove empty rows
- keep original text and label
- inspect class balance
- preserve the official validation split if provided
- record token-length statistics

For `dair-ai/emotion`, also record:

- per-class counts
- per-class average token length
- truncation rate under the chosen tokenizer and max length

Recommended first setup:

1. use the standard train split for extraction
2. use the validation split for labeled evaluation
3. create a small stratified train subset only for fast iteration if needed

### Phase 2: Embedding Extraction

- extract sentence embeddings with the primary encoder
- save raw, normalized, and centered-normalized embeddings
- log batch size, max sequence length, truncation policy, and device

### Phase 3: Geometry Validation

Run all validation on both:

- L2-normalized embeddings
- mean-centered + L2-normalized embeddings

The multiclass question is:

- do the six emotion classes form interpretable local and global geometry rather than just one dominant polarity split

### Phase 4: Comparative Baselines

After the first model is stable:

- repeat on `all-mpnet-base-v2`
- compare results
- only then add a decoder-LLM hidden-state baseline if still needed

## Core Validation Metrics

### Multiclass Classification Quality

Use:

- logistic regression
- linear SVM

Report:

- macro F1
- weighted F1
- overall accuracy
- per-class precision / recall / F1

Purpose:

- test whether the embedding geometry is linearly useful across all classes

Important:

- macro F1 is more important than raw accuracy when class imbalance exists

### Neighborhood Quality

Use:

- k-nearest-neighbor accuracy
- per-class neighborhood purity
- confusion among nearest-neighbor classes

Purpose:

- test whether local geometry preserves class identity

Recommended:

- evaluate `k = 1, 5, 11`

### Cluster Quality

Use:

- silhouette score
- Davies-Bouldin index
- Calinski-Harabasz score

Purpose:

- test whether the dataset forms compact and separated multiclass structure

Important:

- report both overall scores and per-class diagnostics when possible

### Distance Structure

Use:

- class centroid cosine distance matrix
- class centroid Euclidean distance matrix
- average intra-class distance
- average inter-class distance
- per-class compactness

Purpose:

- analyze which classes are geometrically close or far apart

For this dataset, the centroid-distance matrix is especially important because it can reveal:

- `joy` / `love` proximity
- `sadness` / `fear` / `anger` overlap
- `surprise` instability

### Confusion Matrix

Use:

- classifier confusion matrix
- kNN confusion matrix

Purpose:

- identify semantically plausible vs implausible class confusions

Interpretation:

- confusions between nearby emotions may be acceptable and informative
- random confusion patterns suggest poor geometry

### Dimensionality and Spectrum

Use:

- PCA explained variance ratios
- top principal component dominance
- effective rank if convenient

Purpose:

- inspect whether the space is dominated by a few directions
- detect whether one broad polarity axis is overwhelming the multiclass structure

### Isotropy / Uniformity Diagnostics

Use:

- IsoScore if implemented
- average pairwise cosine as a rough diagnostic only
- variance spectrum checks

Important:

- use these as diagnostics, not headline success metrics
- do not optimize the experiment around isotropy alone

### Representation Similarity Across Models

If multiple models are compared, use:

- CKA
- RSA-style comparisons of pairwise distance matrices

Purpose:

- compare whether different encoders produce similar multiclass geometry

## Distance Metrics to Use

### Primary

Use first:

- cosine distance
- Euclidean distance on L2-normalized embeddings

Reason:

- standard and interpretable
- enough for the first multiclass pass

### Secondary

Consider later:

- Mahalanobis distance
- DDR-style similarity
- OT-style distances

Reason:

- potentially useful
- not necessary before the primary multiclass geometry is understood

## Visualization Guidance

Use:

- PCA first
- UMAP second
- t-SNE only if needed

Always:

- color by class
- show class centroids
- annotate the legend with class counts
- avoid making strong claims from 2D plots alone

Recommended plots:

- PCA scatter with class centroids
- UMAP scatter with class centroids
- centroid-distance heatmap
- per-class intra- vs inter-distance boxplots
- confusion matrix heatmap

## Failure Modes to Watch For

### Dominant Valence Collapse

Risk:

- the model may separate examples mostly along a positive / negative axis instead of full emotion structure

Mitigation:

- inspect centroid-distance matrices
- inspect PCA components
- examine whether `joy` and `love` collapse into one side and all negative classes collapse into the other

### Class Imbalance

Risk:

- majority classes can dominate the apparent geometry

Mitigation:

- use macro F1
- report per-class metrics
- inspect per-class neighborhood purity
- consider stratified subsampling for diagnostic experiments

### Lexical Shortcut Separation

Risk:

- classes may separate because of repeated lexical markers rather than richer emotion semantics

Mitigation:

- inspect nearest neighbors
- inspect class prototypes and boundary examples
- check whether a simple lexical baseline already performs unusually strongly

### Over-reading 2D Projections

Risk:

- UMAP or t-SNE may suggest clusters that are weaker in the original space

Mitigation:

- pair every plot with numerical metrics

### Semantically Adjacent Class Overlap

Risk:

- nearby emotions may naturally overlap and be mistaken for model failure

Mitigation:

- use confusion matrices and centroid distances
- interpret overlap semantically, not only numerically

## Recommended Baselines

### Embedding Baselines

- `BAAI/bge-base-en-v1.5`
- `sentence-transformers/all-mpnet-base-v2`

### Lightweight Lexical Baseline

- TF-IDF + logistic regression

Purpose:

- determine whether dense embeddings improve over a simple lexical classifier

### Supervised Upper-Bound Reference

- a standard multiclass classifier fine-tuned directly on `dair-ai/emotion`

Purpose:

- provide a task-accuracy reference
- not the main geometry model

## Recommended First Experiment Matrix

### Dataset

Start with:

- `dair-ai/emotion`

### Models

Run:

1. `BAAI/bge-base-en-v1.5`
2. `sentence-transformers/all-mpnet-base-v2`

Optional later:

3. SimCSE-family model
4. one decoder-LLM hidden-state baseline

### Representation Variants

For each model, evaluate:

1. raw pooled
2. L2-normalized
3. mean-centered + L2-normalized

### Metrics Panel

For each run, report:

- macro F1
- weighted F1
- accuracy
- per-class F1
- kNN accuracy at `k = 1, 5, 11`
- per-class neighborhood purity
- silhouette score
- Davies-Bouldin index
- centroid cosine distance matrix
- centroid Euclidean distance matrix
- PCA explained variance
- optional IsoScore

## Default Prompt Context Summary

If this document is used in later prompts, the default assumptions should be:

- use `dair-ai/emotion` as the current multiclass dataset
- use sentence-level embeddings
- start with `BAAI/bge-base-en-v1.5`
- use masked mean pooling
- use `max_length = 64` first
- save raw, normalized, and centered-normalized embeddings
- preserve the official validation split
- prioritize macro F1, centroid structure, and neighborhood purity
- do not trust 2D plots alone
- do not trust cosine alone
- use TF-IDF + logistic regression as a lexical baseline
- use a fine-tuned classifier only as an upper-bound reference
- treat raw decoder-LLM hidden states as a later comparison

## Current Bottom Line

The safest first multiclass implementation for `vidiq-hpc` is:

- dataset: `dair-ai/emotion`
- model: `BAAI/bge-base-en-v1.5`
- pooling: masked mean pooling
- sequence length: `max_length = 64` unless truncation shows otherwise
- representations: raw, normalized, centered-normalized
- validation: macro F1, confusion structure, centroid-distance matrices, neighborhood purity, and clustering diagnostics

This should be the multiclass baseline against which any later LLM hidden-state or more exotic geometry method is compared.
