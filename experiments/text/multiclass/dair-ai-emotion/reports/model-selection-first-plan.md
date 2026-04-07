# dair-ai/emotion Multiclass Execution Plan

## Purpose

This document defines the first-stage `dair-ai/emotion` experiment plan.

The goal is:

- move directly into the multiclass experiment with the model already selected from the binary setup
- avoid repeating a likely low-value model-selection stage
- save time and avoid unnecessary compute spend

## Principle

The binary SST-2 experiment already selected:

- `BAAI/bge-base-en-v1.5`

That result is being carried forward into `dair-ai/emotion` instead of repeating the same bakeoff.

Reason:

- the multiclass ranking may differ slightly, but the expected benefit of rerunning a full model-selection stage is low
- `BAAI/bge-base-en-v1.5` is already a strong fit for sentence-level geometry analysis
- the more important unknown for the multiclass setup is the embedding representation and class geometry, not the model family itself

So this multiclass plan intentionally skips model selection to save time and avoid wasting computation.

## Fixed Starting Point

Use this as the starting recipe:

- dataset: `dair-ai/emotion`
- text unit: one dataset row
- split usage:
  - full train split for embedding extraction and training-side artifacts
  - validation split for evaluation
- model: `BAAI/bge-base-en-v1.5`
- pooling: masked mean pooling over final hidden states
- sequence length: `max_length = 64`
- save these embedding variants:
  - raw mean pooled
  - L2-normalized mean pooled
  - mean-centered + L2-normalized mean pooled
- do not start with CLS-only pooling

Reason:

- this mirrors the successful binary workflow after model selection
- it gets directly to the multiclass question that actually matters

## Why Model Selection Is Being Skipped

Model selection is not being run first here.

The plan is to:

1. raw mean pooled
2. mean pooled + L2
3. mean pooled + centered + L2
4. optional CLS + L2 later only if needed

Only if the multiclass results are unexpectedly weak or unstable should the repo go back and run:

- `sentence-transformers/all-mpnet-base-v2`
- or a fresh multiclass model-selection bakeoff

## Multiclass-Specific Evaluation Focus

The multiclass experiment should pay special attention to:

- macro F1 rather than accuracy alone
- per-class confusion structure
- whether `joy` and `love` collapse together
- whether `anger`, `fear`, and `sadness` partially overlap
- whether `surprise` behaves as the least stable class

The question is not only whether the embeddings classify well, but whether they form interpretable multiclass geometry.

## Immediate Execution Plan

### Stage 1: Dataset Preparation

- load `dair-ai/emotion`
- preserve the original train and validation splits
- remove empty rows if any
- record:
  - per-class counts
  - token-length statistics
  - truncation rate under the chosen tokenizer

### Stage 2: Embedding Generation

- generate BGE embeddings for the full train split
- generate BGE embeddings for the validation split
- save:
  - raw mean pooled embeddings
  - L2-normalized embeddings
  - mean-centered + L2-normalized embeddings
- save labels, texts, and metadata alongside them

### Stage 3: Multiclass Validation

Run all quantitative metrics on the original embedding dimensionality, not on reduced plots.

Recommended first metric panel:

- multinomial logistic regression
- kNN at `k = 1` and `k = 5`
- macro F1
- confusion matrix
- centroid-distance matrix
- average same-class versus cross-class distance ratio
- silhouette score
- Davies-Bouldin index
- PCA explained variance

### Stage 4: Visualization

Use dimensionality reduction only for visualization:

- PCA 2D
- UMAP 2D or t-SNE fallback
- class-projection plots when useful

These plots are for inspection only and should not replace the original-space metrics.

### Stage 5: Baseline Comparison

After the embedding variants are compared:

- add `TF-IDF + logistic regression`
- optionally add `sentence-transformers/all-mpnet-base-v2` only if the BGE results look weak or ambiguous

## Current Expected Outcome

The expected default model remains:

- `BAAI/bge-base-en-v1.5`

The purpose of this plan is to move directly into embedding generation and multiclass validation without burning time on a likely redundant model-selection stage.
