# dair-ai/emotion Model Selection First Plan

## Purpose

This document defines the first-stage `dair-ai/emotion` experiment whose goal is:

- choose the best embedding model for the multiclass text experiment
- avoid running the full variation matrix before model choice is clear

This stage fixes the embedding recipe and compares only models.

## Principle

Before testing:

- multiple pooling strategies
- centering variants
- alternate normalization strategies
- large visualization and validation panels

first determine which embedding model is strongest under one controlled recipe.

This is the fastest way to reduce the search space while still making a defensible choice.

## Fixed Recipe For Model Comparison

All candidate models must use the exact same embedding recipe:

- dataset: `dair-ai/emotion`
- text unit: one dataset row
- split usage:
  - stratified random subset of the train split for model-selection training artifacts
  - validation split for evaluation
- pooling: masked mean pooling over final hidden states
- sequence length: `max_length = 64`
- normalization: L2 normalization
- no centering in this first stage
- no CLS-only pooling in this first stage

Reason:

- this controls for embedding formation
- differences in results are more attributable to model quality

## Data Budget For Model Selection

Use a small but stable subset for the training side of model selection:

- `10%` stratified random sample from the `dair-ai/emotion` train split
- full validation split for evaluation
- default sample fraction: `0.1`
- default random seed: `42`

Reason:

- large enough to preserve class balance and multiclass structure
- small enough to keep model ranking cheap
- sufficient for comparing the first two encoder candidates

Do not go below `5%` unless the goal is only a rough smoke test.

## Candidate Models

Run these first:

1. `BAAI/bge-base-en-v1.5`
2. `sentence-transformers/all-mpnet-base-v2`

Optional third candidate:

3. one SimCSE-family sentence encoder

Do not include decoder-LLM hidden-state baselines in the first model-selection round.

## Minimal Metrics Panel

For each model, compute:

### 1. Linear Probe

- multinomial logistic regression

Report:

- validation accuracy
- macro F1

Purpose:

- tests whether the six emotion classes are linearly separable in the embedding space

### 2. Neighborhood Quality

- kNN classification

Report:

- accuracy at `k = 1`
- accuracy at `k = 5`
- macro F1 at `k = 1`
- macro F1 at `k = 5`

Purpose:

- tests whether local neighborhoods preserve class labels

### 3. Centroid Separation

Report:

- centroid-distance matrix between all class centroids
- average same-class versus cross-class distance ratio

Purpose:

- checks whether the model forms interpretable class structure rather than only one coarse polarity axis

### 4. Cluster Diagnostic

Report:

- silhouette score
- Davies-Bouldin index

Purpose:

- gives a quick unsupervised view of class compactness and separation

### 5. Spectrum Diagnostic

Report:

- PCA explained variance of the top components

Purpose:

- checks whether the space is dominated by a few directions

### 6. Confusion Structure

Report:

- confusion matrix from the linear probe

Purpose:

- identifies which emotion classes overlap most strongly

## Decision Rule

Select the best model using this priority order:

1. highest macro F1 from multinomial logistic regression
2. strongest kNN macro F1
3. cleaner centroid-distance structure and same-class versus cross-class separation
4. better silhouette and Davies-Bouldin scores
5. healthier PCA spectrum if the other metrics are close

If one model clearly wins on the first three criteria, it becomes the default model for the rest of the multiclass experiment.

## Minimum Recommended Runs

Use these run ids:

1. `run-001-bge-meanpool-l2-model-select`
2. `run-002-mpnet-meanpool-l2-model-select`

Optional:

3. `run-003-simcse-meanpool-l2-model-select`

## What Not To Do Yet

Do not run these until model selection is complete:

- mean-centering ablations
- CLS pooling
- UMAP-heavy analysis
- decoder-LLM hidden-state baselines
- very large run grids

These come after the winning model is chosen.

## Stage 2 After Model Selection

Once the best model is selected, run the second-stage ablations on that model:

1. raw mean pooled
2. mean pooled + L2
3. mean pooled + centered + L2
4. optional CLS + L2

Then add:

- `TF-IDF + logistic regression` baseline
- broader geometry diagnostics
- plots

## Multiclass-Specific Evaluation Focus

The multiclass experiment should pay special attention to:

- macro F1 rather than accuracy alone
- per-class confusion structure
- whether `joy` and `love` collapse together
- whether `anger`, `fear`, and `sadness` partially overlap
- whether `surprise` behaves as the least stable class

The question is not only whether the embeddings classify well, but whether they form interpretable multiclass geometry.

## Current Expected Outcome

The expected front-runner is:

- `BAAI/bge-base-en-v1.5`

The strongest immediate comparison is:

- `sentence-transformers/all-mpnet-base-v2`

This expectation should be treated as a hypothesis, not a final decision.

The purpose of this stage is to confirm or overturn that hypothesis quickly.
