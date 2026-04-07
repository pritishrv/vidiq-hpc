# SST-2 Model Selection First Plan

## Purpose

This document defines the first-stage SST-2 experiment whose goal is:

- choose the best embedding model for the binary text experiment
- avoid running the full variation matrix before model choice is clear

This stage fixes the embedding recipe and compares only models.

## Principle

Before testing:

- multiple pooling strategies
- centering variants
- alternate normalization strategies
- large validation panels

first determine which embedding model is strongest under one controlled recipe.

This is the fastest way to reduce the search space while still making a defensible choice.

## Fixed Recipe For Model Comparison

All candidate models must use the exact same embedding recipe:

- dataset: SST-2
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

- `10%` stratified random sample from the SST-2 train split
- full SST-2 validation split for evaluation
- default sample fraction: `0.1`
- default random seed: `42`

Reason:

- large enough to preserve class balance and sentiment structure
- small enough to keep model ranking cheap
- sufficient for comparing the first two encoder candidates

Do not go below `5%` unless the goal is only a very rough smoke test.

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

- logistic regression

Report:

- validation accuracy
- macro F1

Purpose:

- tests whether sentiment is linearly separable in the embedding space

### 2. Neighborhood Quality

- kNN classification

Report:

- accuracy at `k = 1`
- accuracy at `k = 5`

Purpose:

- tests whether local neighborhoods preserve sentiment labels

### 3. Centroid Separation

Report:

- centroid cosine distance
- centroid Euclidean distance on normalized vectors

Purpose:

- checks whether the model forms a clean polarity axis

### 4. Distance Ratio

Report:

- average intra-class distance
- average inter-class distance
- intra / inter distance ratio

Purpose:

- checks whether same-label examples are tighter than cross-label examples

### 5. Cluster Diagnostic

Report:

- silhouette score

Purpose:

- gives one quick unsupervised view of class compactness and separation

### 6. Spectrum Diagnostic

Report:

- PCA explained variance of the top components

Purpose:

- checks whether a model is dominated by a few directions

## Decision Rule

Select the best model using this priority order:

1. highest macro F1 from logistic regression
2. strongest kNN accuracy
3. strongest centroid separation and best intra / inter distance ratio
4. better silhouette score
5. healthier PCA spectrum if the other metrics are close

If one model clearly wins on the first three criteria, it becomes the default model for the rest of the SST-2 experiment.

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

- TF-IDF + logistic regression baseline
- broader geometry diagnostics
- plots

## Current Expected Outcome

The expected front-runner is:

- `BAAI/bge-base-en-v1.5`

The strongest immediate comparison is:

- `sentence-transformers/all-mpnet-base-v2`

This expectation should be treated as a hypothesis, not a final decision.

The purpose of this stage is to confirm or overturn that hypothesis quickly.
