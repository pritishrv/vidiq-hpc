# SST-2 Binary Emotion Dataset Experiment Report

## Purpose

This document records the current recommendation for:

- which text embedding models to use for `vidiq-hpc`
- how to construct embeddings from text datasets
- how to validate the resulting embedding geometry
- which caveats to keep in mind when interpreting results

It is intended to be reused as prompt context for later implementation, analysis, and reporting work.

## Scope

This recommendation is based on:

- the existing text-dataset reports in this repository
- a full pass over the local literature set in `vidiq/lit-survey/gemini`
- the extracted title / abstract / conclusion report in `vidiq/reports/title-abstract-conclusion.md`
- current SST-2 dataset information from the Stanford Sentiment Treebank page, the Hugging Face dataset card, and GLUE-format dataset metadata

The immediate target use case is now centered on SST-2 and other similar short-text sentiment datasets.

## SST-2 Dataset Profile

SST-2 is the binary sentiment version of the Stanford Sentiment Treebank.

Important practical facts:

- it is an English movie-review sentiment dataset
- it is binary: `0 = negative`, `1 = positive`
- the common Hugging Face / GLUE-formatted version has approximately:
  - `67.3k` train examples
  - `872` validation examples
  - `1.82k` test examples
- the public test split exists, but the labels are hidden in the standard benchmark format
- many examples are short review snippets or phrase-like fragments rather than clean full sentences
- texts are usually short, but there is wide variation in length

Important historical context:

- the original Stanford Sentiment Treebank contains fine-grained phrase labels for `215,154` phrases drawn from `11,855` sentences
- SST-2 is the binary positive / negative task derived from that resource

Operational implications for embedding work:

- SST-2 is good for studying a clean polarity axis
- SST-2 is not ideal for claims about full-document sentiment
- some inputs are incomplete phrases, so sentence encoders may still work well, but fragment sensitivity should be expected
- because the official test labels are hidden, the validation split is the main public labeled evaluation split unless an internal train split is created

## Dataset-Specific Decision For SST-2

For SST-2, keep the same primary model family:

- contrastive sentence encoder

Primary model:

- `BAAI/bge-base-en-v1.5`

Strong alternate:

- `sentence-transformers/all-mpnet-base-v2`

Dataset-specific note:

- SST-2 is simple enough that a supervised sentiment classifier can perform very well, but the objective here is to study embedding geometry rather than maximize task accuracy
- therefore the encoder should remain general-purpose and sentence-oriented
- a supervised classifier fine-tuned on SST-2 should be treated as an upper-bound reference, not the main embedding model

## Model Selection Outcome

The first controlled model-selection stage has been completed for SST-2.

Fixed setup:

- dataset: `glue/sst2`
- train side: stratified `10%` sample of the SST-2 train split
- validation side: full SST-2 validation split
- pooling: masked mean pooling
- normalization: L2
- compared models:
  - `BAAI/bge-base-en-v1.5`
  - `sentence-transformers/all-mpnet-base-v2`

Observed sample sizes:

- train model-selection subset: `6,734`
- validation: `872`

Observed results:

- `BAAI/bge-base-en-v1.5`
  - logistic regression macro F1: `0.8817`
  - kNN@1 accuracy: `0.8337`
  - kNN@5 accuracy: `0.8704`
  - silhouette: `0.0468`
  - intra / inter distance ratio: `0.9100`
- `sentence-transformers/all-mpnet-base-v2`
  - logistic regression macro F1: `0.8875`
  - kNN@1 accuracy: `0.7890`
  - kNN@5 accuracy: `0.8394`
  - silhouette: `0.0293`
  - intra / inter distance ratio: `0.9425`

Interpretation:

- `all-mpnet-base-v2` was slightly stronger on linear probe performance
- `bge-base-en-v1.5` was stronger on neighborhood quality and clustering-oriented geometry
- because this project is centered on embedding-space behavior rather than only probe accuracy, `bge-base-en-v1.5` is the better first main experiment model

Decision:

- use `BAAI/bge-base-en-v1.5` as the default model for the next SST-2 experiment stage
- keep `sentence-transformers/all-mpnet-base-v2` as the completed comparison baseline from model selection
- run the next ablations on:
  - raw mean pooled embeddings
  - L2-normalized mean pooled embeddings
  - mean-centered plus L2-normalized embeddings

## Original Broader Use Case

The broader use case still includes sentence-level embeddings for clean text datasets such as:

- SST-2
- Subjectivity
- GYAFC
- other short-text datasets with simple semantic contrasts

## Executive Decision

For the first experiment pipeline, use a contrastively trained sentence encoder as the primary embedding model.

Do not use raw hidden states from a generative decoder LLM as the main baseline for the first pass.

The current production choice for SST-2 should be:

- `BAAI/bge-base-en-v1.5`

Strong alternative:

- `sentence-transformers/all-mpnet-base-v2`

Optional comparison family for a later second-stage baseline:

- a SimCSE-style sentence model
- an open LLM hidden-state baseline such as Gemma or Llama with the same pooling recipe

## Why This Is the Current Recommendation

The literature sweep supports the following conclusions:

- Vanilla BERT-style sentence vectors are not reliably good for similarity and clustering unless the model is trained for sentence embedding tasks.
- Contrastive or siamese objectives consistently improve sentence-level geometry for similarity, clustering, and retrieval.
- Embedding spaces often show anisotropy, dominant common directions, or other geometry pathologies.
- A single metric such as cosine similarity is not sufficient for validation.
- Performance on STS-style tasks alone does not guarantee strong clustering or dataset geometry behavior.

In practice, this means the first pipeline should favor:

- sentence encoders over raw autoregressive LLM hidden states
- explicit pooling and normalization rules
- explicit geometry diagnostics
- multiple validation metrics rather than one headline score

## Recommended Model Stack

### Tier 1: Primary Model

Use:

- `BAAI/bge-base-en-v1.5`

Reason:

- strong sentence embedding baseline
- practical model size
- good off-the-shelf behavior for semantic similarity and clustering
- simpler and safer for first-pass geometry analysis than raw decoder LLM states

### Tier 2: Practical Alternate

Use:

- `sentence-transformers/all-mpnet-base-v2`

Reason:

- mature sentence-transformers integration
- strong general-purpose sentence embedding behavior
- useful as a robustness comparison against BGE

### Tier 3: Later Comparison Models

Add only after the first pipeline is working:

- a SimCSE-family encoder
- one decoder LLM hidden-state baseline

Reason:

- useful for comparison
- not the safest first choice for clean geometry experiments
- adds ambiguity around pooling, layer selection, and similarity interpretation

## Embedding Construction

### Unit of Embedding

Create one embedding per text example.

For the current datasets, the canonical unit is:

- one sentence
- one short review snippet
- one short formality example

Do not start with token-level embeddings as the primary experimental object.

For SST-2 specifically:

- use one embedding per row in the dataset
- do not attempt to reconstruct parse-tree phrase hierarchies for the first pass
- treat each row as the operational text unit even when it is only a fragment

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
- unpooled token sets as the primary representation

For SST-2, masked mean pooling is preferred because:

- many inputs are short fragments, where single-token pooling can be unstable
- the task is polarity-oriented rather than syntax-probe-oriented
- mean pooling gives a cleaner sentence- or snippet-level representation for geometry analysis

### Sequence Length Rule

Default starting point for SST-2:

- `max_length = 64`

Escalate only if truncation is observed at non-trivial rates.

Fallback:

- use `max_length = 128` if a measurable tail of examples is being cut off

Reason:

- SST-2 examples are generally short
- shorter max length improves throughput and batching efficiency
- there is little reason to pay the cost of very long sequence settings for this dataset

### Normalization Rule

Always save at least two versions of each embedding:

1. pooled embedding
2. L2-normalized pooled embedding

Also save a third version for geometry analysis:

3. mean-centered, then L2-normalized embedding

Reason:

- centering may materially reduce common-direction drift
- normalized vectors are easier to compare with cosine and Euclidean-on-unit-sphere analyses
- storing all three avoids recomputation and makes ablation easier

### Suggested Saved Outputs

For each dataset split, save:

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
- record length statistics
- preserve the official validation split
- sample a manageable but representative subset if the dataset is large

For SST-2 specifically:

- use the standard train / validation split from the dataset source
- do not evaluate primary results on the hidden-label test split
- keep a copy of the raw text exactly as provided
- record:
  - class counts in train and validation
  - token-length distribution
  - truncation rate under the chosen tokenizer and max length
  - proportion of fragment-like examples if manually inspected

Recommended first SST-2 setup:

1. use the full public train split for embedding extraction
2. use the official validation split for labeled evaluation
3. optionally create a stratified train subset for quick iteration

### Phase 2: Embedding Extraction

- extract sentence embeddings with `BAAI/bge-base-en-v1.5`
- save raw, normalized, and centered-normalized embeddings
- log batch size, max sequence length, truncation policy, and device

For SST-2, also log:

- tokenizer truncation count
- average tokenized length
- median tokenized length
- class-wise counts after any filtering

### Phase 3: Geometry Validation

Run all validation on both:

- L2-normalized embeddings
- mean-centered + L2-normalized embeddings

This comparison is important because geometry improvements may come from post-processing rather than the encoder alone.

For SST-2, the main question is:

- does the embedding space exhibit a stable positive / negative polarity separation without relying only on a supervised classifier

### Phase 4: Comparative Baselines

After the first selected-model ablations are stable:

- compare the three BGE embedding variants first
- then compare the chosen BGE setup against the stored `all-mpnet-base-v2` model-selection result
- only then add a decoder-LLM hidden-state baseline if still needed

## Core Validation Metrics

### Cluster Quality

Use:

- silhouette score
- Davies-Bouldin index
- Calinski-Harabasz score

Purpose:

- test whether labeled groups form compact and separated clusters

Notes:

- compute on the same representation variants
- report sensitivity to the number of clusters if using unsupervised methods
- for SST-2, set the primary unsupervised target to `k = 2`
- also report whether the discovered clusters align with the binary labels

### Distance Structure

Use:

- centroid cosine distance
- centroid Euclidean distance
- average intra-class distance
- average inter-class distance
- ratio of intra-class to inter-class distance

Purpose:

- measure whether classes are geometrically separated
- estimate whether a simple semantic axis is present

For SST-2, add:

- class mean difference vector `mu_pos - mu_neg`
- projection histograms of examples onto that vector

This is one of the most interpretable geometry checks for a binary sentiment dataset.

### Linear Separability

Use:

- logistic regression
- linear SVM

Report:

- train / test accuracy
- macro F1
- ROC-AUC when appropriate

Purpose:

- test whether the geometry is linearly usable

Interpretation:

- high linear probe performance is stronger evidence than a pretty 2D plot

For SST-2, this is a primary metric, not a secondary one.

Reason:

- if the embeddings do not support strong linear sentiment separation on SST-2, the geometry is not doing the intended job

### Neighborhood Quality

Use:

- k-nearest-neighbor classification accuracy
- class purity among nearest neighbors

Purpose:

- test whether local geometry preserves label semantics

For SST-2, report:

- kNN accuracy for small `k` values such as `1`, `5`, and `11`
- positive-neighbor purity for positive points
- negative-neighbor purity for negative points

### Dimensionality and Spectrum

Use:

- PCA explained variance ratios
- top principal component dominance
- effective rank if convenient

Purpose:

- measure whether the space is dominated by a few directions
- support interpretation of anisotropy and shared-direction drift

### Isotropy / Uniformity Diagnostics

Use:

- IsoScore if implemented
- average pairwise cosine as a rough diagnostic only
- variance spectrum diagnostics

Important:

- do not rely on brittle isotropy claims from a single simple metric
- treat isotropy as a diagnostic, not a success metric by itself

For SST-2:

- anisotropy diagnostics matter, but only after task-level separation is confirmed
- do not optimize the pipeline around isotropy alone

### Representation Similarity Across Models

If multiple models are compared, use:

- CKA
- RSA-style distance-matrix comparisons

Purpose:

- compare whether different encoders produce similar geometry on the same dataset

## Distance Metrics to Use

### Primary

Use first:

- cosine distance
- Euclidean distance on L2-normalized embeddings

Reason:

- standard
- easy to interpret
- enough for the first experimental pass

### Secondary

Consider later:

- Mahalanobis distance
- Word Mover's Distance style comparisons
- optimal transport style distances
- DDR-style similarity

Reason:

- potentially informative
- more expensive or more specialized
- not necessary for the first implementation

For SST-2, recommended order is:

1. cosine distance
2. Euclidean distance on normalized embeddings
3. optional Mahalanobis distance
4. only later, OT-style or DDR-style analyses

### Important Caution About Cosine

Do not assume cosine similarity is universally meaningful.

The literature suggests cosine can be influenced by representation scaling, regularization, and geometry artifacts. It should remain a primary baseline metric, but not the only one.

## Visualization Guidance

Use:

- PCA first
- UMAP second
- t-SNE only if needed

Always:

- color by label
- show class centroids
- report projection quality caveats
- avoid making strong claims from 2D projections alone

Recommended plots:

- PCA scatter with centroids
- UMAP scatter with centroids
- histogram of projection onto class-difference vector
- pairwise distance histograms for intra-class vs inter-class pairs

## Failure Modes to Watch For

### Lexical Shortcut Separation

Risk:

- clusters may separate because of superficial lexical markers instead of semantics

Mitigation:

- inspect nearest neighbors
- examine misclassified examples
- compare across multiple datasets with different confounds

In SST-2 this matters a lot because:

- many examples are short
- some examples are highly polarized snippets
- a few sentiment-bearing words can dominate the signal

Add these checks:

- inspect examples nearest to each class centroid
- inspect examples near the decision boundary
- inspect very short examples separately

### Dominant Common Direction

Risk:

- the space may be driven by one or a few global directions

Mitigation:

- compare raw vs centered embeddings
- inspect PCA spectrum

### Over-reading 2D Projections

Risk:

- PCA / UMAP / t-SNE can create misleading visual separation

Mitigation:

- pair plots with numerical metrics
- never use plots as the sole evidence

### Model-Specific Metric Fragility

Risk:

- one encoder looks good under one metric and weak under another

Mitigation:

- use a metric panel, not a single score

### Dataset Confounds

Risk:

- subjectivity, formality, and sentiment datasets may include topic or style leakage

Mitigation:

- document known confounds
- compare multiple datasets

For SST-2 specifically, note:

- domain is movie reviews
- many examples are fragments rather than natural full sentences
- lexical polarity cues may be unusually strong
- some compositional phenomena like negation still matter, but the dataset is not cleanly balanced around them

This means SST-2 is strong for a first polarity-axis experiment, but weak as the only evidence for general sentiment geometry.

## SST-2-Specific Validation Priority

For SST-2, prioritize metrics in this order:

1. linear probe accuracy and macro F1 on the validation split
2. centroid separation and intra / inter-class distance ratio
3. kNN accuracy and neighborhood purity
4. silhouette and Davies-Bouldin
5. PCA spectrum and anisotropy diagnostics
6. 2D visualization

Reason:

- SST-2 is binary and label-clean enough that separability metrics should lead
- plots and isotropy diagnostics are useful, but should not be the headline result

## Recommended SST-2 Baselines

Use these baselines for comparison:

### Embedding Baselines

- `BAAI/bge-base-en-v1.5`
- `sentence-transformers/all-mpnet-base-v2`

### Lightweight Non-Embedding Baseline

- TF-IDF + logistic regression

Purpose:

- determine whether dense embeddings are beating a simple lexical baseline in separability and neighborhood structure

### Supervised Upper-Bound Reference

- a standard SST-2 classifier such as DistilBERT or BERT fine-tuned for sentiment classification

Purpose:

- provide a task-accuracy reference point
- not the main geometry model
- useful to calibrate how much signal is available in the dataset

## Recommended First Experiment Matrix

### Datasets

Start with:

- SST-2
- Subjectivity
- GYAFC

If work is currently focused only on SST-2, treat it as the primary dataset and defer the others until the SST-2 pipeline is stable.

Optional later:

- emotion or politeness datasets

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

- silhouette score
- Davies-Bouldin index
- centroid cosine distance
- centroid Euclidean distance
- intra-class / inter-class distance ratio
- linear probe accuracy and macro F1
- kNN accuracy
- PCA explained variance for top components
- optional IsoScore

For SST-2 specifically, also report:

- validation accuracy and macro F1 from a linear probe
- kNN accuracy at `k = 1, 5, 11`
- projection histogram statistics on the positive-minus-negative centroid direction
- truncation rate under the chosen tokenizer settings

## Default Prompt Context Summary

If this document is used in later prompts, the default assumptions should be:

- use sentence-level embeddings
- start with `BAAI/bge-base-en-v1.5`
- use masked mean pooling
- use `max_length = 64` first for SST-2
- save raw, normalized, and centered-normalized embeddings
- preserve the official SST-2 validation split
- validate with both cluster metrics and supervised separability metrics
- do not trust cosine alone
- do not trust 2D plots alone
- treat raw decoder-LLM hidden states as a later comparison, not the primary first model
- use TF-IDF + logistic regression as a lexical baseline
- use a fine-tuned sentiment classifier only as an upper-bound reference

## Current Bottom Line

The safest first implementation for `vidiq-hpc` is:

- a contrastive sentence encoder
- sentence-level mean-pooled embeddings
- explicit normalization and centering variants
- a validation stack that combines cluster, distance, probe, and spectrum metrics

For SST-2 specifically, the safest first implementation is:

- `BAAI/bge-base-en-v1.5`
- masked mean pooling
- `max_length = 64` unless truncation proves otherwise
- evaluation on the official validation split
- linear probe + centroid separation + kNN + cluster metrics as the main validation panel

This is the recommended baseline against which any later LLM hidden-state or more exotic geometry method should be compared.
