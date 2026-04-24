# Experiments

This repository contains three main experimental threads: text embedding studies, geometric analysis of embedding fields, and Qwen-based supervised fine-tuning for multiclass emotion classification. The tracked artifacts show a progression from binary SST-2 embedding experiments, to multiclass emotion embedding analysis, to a supervised Qwen stage that exports embeddings and logits for downstream geometry work. The repo also contains planning and reporting material for future extensions, but those should be kept separate from the experiments that have clear checked-in outputs.

## Experiment Families

### 1. Binary text embedding experiments on SST-2

Primary root: `experiments/text/binary/sst2/`

Nature of the work:

- binary sentiment experiments on the GLUE SST-2 dataset
- sentence-embedding model selection on a sampled training subset
- downstream ablations of embedding formation for the selected BGE model
- evaluation via geometry metrics, k-NN, logistic regression, and projection plots

Tracked runs:

- `runs/run-001-bge-base-en-v1-5-meanpool-l2-model-select/`
  BGE model-selection run
- `runs/run-002-all-mpnet-base-v2-meanpool-l2-model-select/`
  MPNet model-selection run
- `runs/run-101-bge-base-en-v1-5-raw-meanpool/`
  BGE raw mean-pooled variant
- `runs/run-102-bge-base-en-v1-5-l2-meanpool/`
  BGE L2-normalized variant
- `runs/run-103-bge-base-en-v1-5-centered-l2-meanpool/`
  BGE mean-centered plus L2-normalized variant

Inputs:

- GLUE `sst2`
- processed local copies under `data/processed/train/` and `data/processed/validation/`

Outputs:

- run configs and summaries under `runs/`
- reusable embeddings under `artifacts/embeddings/`
- projection and variant-comparison plots under `artifacts/plots/bge-variant-visuals/`

Status:

- clearly tracked and reproducible from checked-in configs, metadata, metrics, and artifacts

### 2. Multiclass text embedding experiments on `dair-ai/emotion`

Primary root: `experiments/text/multiclass/dair-ai-emotion/`

Nature of the work:

- six-class emotion analysis on the `dair-ai/emotion` experiment root
- direct BGE embedding generation
- BGE embedding-variant ablations analogous to the SST-2 work
- dataset-level visual analysis of class geometry
- later bridge analysis of an imported Qwen fine-tuning run

Tracked runs:

- `runs/run-001-bge-base-en-v1-5-embedding-generation/`
  base embedding generation for train and validation splits
- `runs/run-101-bge-base-en-v1-5-raw-meanpool/`
  raw BGE variant
- `runs/run-102-bge-base-en-v1-5-l2-meanpool/`
  L2-normalized BGE variant
- `runs/run-103-bge-base-en-v1-5-centered-l2-meanpool/`
  mean-centered plus L2-normalized BGE variant
- `runs/run-201-qwen3-1-7b-finetune-10e/`
  dataset-level bridge run summarizing the first tracked Qwen fine-tuning result

Inputs:

- processed `dair-ai/emotion` train and validation splits under `data/processed/`
- balanced six-class CSV used for the Qwen stage

Outputs:

- embeddings under `artifacts/embeddings/`
- BGE comparison plots under `artifacts/plots/bge-variant-visuals/`
- Qwen plots under `artifacts/plots/qwen-finetune-10e/`
- Qwen geometry metrics under `artifacts/metrics/qwen-finetune-10e/`
- narrative findings under `reports/`

Status:

- BGE stage is clearly tracked
- one Qwen bridge run is clearly tracked
- additional Qwen long runs are referenced in docs and SLURM scripts but are not yet represented by equivalent checked-in dataset-level bridge outputs

### 3. Qwen fine-tuning and export pipeline

Primary roots: `experiments/text_model/` and `hpc/`

Nature of the work:

- supervised multiclass emotion classification on Hyperion
- fine-tuning a causal language model backbone
- exporting held-out embeddings, logits, centroids, and metrics for downstream geometry analysis
- supporting both full fine-tuning and frozen-backbone variants

Key implementation:

- `experiments/text_model/train_multiclass.py`
- `hpc/train_multiclass.slurm`
- `hpc/train_multiclass_frozen_backbone.slurm`
- `hpc/train_multiclass_balanced_{10e,50e,100e,250e,500e,1000e}.slurm`
- `hpc/train_multiclass_unbalanced_{10e,50e,100e,250e,500e,1000e}.slurm`

Tracked source run:

- `experiments/text_model/runs/tmqb0010_17763/`
  first checked-in balanced Qwen run, sourced from `Qwen/Qwen3-1.7B`, 10 epochs, six balanced emotion labels

Inputs:

- balanced six-class CSV for the tracked run
- unbalanced 20-class CSV workflow is implied by the unbalanced SLURM jobs

Outputs:

- `analysis/eval_embeddings.npy`
- `analysis/eval_logits.npy`
- `analysis/eval_metrics.json`
- `analysis/centroids.json`
- `train_metrics.json`
- `run_metadata.json`

Status:

- one source run is clearly tracked
- the infrastructure for many more balanced and unbalanced jobs exists
- heavy model and tokenizer artifacts are archived externally and represented locally via symlinks

### 4. `embeddings_field` geometric density and overlap analysis

Primary root: `experiments/embeddings_field/`

Nature of the work:

- geometry-focused analysis in a separate experiment tree
- emphasis on density falloff, centroid structure, radial statistics, and pairwise overlap rather than classifier evaluation
- binary SST-2 density analysis plus multiclass all-class and pairwise overlap outputs

Binary branch:

- `experiments/embeddings_field/text/binary/`
- includes a plan, analysis scripts, and tracked metric JSON for SST-2 density overlap
- explicitly references `BAAI/bge-base-en-v1.5` in the plan and code

Multiclass branch:

- `experiments/embeddings_field/text/multiclass/`
- contains output bundles such as `balanced-6-emotions/`, `balanced-6-emotions-mpnet/`, and `balanced_dataset/`
- stores all-class projections, radial-distance summaries, and pairwise class-overlap plots

Status:

- this workstream has substantial checked-in outputs
- its code and artifacts are clearly analysis-oriented
- some of its model provenance is incomplete in top-level metadata, so not every bundle can be named with the same confidence as the `experiments/text/` runs

### 5. Image sentiment experiments (Planned)

Nature of the work:
- Extending embedding geometry research to the visual domain
- Replicating categorical cluster and outlier patterns in image embedding space
- Comparing "full image" geometry against "segmented object" geometry to test the contextual tightening hypothesis
- Investigating cross-modal geometry alignment between text and image sentiments

Target Datasets:
- **EmoSet-118K** (Primary for Phase 1): Large-scale (118K labeled images), balanced 8-class Mikels emotion taxonomy, includes object and action attributes.
- **EmoVerse** (Primary for Phase 2): Features Background-Attribute-Subject (B-A-S) triplets and SAM/DINO segmentation masks for part-whole geometric analysis.

Status:
- Dataset research and selection is complete.
- Six survey reports and a ranked synthesis are checked into the repository.
- Implementation of the Phase 1 embedding generation is pending.

### 6. Supporting planning and reporting work

Primary roots: `reports/`, `prompts/`, and `meetings/`

Nature of the work:

- experiment plans
- methodology notes
- findings writeups
- image-dataset survey material
- meeting records that explain intended next steps

Status:

- useful for interpretation and roadmap context
- not themselves experimental runs

## Models Used

### Executed or clearly tracked in experiment artifacts

| Model | Family | Where it appears | Role |
| --- | --- | --- | --- |
| `BAAI/bge-base-en-v1.5` | sentence embedding model | `experiments/text/binary/sst2/`, `experiments/text/multiclass/dair-ai-emotion/`, `experiments/embeddings_field/text/binary/` | main embedding backbone for binary and multiclass geometry studies |
| `sentence-transformers/all-mpnet-base-v2` | sentence embedding model | `experiments/text/binary/sst2/runs/run-002-all-mpnet-base-v2-meanpool-l2-model-select/`, `experiments/embeddings_field/text/multiclass/balanced-6-emotions-mpnet/` | comparison embedding model for model selection and multiclass geometry outputs |
| `Qwen/Qwen3-1.7B` | causal language model | `experiments/text_model/`, `hpc/`, `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/` | supervised fine-tuning backbone for multiclass emotion classification |

### Model variants rather than new base families

- BGE raw mean pooling
- BGE L2-normalized mean pooling
- BGE mean-centered plus L2-normalized mean pooling
- Qwen fine-tuned backbone plus projection head
- Qwen frozen-backbone variant defined in SLURM and training code

### Mentioned in plans, but not clearly tracked as completed experiment outputs

- `sentence-transformers/all-MiniLM-L6-v2`
- `intfloat/e5-base-v2`
- `thenlper/gte-base`

These appear in planning reports as candidate follow-up models, not as clearly completed checked-in runs.

## Datasets Used

| Dataset | Task type | Where it appears | Notes |
| --- | --- | --- | --- |
| GLUE `sst2` | binary sentiment | `experiments/text/binary/sst2/` | source dataset for model selection and embedding ablations |
| `dair-ai/emotion` | multiclass emotion | `experiments/text/multiclass/dair-ai-emotion/` | main multiclass experiment root with processed train and validation splits |
| `balanced_emotions_6classes.csv` | balanced six-class emotion CSV | Qwen source run and `embeddings_field` multiclass outputs | balanced subset used for the tracked Qwen run and related six-class geometry outputs |
| `20-emotion-dataset.csv` and related raw CSV workflow | multiclass emotion | `experiments/text/multiclass/dair-ai-emotion/data/raw/` and unbalanced SLURM jobs | raw source material for broader emotion-label workflows; checked-in completed unbalanced training results are not obvious |

## How The Workstreams Connect

The repo starts with embedding-first experiments under `experiments/text/`. SST-2 establishes the binary pipeline for selecting an embedding model and comparing embedding variants. The multiclass `dair-ai/emotion` work reuses that pattern with BGE, then adds richer class-level plotting and reporting.

The supervised Qwen stage is separate from the BGE experiments. It lives operationally under `experiments/text_model/` and `hpc/`, where the model is trained and the held-out embeddings and logits are exported. That source run is then bridged back into the dataset-specific multiclass experiment root under `experiments/text/multiclass/dair-ai-emotion/` so the fine-tuned run can be compared and visualized alongside the earlier BGE stage.

`experiments/embeddings_field/` is related but not identical to `experiments/text/`. It is a geometry-analysis branch that concentrates on density decay, overlap volume, radial distance, and pairwise class structure. In practice, it looks like a derived or parallel analysis tree rather than the main run-tracking system used under `experiments/text/`.

## Key Output Locations

- `experiments/text/binary/sst2/runs/`
  SST-2 model-selection and BGE ablation runs
- `experiments/text/binary/sst2/artifacts/embeddings/`
  reusable SST-2 embeddings and metadata
- `experiments/text/binary/sst2/artifacts/plots/bge-variant-visuals/`
  SST-2 comparison plots
- `experiments/text/multiclass/dair-ai-emotion/runs/`
  multiclass BGE runs plus the Qwen bridge run
- `experiments/text/multiclass/dair-ai-emotion/artifacts/embeddings/`
  multiclass reusable embeddings
- `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/`
  BGE and Qwen visual outputs
- `experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/qwen-finetune-10e/`
  Qwen logit-geometry metrics
- `experiments/text_model/runs/tmqb0010_17763/`
  source Qwen fine-tuning run with exported held-out embeddings, logits, and centroids
- `experiments/embeddings_field/text/binary/metrics/`
  SST-2 density-overlap outputs
- `experiments/embeddings_field/text/multiclass/balanced-6-emotions*/`
  multiclass density and overlap outputs, including the MPNet variant

## Gaps / Ambiguities

- `experiments/embeddings_field/text/multiclass/balanced-6-emotions-mpnet/` is clearly an MPNet output bundle, but `balanced-6-emotions/` and `balanced_dataset/` do not record an equally explicit model name in their top-level metadata. They appear to be six-class emotion geometry outputs, but the exact base model is not stated there.
- `experiments/text_model/README.md` says `train_multiclass.py` is not present on the current `main` checkout, but the file is present in the working tree. The code and README are internally inconsistent on that point.
- The repo clearly contains SLURM definitions for many additional Qwen balanced and unbalanced jobs, but only one source run and one dataset-level bridge run are obviously checked in as completed tracked results.
- Planning reports mention additional candidate embedding models, but the checked-in run directories do not show corresponding completed outputs for them.
- The multiclass BGE stage mostly uses `joy`, while the balanced Qwen stage uses `happiness`. The repo documents this caveat, but it still complicates direct cross-stage comparisons.
