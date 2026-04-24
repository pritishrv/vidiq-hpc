# Codex Onboarding Report

## 1. Repository Overview

`vidiq-hpc` is an experiment repository for embedding-geometry research. The original core work is text-based: binary sentiment and multiclass emotion experiments using sentence-embedding backbones, followed by supervised Qwen fine-tuning whose exported embeddings and logits are analyzed geometrically. A newer image workstream has been added for image embedding geometry on EmoSet, but that framework is still partial and does not yet have completed run outputs in this checkout.

Current major workstreams:

- binary text embedding experiments on SST-2
- multiclass text embedding experiments on `dair-ai/emotion`
- Qwen fine-tuning on Hyperion plus dataset-level bridge analysis
- Qwen parity analysis to compare Qwen embeddings against the BGE experiment surface
- `embeddings_field` geometric density / overlap analysis
- image embedding framework for EmoSet Phase 1

## 2. Directory Map

- `experiments/`
  experiment-scoped work. The main output roots live here.
- `experiments/text/`
  dataset-specific text pipelines, configs, reports, artifacts, and run folders.
- `experiments/text_model/`
  Qwen training code and tracked source-run outputs.
- `experiments/embeddings_field/`
  geometry-first analysis outputs and scripts, separate from the main `experiments/text/` run-tracking pattern.
- `hpc/`
  SLURM wrappers for Hyperion.
- `src/`
  top-level reusable code for the image framework only at present.
- `scripts/`
  top-level entrypoints outside the dataset-specific text experiment roots.
- `configs/`
  top-level configs outside the dataset-specific text experiment roots.
- `reports/`
  top-level research reports, audits, and planning documents.
- `prompts/`
  operator prompts for Codex and other agents.
- `work-diary.md`
  historical task log and handoff record. Read this before assuming earlier work is present just because it was discussed.
- `README.md`
  current operational entrypoint for jobs, HPC commands, and push-back rules.
- `experiments.md`
  current repo-level experiment inventory.

## 3. Experiment Families

### Binary text embedding experiments

Purpose:
- select and stress-test sentence embedding models on SST-2
- compare embedding variants in geometry and simple downstream probes

Main entrypoints:
- [experiments/text/binary/sst2/src/run_model_selection.py](/home/daniel/git/vidiq-hpc/experiments/text/binary/sst2/src/run_model_selection.py)
- [experiments/text/binary/sst2/src/run_bge_ablation.py](/home/daniel/git/vidiq-hpc/experiments/text/binary/sst2/src/run_bge_ablation.py)
- [experiments/text/binary/sst2/src/plot_bge_variants.py](/home/daniel/git/vidiq-hpc/experiments/text/binary/sst2/src/plot_bge_variants.py)

Main configs:
- dataset-level configs under [experiments/text/binary/sst2/configs](/home/daniel/git/vidiq-hpc/experiments/text/binary/sst2/configs)

Main outputs:
- [experiments/text/binary/sst2/runs](/home/daniel/git/vidiq-hpc/experiments/text/binary/sst2/runs)
- [experiments/text/binary/sst2/artifacts/embeddings](/home/daniel/git/vidiq-hpc/experiments/text/binary/sst2/artifacts/embeddings)
- [experiments/text/binary/sst2/artifacts/plots/bge-variant-visuals](/home/daniel/git/vidiq-hpc/experiments/text/binary/sst2/artifacts/plots/bge-variant-visuals)

State:
- stable and tracked

### Multiclass `dair-ai/emotion` embedding experiments

Purpose:
- extend the BGE-based geometry workflow to six-class emotion classification
- compare raw / L2 / centered-L2 embedding variants

Main entrypoints:
- [experiments/text/multiclass/dair-ai-emotion/src/run_embedding_generation.py](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/src/run_embedding_generation.py)
- [experiments/text/multiclass/dair-ai-emotion/src/run_bge_ablation.py](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/src/run_bge_ablation.py)
- [experiments/text/multiclass/dair-ai-emotion/src/plot_bge_variants.py](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/src/plot_bge_variants.py)

Main configs:
- [experiments/text/multiclass/dair-ai-emotion/configs/bge-embedding-generation.json](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/configs/bge-embedding-generation.json)
- [experiments/text/multiclass/dair-ai-emotion/configs/bge-ablation-stage.json](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/configs/bge-ablation-stage.json)

Main outputs:
- [experiments/text/multiclass/dair-ai-emotion/runs](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/runs)
- [experiments/text/multiclass/dair-ai-emotion/artifacts/embeddings](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/artifacts/embeddings)
- [experiments/text/multiclass/dair-ai-emotion/artifacts/plots](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/artifacts/plots)

State:
- stable for the BGE stage

### Qwen fine-tuning pipeline

Purpose:
- fine-tune `Qwen/Qwen3-1.7B` on multiclass emotion classification
- export held-out embeddings, logits, centroids, and metrics for downstream analysis

Main entrypoints:
- [experiments/text_model/train_multiclass.py](/home/daniel/git/vidiq-hpc/experiments/text_model/train_multiclass.py)
- `hpc/train_multiclass*.slurm`

Main outputs:
- [experiments/text_model/runs/tmqb0010_17763](/home/daniel/git/vidiq-hpc/experiments/text_model/runs/tmqb0010_17763)

State:
- one tracked source run exists and is the current source of truth for the completed balanced Qwen run

### Qwen bridge / Phase 3 analysis

Purpose:
- project and analyze the tracked fine-tuned Qwen run at the dataset level
- evaluate within-run logit/geometry coherence

Main entrypoints:
- [experiments/text/multiclass/dair-ai-emotion/src/plot_qwen_finetune_run.py](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/src/plot_qwen_finetune_run.py)
- [experiments/text/multiclass/dair-ai-emotion/src/analyze_qwen_logit_geometry.py](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/src/analyze_qwen_logit_geometry.py)

Main outputs:
- [experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e)
- [experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/qwen-finetune-10e](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/qwen-finetune-10e)
- [experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e)

State:
- stable and tracked

### Qwen parity workflow

Purpose:
- build the closest honest analogue of the BGE variant experiment surface using exported Qwen hold-out embeddings

Main entrypoints:
- [experiments/text/multiclass/dair-ai-emotion/src/run_qwen_bge_parity.py](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/src/run_qwen_bge_parity.py)
- [experiments/text/multiclass/dair-ai-emotion/src/plot_qwen_bge_parity.py](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/src/plot_qwen_bge_parity.py)
- [hpc/run_qwen_bge_parity.slurm](/home/daniel/git/vidiq-hpc/hpc/run_qwen_bge_parity.slurm)

Main config:
- [experiments/text/multiclass/dair-ai-emotion/configs/qwen-bge-parity.json](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/configs/qwen-bge-parity.json)

Main outputs currently present:
- parity run dirs:
  - [run-301-qwen3-1-7b-bge-parity-raw](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/runs/run-301-qwen3-1-7b-bge-parity-raw)
  - [run-302-qwen3-1-7b-bge-parity-l2](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/runs/run-302-qwen3-1-7b-bge-parity-l2)
  - [run-303-qwen3-1-7b-bge-parity-centered-l2](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/runs/run-303-qwen3-1-7b-bge-parity-centered-l2)
- parity embedding metadata:
  - [dair_ai_emotion_qwen_holdout_qwen3-1-7b_metadata.json](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/artifacts/embeddings/dair_ai_emotion_qwen_holdout_qwen3-1-7b_metadata.json)
- parity metrics:
  - [experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/qwen-bge-parity/summary.json](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/qwen-bge-parity/summary.json)

State:
- partial

Why partial:
- the run directories exist
- the summary metric bundle exists
- the documented plot bundle `artifacts/plots/qwen-bge-parity/` is absent in this checkout
- the parity `.npy` arrays are not present in Git and are likely intentionally ignored as heavyweight artifacts

### `embeddings_field` geometry analysis

Purpose:
- run geometry-first analyses centered on density decay, radial structure, and overlap

Main outputs:
- [experiments/embeddings_field](/home/daniel/git/vidiq-hpc/experiments/embeddings_field)

State:
- active and tracked, but more heterogeneous than `experiments/text/`

### Image embedding framework

Purpose:
- run Phase 1 image embedding geometry on EmoSet
- stage future image train/eval work

Main entrypoints:
- [scripts/run_image_embeddings.py](/home/daniel/git/vidiq-hpc/scripts/run_image_embeddings.py)
- [hpc/image_embedding_emoset.slurm](/home/daniel/git/vidiq-hpc/hpc/image_embedding_emoset.slurm)

Main code:
- [src/image_experiments/config.py](/home/daniel/git/vidiq-hpc/src/image_experiments/config.py)
- [src/image_experiments/datasets.py](/home/daniel/git/vidiq-hpc/src/image_experiments/datasets.py)
- [src/image_experiments/embeddings.py](/home/daniel/git/vidiq-hpc/src/image_experiments/embeddings.py)
- [src/image_experiments/geometry.py](/home/daniel/git/vidiq-hpc/src/image_experiments/geometry.py)
- [src/image_experiments/io_utils.py](/home/daniel/git/vidiq-hpc/src/image_experiments/io_utils.py)

Main config:
- [configs/emoset_phase1.json](/home/daniel/git/vidiq-hpc/configs/emoset_phase1.json)

Main outputs:
- intended run root is `experiments/image/runs/<run_name>/`
- in this checkout, there is no committed [experiments/image](/home/daniel/git/vidiq-hpc/experiments/image) directory yet

State:
- active but partial

## 4. HPC / Hyperion Workflow

SLURM wrappers live under [hpc](/home/daniel/git/vidiq-hpc/hpc). Important jobs:

- [hpc/check_model_access.slurm](/home/daniel/git/vidiq-hpc/hpc/check_model_access.slurm)
- [hpc/train_multiclass.slurm](/home/daniel/git/vidiq-hpc/hpc/train_multiclass.slurm)
- [hpc/train_multiclass_frozen_backbone.slurm](/home/daniel/git/vidiq-hpc/hpc/train_multiclass_frozen_backbone.slurm)
- [hpc/train_multiclass_balanced_10e.slurm](/home/daniel/git/vidiq-hpc/hpc/train_multiclass_balanced_10e.slurm) and the related balanced/unbalanced long-run variants
- [hpc/run_qwen_bge_parity.slurm](/home/daniel/git/vidiq-hpc/hpc/run_qwen_bge_parity.slurm)
- [hpc/image_embedding_emoset.slurm](/home/daniel/git/vidiq-hpc/hpc/image_embedding_emoset.slurm)

Shared environment:

- text and image jobs currently expect the shared venv at `/users/aczd097/sharedscratch/venvs/main/bin/activate`

Large data / caches:

- Qwen HF caches use `/users/aczd097/sharedscratch/huggingface`
- image dataset staging now defaults to `/users/aczd097/archive/vidiq-hpc/data/image/emoset`
- image HF dataset cache defaults to `/users/aczd097/sharedscratch/huggingface/datasets`

Push-back pattern:

- inspect with `git status --short`
- stage tracked outputs under `experiments/` and any docs that changed
- avoid pushing `outputs/`, ignored embeddings, or heavyweight model/tokenizer directories

## 5. Current Source of Truth

Experiment inventory:
- [experiments.md](/home/daniel/git/vidiq-hpc/experiments.md)

Operational job commands and result-push rules:
- [README.md](/home/daniel/git/vidiq-hpc/README.md)

Historical and recent task context:
- [work-diary.md](/home/daniel/git/vidiq-hpc/work-diary.md)

Current text-model behavior:
- [experiments/text_model/train_multiclass.py](/home/daniel/git/vidiq-hpc/experiments/text_model/train_multiclass.py)
- [experiments/text_model/README.md](/home/daniel/git/vidiq-hpc/experiments/text_model/README.md)

Current multiclass Qwen behavior:
- [experiments/text/multiclass/dair-ai-emotion/README.md](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/README.md)
- [experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e)
- [experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/qwen-finetune-10e](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/qwen-finetune-10e)

Qwen parity status:
- code/config:
  - [experiments/text/multiclass/dair-ai-emotion/src/run_qwen_bge_parity.py](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/src/run_qwen_bge_parity.py)
  - [experiments/text/multiclass/dair-ai-emotion/src/plot_qwen_bge_parity.py](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/src/plot_qwen_bge_parity.py)
  - [experiments/text/multiclass/dair-ai-emotion/configs/qwen-bge-parity.json](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/configs/qwen-bge-parity.json)
- actual committed outputs:
  - parity run dirs
  - parity metadata JSON
  - parity summary metric JSON

Image framework status:
- [reports/image-embedding-audit-report.md](/home/daniel/git/vidiq-hpc/reports/image-embedding-audit-report.md)
- [scripts/run_image_embeddings.py](/home/daniel/git/vidiq-hpc/scripts/run_image_embeddings.py)
- [configs/emoset_phase1.json](/home/daniel/git/vidiq-hpc/configs/emoset_phase1.json)
- [hpc/image_embedding_emoset.slurm](/home/daniel/git/vidiq-hpc/hpc/image_embedding_emoset.slurm)

## 6. Known Gaps / Inconsistencies

- The Qwen parity docs in [experiments/text/multiclass/dair-ai-emotion/README.md](/home/daniel/git/vidiq-hpc/experiments/text/multiclass/dair-ai-emotion/README.md) still describe a parity plot bundle under `artifacts/plots/qwen-bge-parity/`, but that directory is not present in this checkout.
- The Qwen parity workflow is only partially materialized in Git:
  - run dirs are present
  - metadata is present
  - summary metrics are present
  - plots are absent
  - `.npy` arrays are not committed
- The image framework has code, config, and SLURM wiring, but no committed `experiments/image/` run outputs yet.
- The image framework remains extraction-first. The audit report says supervised train/eval support is missing.
- [experiments/text_model/README.md](/home/daniel/git/vidiq-hpc/experiments/text_model/README.md) still says `train_multiclass.py` is not present on the current `main` checkout, but the file does exist in this checkout. That README is historically valuable but not fully synchronized with the filesystem.
- There are tracked `__pycache__` artifacts inside `experiments/text/multiclass/dair-ai-emotion/src/__pycache__/`, which are not meaningful source-of-truth files.

## 7. Safe Working Rules For Future Agents

- Start with [README.md](/home/daniel/git/vidiq-hpc/README.md), [experiments.md](/home/daniel/git/vidiq-hpc/experiments.md), and [work-diary.md](/home/daniel/git/vidiq-hpc/work-diary.md).
- Before trusting any documentation claim about outputs, verify the filesystem with `find` or `git ls-files`.
- Treat code/config presence and generated output presence as different facts.
- Heavy artifacts are often intentionally ignored. Check [.gitignore](/home/daniel/git/vidiq-hpc/.gitignore) before deciding a file is “missing.”
- Do not update README files to describe outputs that have not actually been produced or committed.
- For Hyperion work, prefer SLURM entrypoints in `hpc/` over ad hoc interactive GPU usage.
- Update `work-diary.md` after significant repo-level work, HPC workflow changes, or experiment-surface changes, not for every tiny edit.

## 8. Recommended Next Actions

1. Reconcile the Qwen parity workflow with reality by either generating and committing the missing plot bundle or revising the README so it no longer implies those plots exist.
2. Add supervised training/evaluation to the image framework, as already flagged in [reports/image-embedding-audit-report.md](/home/daniel/git/vidiq-hpc/reports/image-embedding-audit-report.md).
3. Run and verify the staged EmoSet pipeline on Hyperion using the new archive/sharedscratch defaults, then commit the first `experiments/image/` outputs once stable.
4. Clean up repo hygiene issues such as tracked `__pycache__` artifacts and any stale documentation drift.
