# vidiq-hpc

High-performance computing experiments for the `vidiq` embedding-geometry project.

This repo currently covers two main workstreams:

- text experiments for embedding-geometry analysis on sentiment and emotion datasets
- HPC fine-tuning runs for a Qwen-based multiclass emotion classifier

## Main Areas

- `experiments/text/`
  dataset-specific text experiment pipelines, reports, configs, and outputs
- `experiments/text_model/`
  Qwen-based multiclass classifier training code and requirements
- `hpc/`
  SLURM batch scripts for Hyperion runs, including smoke tests and long-run jobs
- `reports/`
  survey reports and research notes, including the current image-dataset survey reports
- `meetings/`
  meeting minutes and transcripts
- `prompts/`
  reusable research prompts, including the image-dataset search prompt

## Text-Model Training

The main multiclass training entrypoint is:

- `experiments/text_model/train_multiclass.py`

Detailed implementation and paper-method notes:

- [experiments/text_model/README.md](experiments/text_model/README.md)

Current default backbone:

- `Qwen/Qwen3-1.7B`

The script supports:

- processed dataset roots via `--data-root`
- direct CSV training via `--csv-path`
- SLURM-derived run ids
- archived large artifacts with local run-directory symlinks
- export of evaluation embeddings, logits, centroid summaries, and metrics

## Hyperion / SLURM

Key batch scripts:

- `hpc/check_model_access.slurm`
  fast smoke test for Qwen tokenizer/model availability
- `hpc/train_multiclass.slurm`
  main multiclass training job
- `hpc/train_multiclass_frozen_backbone.slurm`
  frozen-backbone variant
- `hpc/train_multiclass_balanced_{10e,50e,100e,250e,500e,1000e}.slurm`
  balanced 6-class CSV runs
- `hpc/train_multiclass_unbalanced_{10e,50e,100e,250e,500e,1000e}.slurm`
  unbalanced 20-class CSV runs

Standard Hyperion flow:

```bash
git pull
sbatch hpc/check_model_access.slurm
sbatch hpc/train_multiclass_balanced_10e.slurm
```

Run outputs are written under:

- `experiments/text_model/runs/<run_id>/`

Tracked run artifacts include:

- `analysis/`
- `run_metadata.json`
- `train_metrics.json`

Ignored heavyweight artifacts:

- `model/`
- `tokenizer/`

## Current Tracked Training Result

The first tracked balanced fine-tuning run is:

- `experiments/text_model/runs/tmqb0010_17763/`

It includes:

- `analysis/centroids.json`
- `analysis/eval_embeddings.npy`
- `analysis/eval_logits.npy`
- `analysis/eval_metrics.json`
- `run_metadata.json`
- `train_metrics.json`

Additional Qwen long-run results from the Hyperion `50e`, `100e`, `250e`, `500e`, and `1000e` jobs should be incorporated through the dataset-level bridge workflow documented in:

- `experiments/text/multiclass/dair-ai-emotion/README.md`
- `reports/qwen-finetune-phase3-implementation-report-codex.md`

## Image Dataset Survey Work

The current image-extension planning material includes:

- prompt: `prompts/04-image-sentiment-dataset-search.md`
- reports:
  - `reports/chatgpt-image-datasets.md`
  - `reports/claude-image-datasets.md`
  - `reports/deepseek-image-datasets.md`
  - `reports/gemini-image-datasets.md`
  - `reports/grok-image-datasets.md`
  - `reports/lechat-image-datasets.md`

These reports are intended to support selection of the first image dataset for the cross-modality extension discussed in the 16 April meeting minutes.

## Environment

Original local note:

- Pritish: run this project inside `gemini_env`.

For Hyperion jobs, the current SLURM scripts activate:

- `/users/aczd097/sharedscratch/venvs/main/bin/activate`
