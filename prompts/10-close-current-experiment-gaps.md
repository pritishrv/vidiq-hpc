Implement the current missing pieces in this repository so that the documented experiment surfaces match the actual checked-in outputs and the new image framework is no longer extraction-only.

Repository root: `/home/daniel/git/vidiq-hpc`

## Context

Two concrete shortcomings currently exist:

1. **Qwen parity mismatch**
   - The repo contains the Qwen parity code, config, SLURM job, and README documentation.
   - However, in the current checkout the documented Qwen parity outputs are not present:
     - no `runs/run-301-qwen3-1-7b-bge-parity-raw/`
     - no `runs/run-302-qwen3-1-7b-bge-parity-l2/`
     - no `runs/run-303-qwen3-1-7b-bge-parity-centered-l2/`
     - no `artifacts/metrics/qwen-bge-parity/`
     - no `artifacts/plots/qwen-bge-parity/`
     - no Qwen parity embedding bundle under `artifacts/embeddings/`
   - The documentation currently reads as though those outputs already exist.

2. **Image framework missing training/evaluation**
   - The image experiment framework is implemented under:
     - `src/image_experiments/`
     - `scripts/run_image_embeddings.py`
     - `configs/emoset_phase1.json`
     - `hpc/image_embedding_emoset.slurm`
   - The audit report states that the framework is architecturally sound, but still lacks:
     - a `training.py` module
     - supervised train/eval support
     - standard predictive metrics such as accuracy / F1 / loss
   - Right now it is still an extraction + geometry pipeline rather than a full experiment framework.

## Task

Close these shortcomings end-to-end.

## Part A: Qwen Parity

First inspect:

- `experiments/text/multiclass/dair-ai-emotion/README.md`
- `experiments/text/multiclass/dair-ai-emotion/src/run_qwen_bge_parity.py`
- `experiments/text/multiclass/dair-ai-emotion/src/plot_qwen_bge_parity.py`
- `experiments/text/multiclass/dair-ai-emotion/configs/qwen-bge-parity.json`
- `hpc/run_qwen_bge_parity.slurm`
- the current filesystem state under:
  - `experiments/text/multiclass/dair-ai-emotion/runs/`
  - `experiments/text/multiclass/dair-ai-emotion/artifacts/`

Then do whichever is correct based on repo evidence:

- either generate / restore the documented Qwen parity outputs
- or revise the documentation so it accurately reflects the outputs that actually exist

Preferred outcome:

- the Qwen parity workflow is fully usable and the repository contains the expected parity outputs, including:
  - parity runs
  - parity metrics bundle
  - parity plot bundle
  - parity embedding metadata
- documentation should match the real filesystem exactly

If `.npy` outputs are intentionally ignored, document that clearly and ensure the README does not imply they are committed when they are not.

## Part B: Image Framework

First inspect:

- `reports/image-embedding-audit-report.md`
- `src/image_experiments/config.py`
- `src/image_experiments/datasets.py`
- `src/image_experiments/embeddings.py`
- `src/image_experiments/geometry.py`
- `src/image_experiments/io_utils.py`
- `scripts/run_image_embeddings.py`
- `configs/emoset_phase1.json`
- `hpc/image_embedding_emoset.slurm`

Then implement the missing supervised layer.

Required outcomes:

- add `src/image_experiments/training.py`
- extend the image entrypoint so it can run:
  - extraction-only mode
  - train + eval mode
- add classification evaluation outputs with at least:
  - accuracy
  - macro F1
  - loss where applicable
- store those results in structured artifacts alongside the geometry outputs
- update configs and HPC job(s) as needed so the new train/eval workflow can be run on Hyperion

## Constraints

- Reuse the existing repo patterns; do not invent a parallel architecture unless necessary.
- Keep path handling robust and HPC-friendly.
- Preserve the current modular separation of `src/`, `scripts/`, `configs/`, and `hpc/`.
- Be explicit about which outputs are tracked in Git versus ignored.
- If a missing output cannot be regenerated locally, make the documentation honest rather than leaving it misleading.

## Deliverables

At minimum:

- any code changes needed to complete the image framework
- any code / docs changes needed to reconcile the Qwen parity workflow with reality
- any config changes
- any README updates
- any SLURM updates needed for Hyperion execution

## Acceptance Criteria

- The Qwen parity documentation matches the actual repository state.
- If parity outputs are supposed to exist, the workflow produces them cleanly.
- The image framework supports supervised training/evaluation, not just embedding extraction.
- Structured predictive metrics are saved in the image experiment outputs.
- Hyperion batch execution paths remain clear and documented.

## Final Output

When finished:

- summarize what was implemented
- list the exact files changed
- state whether Qwen parity outputs were generated or the docs were corrected instead
- state what image train/eval functionality now exists that did not exist before
