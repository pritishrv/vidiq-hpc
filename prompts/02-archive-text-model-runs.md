# Archive Text Model Runs

Plan a durable storage workflow for `experiments/text_model/runs` so repeated HPC runs do not duplicate multi-GB model artifacts in the working tree while still leaving analysis outputs easy to inspect and share.

Requirements:

- Keep `analysis/` and `train_metrics.json` directly inside each run directory under `experiments/text_model/runs/<run_id>/`.
- Move heavy `model/` and `tokenizer/` artifacts to `/users/aczd097/archive/vidiq-hpc/text_model/<run_id>/`.
- Create symlinks in the run directory so the existing run layout still exposes `model/` and `tokenizer/`.
- Assume many future experiments, so the solution must scale and should avoid manual post-run file moves.
- Preserve the ability to compare runs and recover exact provenance later.

Proposed design:

1. Add an `--archive-root` argument to `experiments/text_model/train_multiclass.py`.
2. Continue creating the run directory under `experiments/text_model/runs/<run_id>/`.
3. Save lightweight outputs locally in the run directory:
   - `analysis/`
   - `train_metrics.json`
   - `run_metadata.json`
4. Save heavyweight outputs under the archive root:
   - `model/`
   - `tokenizer/`
5. After saving archived artifacts, create symlinks:
   - `experiments/text_model/runs/<run_id>/model -> /users/aczd097/archive/vidiq-hpc/text_model/<run_id>/model`
   - `experiments/text_model/runs/<run_id>/tokenizer -> /users/aczd097/archive/vidiq-hpc/text_model/<run_id>/tokenizer`
6. Write `run_metadata.json` with enough detail to reconstruct the run:
   - run name
   - timestamp
   - git commit if available
   - command-line args
   - dataset path
   - local run root
   - archive root
   - symlink targets
7. Consider a second-stage policy for `eval_embeddings.npy` and `eval_logits.npy` if run volume becomes large enough that analysis storage also needs archiving.

Action:

1. Update `RunConfig` so it tracks local and archived output locations separately.
2. Add `--archive-root` to `parse_args()` with default `/users/aczd097/archive/vidiq-hpc/text_model`.
3. Change run-config creation so:
   - local run directory contains `analysis/` and metrics files
   - archive directory contains `model/` and `tokenizer/`
4. Save model and tokenizer into the archive directory instead of directly inside the run directory.
5. Add helper logic to recreate symlinks safely when the run finishes:
   - remove stale symlink if present
   - fail loudly if a real conflicting directory already exists
6. Add `run_metadata.json` emission.
7. Add a `.gitignore` rule for `experiments/text_model/runs/` if the intent is to keep generated run outputs out of git.
8. Keep the HPC SLURM scripts unchanged for now unless they should pass an explicit `--archive-root`.

Deliverable:

Implement the archive-root and symlink workflow in `experiments/text_model/train_multiclass.py`, keeping the visible run layout stable for downstream users while preventing repeated storage of heavyweight model artifacts inside the repo tree.
