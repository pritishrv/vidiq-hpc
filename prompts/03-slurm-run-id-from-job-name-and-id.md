# Use SLURM Job Name And Job ID For Run IDs

Update the text-model training workflow so every run id is derived from the SLURM job metadata instead of a random suffix.

Requirements:

- Treat the SLURM job name as the primary experiment identifier.
- Treat the numeric `SLURM_JOB_ID` as the uniqueness suffix.
- The run id that defines output locations must be:
  - `${SLURM_JOB_NAME}_${SLURM_JOB_ID}`
- This run id must be used for:
  - `experiments/text_model/runs/<run_id>/`
  - `/users/aczd097/archive/vidiq-hpc/text_model/<run_id>/`
- The same exact SLURM job may be submitted more than once, so the numeric job id must remain part of the run id.
- `run_metadata.json` must record both:
  - `slurm_job_name`
  - `slurm_job_id`
- Keep the archive-and-symlink workflow already implemented.

Implementation plan:

1. Update `experiments/text_model/train_multiclass.py` to read `SLURM_JOB_NAME` and `SLURM_JOB_ID` from the environment.
2. If `--run-name` is not explicitly provided:
   - when both SLURM values exist, set `run_name` to `${SLURM_JOB_NAME}_${SLURM_JOB_ID}`
   - otherwise fall back to the current non-SLURM naming scheme
3. Extend `run_metadata.json` to store:
   - `slurm_job_name`
   - `slurm_job_id`
4. Update the SLURM batch scripts so they pass an explicit run name:
   - `--run-name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}"`
5. Keep the SLURM `#SBATCH --job-name` value short and within the 8-character scheduler limit.

Action:

1. Patch `experiments/text_model/train_multiclass.py` to capture SLURM env vars and write them into `run_metadata.json`.
2. Change the default run-name logic to prefer `${SLURM_JOB_NAME}_${SLURM_JOB_ID}` whenever available.
3. Update:
   - `hpc/train_multiclass.slurm`
   - `hpc/train_multiclass_frozen_backbone.slurm`
4. In each SLURM script:
   - use a short 8-character `#SBATCH --job-name`
   - add `--run-name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}"` to the Python command
5. Verify that a dry-run example would create:
   - local run dir: `experiments/text_model/runs/tmq3u001_13204/`
   - archive dir: `/users/aczd097/archive/vidiq-hpc/text_model/tmq3u001_13204/`
   - metadata fields:
     - `"slurm_job_name": "tmq3u001"`
     - `"slurm_job_id": "13204"`
     - `"run_name": "tmq3u001_13204"`

Deliverable:

Implement SLURM-aware run naming so repeated submissions of the same experiment code remain uniquely identifiable and all output locations are keyed by `<job_name>_<job_id>`.
