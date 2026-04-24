# Audit Plan: Image Embedding Experiment Framework (Prompt 08)

Instruct an agent to audit the implementation of the image embedding experiment framework executed in the previous stage. The auditor must ensure the code is technically sound, follows the specified modular patterns, and fulfills the core requirements for training, evaluation, and logging.

## Audit Objectives

### 1. Verification of Training and Evaluation Logic
- **Training:** Verify if a training loop exists (e.g., in a `src/image_experiments/training.py` module or similar). Even if the current phase focuses on pre-trained embeddings, check if the framework supports fine-tuning or supervised training as per the `kai-centroid-experiments` reference.
- **Evaluation:** Confirm that evaluation logic is implemented to assess model performance or embedding quality (e.g., accuracy, loss, or geometry-based metrics like cluster separation).
- **Execution:** Check if the entrypoint (`scripts/run_image_embeddings.py`) correctly invokes these training and evaluation stages.

### 2. Metrics and Artifact Storage
- **Storage:** Verify that training and evaluation metrics are stored in a structured, parseable format (e.g., `JSON`).
- **Completeness:** Ensure that metrics include at least:
  - Loss/Accuracy (if training/classification is performed).
  - Centroid distances and radial density falloff (for geometry experiments).
  - Metadata (timestamps, duration, infrastructure).
- **Persistence:** Confirm that these artifacts are saved in the correct run-level directory under `experiments/image/runs/<run_name>/`.

### 3. Architectural Alignment
- **Modality:** Ensure the code strictly adheres to the `src/`, `scripts/`, `configs/` separation.
- **Path Handling:** Verify that the SLURM batch file (`hpc/image_embedding_emoset.slurm`) and entrypoint script handle paths dynamically using `REPO_ROOT` and `PYTHONPATH` exports, avoiding hardcoded absolute paths.
- **Data Acquisition:** Check that dataset classes in `src/image_experiments/datasets.py` correctly implement the `download=True` logic and encapsulate data-root handling.

### 4. Logging & Readability
- **Human-Readable Logs:** Verify that the console output (stdout) is clean and tabular, providing immediate insight into the experiment's status and results.
- **Structured Logs:** Verify the presence of `run_metadata.json` and its contents (timestamp, duration, etc.).

## Deliverables
- A detailed audit report saved to `reports/image-embedding-audit-report.md`, highlighting any missing components or deviations from the `kai-centroid-experiments` patterns.
- Specific recommendations for correcting any identified gaps, particularly regarding training/evaluation loops and metrics storage.
- (Optional) A verification run command to confirm the framework operates correctly on a small subset of data.
