# Implementation Plan: Image Embedding Geometry Experiments (EmoSet-118K & EmoVerse)

Instruct an agent to implement the data acquisition, experiment scripts, and HPC batch files for the upcoming image-embedding geometry research.

## Reference Patterns
The implementation MUST follow the conventions found in the `dsikar/kai-centroid-experiments` repository:
- **Project Structure:** 
  - Core logic and dataset classes in `src/` (e.g., `src/image_experiments/`).
  - Executable entrypoints in `scripts/` (e.g., `scripts/run_image_embeddings.py`).
  - Configuration files in `configs/` passed via `--config`.
- **SLURM Batch Files:** Batch files in `hpc/` must dynamically resolve the repository root and set `PYTHONPATH` appropriately:
  ```bash
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  cd "${REPO_ROOT}"
  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
  ```
- **Data Download:** Follow the pattern in `src/kai_experiments/data.py`, where data is downloaded and loaded via dataset classes (e.g., using a `download=True` flag and a `data_root` from the config), rather than using standalone download scripts.

## Tasks

### 1. Data Acquisition & Processing
- Implement dataset classes in `src/` to handle EmoSet-118K and EmoVerse. 
- The dataset classes should automatically download the datasets to `data_root` if they are not present.
- **EmoSet-118K:** Handle URL-based image downloading and parsing of the 8 emotion classes and 6 attributes.
- **EmoVerse:** Handle download of the B-A-S triplets and the associated SAM/DINO segmentation masks.

### 2. Experiment Implementation
- Implement the embedding generation and geometry logic within `src/`.
- Create the entrypoint script `scripts/run_image_embeddings.py` that takes a `--config` argument pointing to a JSON file in `configs/`.
- **Phase 1 (Full Image):** Generate embeddings for the raw images in EmoSet-118K.
- **Phase 2 (Segmented):** Generate embeddings for segmented objects vs. full backgrounds using EmoVerse masks.
- Implement the geometry analysis stage (centroids, radial density falloff, and inter-class overlap).

### 3. HPC Batch Files
- Create SLURM scripts in `hpc/` (e.g., `hpc/image_embedding_emoset.slurm`).
- Configure for the `gpu-a100` partition.
- Ensure the batch files use the dynamic path resolution and `PYTHONPATH` setup from the reference repository.

### 4. Logging & Reporting
- Implement structured logging for training and evaluation.
- **Job Metadata:** The log must include:
    - `timestamp`: Start and completion times.
    - `duration`: Total wall-clock time.
    - `infrastructure`: GPU model, CUDA version, and node name (leave as `TBC` in the report if you cannot access this info during development).
- **Format:** Make the log easily parseable. It should be structured (e.g., JSON) but also printed in the most human-readable tabular format possible to stdout for immediate review.

## Verification
- Run a smoke test on the HPC to verify vision model loading and dataset downloading.
- Ensure the outputs match the structured logging requirements.
