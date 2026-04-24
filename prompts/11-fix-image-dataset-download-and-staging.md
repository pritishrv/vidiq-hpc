Fix the image dataset acquisition and staging pipeline so the image embedding experiment can actually run on Hyperion instead of failing at the first metadata lookup.

Repository root: `/users/aczd097/git/vidiq-hpc`

## Current Failure

Running:

```bash
python3 scripts/run_image_embeddings.py --config configs/emoset_phase1.json
```

currently fails with:

```text
FileNotFoundError: EmoSet metadata not found at data/image/emoset/metadata.csv. Please ensure download=True or data is placed correctly.
```

The root cause is that:

- `scripts/run_image_embeddings.py` always constructs `EmoSetDataset(..., download=True)`
- `src/image_experiments/datasets.py` calls `_download()`
- `EmoSetDataset._download()` is only a placeholder `pass`
- `configs/emoset_phase1.json` points at a repo-local path (`data/image/emoset`) rather than an HPC-friendly storage location

This is not acceptable. The dataset path and acquisition flow need to be made real.

## Important Environment Context

On Hyperion, you have unquoted storage available under:

- `~/archive`
- `~/sharedscratch`

In practice, prefer the explicit HPC paths already used elsewhere in this repo when appropriate:

- `/users/aczd097/archive`
- `/users/aczd097/sharedscratch`

Use those locations for dataset storage / caching / staging instead of bloating the Git checkout with large image data.

## Existing Repo Patterns To Follow

Base your implementation on successful patterns already present in this project:

1. **Dynamic path handling in HPC scripts**
   - `hpc/image_embedding_emoset.slurm`
   - existing SLURM scripts under `hpc/`

2. **Archive / scratch usage for large artifacts**
   - `experiments/text_model/train_multiclass.py`
   - its archive-root / local-visible-run pattern

3. **Config-driven entrypoints**
   - `scripts/run_image_embeddings.py`
   - `configs/emoset_phase1.json`

4. **Robust metadata / logging**
   - `src/image_experiments/io_utils.py`
   - run metadata patterns in the text experiments

The fix should feel like a real continuation of this repo, not a one-off patch.

## What You Must Do

### Part 1: Inspect current code and storage assumptions

Read at least:

- `scripts/run_image_embeddings.py`
- `src/image_experiments/datasets.py`
- `src/image_experiments/config.py`
- `configs/emoset_phase1.json`
- `hpc/image_embedding_emoset.slurm`
- `reports/image-embedding-audit-report.md`

Also inspect the current contents of:

- `~/archive`
- `~/sharedscratch`

and determine whether any EmoSet-related files are already present that should be reused instead of redownloaded.

### Part 2: Implement real dataset acquisition / staging

You must replace the placeholder dataset logic with a real, working flow.

For `EmoSetDataset`, implement one of the following, in this order of preference:

1. **Reuse existing staged data** if EmoSet metadata/images already exist under archive/sharedscratch.
2. **Download the metadata and/or dataset indices automatically** into a stable storage location if direct public download is possible.
3. **If the full image payload cannot be downloaded programmatically**, implement a proper staged-data bootstrap flow that:
   - downloads whatever can be downloaded automatically
   - resolves metadata/image roots from config or environment
   - fails with a precise, actionable message describing exactly what file(s) are missing and where they should be placed

What you must not do:

- do not leave `_download()` as a stub
- do not silently assume repo-local `data/image/emoset`
- do not fail with a vague “metadata not found” error if you can provide a concrete recovery path

### Part 3: Make storage HPC-appropriate

Refactor config and code so large image data lives outside the repo checkout by default.

The final design should support:

- a repo config file with explicit dataset root / cache root / staging root
- defaults that make sense on Hyperion
- reuse of previously downloaded assets under `~/archive` or `~/sharedscratch`

If needed, add config keys such as:

- `data_root`
- `download_root`
- `archive_root`
- `metadata_path`
- `images_root`

but keep the design coherent and minimal.

### Part 4: Update the SLURM path

Update `hpc/image_embedding_emoset.slurm` if necessary so the job:

- uses the corrected config
- resolves the correct storage location
- does not assume the repo checkout itself contains the dataset payload

### Part 5: Improve failure messages

If automatic download is still partly constrained by upstream packaging/licensing, the code must:

- detect exactly what is missing
- print the exact expected paths
- explain whether metadata, images, or both are absent
- explain whether the dataset can be resumed from a partially staged location

The user should not have to read the source code to know how to recover.

## Desired End State

After your changes, this workflow should be credible on Hyperion:

```bash
source /users/aczd097/sharedscratch/venvs/main/bin/activate
python3 scripts/run_image_embeddings.py --config configs/emoset_phase1.json
```

and this should also be credible:

```bash
sbatch hpc/image_embedding_emoset.slurm
```

“Credible” means one of:

- it runs successfully using discovered/staged data
- or it stops with a sharply actionable message that identifies the missing external dataset assets and the intended storage paths

It must not fail because the code never implemented download/staging.

## Files You Are Likely To Change

At minimum, expect to inspect and possibly edit:

- `src/image_experiments/datasets.py`
- `src/image_experiments/config.py`
- `configs/emoset_phase1.json`
- `scripts/run_image_embeddings.py`
- `hpc/image_embedding_emoset.slurm`
- `README.md` and/or experiment docs if storage instructions need documenting

You may add helper modules if justified, but keep the architecture disciplined.

## Acceptance Criteria

- `EmoSetDataset._download()` or its replacement contains real logic, not a stub.
- Dataset metadata/image resolution uses HPC-appropriate storage, not just repo-local paths.
- Existing staged data under `~/archive` or `~/sharedscratch` is reused when available.
- The config and SLURM workflow are consistent with the new storage design.
- Failure messages are operationally useful.
- The implementation follows the same pragmatic HPC patterns already used successfully elsewhere in this repository.

## Final Output

When finished:

- summarize the dataset acquisition design
- list the exact files changed
- state where EmoSet data is expected to live on Hyperion
- state whether the code now auto-downloads, auto-stages, reuses existing storage, or requires a one-time manual placement step
