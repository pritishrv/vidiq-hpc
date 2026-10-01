# FER2013 CLIP ViT-L/14 Experiment Plan

## Purpose

This document defines the image-side analogue of the earlier text embedding-field experiments. The goal is to generate CLIP image embeddings for FER2013 face images, then run the same geometry analysis pattern used in the text experiments:

1. all-class projection with centroids,
2. per-class radial/density analysis,
3. pairwise overlap analysis,
4. publication-ready plots and metrics.

Model choice for this phase:
- backbone: `CLIP ViT-L/14`
- modality used for geometry: image encoder embeddings only
- primary distance metric: Euclidean distance in native embedding space

---

## Dataset Summary

Dataset root:
- `/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/data/Image_Dataset/FER2013`

Observed layout:
- `train/<class>/*.jpg`
- `test/<class>/*.jpg`

Observed raw classes:
- `angry`
- `disgust`
- `fear`
- `happy`
- `neutral`
- `sad`
- `surprise`

Observed counts:

| Split | angry | disgust | fear | happy | neutral | sad | surprise |
|------|------:|--------:|-----:|------:|--------:|----:|---------:|
| train | 3995 | 436 | 4097 | 7215 | 4965 | 4830 | 3171 |
| test | 958 | 111 | 1024 | 1774 | 1233 | 1247 | 831 |

Working decision for this experiment:
- remove `disgust` from the study
- balance the remaining six classes:
  - `angry`
  - `fear`
  - `happy`
  - `neutral`
  - `sad`
  - `surprise`

Reason:
- `disgust` is too sparse and would force an unnecessarily small balanced subset
- removing it preserves much more data while keeping the label set close to the prior six-class experiments

Balanced per-class caps after removing `disgust`:

| Split | Minority count after removing disgust |
|------|--------------------------------------:|
| train | 3171 |
| test | 831 |

---

## Principle

Preserve the same core measurement choices as the text experiments:

- use raw CLIP image embeddings, not softmax outputs
- use Euclidean distance in native embedding space for density and overlap analysis
- treat PCA or t-SNE only as visualisation layers, not as primary evidence
- save all intermediate metadata needed to reproduce the split, class mapping, and source-image provenance

---

## Experimental Structure

We should run the FER2013 study in three layers.

### Layer 1: Dataset preparation

Create a repo-local prepared representation that mirrors the text pipeline structure:

- `data/processed/train/`
- `data/processed/test/`
- `metadata.json`
- `label_to_id.json` or equivalent metadata payload
- image manifests containing:
  - source path
  - split
  - class name
  - class id
  - optional original filename

Recommended experiment roots:

- balanced train:
  - `experiments/embeddings_field/image/emotions/fer2013_balanced_no_disgust_clip_vitl14_train/`
- balanced test:
  - `experiments/embeddings_field/image/emotions/fer2013_balanced_no_disgust_clip_vitl14_test/`

Preparation outputs should include:

- `data/processed/train/manifest.jsonl`
- `data/processed/test/manifest.jsonl`
- `data/processed/train/labels.npy`
- `data/processed/test/labels.npy`
- `data/processed/metadata.json`

For the balanced variant:
- exclude `disgust` before any balancing step
- downsample each train/test class separately to the split-specific minority count
- record the sampled file list explicitly so reruns are deterministic
- use a fixed seed

### Layer 2: Embedding generation

Generate one CLIP embedding per image with `ViT-L/14`.

Expected outputs:

- `artifacts/embeddings/train_raw.npy`
- `artifacts/embeddings/test_raw.npy`
- `artifacts/embeddings/train_metadata.json`
- `artifacts/embeddings/test_metadata.json`

Metadata should include:

- model id / checkpoint
- preprocessing recipe
- embedding dimensionality
- batch size
- device used
- split sizes
- class names and ids
- manifest source path

### Layer 3: Geometry analysis and plotting

Repeat the text experiment pattern on image embeddings:

1. all-class cluster projection with centroids
2. per-class cluster snapshots
3. radial distance scatter
4. all-class density decay
5. all-class overlap ratio
6. pairwise density decay
7. pairwise overlap volume
8. pairwise scatter with centroids

Expected output tree:

- `all_class/`
- `all_class/cluster_snapshots/`
- `all_class/radial_distance/`
- `pairwise/<class_a>_vs_<class_b>/`
- `summary.json`

---

## Planned Runs

### Run A: Balanced FER2013 six-class train split

Goal:
- match the logic of the balanced text experiments and get cleaner pairwise comparisons

Why:
- removes count-driven artefacts from density and overlap plots
- removes the unstable `disgust` tail from the study
- makes pairwise plots easier to compare across classes

Recommendation:
- use this as the main geometry figure set

### Run B: Balanced FER2013 six-class test split

Goal:
- verify that the geometry seen in train is not just a split-specific artefact

Recommendation:
- run at least the all-class plots and a reduced pairwise subset on `test`
- full pairwise test plots are optional if runtime becomes heavy

---

## Implementation Plan

### Step 1: Add image dataset preparation script

Create a script along the lines of:
- `experiments/embeddings_field/image/emotions/src/prepare_fer2013_dataset.py`

Responsibilities:
- scan `train/` and `test/` folders
- exclude the `disgust` directory
- map class names to stable integer ids
- emit manifests and labels arrays
- optionally create balanced manifests with deterministic downsampling

### Step 2: Add CLIP embedding generation script

Create:
- `experiments/embeddings_field/image/emotions/src/generate_clip_image_embeddings.py`

Responsibilities:
- load manifests
- open and preprocess images with CLIP ViT-L/14 transforms
- batch inference over images
- save raw embeddings and metadata

Practical choices:
- use `open_clip` if already preferred in this environment
- otherwise use Hugging Face `transformers` CLIP implementation
- use batched GPU inference where available
- save float32 embeddings

### Step 3: Add plotting/analysis scripts

Create image equivalents of the text plotting scripts, for example:

- `plot_all_class_scatter.py`
- `plot_class_snapshots.py`
- `plot_radial_distance_scatter.py`
- `run_all_class_density_overlap.py`
- `run_pairwise_density_overlap.py`
- `plot_pairwise_scatter_with_centroids.py`

These should reuse the same output naming conventions as the text experiments where possible.

### Step 4: Add one orchestration script

Create:
- `experiments/embeddings_field/image/emotions/src/run_fer2013_clip_experiment.py`

Responsibilities:
- load embeddings + labels
- run all-class analysis
- run pairwise analysis
- write summary metadata

This keeps the experiment reproducible with one command per dataset variant.

---

## Output Conventions

Recommended output roots:

- balanced train:
  - `experiments/embeddings_field/image/emotions/fer2013_balanced_no_disgust_clip_vitl14_train/`
- balanced test:
  - `experiments/embeddings_field/image/emotions/fer2013_balanced_no_disgust_clip_vitl14_test/`

Within each root:

- `artifacts/embeddings/`
- `data/processed/`
- `all_class/`
- `pairwise/`
- `summary.json`

---

## Key Analysis Questions

The FER2013 CLIP study should answer the following:

1. Do facial-emotion image embeddings show the same belt-density pattern observed in text embeddings?
2. Which emotion pairs are naturally closest in CLIP image space?
3. Are some classes intrinsically diffuse (`fear`, `neutral`, `sad`) while others form tighter clusters (`happy`, `surprise`)?
4. Does removing `disgust` produce a cleaner and more interpretable six-class geometry?

---

## Risks and Mitigations

### Risk 1: Residual sample imbalance after removing disgust

Problem:
- even after removing `disgust`, balancing still requires downsampling the larger classes

Mitigation:
- build a deterministic balanced subset from the remaining six classes
- record exact sampled manifests for train and test

### Risk 2: Low-resolution grayscale images

Problem:
- FER2013 images are small and visually degraded compared with CLIP pretraining data

Mitigation:
- document preprocessing clearly
- keep the study focused on geometry, not top-line classification quality
- note that noisy clustering may be a dataset property, not necessarily a CLIP failure

### Risk 3: Runtime / storage

Problem:
- full-train embeddings plus many pairwise plots may be heavy

Mitigation:
- save one embedding per image only
- avoid committing bulky temporary caches
- allow reduced test-pair plotting if needed

---

## Recommended Run Order

1. Prepare balanced FER2013 manifests after excluding `disgust`.
2. Generate balanced-train CLIP ViT-L/14 embeddings.
3. Run all-class and pairwise plots on balanced train.
4. Generate balanced-test CLIP ViT-L/14 embeddings.
5. Run reduced all-class plus selected pairwise analysis on balanced test.
6. Write findings comparing:
   - balanced train vs balanced test
   - image geometry vs prior text geometry

---

## Minimum Deliverables

Code:
- FER2013 preparation script
- CLIP embedding generation script
- image embedding-field plotting scripts
- one orchestration script

Artifacts:
- raw train/test CLIP embeddings
- manifests and label metadata
- all-class plots
- pairwise plots
- metrics JSON files

Report:
- a short findings note summarising:
  - closest/farthest class pairs
  - radial spread by class
  - effect of removing `disgust`
  - whether image embeddings show the same density/overlap pattern as text embeddings

---

## Recommendation

Primary figure set for the paper or internal review should come from:
- `FER2013 balanced six-class (no disgust) + CLIP ViT-L/14 + train split`

Sanity check:
- balanced six-class test split

This keeps the study aligned with the earlier six-class setup while avoiding the heavy distortion introduced by the tiny `disgust` class.
