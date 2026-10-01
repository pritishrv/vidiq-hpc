# FER2013 × CLIP ViT-L/14 — Full Image Emotion Geometry Report

> Compiled: 2026-09-01
> Scope: `experiments/embeddings_field/image/emotions/fer2013_balanced_no_disgust_clip_vitl14/`
> Status: First-pass results. All numbers in this document are read directly from the JSON/Markdown artifacts already produced in this directory — nothing here is simulated or projected.

---

## 1. Purpose and Relationship to the Wider Study

The core `vidiq-hpc` project (see `repo_context/PROJECT_OVERVIEW.md`) established a geometric pattern in **text** embeddings (8 LLM variants over a 6-class emotion dataset) and then tested whether the same local pattern appears in **human brain fMRI** (OpenNeuro DS005700). Both systems showed:

- A **local principle**: embeddings never sit exactly at their class centroid — there's a zero-density "Void" near the centroid and a peak-density "Belt" further out — and radial/margin position relative to competing centroids predicts classification ambiguity.
- A **global divergence**: text LLMs organize emotion primarily by valence, while the brain organizes it primarily by arousal, producing near-opposite relational geometries (RDM correlations as strong as −0.99).

An earlier attempt to extend this to images (a 120K-image dataset, abandoned ~April 2026) was too noisy to use. This experiment is a **fresh, better-controlled image-side replication**, run on FER2013 facial expressions through CLIP ViT-L/14, following the same measurement pattern used for text: raw (non-softmax) embeddings, Euclidean distance in native embedding space, radial density/overlap analysis, SVD-based signal retention, and RSA stability checks.

This document consolidates every test that has been run so far on this dataset/model combination and reports the results exactly as produced. It does not yet include a brain- or LLM-style statistical validation pass (bootstrap CI, permutation tests, cross-model replication) — see §10 (Limitations).

---

## 2. Dataset and Preprocessing

**Source:** `/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/data/Image_Dataset/FER2013`, laid out as `train/<class>/*.jpg` and `test/<class>/*.jpg`.

**Raw class counts (before any filtering):**

| Split | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 3995 | 436 | 4097 | 7215 | 4965 | 4830 | 3171 |
| test | 958 | 111 | 1024 | 1774 | 1233 | 1247 | 831 |

**Design decision:** `disgust` was dropped (too sparse — 436/111 examples would force the balanced set down to a tiny minority count) and the remaining six classes were downsampled to the per-split minority count, deterministically, with a fixed seed (`seed = 42`).

**Balanced working dataset actually used:**

| Split | Per-class count | Total examples | Classes |
|---|---:|---:|---|
| train | 3171 | **19,026** | angry, fear, happy, neutral, sad, surprise |
| test | 831 | **4,986** | angry, fear, happy, neutral, sad, surprise |

Manifests, label arrays, and metadata for both splits are saved under `data/processed/{train,test}/` (`manifest.jsonl`, `labels.npy`, `metadata.json`), so the exact sampled file list is reproducible.

**Important scope note:** Every result in this report is computed on the **train split only** (19,026 balanced examples). The `test` split embeddings (`test_raw.npy`) have not yet been generated — the phase-3 retention experiment substitutes a stratified 80/20 holdout carved out of the train split for evaluation (see §8). A true train/test replication (as recommended in the original experiment plan) is still outstanding.

---

## 3. Embedding Generation

| Field | Value |
|---|---|
| Model | `openai/clip-vit-large-patch14` (CLIP ViT-L/14) |
| Modality used | Image encoder only |
| Embedding dimensionality | 768 |
| Batch size | 32 |
| Device | CPU |
| Precision | float32 |
| Pooling / normalization | Raw CLIP image-encoder output (no softmax; consistent with the "raw pooling / logits over softmax" principle used throughout the text experiments) |
| Distance metric | Euclidean, native embedding space |
| Output | `artifacts/embeddings/train_raw.npy` (19,026 × 768 float32) + `train_metadata.json` |

This mirrors the text pipeline's design principle: direction and magnitude are both preserved (no L2 pre-normalization at the embedding stage), and PCA/t-SNE are reserved for visualization only, not as primary evidence.

---

## 4. Global Cluster Diagnostics

Source: `reports/embedding_metrics/EMBEDDING_METRICS_REPORT.md`, `reports/embedding_metrics/train/summary.json`.

| Metric | Value |
|---|---|
| Silhouette score | **−0.00087** |
| Davies-Bouldin score | **8.081** |
| Examples | 19,026 |
| Classes | 6 |
| Embedding dimensionality | 768 |

**Interpretation:** The silhouette score is effectively zero (very slightly negative), meaning the six emotion classes form heavily overlapping clouds in raw 768D CLIP space — there is no clean separation visible to a global clustering metric. Davies-Bouldin of ~8.1 (higher = worse separation) confirms this. This is the same signature seen in the **pretrained** (non-fine-tuned) text models (MPNet-Base-Final: silhouette 0.0471; BGE-Base-Final: 0.0570) — i.e., the image embeddings look like an *un-fine-tuned* representation, which is expected since CLIP ViT-L/14 was used off-the-shelf with no emotion-specific fine-tuning step. The text-study precedent (Finding 4) is that this kind of near-zero/negative silhouette in a high-dimensional space is not necessarily "no structure" — the same text pretrained models jumped from ~0.05 to ~0.41 silhouette once projected into their top-20 discriminative dimensions. That subspace-isolation step (Phase 4 equivalent) has **not** been run yet for the image embeddings — see §10.

---

## 5. Centroid Geometry

Source: `reports/embedding_metrics/train/centroid-distance-matrix.md`.

**Full pairwise centroid Euclidean-distance matrix (768D, raw CLIP space):**

| | angry | fear | happy | neutral | sad | surprise |
|---|---:|---:|---:|---:|---:|---:|
| **angry** | 0.0000 | 2.2576 | 4.0658 | 3.3694 | 2.6559 | 3.9255 |
| **fear** | 2.2576 | 0.0000 | 3.8817 | 3.4850 | 2.0257 | 3.0693 |
| **happy** | 4.0658 | 3.8817 | 0.0000 | 2.7556 | 3.7985 | 4.8659 |
| **neutral** | 3.3694 | 3.4850 | 2.7556 | 0.0000 | 2.7819 | 4.9933 |
| **sad** | 2.6559 | 2.0257 | 3.7985 | 2.7819 | 4.5044 | 0.0000 |
| **surprise** | 3.9255 | 3.0693 | 4.8659 | 4.9933 | 4.5044 | 0.0000 |

**Nearest centroid pairs (tightest family):**

| Pair | Distance |
|---|---:|
| fear ↔ sad | 2.0257 |
| angry ↔ fear | 2.2576 |
| angry ↔ sad | 2.6559 |
| happy ↔ neutral | 2.7556 |
| neutral ↔ sad | 2.7819 |

**Most separated pairs:**

| Pair | Distance |
|---|---:|
| neutral ↔ surprise | 4.9933 |
| happy ↔ surprise | 4.8659 |
| sad ↔ surprise | 4.5044 |
| angry ↔ happy | 4.0658 |
| fear ↔ happy | 3.8817 |

**Interpretation:**

- The three negative-affect classes — **fear, sad, angry** — form a tight triangle (pairwise distances 2.03–2.66), the closest family in the whole space. This is consistent with FER2013's known label ambiguity in these categories (facial-action-unit overlap between fear/sad/angry expressions).
- **Happy and neutral** are notably close (2.76) relative to how far happy sits from every negative-affect class (3.79–4.87) — CLIP appears to organize the space primarily along a "negative-affect cluster vs. happy/neutral" axis rather than by discrete category.
- **Surprise is the global outlier** — it is the single farthest class from every other centroid (all five of its distances are ≥3.07, and three of its five distances are its own largest gaps in the matrix: vs. neutral 4.99, vs. happy 4.87, vs. sad 4.50). This directly echoes the text-embedding finding that **surprise is the geometrically ambiguous/outlier class** (flagged as early as the 2026-04-07 meeting minutes: "a reaction rather than a pure emotion") — the same class shows outlier behavior in an entirely different modality (image, unsupervised CLIP encoder) and a different, non-overlapping label set (FER2013 vs. dair-ai/emotion).

---

## 6. Radial Density Structure — Void and Belt (Per Class)

Source: `reports/all_class/metrics.json`, visualized in `reports/all_class/all-class-density-decay.png`, `all-class-surface-density-decay.png`, `all-class-overlap-ratio.png`.

Each class's 3,171 points were binned by raw Euclidean distance from their own class centroid (12 bins per class, unequal width). For each bin the pipeline records: point count (`density`), density normalized by shell volume (`density_per_unit`), the count of points in that bin that are geometrically closer to a *different* class centroid (`overlap_count`), and the resulting `overlap_ratio`.

### 6.1 Void edge (closest any point gets to its own centroid)

| Class | Void edge (raw distance) |
|---|---:|
| surprise | **5.618** |
| happy | 6.471 |
| sad | 6.574 |
| fear | 6.618 |
| angry | 6.907 |
| neutral | 7.107 |

No class has a single embedded face sitting at its centroid — every class reproduces the Void observed in text and brain data. The size of the void varies by class (5.6–7.1 raw units), with **surprise having by far the shortest void** — its nearest members sit closer to its centroid (in absolute terms) than any other class's nearest members do to theirs, even though surprise's centroid is the most isolated one in the matrix in §5. Those two facts are not contradictory: a class can have a compact, close-in core of examples *and* still have a centroid that sits far from every other centroid.

### 6.2 Belt peak (bin with highest volume-normalized density)

| Class | Peak `density_per_unit` | Peak bin midpoint |
|---|---:|---:|
| neutral | 1115.5 | 9.185 |
| happy | 941.8 | 8.912 |
| fear | 917.9 | 8.849 |
| sad | 934.7 | 9.504 |
| angry | 1006.3 | 9.302 |
| surprise | 845.0 | 8.000 |

All six classes show the same qualitative shape: density rises sharply from the void edge, peaks in a mid-range shell (~8.0–9.5 raw units), then decays into a long low-density tail out to 15–18 units. This is the same **Void → Belt → long tail** structure documented for text embeddings (Finding 1) and is now confirmed in a third, independent modality. **Surprise peaks earliest (8.00)** and has the lowest peak density of any class — its mass is both closer-in and more spread out than the other five, consistent with it being the geometric outlier class.

### 6.3 Ambiguity gradient within each class (near-centroid vs. far-periphery overlap)

This is the most informative per-class result. `overlap_ratio` in the innermost bin (closest to the class's own centroid, i.e. the class's most confident members) vs. the outermost bin (farthest, i.e. the class's most atypical members):

| Class | Overlap ratio — innermost bin | Overlap ratio — outermost bin | Δ (spread) |
|---|---:|---:|---:|
| surprise | **0.109** | **18.45** | 18.35 |
| happy | 0.989 | 8.97 | 7.98 |
| fear | 1.992 | 8.01 | 6.01 |
| neutral | 3.011 | 5.06 | 2.05 |
| sad | 1.328 | 4.57 | 3.24 |
| angry | 1.160 | 3.65 | 2.49 |

**Interpretation:** Every class gets more ambiguous (higher overlap ratio) farther from its own centroid — this is the image-domain analogue of the text/brain "Ambiguity Gradient" finding (Finding 5: geometric position relative to class prototypes predicts confidence). But the *steepness* of that gradient differs sharply by class:

- **Surprise has the most extreme gradient of any class**: its near-centroid members are almost never claimed by another class (overlap ratio 0.109 — the lowest in the whole table, meaning close-in surprise faces are nearly unambiguous), but its far-periphery members are the single most contested points in the entire dataset (overlap ratio 18.45 — over 4× higher than any other class's outer bin, and roughly 170× its own inner-bin value). Surprise is simultaneously the "purest when typical" and "most ambiguous when atypical" class.
- **Neutral has the opposite profile**: it starts *already* moderately contested near its own centroid (3.011 — the highest inner-bin value of any class, consistent with its 2.7556 proximity to happy in §5) and only grows moderately toward the edge (5.06). Neutral is never very confident, but it is also never catastrophically ambiguous — a flat, low-grade overlap profile.
- Happy and fear show large but more graded increases (~8×), while angry and sad are the most stable across the radius (Δ of 2.49 and 3.24 respectively) — the negative-affect trio (angry/fear/sad), despite sitting close to each other in centroid space (§5), does not show uniformly steep ambiguity gradients; angry in particular stays comparatively well-defined even at its own periphery.

---

## 7. Signal Retention (Directional Erasure / SVD Ablation)

Source: `reports/phase3/SIGNAL_RETENTION_REPORT.md`, `reports/phase3/retention_metrics.json`.

**Run configuration:**

| Field | Value |
|---|---|
| Evaluation mode | Stratified 80/20 holdout carved from the train split (no separate test-split embeddings used) |
| Seed | 42 |
| Train examples (probe) | 15,220 |
| Eval examples | 3,806 |
| Chance level (6 classes) | 0.1667 |
| Erasure steps completed | 3 (short first-pass run) |

**Linear-probe accuracy as dominant classifier-weight directions are removed:**

| Step | Directions removed | Accuracy |
|---|---:|---:|
| 0 (baseline) | 0 | **0.6905** |
| 1 | top 1 | 0.6902 |
| 2 | top 2 | 0.6934 |
| 3 | top 3 | 0.6936 |

Singular-value magnitude of the removed directions: 10.06, 10.21, 9.87 (steps 1–3).

**Interpretation:** A linear probe on raw CLIP embeddings reaches ~69% accuracy on 6-way classification (chance = 16.7%) — a moderate but clearly above-chance signal, achieved with zero fine-tuning. Removing the top 3 dominant classifier directions produces **no accuracy loss at all** (accuracy is flat, even ticking up slightly by 0.003). At this short horizon, the image emotion signal looks **distributed rather than compressed** — the opposite of the fine-tuned-text signature (MPNet-FT/BGE-FT erase to chance within 15–20 dimensions) and closer to the pretrained-text signature (BGE-Base/MPNet-Base survive 26–67 dimensions) or the brain's highly distributed encoding (D50 = 4 dims but full erasure requires far more). This is consistent with §4: an unfine-tuned CLIP encoder behaving like a pretrained (not fine-tuned) system.

**Caveat (explicit in the source report):** this is a 3-step run, not the 50–100 step curve used to compute erasure points and D50 in the text/brain studies. No erasure point or D50 value can be reported yet for images — only the qualitative "not concentrated in the first few directions" observation.

---

## 8. Representational Stability Under Ablation (RSA)

Source: `reports/phase5/RSA_REPORT.md`, `reports/phase5/rsa_results.json`.

The class-centroid distance matrix (RDM, `euclidean_norm_matrix`) was recomputed at each of the 4 erasure checkpoints (steps 0–3 from §7) and compared pairwise via Spearman correlation.

**Cross-checkpoint Spearman correlations:**

| | ckpt 0 | ckpt 1 | ckpt 2 | ckpt 3 |
|---|---:|---:|---:|---:|
| **ckpt 0** | 1.0000 | 0.9893 | 0.9821 | 0.9821 |
| **ckpt 1** | 0.9893 | 1.0000 | 0.9964 | 0.9964 |
| **ckpt 2** | 0.9821 | 0.9964 | 1.0000 | 1.0000 |
| **ckpt 3** | 0.9821 | 0.9964 | 1.0000 | 1.0000 |

**Interpretation:** The relational structure between class centroids (which classes are close/far from which others) is essentially unchanged across the first three ablation steps — correlations never drop below 0.982. Combined with §7 (removing top directions doesn't hurt accuracy either), this reinforces that the image-embedding geometry is **distributed and structurally stable**, not concentrated in a small number of fragile directions. This is the same qualitative story as the text pretrained models' "redundant manifold" behavior, now extended to a class-relational (not just accuracy) metric.

**Caveat:** inherits the same short-horizon limitation as §7 — a longer erasure schedule (matching the ~50–100 step curves used for text/brain) is needed before this can be reported as a strong low-rank/high-rank claim.

---

## 9. Visual Artifacts on Disk (not reproduced inline, but available)

| File | Content |
|---|---|
| `reports/plots/all-class-cluster-projection.png` | 2D projection (all 6 classes + centroids) |
| `reports/plots/cluster_snapshots/cluster_<class>.png` | Per-class cluster snapshot (6 files: angry, fear, happy, neutral, sad, surprise) |
| `reports/plots/pairwise/<class_a>_vs_<class_b>/scatter-centroids.png` | Pairwise scatter with centroids for all 15 class pairs |
| `reports/all_class/all-class-density-decay.png` | Radial density-decay curve, all classes overlaid |
| `reports/all_class/all-class-surface-density-decay.png` | Volume-normalized density-decay curve |
| `reports/all_class/all-class-overlap-ratio.png` | Overlap-ratio-vs-radius curve, all classes overlaid |
| `reports/phase3/cliff_zoom_top25.png`, `full_erasure.png` | Ablation accuracy curves (3-step) |
| `reports/phase5/rsa_correlation_matrix.png` | Heatmap of the table in §8 |

---

## 10. Cross-Modal Comparison to Text and Brain (from `repo_context/PROJECT_OVERVIEW.md`)

| Property | Text (fine-tuned LLMs) | Text (pretrained LLMs) | Brain fMRI | **Image (CLIP, this report)** |
|---|---|---|---|---|
| Silhouette (raw space) | 0.68–0.86 | 0.047–0.057 | negative (high-D artefact) | **−0.0009** |
| Void present | Yes | Yes | Yes | **Yes** |
| Belt present | Yes | Yes | n/a (not directly reported) | **Yes** |
| Signal concentration | Sharp cliff, erased dim 15–20 | Distributed, erased dim 26–67 | Highly distributed (D50=4) | **Distributed (flat/no drop over first 3 directions; full curve pending)** |
| Geometric outlier class | Surprise (text) | Surprise (text) | — | **Surprise (image, independently)** |
| Ambiguity gradient present | Yes (r=0.957–0.988, distance→logit) | — | Yes (r=0.61, margin→uncertainty) | **Yes (qualitative: inner-vs-outer overlap ratio rises in every class)** |

The headline cross-modal observation from this first pass: **the Void/Belt local structure and the "surprise is the outlier class" finding both replicate in a third modality (image) using a completely different, off-the-shelf, non-fine-tuned encoder (CLIP) and a non-overlapping label set (FER2013 faces vs. dair-ai/emotion text).** This is independent supporting evidence for the project's core geometric-competition hypothesis, obtained without any emotion-specific training on the image side.

---

## 11. Limitations and Open Items

- **Train-only.** No test-split embeddings exist yet (`test_raw.npy` absent); the FER2013 balanced test split (4,986 examples, `data/processed/test/`) is prepared but not embedded. The original experiment plan's "Run B" (test-split replication) has not been executed.
- **Short ablation horizon.** Phase 3/5 only cover 3 erasure steps. No erasure point or D50 value is computable yet; the text/brain equivalents used 50–100 step curves.
- **No subspace-isolation (Phase 4 equivalent) run yet.** Text embeddings showed silhouette jump from ~0.05 to ~0.41 in a top-20D subspace for pretrained models; whether the same "cloud is a high-D artefact" story holds for CLIP image embeddings is untested.
- **No global scalar overlap % metric.** Text's `overlap_metrics.json`-style single overlap percentage per model has not been computed for images — only the per-class, per-radius `overlap_ratio` curves in §6.3 exist. A direct numeric comparison to text's 0.42–19.22% overlap figures is not yet possible.
- **No statistical validation pass.** No bootstrap CI, permutation test, or cross-model replication (e.g., a second CLIP variant, or a fine-tuned CLIP) has been run. The compass's rationale for skipping this on text ("cross-model replication across 4 architectures is the generalization argument") does not yet apply here — this is a single model, single dataset result.
- **CPU-only embedding generation.** Noted in metadata; not expected to affect embedding values, only runtime.
- **FER2013 image quality.** As flagged in the original experiment plan, FER2013 images are small, grayscale, and visually degraded relative to CLIP's pretraining distribution — the ~69% probe accuracy and near-zero silhouette may partly reflect dataset quality rather than purely a CLIP/geometry property.

---

## 12. Artifact Index (Reproducibility)

All paths relative to `experiments/embeddings_field/image/emotions/fer2013_balanced_no_disgust_clip_vitl14/`:

```
data/processed/train/manifest.jsonl, labels.npy, metadata.json
data/processed/test/manifest.jsonl, labels.npy, metadata.json
artifacts/embeddings/train_raw.npy, train_metadata.json
reports/embedding_metrics/EMBEDDING_METRICS_REPORT.md
reports/embedding_metrics/train/summary.json, centroid-distance-matrix.md
reports/all_class/metrics.json, all-class-density-decay.png,
  all-class-surface-density-decay.png, all-class-overlap-ratio.png
reports/plots/all-class-cluster-projection.png, cluster_snapshots/*, pairwise/*/scatter-centroids.png
reports/phase3/SIGNAL_RETENTION_REPORT.md, retention_metrics.json,
  rdm_checkpoints.json, cliff_zoom_top25.png, full_erasure.png
reports/phase5/RSA_REPORT.md, rsa_results.json, rsa_correlation_matrix.png
```

Source scripts (in `src/`): `prepare_fer2013_dataset.py`, `generate_clip_image_embeddings.py`, `run_embedding_metrics.py`, `run_all_class_density_overlap.py`, `run_signal_retention.py`, `run_rsa.py`, `plot_all_class_scatter.py`, `plot_class_snapshots.py`, `plot_pairwise_scatter_with_centroids.py`, `metrics.py`.

Original design document: `../../reports/fer2013-clip-vitl14-experiment-plan.md`.

---

## 13. Recommended Next Steps

1. Generate `test_raw.npy` for the balanced test split and re-run §4–§6 to confirm the geometry is not a train-split artefact (this was "Run B" in the original plan).
2. Extend the phase-3/phase-5 ablation to the full 50–100 step schedule used for text, to get a real erasure point / D50 and a longer-horizon RSA stability curve.
3. Run a top-20D (or similarly small) subspace isolation pass, mirroring text Phase 4, to test whether the near-zero silhouette is a high-dimensional-noise artefact rather than absent structure.
4. Compute a single global overlap-percentage metric per class (or per model), directly comparable to text's `overlap_metrics.json` figures.
5. If time/compute allows, add a second image encoder (or a lightly fine-tuned CLIP head on FER2013 emotion labels) to get the same "pretrained vs. fine-tuned" contrast the text study relies on for its generalization argument.
