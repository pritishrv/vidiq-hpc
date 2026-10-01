# RAVDESS Video + Speech/Song Emotion Geometry — Experiment Plan

Date drafted: 2026-09-01
Status: **PLAN ONLY — nothing in this document has been run.** No RAVDESS files have been downloaded into this repo yet (`data/Video_Datasets/` is currently empty). This mirrors the format of `experiments/embeddings_field/image/emotions/reports/fer2013-clip-vitl14-experiment-plan.md` and is written before any execution, exactly like that document was.

Grounded in: `Final_Project_Proposal.pdf` §3.5 ("Proposed Video Representation and Temporal Emotion Geometry") and §3.6 ("Evaluation and Statistical Validation"), plus `repo_context/PROJECT_OVERVIEW.md` for the text/brain/image methodology this must stay consistent with.

---

## 1. Purpose

The proposal's video component (§3.5) commits to three things:

1. Extract embeddings with a **transformer-based video architecture** capable of encoding **both spatial and temporal information** across frame sequences (not just per-frame image embeddings).
2. Add **action variables / motion descriptors** computed across neighbouring frames or short frame ranges, alongside the embeddings, to preserve temporal continuity.
3. Test whether **temporal emotional trajectories** reproduce the same geometric structures already found in text and brain data — centroid competition, overlap regions, ambiguity gradients, low-dimensional emotional manifolds — and whether temporal context **reduces or amplifies** ambiguity relative to the static (image) case.

This plan operationalizes those three commitments into an exact, runnable pipeline, using RAVDESS as the dataset, and reuses the metric definitions already implemented for text (`experiments/understanding_text_embeddings/`) and image (`experiments/embeddings_field/image/emotions/`) wherever the concept transfers directly.

---

## 2. Dataset

**Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**

- Source: Zenodo record 1188976 (full Speech + Song, Audio + Video, ~24.8 GB) — https://zenodo.org/record/1188976
- Audio-only speech subset also mirrored on Kaggle (1440 `.wav` files) — this is the subset described in your message.
- License: **CC BY-NC-SA 4.0** (non-commercial). This must be flagged in any coursework/paper write-up — it constrains redistribution and commercial use of derived artifacts.
- Academic citation: Livingstone SR, Russo FA (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391

**Actors:** 24 professional actors (12 female, 12 male; odd actor IDs = male, even = female), two lexically-matched statements ("Kids are talking by the door" / "Dogs are sitting by the door"), North American accent, 2 repetitions per statement.

**Emotion categories (speech subset, 8 classes):** neutral, calm, happy, sad, angry, fearful, disgust, surprised. Two intensity levels (normal, strong) for every emotion except neutral (normal only).

**Filename identifier (7-part, `-` separated):**

| Position | Field | Values |
|---|---|---|
| 1 | Modality | 01 = full audio-video, 02 = video-only, 03 = audio-only |
| 2 | Vocal channel | 01 = speech, 02 = song |
| 3 | Emotion | 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised |
| 4 | Intensity | 01 normal, 02 strong (no strong for neutral) |
| 5 | Statement | 01 "kids", 02 "dogs" |
| 6 | Repetition | 01 or 02 |
| 7 | Actor | 01–24 (odd = male, even = female) |

**Per-actor trial count (speech):** 4 trials for neutral (1 intensity × 2 statements × 2 reps) + 8 trials × 7 other emotions = 60 trials/actor × 24 actors = 1440 files — matches the description you pasted. Expected balanced-by-construction speech-subset counts across 24 actors: neutral = 96, each of the other 7 emotions = 192.

**Song subset caveat:** the RAVDESS song set is known to use a reduced emotion label set (commonly reported as neutral/calm/happy/sad/angry/fearful — i.e. *no disgust or surprised*) and a slightly different actor pool. **Do not assume this without checking** — the exact song-subset class list and actor coverage must be read directly off the downloaded file manifest during Layer 1 (§4) before any class-balancing decision is made, exactly the way the image pipeline discovered FER2013's true per-class counts empirically before deciding to drop `disgust`.

**Modality decision for this project — two parallel tracks, both from RAVDESS, both addressing what you described wanting to use:**

- **Track V (Video):** the full audio-video (`Modality=01`) or video-only (`Modality=02`) files — this is what satisfies proposal §3.5's "transformer-based video architecture... spatial and temporal information across frame sequences."
- **Track A (Audio — Speech + Song):** the audio-only (`Modality=03`) files, vocal channel split into speech (`01`) and song (`02`) — this is the specific subset you referenced wanting to sample from ("some samples of speech and some samples of song from similar actors").

Both tracks are run through the **same downstream geometry pipeline** (§6), so they are directly comparable to each other and to text/image/brain — this is the point of the "unified geometric analysis framework" the proposal commits to in §1.

---

## 3. Label Scheme and Cross-System Mapping

To stay comparable with the rest of the study, RAVDESS's 8 classes need an explicit mapping onto the label vocabularies already in use elsewhere in the repo:

| RAVDESS | Text (dair-ai, 6-class) | Brain (DS005700, 5-class) | Image (FER2013, 6-class) |
|---|---|---|---|
| neutral | — (not in text set) | — | neutral |
| calm | — | calm | — |
| happy | happiness | delighted | happy |
| sad | sadness | depressed | sad |
| angry | anger | — | angry |
| fearful | fear | afraid | fear |
| disgust | — (dropped from image study too) | — | dropped |
| surprised | surprise | — | surprise |

**Decision:** keep all 8 RAVDESS classes for the within-video-study analysis (§6), matching the "keep the native label set, exclude only truly sparse classes after checking real counts" precedent from the image study. For **cross-system RSA** (comparing RAVDESS's RDM to the text/brain/image RDMs — proposal §3.3's method extended to video), restrict to the **overlapping triplet-or-larger set** that exists in all systems being compared, exactly as the brain/text comparison already restricts to the Fear/Happiness/Sadness triplet (`repo_context/PROJECT_OVERVIEW.md` Finding 6b). For RAVDESS vs. text vs. image the natural overlap is **{happy, sad, angry, fear, surprise}** (5 classes) since neutral/calm/disgust don't have full support across every system. This mapping table needs to be logged with the same "conceptual equivalence — flag as a potential reviewer question" caveat already used for the brain/text mapping.

**Valence-Arousal coordinates:** `repo_context/PROJECT_OVERVIEW.md` already has V/A values for Happiness, Sadness, Calm, Fear — but **not** for Anger, Disgust, Surprise, or Neutral in numeric form (the existing table lists them as `—`). Before running the V/A-alignment step (§6, step 6.7) these need to be filled in from the same citation chain already in use (Russell 1980 circumplex + DEAP/Koelstra numerical anchor + ANEW/IAPS), not invented ad hoc. This is a required prep step, not an assumption to skip.

---

## 4. Data Preparation Pipeline (Layer 1)

Mirrors `prepare_fer2013_dataset.py`'s structure.

**Step 4.1 — Acquire and verify.**
Download the relevant RAVDESS archives from Zenodo (full AV + video-only for Track V; audio-only speech + song for Track A). Verify file counts against the known totals (1440 for audio-only speech) before proceeding — if the video-only/full-AV or song archives differ from documented totals, record the actual counts rather than assuming them.

**Step 4.2 — Parse filenames into structured metadata.**
For every file, decode the 7-part identifier into: `modality`, `vocal_channel` (speech/song), `emotion`, `intensity`, `statement`, `repetition`, `actor_id`, `actor_sex` (derived: odd=male, even=female). Write one manifest row per file.

**Step 4.3 — Actor-level train/test split (not clip-level).**
Unlike FER2013 (independent images, split at random), RAVDESS has exactly 24 repeated "subjects." Splitting clips at random risks the same actor's voice/face appearing in both train and test, inflating apparent accuracy. **Split by actor ID**, e.g. 18 actors train / 6 actors test (75/25, stratified by sex — 9M/9F train, 3M/3F test), fixed seed = 42. This also sets up a **leave-one-actor-out (LOAO) cross-validation** option later, directly analogous to the brain study's LOSO (leave-one-subject-out) protocol (`PROJECT_OVERVIEW.md` Finding 5, brain LOSO accuracy 0.56).

**Step 4.4 — Class balancing.**
After the actor split, compute real per-class counts within each split (do not assume 192/96 — intensity and song-subset irregularities may change this). Downsample to the per-split minority count, deterministic seed, and log the exact sampled file list — same discipline as the image pipeline's balanced-subset step.

**Step 4.5 — Track V clip preparation (video).**
For each selected video file: decode to frames, trim/pad to a fixed clip length. Proposal §4 (Work Plan) specifies **5–10 second clips with limited contextual complexity** — RAVDESS clips are naturally short (a few seconds per spoken statement), so use the native clip length, capped at 10s, no artificial extension needed. Uniformly sample **N=16 frames** per clip (a standard choice for video transformers; adjustable based on compute budget), resized to the target encoder's expected input resolution (224×224 for CLIP/VideoMAE/X-CLIP).

**Step 4.6 — Track A clip preparation (audio).**
For each selected audio file: resample to 16 kHz mono (the standard input rate for Wav2Vec2-family models), trim leading/trailing silence, cap/pad to a fixed duration matching the video clip length for cross-track comparability.

**Step 4.7 — Outputs (mirrors the existing `data/processed/` convention).**

```
data/processed/track_video/{train,test}/manifest.jsonl
data/processed/track_video/{train,test}/labels.npy
data/processed/track_video/{train,test}/metadata.json
data/processed/track_audio/{train,test}/manifest.jsonl
data/processed/track_audio/{train,test}/labels.npy
data/processed/track_audio/{train,test}/metadata.json
```

Each manifest row: source path, split, actor_id, actor_sex, vocal_channel (Track A only), class name, class id, intensity, statement, repetition.

---

## 5. Embedding Generation (Layer 2)

### Track V — Video

**Two parallel embedding methods, both run, both compared** (this directly tests the proposal's own premise — that a real spatio-temporal model should behave differently than a naively-pooled per-frame model):

**V1 — Baseline: CLIP-frame-mean-pool.**
- Model: `openai/clip-vit-large-patch14` (same encoder already used for the FER2013 image study — direct comparability to Finding-set in `IMAGE_EMOTION_GEOMETRY_FULL_REPORT.md`).
- Method: encode each of the 16 sampled frames independently, mean-pool across the frame axis → one 768D vector per clip.
- Rationale: cheap, reuses existing infra, and is the necessary control to check whether a "real" video transformer (V2) actually adds anything beyond naive frame averaging.

**V2 — Proposal-faithful: dedicated video transformer.**
- Model: `MCG-NJU/videomae-base` (VideoMAE, native spatiotemporal attention, 768D) **or** `microsoft/xclip-base-patch32` (X-CLIP, 512D, contrastive video-language pretraining, architecturally a temporal extension of CLIP — preferred if closer lineage to the image-study's CLIP encoder is wanted for comparability).
- Method: feed the full 16-frame clip tensor in one forward pass, take the pooled clip-level embedding directly from the model (no manual mean-pooling — the temporal attention is internal to the model).
- This is the track that actually satisfies "transformer-based video architectures capable of encoding both spatial and temporal information across frame sequences" (§3.5) as written.

**Motion descriptors (appended to both V1 and V2 embeddings, not replacing them):**
- Compute dense optical flow between consecutive sampled frames using RAFT (`torchvision.models.optical_flow.raft_large`, pretrained) or, if compute-constrained, Farneback optical flow via OpenCV as a lightweight fallback.
- Reduce to a compact descriptor per clip: mean flow magnitude, variance of flow magnitude, and total motion energy (sum of squared flow vectors) across the frame sequence — a small (~3–8 dimensional) vector, not a full flow field.
- Concatenate this motion descriptor to the clip embedding (768D + motion-dims), producing an "embedding + action variable" representation matching §3.5's exact phrasing ("action variables and motion descriptors computed across neighbouring frames... alongside the extracted video embeddings").
- Keep the raw (un-concatenated) embedding and the concatenated version both on disk, so the geometry pipeline (§6) can be run on each separately — this is the direct empirical test of whether motion information changes the emotional geometry.

### Track A — Audio (Speech + Song)

- Model: `facebook/wav2vec2-large-xlsr-53` (self-supervised speech representation, robust to accent/microphone variation) — chosen over a supervised speech-emotion-recognition (SER) model to stay consistent with the project's "raw pooling / logits over softmax" principle: extract *representations*, not classifier outputs, matching the text pipeline's mean-pooled hidden states rather than pre-baked emotion probabilities.
- Method: attention-mask-weighted mean pooling over the time axis of the hidden states — **the exact pooling strategy already documented for text** (`repo_context/project_context/embeddings_extraction.md`), applied here to the audio time dimension instead of the token dimension.
- **Final-layer and mid-layer variants**, mirroring the text study's 8-variant design (final layer 12 vs. mid layer 6 — for wav2vec2-large-xlsr-53, use the final transformer block vs. block 12 of 24 as the mid-layer analogue). This gives the same "how does emotional geometry evolve across the representational hierarchy" axis that the text study already has.
- Keep speech (`vocal_channel=01`) and song (`vocal_channel=02`) as separate labeled splits within Track A so they can be analyzed independently or pooled — since you specifically want "some samples of speech and some samples of song," both should be embedded and the geometry pipeline should be run once per vocal channel and once on the pooled set, to see whether speech-emotion and song-emotion occupy the same or different regions of the embedding space for the *same actors*.

### Output convention (mirrors `artifacts/embeddings/` in the image study)

```
artifacts/embeddings/track_video_v1_clip_meanpool/{train,test}_raw.npy, {train,test}_metadata.json
artifacts/embeddings/track_video_v2_videotransformer/{train,test}_raw.npy, {train,test}_metadata.json
artifacts/embeddings/track_video_motion_descriptors/{train,test}.npy
artifacts/embeddings/track_audio_speech_final/{train,test}_raw.npy, {train,test}_metadata.json
artifacts/embeddings/track_audio_speech_mid/{train,test}_raw.npy, {train,test}_metadata.json
artifacts/embeddings/track_audio_song_final/{train,test}_raw.npy, {train,test}_metadata.json
artifacts/embeddings/track_audio_song_mid/{train,test}_raw.npy, {train,test}_metadata.json
```

Metadata for every file: model id, preprocessing recipe, embedding dimensionality, batch size, device, split sizes, class names/ids, manifest source path — same fields already used in `train_metadata.json` for the image study.

---

## 6. Metrics Calculation (Layer 3)

Every metric below already has a working implementation pattern in this repo (text Phases 1–5, or the FER2013 image study) — this section states exactly which existing method transfers, applied per track/variant (V1, V2, V1+motion, V2+motion, audio-speech-final, audio-speech-mid, audio-song-final, audio-song-mid — 8 embedding sets in total, each run through the same pipeline below).

**6.1 — Cluster diagnostics.** Silhouette score + Davies-Bouldin score on the raw embedding space. Direct analogue of `run_embedding_metrics.py` / `EMBEDDING_METRICS_REPORT.md` (image study §4).

**6.2 — Centroid geometry.** Full pairwise Euclidean centroid-distance matrix across all classes present in that track/split; report nearest/farthest pairs. Direct analogue of `centroid-distance-matrix.md` (image study §5). This is where the surprise-outlier / negative-affect-cluster pattern found in image and text can be directly checked for replication in video and audio.

**6.3 — Radial density (Void + Belt) and per-class ambiguity gradient.** Bin each class's points by distance from its own centroid; report void edge, peak (belt) bin, and the inner-bin-vs-outer-bin overlap ratio spread per class. Direct analogue of `run_all_class_density_overlap.py` / `all_class/metrics.json` (image study §6).

**6.4 — Signal retention (SVD ablation).** Iteratively remove the top dominant classifier-weight directions and track linear-probe accuracy. **Run the full 50–100 step schedule** (the image study explicitly flagged its 3-step run as a limitation — this plan should not repeat that shortfall) to get a real erasure point and D50 value, directly comparable to the text/brain numbers already in the compass (`PROJECT_OVERVIEW.md` Finding 3). Direct analogue of `run_signal_retention.py`.

**6.5 — Subspace isolation.** Project into the top-k (e.g. top-20) SVD/classifier-weight directions and recompute silhouette + accuracy, to test whether a near-zero raw-space silhouette (expected for Track V1/A, by analogy with pretrained CLIP/image results) is a high-dimensional-noise artefact rather than absent structure — direct analogue of the text study's Phase 4 (not yet run for image either — flagged as an open item in `IMAGE_EMOTION_GEOMETRY_FULL_REPORT.md` §11, should be built once for both image and video/audio).

**6.6 — Representational Similarity Analysis (RSA).**
- *Within-track stability:* Spearman correlation of the centroid-distance RDM across the SVD-ablation checkpoints from 6.4 — analogue of `run_rsa.py` (image study §8).
- *Cross-modal RSA:* correlate the RAVDESS RDM (restricted to the 5-class overlap set from §3) against the existing text RDM and brain RDM already computed in `experiments/brain_embedding_understanding/checking_centroids/` and `experiments/understanding_text_embeddings/reports/phase5/`. This is the direct video-side analogue of the compass's Finding 6 ("Relational Paradox") and should be reported the same way: raw correlation, plus a Manhattan-distance sensitivity check, plus comparison against the existing brain-brain noise ceiling (~0.47) for context.

**6.7 — Valence-Arousal alignment.** PCA on class centroids; correlate PC1/PC2 against the V/A reference coordinates (once §3's gap-filling step is done). Direct analogue of `valence-arousal-dimensional_reduction/` (brain study) — this tests whether video/audio, like the brain, ends up arousal-dominant, or whether it patterns with the (valence-dominant) LLM result, which would be a genuinely new finding either way.

**6.8 — Ambiguity gradient validation.** Train a linear classifier per track; compute Spearman correlation between (a) centroid margin [distance to own centroid − distance to nearest competing centroid] and (b) classifier uncertainty (1 − max softmax probability, or logit margin). Direct analogue of the brain study's margin-vs-uncertainty result (r=0.61, AUC=0.81) and the text study's distance-vs-logit result (r=0.957–0.988).

**6.9 — Statistical validation (proposal §3.6, explicit requirement).**
- **Bootstrap confidence intervals**: resample clips with replacement (1000–5000 resamples) around every headline correlation (6.6 cross-modal RSA, 6.8 ambiguity gradient) and report 95% CI, matching the format already used for the cross-system ambiguity gradient in the compass (r=0.9565, 95% CI [0.9370, 0.9725]).
- **Permutation testing**: shuffle class labels (5000 permutations) and recompute each headline statistic to build a null distribution; report permutation p-value, matching the existing brain-study format.
- **Leave-one-actor-out (LOAO) cross-validation**: train the linear probe used in 6.4/6.8 with each actor held out in turn (24 folds for the full actor pool, or within the 18-actor train split), report mean accuracy ± 95% CI — this is RAVDESS's direct structural analogue of the brain study's LOSO protocol, and is the correct generalization check here (the compass explicitly reserved LOSO-style validation for data with repeated subjects, and RAVDESS — unlike the single-shot text embeddings — has exactly that structure).

**6.10 — Motion-effect test (the proposal's own stated question, §3.5's last sentence).** Directly compare the metrics in 6.1–6.3 computed on the raw clip embedding vs. the same embedding concatenated with the motion descriptor (§5, Track V). If silhouette improves / overlap ratio drops / void widens with motion appended, that is evidence temporal dynamics **reduce** ambiguity; the reverse would be evidence they **amplify** it. This is the one metric in this plan with no existing precedent elsewhere in the repo — it is genuinely new and specific to video.

### Output convention (mirrors the image study's `reports/` tree, once per track/variant)

```
reports/<track_variant>/embedding_metrics/{train,test}/summary.json, centroid-distance-matrix.md
reports/<track_variant>/all_class/metrics.json, *.png
reports/<track_variant>/phase3/retention_metrics.json, rdm_checkpoints.json, *.png
reports/<track_variant>/phase4/subspace_metrics.json
reports/<track_variant>/phase5/rsa_results.json, *.png
reports/<track_variant>/cross_modal_rsa/rsa_vs_text.json, rsa_vs_brain.json
reports/<track_variant>/va_alignment/alignment_metrics.json
reports/<track_variant>/statistical_validation/bootstrap_ci.json, permutation_test.json, loao_accuracy.json
reports/motion_effect_comparison/motion_vs_no_motion.json
```

---

## 7. Source Scripts to Build (mirrors `src/` in the image study)

```
src/prepare_ravdess_manifest.py        # §4.2–4.4: parse filenames, actor split, balance
src/extract_video_frames.py            # §4.5: decode + uniformly sample frames per clip
src/prepare_audio_clips.py             # §4.6: resample/trim audio
src/generate_clip_frame_embeddings.py  # §5 Track V1
src/generate_video_transformer_embeddings.py  # §5 Track V2
src/compute_motion_descriptors.py      # §5 motion descriptors
src/generate_wav2vec_embeddings.py     # §5 Track A
src/run_embedding_metrics.py           # §6.1–6.2 (reuse image-study version, generalized)
src/run_all_class_density_overlap.py   # §6.3 (reuse)
src/run_signal_retention.py            # §6.4 (reuse, extended to full step schedule)
src/run_subspace_isolation.py          # §6.5 (new — also backport to image study)
src/run_rsa.py                         # §6.6 within-track (reuse)
src/run_cross_modal_rsa.py             # §6.6 cross-modal (new)
src/run_va_alignment.py                # §6.7 (adapt from brain study)
src/run_ambiguity_gradient.py          # §6.8 (adapt from brain study)
src/run_statistical_validation.py      # §6.9 (adapt from brain study's bootstrap/permutation code)
src/run_motion_effect_comparison.py    # §6.10 (new)
```

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| RAVDESS license is CC BY-NC-SA (non-commercial) | Flag explicitly in any submission; fine for coursework/academic use, but do not treat as unrestricted |
| Only 24 actors — small, and song subset may cover fewer | Actor-level split + LOAO CV (§4.3, §6.9) makes the generalization limit explicit rather than hidden; report it the same way the brain study reports its N=40 caveat |
| Song-subset class list may not match speech (disgust/surprised possibly absent) | Verify empirically at Layer 1 (§4.1) before any balancing decision; do not assume symmetry with speech |
| Video decode + VideoMAE/X-CLIP inference is far heavier than CLIP-on-images | Keep V1 (CLIP-frame-mean-pool) as a cheap fallback track that can ship even if V2 (real video transformer) is compute-constrained |
| Speaker/channel leakage between train and test | Actor-level split, never clip-level (§4.3) |
| Motion descriptors mostly reflect speaking mouth movement, not "emotional" motion | Report this limitation explicitly if 6.10 shows an effect — cannot claim it's affect-specific without a control (e.g. neutral-statement motion baseline) |
| Cross-modal label mapping (RAVDESS → text/brain vocabularies) is conceptual, not exact | Same caveat already applied to the existing brain↔text triplet mapping — carry it forward, don't strengthen the claim beyond what the mapping supports |

---

## 9. Minimum Deliverables

- Prepared manifests + actor-level balanced splits for both tracks (§4.7)
- Embeddings for all 8 variants listed in §5's output convention
- Full metrics run (§6.1–6.10) for at least the two primary variants: V2 (video transformer, no motion) and audio-speech-final — these two most directly answer the proposal's stated question
- A findings write-up in the same style as `IMAGE_EMOTION_GEOMETRY_FULL_REPORT.md`, once real numbers exist
- Explicit before/after comparison for §6.10 (motion effect) — this is the one result unique to the video track and should be the headline of the eventual report

---

## 10. Recommended Run Order

1. Download and verify RAVDESS archives (§4.1); confirm song-subset class/actor coverage empirically.
2. Build manifests, actor split, balanced subsets (§4.2–§4.4).
3. Generate Track A (audio) embeddings first — smallest compute footprint, fastest path to a first real result.
4. Run the full metrics pipeline (§6) on Track A speech-final as a pilot, to shake out bugs in the generalized metrics scripts before spending compute on video.
5. Generate Track V1 (CLIP-frame-mean-pool) — reuses existing image-study infra almost directly.
6. Generate motion descriptors and the V1+motion concatenated variant; run §6.10 motion-effect test.
7. Generate Track V2 (dedicated video transformer) if compute allows; repeat §6.10 with V2.
8. Run cross-modal RSA (§6.6) and statistical validation (§6.9) last, once all individual-track results are stable.
9. Write the findings report.
