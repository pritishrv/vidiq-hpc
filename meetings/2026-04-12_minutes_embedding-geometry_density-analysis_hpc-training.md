# Meeting Minutes — 12 April 2026

**Project:** LLM Embedding Geometry / NeurIPS submission  
**Attendees:** Daniel Sikar, Pritish Ranjan PG-Verma  
**Sessions:** Three recorded sessions (17:48, 19:09, 20:43)  
**Total duration:** ~23 minutes

---

## Session 1 — Density Analysis Correction & Key Geometric Findings (17:48, 13m 36s)

### Density Calculation Correction

Pritish identified an error in how density was being framed. The previous approach computed
the number of points in a radial band (between two radii R₁ and R₂) divided by the band
width. This was labelled "density per unit". However, in 768-dimensional space the *volume*
of a spherical shell grows by a very large factor as radius increases, meaning the band
captures more points simply because it has more capture area — not because the region is
genuinely denser.

**Corrected understanding:**
- **Y-axis previously labelled:** density per unit ← **incorrect label**
- **Y-axis should be:** number of points per band
- When normalising by shell volume in 768D, density per unit volume *only drops* as
  distance from the centroid increases — it never increases.
- The peak in raw point counts near radius 9 is a volume artefact, not a true density peak.

### Key Finding: The Void

When examining the number of points per band:

> **There are no data points between the centroid and a radius of ~7.5 for any emotion.**

This holds across all six emotion classes. Every class has a hollow core — a void — from
the centroid out to approximately r = 7.5. This was confirmed as consistent across all
emotions.

### Key Finding: No Pure Emotion

Because all data points sit at a non-zero distance from their class centroid, and because
the centroids are surrounded by mixed-class neighbourhoods, the data supports the observation:

> **There is no pure emotion** — every data point has non-zero proximity to other emotion
> centroids. No instance of, e.g., anger is purely anger. This mirrors real-world intuition
> (anger is always about something and carries other affect).

### Overlap Definition Clarified

A previous informal definition of overlap (anything beyond a certain radius counts as
overlap) was corrected. The actual definition in use:

> A point P (with ground-truth class C) **overlaps** class C′ if the Euclidean distance
> from P to the centroid of C′ is **less than** the distance from P to the centroid of its
> own class C.

A point can overlap one, several, or all other classes simultaneously. There is no radius
threshold in this definition.

---

## Session 2 — Centroid Distance Matrix & Radial Scatter Plot (19:09, 3m 12s)

### Centroid Distance Matrix (Balanced Dataset)

A 6 × 6 table of pairwise Euclidean distances between class centroids was produced on the
balanced dataset. Key observations:

- Sadness centroid is **nearest** to the anger centroid.
- Sadness centroid is **furthest** from the joy centroid.
- (Full matrix values to be included in the experiment report.)

### Radial Scatter Plot

Plot axes: emotions on x-axis; distance to own class centroid on y-axis. Per-class ranges:

| Emotion | Min distance to centroid | Max distance to centroid |
|---------|--------------------------|--------------------------|
| Love    | 7.44                     | 11.32                    |
| Anger   | 7.32                     | 12.70                    |
| Fear    | ~7.3                     | **12.77** (global max)   |

- **Love** has the tightest cluster — its furthest points are closer than the furthest
  points of any other emotion.
- **Fear** has the most dispersed points, reaching the greatest distance from its centroid.
- Anger and fear are the most spread out overall.
- 10th / 50th / 90th percentile distances were also computed per class.

---

## Session 3 — Fine-Tuned Model Architecture & HPC Training Plan (20:43, 6m 36s)

### New Code Pushed to Repo

Pritish added `experiments/text_model/` to `vidiq-hpc` (already pushed):

| File | Purpose |
|------|---------|
| `train_multiclass.py` | Full model training pipeline |
| `requirements.txt` | Python dependencies for training |
| `README.md` | Context for HPC operator / Claude Code |

### Model Architecture

- **Base model:** Qwen3, 1.7 B parameters
  (600 M was considered too small; 1.7 B is the agreed starting point)
- **Embedding layer:** 768-dimensional hidden state extracted after the transformer
- **Head:** Fully-connected layers → 5-class logits → softmax
- **Target classes (5):** sadness, anger, fear, love, surprise
  (Joy is labelled as *happiness* in the updated dataset; treated as equivalent)
- **Estimated VRAM:** ~8 GB; HPC has 40 GB and 80 GB cards — well within capacity

### Experiment Goal

Train the model on `dair-ai/emotion`, then:
1. Extract embeddings at the 768-dimensional layer (same position as the pre-trained
   BGE model used previously).
2. Repeat the full geometry experiment suite (centroid distances, overlap analysis,
   density/radial analysis, visualisations) on the fine-tuned model's embedding space.
3. Compare geometry of fine-tuned vs. pre-trained embeddings.

### Flexibility Note

The architecture is not rigid. Any model can be substituted provided:
- A 768-dimensional embedding layer remains accessible for extraction.
- The output head retains 5-class logits.

---

## Action Points

### Pritish

| # | Action | Notes |
|---|--------|-------|
| P1 | Fix y-axis label on radial density plot | Change from "density per unit" → "number of points per band" |
| P2 | Update any figures / report text that refers to "density" in the band-counting sense | Use "points per band" or "point count per shell" throughout |
| P3 | Add the full 6×6 centroid distance matrix values to the experiment report | Already computed; needs to be recorded in writing |

### Daniel

| # | Action | Notes |
|---|--------|-------|
| D1 | Verify HPC access end-to-end | Currently routing through a Windows machine; confirm full login and job submission |
| D2 | Run a smoke-test job on HPC | Confirm the environment works before launching the full training run |
| D3 | Clone `vidiq-hpc` onto HPC and set up `text_model` environment | Follow `experiments/text_model/README.md` and `requirements.txt` |
| D4 | Add HPC batch scripts to repo | Create an `hpc/` or `experiments/text_model/scripts/` folder with SLURM / PBS job scripts as needed |
| D5 | Launch `train_multiclass.py` on HPC | Can substitute model (e.g. swap Qwen3 for another) as long as 768-dim layer and 5-class head are preserved |
| D6 | After training: extract embeddings and run geometry experiments | Same pipeline as pre-trained BGE runs; compare results |

---

## Next Meeting

To be scheduled once Daniel has confirmed HPC access and the training job is queued or
running.
