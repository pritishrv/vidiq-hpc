# Meeting Minutes — NeurIPS Paper: Results & Next Steps

**Date:** 16 April 2026  
**Present:** Daniel Sikar, Pritish Ranjan PG-Verma, Josh, Andrew (joined late)  
**Topic:** New dataset / new model experiments; fine-tuning plan; image extension

---

## Results

### Experiment 1: New Dataset, Same Model
- Previous dataset had ~564 samples per class (after balancing). New dataset (the 20K set shared by email) has 4–5× more data points per class.
- Clusters plotted with the same pre-trained model. Emotions continue to form the same spatial groupings:
  - **Anger, sadness, fear** cluster together.
  - **Joy and love** cluster together.
  - **Surprise** remains an outlier, not blending with either group.
- Minimum/maximum distances from centroid remain comparable (~6.5 vs ~7 on the previous dataset).
- **Conclusion:** Geometry is not dataset-specific. The pattern generalises across datasets.

### Experiment 2: New Dataset, New Model
- Same larger dataset, different pre-trained model.
- Centroid positions shift somewhat, but the relational pattern is preserved: anger/sadness remain close, joy/love remain close, surprise sits apart.
- Density decay (number of data points per radial distance, corrected axis label) follows the same linear decay pattern.
- Overlap ratio peak occurs just after density peak in both models (e.g. density peak at ~2 units, overlap onset at ~2.4).
- Absolute magnitudes differ (peak at ~9 units for model 1; ~4–5 units for model 2) but the functional pattern is unchanged.
- **Conclusion:** Cross-model, cross-dataset validation achieved. The geometric patterns are universal properties, not model or dataset artefacts.

### Key Observation: No Pure Emotion
- Every data point lies somewhere between clusters, reflecting mixed emotional content.
- Vector arithmetic analogy discussed (Italy − Rome + France ≈ Paris): the same principle applies — emotional embeddings can be decomposed and composed geometrically.

### Axis Label Correction
- The y-axis previously labelled "density" should read **"number of data points per radial distance"** (normalised by shell surface area at that radius). Pritish to update plots before submission.

---

## Discussion: Future Work Flagged for Paper

- **Experience as geometry:** A human mind draws on distributed memories to answer a question (e.g. Mahatma Gandhi's birthday via a school essay). LLMs appear to do the same — each transformer layer produces an embedding that can be located in this geometric space. Tracking the path through layers could reveal what context a model draws on. This was flagged as a **future work** discussion point for the paper.
- **Fine-tuning hypothesis:** Fine-tuning on a labelled subset (e.g. classes A–W, tested on X–Z) may tighten clusters because the model has "learned" the label. Alternatively, geometry may remain unchanged — either outcome is publishable.
- **Schrodinger's cat argument for images:** Unlike emotions, concrete objects (a cat, a dog) have a definite referent. We expect image embeddings to sit closer to their centroid, giving tighter clusters. This contrast is a discussion point in the images section.

---

## Next Steps

### Immediate (before next weekly meeting)

| Action | Owner |
|--------|-------|
| Run 6 fine-tuning jobs on HPC: multiclass classifier, 10 / 50 / 100 epochs × balanced / unbalanced data | Daniel |
| Add learning-rate sweep experiments: one low-LR and one high-LR run to bracket fine-tuning sensitivity | Pritish / Daniel |
| Correct axis label ("number of data points per radial distance") on all existing plots | Pritish |
| Report fine-tuning results (cluster tightness, logit–centroid correlation) at next meeting | Pritish |

### Images Phase (to begin next meeting, target before abstract deadline 5 May)

| Action | Owner |
|--------|-------|
| Identify image dataset — preferred: multi-category with activity labels (e.g. dog/cat/horse × walking/running/standing) to replicate cluster + outlier geometry | Josh / Daniel |
| Phase 1 images: no segmentation — embed full images, plot clusters, check density decay | Pritish |
| Phase 2 images: segment object from background, repeat — test whether reduced context tightens centroid proximity | Pritish |
| Share Overleaf link with Andrew and Neil so they can contribute | Daniel |

### HPC Housekeeping
- Pritish resolved SSH access (must connect to university VPN, then SSH into Hyperion — this is undocumented; note it for the paper's reproducibility section).
- Jobs use Qwen model, batch size 8, learning rate 5×10⁻⁵. Scripts are committed to the HPC GitHub repo.
- A heavy user (likely mechanical engineering / fluid simulation) is occupying most GPU slots; jobs will queue but will run.

---

## Abstract Deadline
**May 5, 2026** — approximately 19 days from this meeting. Images work must be complete and results interpretable before then.
