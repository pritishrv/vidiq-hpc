# Meeting Minutes — Project Explanation and Adding Images

**Date:** 21 April 2026  
**Present:** Daniel Sikar, Pritish Ranjan PG-Verma, Josh, Andrew  
**Topic:** Project walkthrough for newer team members; embedding methodology; image and video extension; HPC coordination

---

## 1. Project Walkthrough and Core Hypothesis

- Daniel gave a from-scratch explanation of transformer architecture, tokenisation, embeddings, and the motivation for the project.
- The working hypothesis remains that embeddings retain contextual information from the input, even when the model behaves as a black box at output level.
- Current validation work is based on text embeddings from emotion-labelled sentences. Each sentence is passed through a pre-trained model, an embedding is extracted, and the resulting points are analysed geometrically.
- The team clarified that the current phase does **not** train the base model for these initial experiments; it uses pre-trained models to generate embeddings and studies the resulting geometry.
- A key methodological point raised in discussion was that the paper must explicitly state **which layer / extraction point** the embedding comes from in each model architecture.

**Action — Daniel / Pritish:** Make sure the methods section records the exact embedding extraction point for each model used.

---

## 2. Text Embedding Results Recap

- The active text work remains focused on emotion-labelled sentence datasets, especially the six-emotion multiclass setting.
- Each sentence corresponds to one point in embedding space; centroids and density-style analyses are computed in the full embedding dimension, with PCA used only for visualisation.
- The team revisited the interpretation of the radial plots:
  - the previous "density" wording was misleading for one plot;
  - that plot is better described as **number of data points per radial-distance band**;
  - absolute density, when normalised by available volume, behaves as originally expected near the centroid.
- Discussion reaffirmed the conceptual result that there appears to be no "pure emotion" point at the centre; instead, points occupy regions at non-zero distance from the centroid.

**Action — Pritish:** Update plot wording where needed so radial-band counts are not labelled as density.

---

## 3. Image Extension

- The next major extension is from text into images, using the same general idea: generate embeddings, plot them, and test whether contextual properties can be recovered geometrically.
- One line of discussion focused on whether image embeddings could reveal multiple independent contextual attributes at once, for example:
  - object category, such as car / bus / truck;
  - colour, such as black / non-black.
- The proposed logic is to reuse the **same embeddings** and test different labelings against them, rather than generating a separate representation for each concept.
- This would allow the team to test whether the same image embedding space simultaneously supports clustering by more than one valid contextual attribute.
- Daniel showed the current experiment repository structure and explained the expected folder layout for image work, including dedicated folders for experiments, reports, source code, and datasets.

**Action — Team:** Implement image experiments inside the shared repository structure under the dedicated images area.

---

## 4. Video Extension

- The group also discussed how video could be handled as the next extension after images.
- The working idea is to segment a video over time and generate embeddings for each segment rather than treating the whole video as a single unit.
- A frame-by-frame approach may be too granular, so a coarser segmentation such as fixed timestamps or short intervals was suggested as a more practical starting point.
- The implementation sessions for video will be **vibe coded as a group** with **Andrew, Daniel, Josh, and Pritish** working together.

**Action — Andrew / Daniel / Josh / Pritish:** Organise group vibe-coding sessions to implement the first video pipeline.

---

## 5. HPC Access and Coordination

- The team confirmed that all four members either now have HPC access or are in the process of getting fully set up.
- Daniel noted that, with four accounts available, the group should be able to run multiple experiments in parallel and make fuller use of available GPU resources.
- Setup remains a practical barrier for some members, particularly in preparing projects correctly for HPC execution rather than just obtaining account access.
- Daniel offered to share the commands needed to get others running more quickly.

**Action — Daniel:** Share / resend the essential HPC setup commands to the team.

**Action — Josh / Andrew:** Complete HPC setup and confirm they can run project code there.

---

## 6. Current Qwen Run and Repository Follow-Up

- During the meeting, a Qwen-based run on the HPC completed and pushed JSON outputs.
- The initial push did **not** include the required `.npy` embedding files because they had been excluded.
- The team identified that the embeddings are still needed for downstream plotting and analysis, so the run output handling needs to be corrected.
- Once the embedding files are available in the repository, the existing analysis pipeline can be run to generate the corresponding plots.

**Action — Daniel:** Force the required `.npy` embedding files into the repository output for the Qwen run.

**Action — Pritish:** Run the analysis pipeline on the completed outputs once the embedding files are available.

---

## General Notes

- This meeting mainly served to bring newer members up to speed on the conceptual basis of the project before moving fully into image and video experiments.
- The repository structure and workflow are now clear enough that image tasks can be split across the group while keeping outputs in a consistent layout.
- Coursework discussion at the end of the transcript was intentionally excluded from these minutes.
