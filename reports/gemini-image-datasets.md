https://gemini.google.com/app/fd6bab07b9e77efc

# **Topological and Geometric Evaluation of Affective Image Corpora for High-Dimensional Sentiment Embedding Research**

The advancement of affective computing has transitioned from a preoccupation with categorical classification accuracy toward a deeper investigation of the latent manifolds that represent human emotion within neural networks. For researchers focused on the geometric properties of these embeddings—specifically class compactness, centroid distances, and the semantic overlap between sentiment categories—the choice of dataset is no longer merely a matter of scale or "state-of-the-art" performance metrics. Instead, it becomes a question of label granularity, attribute richness, and the structural fidelity of the affective space. A robust embedding-geometry experiment requires datasets that can survive the transition from high-dimensional feature vectors to low-dimensional manifold representations without collapsing the nuanced differences between adjacent affective states, such as "awe" and "amusement" or "anger" and "disgust."

## **Theoretical Foundations of Affective Manifolds**

The analysis of embedding geometry is predicated on the manifold hypothesis, which posits that high-dimensional data, such as images, lies on a lower-dimensional manifold embedded within the high-dimensional space.1 In the context of visual sentiment analysis (VSA), this manifold is expected to reflect the psychological structure of affect. Two primary models dominate the annotation schemas of the datasets reviewed: the Categorical Emotion States (CES) model, often following Ekman’s six basic emotions or Mikels’ eight-emotion extension, and the Dimensional Emotion Space (DES), which utilizes the Valence-Arousal-Dominance (VAD) framework.3  
For geometric analysis, the DES model is inherently more conducive to measuring distances and trajectories, as it provides a continuous coordinate system.3 However, the CES model remains essential for cluster-based analysis, where the goal is to determine if a neural network’s latent space naturally segregates "Sadness" from "Fear" with sufficient margin. The most valuable datasets for modern research are those that provide "dual-space" annotations, allowing for a mapping between the categorical clusters and their underlying dimensional coordinates.7

## **Ranking of Candidate Datasets for Geometric Analysis**

The following table ranks the most prominent datasets based on their utility for embedding-geometry experiments. The criteria for this ranking include the precision of the labeling schema, the scale of the human-annotated portion, the availability of object-level or attribute-level metadata, and the minimization of label noise.

### **Table 1: Global Ranking of Affective Datasets for Geometric Research**

| Dataset Name | Rank | Primary Domain | Label Schema | Granularity | Scale (Human-Labeled) | Geometric Utility Score |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| EmoVerse | 1 | General / Multi-scene | B-A-S Triplet \+ VAD \+ CES | Object-level | 219,000+ | 9.8 / 10 |
| EmoSet | 2 | Social / Artistic | CES \+ 6 Perceptual Attributes | Image-level | 118,102 | 9.4 / 10 |
| OASIS (Kurdi) | 3 | Normative Scenes | Continuous Valence / Arousal | Image-level | 900 | 9.0 / 10 |
| EMOTIC | 4 | Contextual People | 26 Categories \+ VAD | Person-level | 23,571 | 8.7 / 10 |
| AffectNet | 5 | Facial Expressions | 8 Categories \+ V / A | Face-level | 450,000+ | 8.2 / 10 |
| EmoArt | 6 | Fine Arts | CES \+ V / A \+ Style Attributes | Image-level | 132,664 | 8.0 / 10 |
| FI (Flickr/Insta) | 7 | Social Media | 8 Categories (Mikels) | Image-level | 23,308 | 7.5 / 10 |
| WEBEmo | 8 | Web Images | 25 Categories | Image-level | 268,000 | 7.2 / 10 |
| T4SA (B-T4SA) | 9 | Multimodal Social | Positive / Neutral / Negative | Image-level | 1,400,000+ | 6.8 / 10 |
| EmotionROI | 10 | Low-level Saliency | 6 Categories (Ekman) | Region-level | 1,980 | 6.5 / 10 |

## **Primary Candidate Analysis: EmoVerse and the B-A-S Paradigm**

EmoVerse stands at the vanguard of affective datasets, specifically designed to bridge the gap between high-level semantics and low-level visual perception through a "knowledge-graph-inspired" annotation schema.7 Its most salient feature for geometry research is the Background-Attribute-Subject (B-A-S) triplet.7 By deconstructing the emotional content of an image into these three components, EmoVerse allows for the calculation of disentangled embeddings. For instance, researchers can evaluate the "centroid distance" between the sentiment evoked by a "funeral scene" (background) versus a "smiling child" (subject) within that scene.

### **Table 2: EmoVerse Technical Profile**

| Feature | Specification |
| :---- | :---- |
| **Publisher** | Various (VCC Tech / ArXiv 2024-2025) |
| **Task Type** | Affective Grounding and Reasoning |
| **Label Schema** | Dual-space: 8 categories (CES) \+ VAD (DES) |
| **Image Count** | 218,522 finely annotated |
| **Label Type** | Continuous (VAD) and Discrete (8-class) |
| **Granularity** | Object-level (Bounding boxes and segmentation masks) |
| **Accessibility** | Public / Open-source |
| **Image Domain** | Art, Nature, Social Media, Synthetic (AIGC) |

The utility of EmoVerse for geometry experiments is amplified by its use of the Segment Anything Model (SAM) and Grounding DINO to provide precise spatial annotations.8 This enables "part-whole" geometric comparisons: the embedding of a specific affective object can be compared to the embedding of the entire scene to determine the degree of "sentiment leakage" or "contextual interference." This is a crucial control for experiments investigating whether a model's sentiment space is truly capturing affective objects or merely responding to global scene statistics.10  
Furthermore, EmoVerse addresses the problem of "cluster overlap" by providing a confidence score for each emotion category.8 This allows researchers to filter for "clean" samples when establishing centroids or to intentionally use "ambiguous" samples to investigate the topological boundaries between emotion classes. Its inclusion of high-dimensional DES annotations facilitates the study of whether the latent space preserves the manifold curvature predicted by dimensional emotion theories.8

## **Large-Scale Attribute Analysis: EmoSet**

EmoSet addresses the "black box" nature of visual emotion by providing 118,102 human-labeled images across eight categories, enriched with six describable emotion attributes: brightness, colorfulness, scene type, object class, facial expression, and human action.5 For geometric analysis, these attributes serve as "nuisance variables" that must be accounted for to ensure the sentiment clusters are not merely proxies for low-level visual cues.

### **Table 3: EmoSet Technical Profile**

| Feature | Specification |
| :---- | :---- |
| **Publisher** | Yang et al. (ICCV 2023 / VCC Tech) |
| **Task Type** | Attribute-rich Visual Emotion Analysis |
| **Label Schema** | Mikels (8 classes) \+ 6 Descriptive Attributes |
| **Image Count** | 3.3 Million total (118,102 human-labeled) |
| **Label Type** | Multiclass (8 classes) \+ Continuous (Attributes) |
| **Granularity** | Image-level \+ Attribute tags |
| **Accessibility** | Public (Non-commercial) |
| **Image Domain** | Social Networks and Artistic Images |

The strength of EmoSet lies in its data balance. Each of the eight emotion categories is represented by 10,660 to 19,828 images, providing the statistical density required for robust centroid calculation.5 In embedding space, this balance prevents the "centroid drift" that occurs when one class is significantly over-represented, allowing for a fair comparison of inter-class distances.  
Researchers can utilize EmoSet to measure "cluster compactness" as a function of specific attributes. For example, one can test if "Fear" clusters are more compact in images with low brightness ($\<0.3$) versus high brightness, confirming whether the embedding space has learned a "universal" fear concept or if it is anchored to specific visual profiles.13 The dataset’s inclusion of artistic imagery alongside social media photos also permits experiments on "domain invariance" within the affective manifold.13

## **Normative Standards and Manifold Reliability: OASIS**

The Open Affective Standardized Image Set (OASIS) is distinct from the larger web-scraped datasets in its focus on "normative reliability".6 While it contains only 900 images, these images are matched for valence and arousal by a large participant pool ($N=822$), making the labels significantly less noisy than those found in crowd-sourced social media sets.6

### **Table 4: OASIS Technical Profile**

| Feature | Specification |
| :---- | :---- |
| **Publisher** | Kurdi, Lozano, & Banaji (2017) |
| **Task Type** | Normative Affective Rating |
| **Label Schema** | Continuous Valence and Arousal (1-7 scale) |
| **Image Count** | 900 |
| **Label Type** | Continuous |
| **Granularity** | Image-level |
| **Accessibility** | Public (Open access) |
| **Image Domain** | Humans, Animals, Objects, and Scenes |

OASIS is uniquely suited for testing the "circumplex model" of affect—the theory that emotions form a circular structure in the valence-arousal plane.6 Because the valence and arousal ratings cover much of the circumplex space, researchers can perform a "procrustes analysis" to see if the model's high-dimensional latent space can be linearly or non-linearly projected back onto this psychological ground truth.  
The primary weakness of OASIS for modern deep learning research is its small scale, which may lead to unstable geometry in models with extremely high-dimensional embeddings. However, as a "validation set" for the fundamental structure of the affective manifold, it is peerless.15 It also avoids the copyright restrictions associated with the older International Affective Picture System (IAPS), making it a superior choice for reproducible research.6

## **Contextual and Fine-Grained Affect: EMOTIC**

EMOTIC (Emotion Recognition in Context) provides a dataset of 23,571 images labeled with 26 emotion categories and the VAD continuous model.16 It focuses on images where emotions are inferred from the context of the scene rather than just facial expressions, often involving severe facial occlusion or subjects not facing the camera.16

### **Table 5: EMOTIC Technical Profile**

| Feature | Specification |
| :---- | :---- |
| **Publisher** | Kosti et al. (2017 / 2019 versions) |
| **Task Type** | Context-Aware Emotion Recognition |
| **Label Schema** | 26 Categories \+ VAD Continuous |
| **Image Count** | 23,571 (34,320 annotated subjects) |
| **Label Type** | Multilabel (Categories) \+ Continuous (VAD) |
| **Granularity** | Person-level within a scene |
| **Accessibility** | Public |
| **Image Domain** | Web-scraped and curated from MSCOCO and ADE20K |

EMOTIC is highly effective for "overlap analysis." With 26 categories, many of which are semantically similar (e.g., "Peace," "Happiness," "Pleasure"), it provides a testbed for measuring the "semantic resolution" of an embedding space.16 Researchers can investigate if these adjacent categories form overlapping "super-clusters" or if the model distinguishes them through subtle directional shifts in the latent space.  
The presence of "person-level" annotations within the context of MSCOCO and ADE20K images allows for a comparison between the embedding of the "affective actor" and the "contextual background".16 This is vital for understanding the "compositionality" of sentiment: does the total sentiment embedding $z\_{total}$ equal the sum of its parts $z\_{actor} \+ z\_{context}$?

## **Facial Manifolds and Scale: AffectNet**

For experiments specifically targeting the "facial affect manifold," AffectNet is the most extensive resource available.17 With over 1,000,000 facial images, it provides the statistical density required for analyzing the "fractal structure" or "local density" of emotion clusters.1

### **Table 6: AffectNet Technical Profile**

| Feature | Specification |
| :---- | :---- |
| **Publisher** | Mollahosseini et al. (2017 / 2019\) |
| **Task Type** | Facial Emotion Recognition |
| **Label Schema** | 8 categories \+ Valence/Arousal |
| **Image Count** | 1,000,000+ total (450,000+ manually annotated) |
| **Label Type** | Discrete and Continuous |
| **Granularity** | Face-level |
| **Accessibility** | Restricted (Requires license agreement) |
| **Image Domain** | "In-the-wild" internet search images |

AffectNet's primary geometric advantage is its scale, which enables the use of high-order dimensionality reduction techniques like UMAP or t-SNE without the risk of "empty space" artifacts.1 However, its domain is strictly limited to faces, meaning any geometric insights gained may not generalize to the "sentiment of objects" or "sentiment of scenes." Furthermore, the "in-the-wild" nature of the data introduces significant pose and lighting variance, which can act as a major confound if not properly neutralized through pre-processing or the use of pose-invariant feature extractors.18

## **Artistic Sentiment: EmoArt and ArtEmis**

The transition from realistic photography to artistic expression introduces significant challenges for embedding geometry, as the relationship between "visual cues" and "affective response" becomes more abstract. EmoArt (132,664 images) and ArtEmis (80,000 images) are the leading datasets in this domain.21

### **Table 7: EmoArt Technical Profile**

| Feature | Specification |
| :---- | :---- |
| **Publisher** | Zhang et al. (2024-2025) |
| **Task Type** | Affective Art Understanding / Generation |
| **Label Schema** | 12 categories \+ V / A \+ Style attributes |
| **Image Count** | 132,664 |
| **Label Type** | Discrete and Binary (V/A) |
| **Granularity** | Image-level |
| **Accessibility** | Public (CC BY-NC-ND 4.0) |
| **Image Domain** | Fine Art (Painting styles: Impressionism, etc.) |

EmoArt is particularly useful for studying "stylistic manifolds." Each image is annotated with five key visual attributes: brushwork, composition, color, line, and light.22 This allows researchers to perform a "sensitivity analysis" on the embedding space: how does the "line" attribute influence the position of an image within the "Sadness" cluster? In artistic data, the "sentiment" often resides in the "form" rather than the "content," and EmoArt’s labeling schema supports the isolation of these formal properties in high-dimensional space.21

## **Weakly Supervised and Multimodal Candidates: T4SA and B-T4SA**

The Twitter for Sentiment Analysis (T4SA) and its balanced version (B-T4SA) are massive social media datasets (1.4 million images) where labels are often automatically extracted from the associated text.23 While this introduces a high degree of label noise, it provides a unique opportunity for "cross-modality comparison."

### **Table 8: T4SA / B-T4SA Technical Profile**

| Feature | Specification |
| :---- | :---- |
| **Publisher** | Vadicamo et al. (2017) |
| **Task Type** | Multimodal Sentiment Analysis |
| **Label Schema** | Positive / Neutral / Negative |
| **Image Count** | 1,400,000+ images |
| **Label Type** | Discrete (3-class) |
| **Granularity** | Image-level |
| **Accessibility** | Public |
| **Image Domain** | Social media tweets (User-generated content) |

The research utility of T4SA for geometry lies in the "modality alignment" experiment.25 Researchers can investigate if the "Positive" image centroid and the "Positive" text centroid converge in a joint multimodal space like CLIP.27 The "Sentiment Discrepancy" labels in some sub-sets of B-T4SA are particularly interesting; they mark cases where the text and image convey different polarities, providing "adversarial points" for geometric analysis that can help define the robust boundaries of the sentiment manifold.24

## **Critical Discretization of Continuous Affective Data**

For datasets utilizing continuous Valence-Arousal (VA) or Valence-Arousal-Dominance (VAD) labels (OASIS, AffectNet, EMOTIC), the question of how to discretize these values for "cluster-centroid analysis" is non-trivial. A simple quadrant-based discretization is often employed:

1. **HVHA (High Valence, High Arousal):** $V \> 0.5, A \> 0.5$ (e.g., Excitement).  
2. **LVHA (Low Valence, High Arousal):** $V \< \-0.5, A \> 0.5$ (e.g., Anger, Fear).  
3. **LVLA (Low Valence, Low Arousal):** $V \< \-0.5, A \< \-0.5$ (e.g., Sadness, Boredom).  
4. **HVLA (High Valence, Low Arousal):** $V \> 0.5, A \< \-0.5$ (e.g., Contentment, Serenity).

However, this linear partitioning ignores the "parabolic" or "U-shaped" nature of the affective manifold frequently observed in human ratings.3 A more sophisticated geometric approach involves calculating the "Mahalanobis distance" from the normative centroids defined by OASIS to determine class membership.28 Discretizing VAD values allows researchers to utilize discrete metrics like the "Silhouette Coefficient" or "Davies-Bouldin Index" to evaluate the quality of the latent space representation.29

## **Methodological Framework for Geometric Analysis**

To conduct a rigorous geometric analysis of embeddings using these datasets, researchers must look beyond simple Euclidean distance. The following metrics are essential for a comprehensive evaluation:

### **1\. Class Compactness and Variance**

Compactness is quantified by the intra-class dispersion $S\_w$, which measures how tightly images of the same sentiment are grouped around their centroid $\\mu\_c$:

$$S\_w \= \\frac{1}{N\_c} \\sum\_{i=1}^{N\_c} ||z\_i \- \\mu\_c||^2$$  
A low $S\_w$ relative to the inter-class distance is indicative of a "well-resolved" sentiment space.28 Datasets like EmoSet and EmoVerse, with their large-scale human verification, provide the "gold standard" samples required to calculate a baseline $S\_w$ that is not inflated by label noise.

### **2\. Centroid Separation and Sentiment Gradients**

The distance between the centroids of different emotions (e.g., $d(\\mu\_{happy}, \\mu\_{sad})$) determines the "discriminative margin" of the model. In a continuous VAD space, one should expect a "Sentiment Gradient" where the distance between centroids correlates with the distance in VAD coordinates:

$$d(z\_1, z\_2) \\propto d(VAD\_1, VAD\_2)$$  
If a model places "Fear" and "Anger" close together in embedding space but far apart in VAD space, it suggests a geometric failure to capture "Arousal" or "Dominance" nuance.3

### **3\. Cluster Overlap and Manifold Ambiguity**

Overlap can be measured using the "Bhattacharyya distance" between the distributions of two sentiment classes. In social media datasets like FI-8 or T4SA, overlap is often a result of "semantic ambiguity".31 In a geometric experiment, identifying the "points of maximum overlap" can reveal which visual features are most prone to sentiment confusion (e.g., "high-energy" negative events being confused for "high-energy" positive events).32

## **Analysis of Confounding Variables in Affective Embedding Spaces**

A significant challenge in visual sentiment geometry is the "low-level shortcut." Neural networks frequently learn to cluster sentiments based on "low-level visual statistics" rather than "affective meaning".13

### **The Color-Brightness Confound**

There is a documented correlation between "Positive Sentiment" and "High Brightness/Saturation" and "Negative Sentiment" and "Darker/Greyer tones".13

* **EmoSet** is the best tool for auditing this confound because it provides explicit "brightness" and "colorfulness" labels.13  
* **EmotionROI** focuses on how these low-level features *drive* emotion, providing region-level masks where specific color/texture patterns evoke affect.34

### **The Domain-Content Confound**

Certain sentiment classes are "conceptually anchored" to specific domains. For example, "Awe" is often anchored to "Mountain/Nature" scenes, while "Excitement" is anchored to "Sport/People" scenes.10 If a model's "Awe" cluster is merely a "Mountain" cluster, the geometric representation has failed to capture the abstract affect. **EmoVerse** mitigates this by deconstructing the background from the subject, allowing for the calculation of the "Subject-Independent Affective Vector".7

## **Rejected and Poorly Suited Candidates**

Several datasets, while famous in the field of computer vision or psychology, are poorly suited for high-dimensional geometric analysis.

1. **IAPS (International Affective Picture System):** The most famous database in psychology, but it is effectively "deprecated" for computational research due to prohibitive copyright restrictions and dated image quality.6 Its small size also makes it prone to "overfitting" in deep embedding spaces.  
2. **OASIS (Medical Dataset):** Researchers must be careful not to confuse the **Open Affective Standardized Image Set** 6 with the **Open Access Series of Imaging Studies** (OASIS-1 through OASIS-4), which contains neuroimaging data for Alzheimer's research.36 The latter has no relevance to visual sentiment analysis of natural scenes.  
3. **Small Polarity Sets (Twitter I & II):** Containing only a few hundred or thousand images with binary labels, these datasets do not provide the "sample density" required for stable manifold analysis or centroid estimation in high dimensions.34  
4. **MVSA-Single:** A multimodal dataset that is often criticized for high label noise and "modality mismatch," where the text label does not correspond to the image content, leading to "diffuse" and uninterpretable clusters in joint embedding spaces.26

## **Integration of Multimodal and Textual Sentiment Geometry**

The final frontier of embedding-geometry analysis is the "Cross-Modal Alignment." This involves comparing the latent space of an image model with that of a text model.

### **1\. Comparative Manifold Analysis**

Using datasets like **T4SA** or **Artemis** (which contains 80,000 image-caption pairs), one can calculate the "Centroid Consistency" across modalities.23 For example, does the vector $v \= \\mu\_{happy} \- \\mu\_{sad}$ have the same direction and magnitude in the visual embedding space as it does in the text embedding space?

### **2\. Identifying the "Modality Gap"**

Research on "joint embedding" models like CLIP has shown that a "modality gap" often exists, where all image embeddings are clustered separately from all text embeddings.27 Affective datasets allow researchers to investigate if this gap is "concept-dependent." For instance, is the modality gap smaller for "concrete" sentiments (e.g., "Sadness" often depicted by crying faces) than for "abstract" sentiments (e.g., "Awe" depicted by vast landscapes)? **EmoVerse** is uniquely suited for this due to its "Background-Attribute-Subject" triplets, which can be mapped directly to fine-grained textual descriptions.8

## **Final Recommendation: The "Top 3" Selection**

Based on the objective of supporting "centroid, cluster, and overlap analysis" in high-dimensional embedding space, the following three datasets are recommended as the primary resources for any sophisticated geometric experiment.

### **Recommendation 1: EmoVerse**

**Why:** EmoVerse provides the most "interpretable" geometry currently available.7 Its multi-layered annotations (B-A-S triplets, grounding masks, dual CES/DES labels) allow for the disentanglement of sentiment from scene context—the holy grail of affective embedding research. It is the only dataset that truly supports "object-level" vs. "scene-level" geometric comparisons on a massive scale.8

### **Recommendation 2: EmoSet**

**Why:** EmoSet is the essential tool for "confound control".5 For experiments where the goal is to prove that a model has learned a "semantic sentiment space" rather than a "low-level visual space," EmoSet's attribute labels (brightness, color, scene) provide the necessary metadata for regression or stratified sampling.13 Its balance across eight categories ensures that cluster density is a reflection of semantic properties rather than sample-size bias.5

### **Recommendation 3: OASIS (Kurdi et al.)**

**Why:** OASIS serves as the "normative anchor".6 While its scale is small, its high label reliability and coverage of the V-A circumplex make it the perfect "ground truth" for manifold validation. Any geometric analysis on a large dataset like EmoSet or AffectNet should be cross-validated against OASIS to ensure the discovered high-dimensional structures correspond to the psychological reality of human affective perception.6

## **Conclusion: Toward a Geometrically Consistent Affective AI**

The transition from "accuracy-driven" to "geometry-driven" sentiment analysis represents a maturation of the field, moving from superficial labeling toward a foundational understanding of how affect is encoded in neural architectures. The datasets reviewed—ranging from the object-grounded precision of **EmoVerse** to the normative reliability of **OASIS**—provide the necessary scaffolding for this research. By focusing on metrics such as "centroid distance," "cluster compactness," and "manifold alignment," researchers can ensure that the next generation of affective models is not only more accurate but also more geometrically consistent with the complex structure of human emotion. The use of "dual-space" annotations and "attribute-aware" labeling is no longer a luxury but a requirement for any study that seeks to claim a robust and interpretable representation of sentiment in high-dimensional space.

#### **Works cited**

1. Latent space \- Wikipedia, accessed on April 17, 2026, [https://en.wikipedia.org/wiki/Latent\_space](https://en.wikipedia.org/wiki/Latent_space)  
2. Visualizing LLM Latent Space Geometry Through Dimensionality Reduction \- arXiv, accessed on April 17, 2026, [https://arxiv.org/html/2511.21594v2](https://arxiv.org/html/2511.21594v2)  
3. A-Situ: a computational framework for affective labeling from psychological behaviors in real-life situations \- PMC, accessed on April 17, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7522975/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7522975/)  
4. Datasets for Valence and Arousal Inference: A Survey \- arXiv, accessed on April 17, 2026, [https://arxiv.org/html/2510.00738v1](https://arxiv.org/html/2510.00738v1)  
5. EmoSet: A Large-scale Visual Emotion Dataset with Rich Attributes \- CVF Open Access, accessed on April 17, 2026, [https://openaccess.thecvf.com/content/ICCV2023/papers/Yang\_EmoSet\_A\_Large-scale\_Visual\_Emotion\_Dataset\_with\_Rich\_Attributes\_ICCV\_2023\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_EmoSet_A_Large-scale_Visual_Emotion_Dataset_with_Rich_Attributes_ICCV_2023_paper.pdf)  
6. (PDF) Introducing the Open Affective Standardized Image Set (OASIS) \- ResearchGate, accessed on April 17, 2026, [https://www.researchgate.net/publication/295845334\_Introducing\_the\_Open\_Affective\_Standardized\_Image\_Set\_OASIS](https://www.researchgate.net/publication/295845334_Introducing_the_Open_Affective_Standardized_Image_Set_OASIS)  
7. EmoSet: A Large-scale Visual Emotion Dataset with Rich Attributes \- ResearchGate, accessed on April 17, 2026, [https://www.researchgate.net/publication/377420602\_EmoSet\_A\_Large-scale\_Visual\_Emotion\_Dataset\_with\_Rich\_Attributes](https://www.researchgate.net/publication/377420602_EmoSet_A_Large-scale_Visual_Emotion_Dataset_with_Rich_Attributes)  
8. EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis \- arXiv, accessed on April 17, 2026, [https://arxiv.org/html/2511.12554v1](https://arxiv.org/html/2511.12554v1)  
9. What's the Point: Semantic Segmentation with Point Supervision \- ResearchGate, accessed on April 17, 2026, [https://www.researchgate.net/publication/308188996\_What's\_the\_Point\_Semantic\_Segmentation\_with\_Point\_Supervision](https://www.researchgate.net/publication/308188996_What's_the_Point_Semantic_Segmentation_with_Point_Supervision)  
10. UniEmoX: Cross-Modal Semantic-Guided Large-Scale Pretraining for Universal Scene Emotion Perception | Request PDF \- ResearchGate, accessed on April 17, 2026, [https://www.researchgate.net/publication/393725069\_UniEmoX\_Cross-modal\_Semantic-Guided\_Large-Scale\_Pretraining\_for\_Universal\_Scene\_Emotion\_Perception](https://www.researchgate.net/publication/393725069_UniEmoX_Cross-modal_Semantic-Guided_Large-Scale_Pretraining_for_Universal_Scene_Emotion_Perception)  
11. Multi-Output Learning Based on Multimodal GCN and Co-Attention for Image Aesthetics and Emotion Analysis \- MDPI, accessed on April 17, 2026, [https://www.mdpi.com/2227-7390/9/12/1437](https://www.mdpi.com/2227-7390/9/12/1437)  
12. EmoVid: A Multimodal Emotion Video Dataset for Emotion-Centric Video Understanding and Generation \- arXiv, accessed on April 17, 2026, [https://arxiv.org/html/2511.11002v1](https://arxiv.org/html/2511.11002v1)  
13. EmoSet-118K \- VCC, accessed on April 17, 2026, [https://vcc.tech/EmoSet](https://vcc.tech/EmoSet)  
14. (PDF) Introducing the Open Affective Standardized Image Set (OASIS). (2017) | Benedek Kurdi | 524 Citations \- SciSpace, accessed on April 17, 2026, [https://scispace.com/papers/introducing-the-open-affective-standardized-image-set-oasis-5ciekf4d1c](https://scispace.com/papers/introducing-the-open-affective-standardized-image-set-oasis-5ciekf4d1c)  
15. Replacing the Classics? A Comparison of the ERPs Evoked by IAPS and OASIS Images During Emotional Processing \- PMC, accessed on April 17, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12819365/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12819365/)  
16. A Survey on Datasets for Emotion Recognition from Vision ... \- MDPI, accessed on April 17, 2026, [https://www.mdpi.com/2076-3417/13/9/5697](https://www.mdpi.com/2076-3417/13/9/5697)  
17. AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild, accessed on April 17, 2026, [https://www.computer.org/csdl/journal/ta/2019/01/08013713/13rRUwjXZQG](https://www.computer.org/csdl/journal/ta/2019/01/08013713/13rRUwjXZQG)  
18. Top 6 Datasets For Emotion Detection \- Analytics Vidhya, accessed on April 17, 2026, [https://www.analyticsvidhya.com/blog/2024/04/top-datasets-for-emotion-detection/](https://www.analyticsvidhya.com/blog/2024/04/top-datasets-for-emotion-detection/)  
19. Emotion Recognition Datasets. We express our emotions via facial… | by Kayathiri Mahendrakumaran | Analytics Vidhya | Medium, accessed on April 17, 2026, [https://medium.com/analytics-vidhya/emotion-recognition-datasets-8a397590c7d1](https://medium.com/analytics-vidhya/emotion-recognition-datasets-8a397590c7d1)  
20. Exploring Facial Emotion Recognition Datasets \- MorphCast, accessed on April 17, 2026, [https://www.morphcast.com/blog/facial-emotion-recognition-datasets/](https://www.morphcast.com/blog/facial-emotion-recognition-datasets/)  
21. AffectiveArt Challenge 2026: Fine-Grained Emotion Understanding and Generation in Artistic Images \- OpenReview, accessed on April 17, 2026, [https://openreview.net/pdf/9110652108c520870f912cce5f5efeb745f549cf.pdf](https://openreview.net/pdf/9110652108c520870f912cce5f5efeb745f549cf.pdf)  
22. (PDF) EmoArt: A Multidimensional Dataset for Emotion-Aware Artistic Generation, accessed on April 17, 2026, [https://www.researchgate.net/publication/392406334\_EmoArt\_A\_Multidimensional\_Dataset\_for\_Emotion-Aware\_Artistic\_Generation](https://www.researchgate.net/publication/392406334_EmoArt_A_Multidimensional_Dataset_for_Emotion-Aware_Artistic_Generation)  
23. Cross-Media Learning for Image Sentiment Analysis in the Wild \- CVF Open Access, accessed on April 17, 2026, [https://openaccess.thecvf.com/content\_ICCV\_2017\_workshops/papers/w5/Vadicamo\_Cross-Media\_Learning\_for\_ICCV\_2017\_paper.pdf](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w5/Vadicamo_Cross-Media_Learning_for_ICCV_2017_paper.pdf)  
24. Multimodal Sentiment Classifier Framework for Different Scene Contexts \- MDPI, accessed on April 17, 2026, [https://www.mdpi.com/2076-3417/14/16/7065](https://www.mdpi.com/2076-3417/14/16/7065)  
25. Multimodal Social Media Sentiment Analysis \- Stanford University, accessed on April 17, 2026, [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/MubarakAliSeyedIbrahimPratyushMuthukumar.pdf](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/MubarakAliSeyedIbrahimPratyushMuthukumar.pdf)  
26. Multimodal Sentiment Analysis Representations Learning via Contrastive Learning with Condense Attention Fusion \- PMC, accessed on April 17, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10007095/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10007095/)  
27. Joint Text-and-Image Clustering for Social Science Research \- Han Zhang, accessed on April 17, 2026, [https://hanzhang.xyz/files/Zhang%20and%20Leung%20-%202025%20-%20Joint%20Text-and-Image%20Clustering%20for%20Social%20Science%20Research%20accepted%20version.pdf](https://hanzhang.xyz/files/Zhang%20and%20Leung%20-%202025%20-%20Joint%20Text-and-Image%20Clustering%20for%20Social%20Science%20Research%20accepted%20version.pdf)  
28. Cluster validity indices for automatic clustering: A comprehensive review \- ResearchGate, accessed on April 17, 2026, [https://www.researchgate.net/publication/388026694\_Cluster\_validity\_indices\_for\_automatic\_clustering\_A\_comprehensive\_review](https://www.researchgate.net/publication/388026694_Cluster_validity_indices_for_automatic_clustering_A_comprehensive_review)  
29. Transforming Complex Problems Into K-Means Solutions | Request PDF \- ResearchGate, accessed on April 17, 2026, [https://www.researchgate.net/publication/367235224\_Transforming\_Complex\_Problems\_into\_K-means\_Solutions](https://www.researchgate.net/publication/367235224_Transforming_Complex_Problems_into_K-means_Solutions)  
30. Efficient Data Reduction Through Maximum-Separation Vector Selection and Centroid Embedding Representation \- MDPI, accessed on April 17, 2026, [https://www.mdpi.com/2079-9292/14/10/1919](https://www.mdpi.com/2079-9292/14/10/1919)  
31. Seven commonly used visual sentiment analysis datasets. Most datasets... \- ResearchGate, accessed on April 17, 2026, [https://www.researchgate.net/figure/Seven-commonly-used-visual-sentiment-analysis-datasets-Most-datasets-are-from-social\_tbl1\_350180656](https://www.researchgate.net/figure/Seven-commonly-used-visual-sentiment-analysis-datasets-Most-datasets-are-from-social_tbl1_350180656)  
32. An Overview on Image Sentiment Analysis: Methods, Datasets and Current Challenges \- iris@unict.it, accessed on April 17, 2026, [https://www.iris.unict.it/retrieve/dfe4d22b-4ebf-bb0a-e053-d805fe0a78d9/An%20overview%20on%20image%20sentiment%20analysis.pdf](https://www.iris.unict.it/retrieve/dfe4d22b-4ebf-bb0a-e053-d805fe0a78d9/An%20overview%20on%20image%20sentiment%20analysis.pdf)  
33. SafeConstellations: Mitigating Over-Refusals in LLMs Through Task-Aware Representation Steering \- arXiv, accessed on April 17, 2026, [https://arxiv.org/html/2508.11290v3](https://arxiv.org/html/2508.11290v3)  
34. Object-scene semantics correlation analysis for image ... \- Frontiers, accessed on April 17, 2026, [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1657562/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1657562/full)  
35. EmoScene: A Dual-space Dataset for Controllable Affective Image Generation \- arXiv, accessed on April 17, 2026, [https://arxiv.org/html/2604.00933v1](https://arxiv.org/html/2604.00933v1)  
36. Request Access to All OASIS Datasets, accessed on April 17, 2026, [https://sites.wustl.edu/oasisbrains/home/access/](https://sites.wustl.edu/oasisbrains/home/access/)  
37. OASIS Brain Dataset: MRI for Alzheimer's research \- Innovatiana, accessed on April 17, 2026, [https://www.innovatiana.com/en/datasets/oasis-open-access-series-of-imaging-studies](https://www.innovatiana.com/en/datasets/oasis-open-access-series-of-imaging-studies)  
38. Open Access Series of Imaging Studies (OASIS), accessed on April 17, 2026, [https://sites.wustl.edu/oasisbrains/](https://sites.wustl.edu/oasisbrains/)  
39. ArtEmis: Affective Language for Visual Art | Request PDF \- ResearchGate, accessed on April 17, 2026, [https://www.researchgate.net/publication/355882260\_ArtEmis\_Affective\_Language\_for\_Visual\_Art](https://www.researchgate.net/publication/355882260_ArtEmis_Affective_Language_for_Visual_Art)
