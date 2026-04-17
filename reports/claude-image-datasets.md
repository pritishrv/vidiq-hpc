https://claude.ai/chat/6a44dac1-c49e-4383-8388-edb11e7a3521

Now I have comprehensive data to write the full analysis. Let me compose the report.---

## Executive Summary

The field has a genuine scarcity problem: most commonly cited visual sentiment datasets are either too small for stable geometry analysis, too noisy for meaningful cluster interpretation, or confound sentiment label with domain/content in ways that will seriously distort embedding geometry. Only a handful of datasets are worth serious consideration. The best general-purpose choice for embedding-geometry research is **EmoSet-118K**; the best choice if you need person-level, context-grounded labels is **EMOTIC**; the best choice for cross-modal text/image comparison is **ArtEmis**.

---

## Ranked Dataset Table

See the interactive table and scatter chart above. Scores reflect: label quality/inter-annotator reliability, dataset scale sufficient for stable centroids, class balance, domain diversity, and geometry-relevant features (attribute richness, continuous dimensions, region annotations).

---

## Detailed Notes on Strong Candidates

### 1. EmoSet-118K — Rank 1 (Best overall)

EmoSet, presented at ICCV 2023, comprises 3.3 million images in total, with 118,102 carefully labeled by human annotators — roughly five times larger than its nearest predecessor. Labels follow the Mikels 8-class emotion model (amusement, anger, awe, contentment, disgust, excitement, fear, sadness), and each image is additionally annotated with six emotion attributes: brightness, colorfulness, scene type, object class, facial expression, and human action.

For geometry analysis, EmoSet-118K is exceptionally well suited. The eight emotion classes map naturally to four positive / four negative, enabling both coarse polarity and fine-grained cluster analysis. The attribute annotations (brightness, colorfulness, object class) let you probe whether cluster separation in embedding space is genuine sentiment structure or visual shortcut. The dataset is well balanced across categories, each represented by 10,660 to 19,828 images, which is critical for avoiding degenerate centroids. The mix of social-media and artistic images adds domain diversity that reduces trivial visual confounds.

Weaknesses: emotion labels were collected by querying emotion keywords, introducing some selection bias. The 3.3M weakly labeled pool is useful for semi-supervised experiments but should not be used for geometry benchmarking without careful filtering. Access is free for non-commercial use via the project page.

Suitability ratings: clean polarity ✓✓, fine affect ✓✓, object-level ✓ (partial via object class attribute), cross-modal ✓✓ (attribute labels support text alignment).

---

### 2. FI (Flickr and Instagram) — Rank 2

The FI dataset is described as the first large-scale well-labeled visual emotion dataset, built from images scraped from Flickr and Instagram and annotated with Mikels 8-class categories, reaching roughly 90,000 images and substantially exceeding the size of prior datasets. It is the most-cited benchmark in the visual sentiment literature and has been used in virtually every state-of-the-art comparison paper.

For geometry analysis, FI offers the best combination of scale and established label schema among purely image-level social-photo datasets. The Mikels wheel structure provides a natural theoretical basis for expecting geometric separation (positive cluster vs. negative cluster) while still permitting finer 8-way analysis.

Weaknesses: images sourced by hashtag/search queries mean that image content is partly confounded with emotion category. A large fraction of FI images are stock-style photographs, and researchers have noted that models trained on FI tend to exploit low-level color and scene cues rather than genuine affective content. This is actually useful for geometry experiments — you can measure how much cluster separation is artifact — but be aware it limits interpretive claims. Access requires a request to the original Cornell team.

Suitability: clean polarity ✓✓, fine affect ✓✓, object-level ✗, cross-modal ✓.

---

### 3. EMOTIC — Rank 3 (Best for person/context-level and VAD experiments)

EMOTIC annotates images of people in diverse natural environments with two representation types: 26 discrete emotion categories and continuous valence, arousal, and dominance dimensions. The dataset contains approximately 24,600 images with bounding boxes marking each annotated person, providing both person-level and full-scene context.

EMOTIC is the only strong candidate with region-level (person bounding box) sentiment labels, making it uniquely valuable for experiments comparing whole-image embedding geometry against cropped-region geometry. The dual representation — 26 discrete categories plus continuous VAD — gives several discretization options and supports analysis of how well ordinal label structure is recovered in embedding space. The VAD dimensions are particularly useful: you can ask whether valence and arousal axes manifest as separable geometric dimensions in a pretrained embedding, which is a clean theoretical question.

Weaknesses: the 26 categories are person-apparent emotions ("excitement," "disconnection," "fatigue") rather than image-evoked sentiment, which is a different construct than what FI or EmoSet measure. Images are constrained to scenes with identifiable people, which limits domain diversity. The dataset is open-sourced and available on GitHub.

Suitability: clean polarity ✓, fine affect ✓✓, object-level ✓✓ (unique), cross-modal ✓✓.

---

### 4. ArtEmis — Rank 4 (Best for cross-modal geometry)

ArtEmis contains 455,000 emotion attributions and explanations from human annotators covering 80,000 artworks from WikiArt. Annotators chose from the 8 Mikels emotion categories and provided grounded verbal explanations for their choice.

The paired image-and-text structure makes ArtEmis exceptional for cross-modal geometry experiments: you can compute embedding centroids in both visual and text spaces and directly compare cluster geometry across modalities. ArtEmis has significantly more abstract, subjective, and sentimental language than caption datasets like COCO, meaning that text-space sentiment geometry from ArtEmis explanations will be richer and more interpretable than from neutral captioning data.

Weaknesses: the artistic image domain (paintings, drawings) is very different from natural photos, so geometry findings may not transfer. The "something else" category (used ~12% of the time) is a catch-all that pollutes label space. Annotator disagreement is higher for art than for unambiguous photographic scenes. The dataset is open access.

Suitability: clean polarity ✓, fine affect ✓, object-level ✗, cross-modal ✓✓✓ (uniquely strong).

---

### 5. T4SA — Rank 5 (Scale benchmark, weak labels)

T4SA (Twitter for Sentiment Analysis) consists of approximately 3.4 million tweets with around 4 million images, labeled for sentiment polarity (negative, neutral, positive) using a tandem LSTM-SVM architecture trained on text sentiment. A balanced subset, B-T4SA, contains 470,586 images.

At nearly half a million images, T4SA is the largest labeled set available. However, sentiment labels are derived from associated tweet text, not from human image annotation — meaning the image may not visually express the sentiment the text conveys. This introduces structural label noise that is specifically damaging for embedding-geometry experiments, where you want labels that actually correspond to image content. Useful for large-scale polarity geometry as a stress test, but centroid distances will be compressed by label noise and cross-image domain variance.

Suitability: clean polarity ✓ (with caveats), fine affect ✗, object-level ✗, cross-modal ✓ (labels are text-derived, making it an interesting negative control).

---

### 6. IAPS — Rank 6 (Highest quality labels, prohibitively small)

The International Affective Picture System contains approximately 1,000 color photographs with normative ratings along valence, arousal, and dominance, collected from standardized laboratory samples. Access is restricted to members of recognized academic institutions.

Each image in IAPS is accompanied by norms along three dimensions: arousal (physiological activation), valence (pleasantness), and dominance (degree of control), with extensive cross-cultural validation.

The continuous VAD annotations are the gold standard for affect research, and the C-shaped distribution in valence-arousal space is well documented. For geometry analysis, this means you can probe whether high-arousal positive and high-arousal negative form geometrically distinct clusters from low-arousal neutral, which is a theoretically motivated hypothesis. The problem is scale: ~1,000 images cannot support stable centroid estimates or robust overlap measurement, especially once discretized into sub-categories. IAPS is best used as a validation set alongside a larger dataset, not as a primary geometry corpus.

---

### 7. OASIS — Rank 7 (IAPS alternative, public domain)

The Open Affective Standardized Image Set (OASIS) contains 900 images depicting humans, animals, objects, and scenes, with normative valence and arousal ratings collected online. It was explicitly designed as a public-domain alternative to IAPS.

Same virtues and limitations as IAPS: excellent label quality, continuous VA space, but too small for robust geometry estimation. OASIS is fully open (public domain images, published norms), which makes it immediately accessible and suitable for pilot geometry experiments before scaling up to FI or EmoSet.

---

### 8. Emotion6 — Rank 8

Emotion6 contains 1,980 images collected from Flickr, labeled across six categories: joy, surprise, anger, disgust, fear, and sadness. Each image was annotated by 15 annotators, and continuous valence-arousal scores are also available.

The 15-annotator setup gives inter-annotator agreement statistics that are useful for understanding label uncertainty as a geometric property. At 1,980 images the dataset is marginal for geometry analysis — centroid estimates from ~330 images per class will have meaningful variance. Best used as a comparison point or to validate findings on FI.

---

## Rejected / Weak Candidates

**VSO / SentiBank**: VSO contains 3,244 adjective-noun pairs with associated Flickr images; SentiBank provides 1,200 trained ANP detectors. The ANP-based binary labels are derived from linguistic polarity of the adjective, not direct human image annotation. Label noise is high (a "beautiful sunset" image might encode "beautiful" as positive while the actual sunset composition is neutral). Useful for mid-level feature experiments but not for clean geometry benchmarking.

**ArtPhoto / Abstract Paintings**: Each contains fewer than 1,000 images, making them statistically inadequate for centroid or cluster analysis. The single-annotator (artist-determined) ground truth in ArtPhoto is especially problematic for inter-class boundary analysis.

**Twitter I / II (You et al.)**: Twitter I has 1,269 images (with subsets "3-agree," "4-agree," "5-agree" containing 1,269, 1,116, and 882 images respectively). Consistently cited as a benchmark but far too small for geometry experiments.

**WikiArt Emotions (Mohammad & Kiritchenko)**: Approximately 4,000 artworks annotated for 20 emotion categories. Interesting label richness but domain-constrained (Western art) and too small for stable geometry.

**FER2013 / AffectNet (facial expression)**: These are facial expression datasets, not image sentiment datasets. Sentiment is attributed to depicted faces; they do not support the kind of scene-level or object-level sentiment geometry that is the goal of this research. Their geometry would trivially collapse to the geometry of facial action units.

---

## Final Recommendation

**Top 3 datasets for embedding-geometry experiments:**

1. **EmoSet-118K** as your primary corpus. It has the right combination of scale (statistically robust centroids), balanced class distribution, theoretically grounded labels (Mikels), and rich attributes that allow you to decompose what drives cluster separation. The 8-class structure collapses naturally to binary polarity when needed but preserves fine structure for detailed analysis.

2. **EMOTIC** as the person-level and VAD supplement. No other dataset offers bounding-box-level sentiment labels, which uniquely enables comparing whole-image embedding geometry against region-cropped embedding geometry — a core question in your research agenda. The 26 categories plus continuous VAD also let you probe dimensionality of the affective space directly in embedding geometry.

3. **ArtEmis** for cross-modal comparison. The paired image + natural-language-explanation structure is irreplaceable for directly comparing visual embedding geometry with text embedding geometry using the same underlying sentiment labels. This is the only dataset where you can ask: does the centroid of "awe" images in CLIP visual space correspond geometrically to the centroid of "awe" explanation texts in CLIP text space?

The strategic play is to run geometry experiments on EmoSet-118K first (scale, balance, attribute decomposition), replicate polarity findings on FI as a second social-photo source, test person/region geometry on EMOTIC, and run cross-modal comparisons on ArtEmis. OASIS is worth using for pilot calibration given its zero-friction access and high label quality, even though it is too small for primary analysis.
