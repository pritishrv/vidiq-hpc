https://chatgpt.com/c/69e28efb-2824-83eb-8ff8-c90299ea9b3b

# Executive Summary

We identified a range of **image sentiment/affect datasets** from the literature and repositories. Top candidates include **EmoSet-118K** (Yang *et al.*, ICCV’23)【78†L96-L104】【78†L115-L120】, **You’s 8‑emotion dataset (FI)**【26†L255-L258】【77†L7-L11】, **FindingEmo**【30†L233-L240】, **MVSO/MVSO‑sentiment**【74†L28-L36】【74†L68-L76】, **VSO/SentiBank**【79†L43-L46】, and **EMOTIC**【32†L16-L20】. These cover a spectrum of label types (discrete emotions, continuous valence/arousal, adjective-noun concepts), sizes (from ~10³ up to millions), and annotation styles. We evaluated them for embedding‑geometry analysis: cluster compactness, centroid separations, and overlap among sentiment classes, as well as applicability to sub-image (object/person) sentiment and comparison with text. Datasets were assessed for label clarity, balance, domain bias, license, and ease of access. 

In summary:

- **EmoSet-118K (Yang et al. ICCV 2023)** is a newly released, balanced *8‑class emotion* dataset (118K images) with detailed annotations【78†L96-L100】【78†L115-L120】. Its size and balance make it ideal for cluster/centroid analysis.  
- **You et al.’s 8‑emotion dataset (2016)** has 23K images labeled with 8 Mikels emotions【26†L255-L258】【77†L7-L11】. This well‑known dataset is large enough for stable geometry but is somewhat imbalanced.  
- **FindingEmo (Mertens et al. 2024)** provides 25.9K *scene-level* images annotated with *Valence/Arousal* and Plutchik emotion labels【30†L233-L240】. It offers both continuous and fine-grained categorical labels, useful for nuanced geometry analyses.  
- **MVSO (Jou et al. 2015)** comprises ~7.4M Flickr images tagged with sentiment **ANP** concepts【74†L28-L36】; a subset (11.7K images) has human‑annotated sentiment scores【74†L66-L75】. Its scale and multilingual ANPs are valuable but labels are indirect and noisy.  
- **VSO/SentiBank (Borth et al. 2013)** is a smaller Flickr corpus (≈90K CC images across 1,553 ANPs)【79†L43-L46】. It’s easy to access (CC‑licensed) but ANP labels mix sentiment with objects/attributes, complicating pure sentiment geometry.  
- **EMOTIC (Kosti et al. 2017, 2019)** has 23.6K images with *person-level* annotations: 26 emotion categories + continuous Valence/Arousal/Dominance【32†L16-L20】. It’s useful for object‑level affect analysis but images often contain multiple people (multi-label).  
- **MVSA (Niu et al. 2016)** offers ~5K Twitter image-text pairs labeled (positive/neutral/negative)【68†L1-L4】. It’s explicitly multi-modal and small, so clusters may be noisy, but allows image-vs-text sentiment comparisons.  

Many older datasets (IAPS, GAPED, ArtPhoto, Emotion6, Twitter I/II, etc.) are **small (hundreds to low thousands of images)** and imbalanced, limiting reliable geometry analysis; we mention them as *weak candidates*. We also note licenses: many large datasets are CC or free for research【32†L16-L20】【79†L43-L46】, but some multi-view sets (MVSA) have more restrictive terms. 

We rank datasets by usefulness to sentiment-embedding geometry. The **top 3 recommended datasets** are **EmoSet-118K**, **You et al.’s 8‑emotion (FI)**, and **FindingEmo**. These combine substantial size, clear affect labels, and diversity. They offer both *polar sentiment* (positive/negative) and *fine-grained emotions*, and their well‑defined label spaces support centroid/overlap analyses (unlike very noisy large sets or very tiny ones).

# Ranked Dataset Table

| Rank | Dataset (Source)                      | Size       | Labels (type)                     | Granularity     | Domain            | Access/License | Strengths for Geometry                     | Weaknesses                            |
|------|---------------------------------------|------------|-----------------------------------|-----------------|-------------------|----------------|-------------------------------------------|---------------------------------------|
| 1    | **EmoSet-118K** (Yang *et al.*, ICCV’23)【78†L96-L100】【78†L115-L120】 | 118,102    | 8 emotion classes (Mikels model); **balanced** categorical (each ~10–19K images)【78†L115-L120】 | Image-level     | Diverse (social & artistic) | Public (CVF release) | Large and balanced 8-class dataset; uniform distribution of positives (4 classes) vs negatives (4 classes); suited for clear cluster/centroid analysis. Rich “attributes” available for interpretability. | Must download images via URLs; no object-level labels. |
| 2    | **You *et al.* (2016) “FI”** (AAAI)【26†L255-L258】【77†L7-L11】 | 23,308     | 8 emotion categories (Mikels model); multi‑class (image tag filtered + AMT consensus)【26†L255-L258】 | Image-level     | Social (Flickr/Instagram) | Public on request (research use) | High-quality labels (5 annotators per image) in 8 classes; large enough for clustering; contains positive vs negative classes. | Imbalanced (some classes ~1K images); domain biases (query-based search).  |
| 3    | **FindingEmo** (Mertens *et al.*, 2024)【30†L233-L240】      | 25,869     | Plutchik-based emotions (8 “leaves” in 4 quadrants), **continuous** Valence (−3 to +3) & Arousal (0–6)【30†L239-L240】【28†L19-L28】 | Image-level     | Complex social scenes     | Public (image URLs provided) | Multi-dimensional labels: both fine emotions and continuous valence/arousal. Largest dataset focusing on scene-level, multi-person emotions. Good for analyzing clustering by valence or by emotion category. | Many images are multi-label; some labels ambiguous; requirement to fetch images from web. |
| 4    | **MVSO (English)** (Jou *et al.*, ACM MM’15)【74†L28-L36】【74†L66-L73】 | ~7,400,000 (gathered); **11,733** annotated subset | 3,911 ANP concepts (adjective–noun pairs)【74†L28-L36】; *sentiment score* (−6 to +6 sum of 3 annotators) for 11,733 sampled images【74†L66-L74】 | Image-level     | Social (Flickr, 12 languages) | Public (downloadable image URLs) | Extremely large collection of sentiment-related concepts; human sentiment ratings (−2..+2) exist for a subset, allowing polarity analysis. Multi-language ANPs capture diverse affective cues. | ANP labels mix sentiment and objects (no simple polarity label). Only a small subset has human annotations. Metadata required for higher consistency【74†L35-L43】.  |
| 5    | **VSO / SentiBank** (Borth *et al.*, ACM MM’13)【79†L43-L46】   | ~90,000 (CC images) | 1,553 ANP classes【79†L43-L46】 | Image-level     | Social (Flickr CC)       | Public (dataset & code)【79†L43-L46】 | Well-curated CC-licensed images and ANP labels; covers both object and affect concepts; easy to access.  | ANP labels are many (1,553) – too fine-grained for sentiment polarity. ANPs are partly aesthetic, partly affective, so clusters may not cleanly map to positive/negative. |
| 6    | **EMOTIC** (Kosti *et al.*, PAMI’19)【32†L16-L20】     | 23,571     | 26 emotion categories + continuous Valence/Arousal/Dominance | Person-level (regions) | Social (COCO/ADE + web) | Public (research use)【33†L13-L21】 | Provides *object-level* (person-centric) affect labels with context. Useful for embedding geometry of individual emotional expressions. Contains both discrete categories and continuous scores. | Multi-person scenes – image has multiple labels (hard to assign global sentiment). License: research-only; some images overlap with large corpora (COCO). |
| 7    | **MVSA (tweets)** (Niu *et al.*, MMM’16)【68†L1-L4】   | 5,129 (single-label); 19,600 (multi-label) | 3 sentiment classes (positive/neutral/negative)【68†L1-L4】 | Image/text-level | Social (Twitter)        | Public (download via links) | Multi-modal (image+text); direct sentiment labels allow comparison between image and text embeddings. Small size manageable. | Very small (only ~5K pairs); labels coarse (3 classes); potential domain bias (Twitter memes, ads). |
| 8    | **Others (small)** – IAPS, GAPED, ArtPhoto, Emotion6, etc. | 200–3,000 each | Various (e.g. IAPS: valence/arousal ratings; Emotion6: 6 emotions) | Image-level     | Varies (stock photos, art) | Public (some) | Historically important, well-annotated by psychology community. | **Very small**, imbalanced, often subjective labels. Poor for statistical geometry or clustering. Eg. IAPS has only 395 images【78†L159-L164】, GAPED 730, ArtPhoto 806. These are inadequate for robust geometry analyses. |

# Detailed Notes on Top Candidate Datasets

- **EmoSet-118K**【78†L96-L100】【78†L115-L120】 (Yang *et al.*, ICCV’23): A new, large visual emotion dataset. Images (3.3M candidates, 118K annotated) were collected via 810 emotion-related keywords (Mikels eight emotions) and human‑annotated by trained crowdworkers. There are 8 discrete emotion classes (Amusement, Awe, Contentment, Excitement – considered positive; Anger, Disgust, Fear, Sadness – negative), **balanced** (≈10–20K images each)【78†L115-L120】. Labels are **image-level, multiclass**. The domain is mixed “social and artistic” images; most likely Flickr/Instagram content plus some creative sources. The dataset is public (download via URLs) for non-commercial research, and is **the first to combine large scale, balanced classes, and descriptive visual “attributes”** (e.g. brightness, colorfulness, objects)【78†L115-L120】. 

  *Strengths:* Balanced class sizes yield stable cluster centroids. The clear positive/negative dichotomy among the 8 classes allows clean polarity analysis. The large size (118K) and diversity mean geometry (cluster compactness, centroid distances) will be statistically robust. The rich annotations (including object and scene labels) aid interpreting cluster structure.  
  *Weaknesses:* Only image-level labels (no object/person breakdown). Images are retrieved via queries, so some thematic bias may exist. Access requires fetching via URLs (some images may drop). Not explicitly designed for sentiment polarity (focus is “emotion”), but positive/negative mapping is straightforward (four positive vs four negative emotions).  

  *Suitability:* Excellent for **clean sentiment polarity** (group emotions into positive/negative) and **fine-grained affect** (8 classes). Not for object-level sentiment (whole-scene labels only). Multi-label cases are rare (annotators largely agreed on one emotion). No direct text component for cross-modality, though it could be paired with captions or tags externally. Citation: human-labeled emotion categories and dataset size【78†L96-L100】【78†L115-L120】.

- **You *et al.* (2016) “FI” dataset**【26†L255-L258】【77†L7-L11】: This dataset contains 23,308 Flickr/Instagram images labeled with 8 emotion categories (Mikels model)【26†L255-L258】. Initially 90K “noisy” images were collected via keyword queries on 8 emotion terms, then 23,308 images were retained after AMT verification (each image labeled by 5 workers, accepted if ≥3 agreed)【26†L255-L258】. Classes are discrete single‑labels (Amusement, Anger, Awe, Contentment, Disgust, Excitement, Fear, Sadness). 

  *Task:* Multi-class emotion classification (8 categories).  
  *Label schema:* Categorical, single-label.  
  *Images:* 23,308 (with at least 1,000 images per category)【26†L255-L258】.  
  *Granularity:* Image-level.  
  *Public Access:* Yes (authors have made it available to researchers; uses CC-licensed images).  
  *Domain:* Social photos (varied scenes relevant to each emotion query).  
  *Strengths:* One of the largest human‑labeled emotion datasets at publication, with **well-verified labels**. Suitable for studying geometry of positive vs negative classes (4 vs 4). Data are concrete “emoji-like” emotions, easily interpreted.  
  *Weaknesses:* Classes are somewhat imbalanced (e.g. Fear and Anger have fewer images)【26†L255-L258】, so cluster sizes vary. The dataset was collected via keyword search, so classes may carry query bias (e.g. “sadness” images may share content). It’s not a “sentiment polarity” dataset per se, but the label set splits into positive (Amusement/Awe/Contentment/Excitement) vs negative (Anger/Disgust/Fear/Sadness).  
  *Suitability:* Good for both **coarse polarity** (grouping positive vs negative classes) and **fine affect** (8 distinct emotions). All labels are image-level. No direct text annotations, but one could attach the original query words if needed (for cross-modality). Citations: dataset size and label schema【26†L255-L258】【77†L7-L11】.

- **FindingEmo**【30†L233-L240】 (Mertens *et al.*, 2024): A **multi-dimensional emotion** dataset with 25,869 images of *social scenes* (often multiple people). Each image has (a) **continuous Valence (negative→positive)** on a 7-point scale and **Arousal** (calm→active), and (b) an emotion label from **Plutchik’s wheel** (24 fine-grained emotions, organized into 8 “leaves”)【28†L19-L28】【30†L231-L240】. Annotators gave one or more “emotion leaves” per image; about 80% of images ended up with a single label, the rest with 2-3 (adjacent) emotions【30†L274-L282】. The public “clean” set has 25,869 images (20% of collected images; the rest are private multi-label)【30†L231-L240】. 

  *Task:* Multi-class emotion classification + valence/arousal regression.  
  *Label schema:* 24 discrete emotions (multi-label) + continuous Valence/Arousal【28†L19-L28】【30†L231-L240】.  
  *Granularity:* Image-level.  
  *Public Access:* Yes (URLs and annotations released).  
  *Domain:* Diverse “in the wild” scenarios with multiple people (family, work, events)【27†L66-L74】.  
  *Strengths:* Rich labels: allows geometric analysis both in discrete (24 emotions) and continuous (2D VA) spaces. Cluster analysis can examine whether e.g. “joy vs sadness” images separate in embedding space and correspond to Va distribution【28†L55-L64】. The continuous labels support gradations (useful for centroid positioning by valence/arousal).  
  *Weaknesses:* Many images have **multi-label** emotions, complicating “pure” class geometry (46.6% had 2 labels)【30†L299-L307】. The label distribution is imbalanced (e.g. “joy” and “anticipation” common, “disgust” rare)【30†L321-L329】. Scenes are complex, so visuals of one emotion might co-occur with cues for another, muddying clusters. Also license is non-uniform (collected from web).  
  *Suitability:* Excellent for **fine affect categories** and **dimensional analysis**. One can embed images and color by valence or by Plutchik emotion to study overlap. It also allows mapping to sentiment polarity (via valence). Not suitable for object-level sentiment (no object annotations). For cross-modality, one could align the Plutchik categories with emotional word embeddings or use valence/arousal from text studies. Citations: dataset size, labels【30†L233-L240】【28†L19-L28】.

- **MVSO (English subset)**【74†L28-L36】【74†L66-L74】 (Jou *et al.*, ACM MM’15): A *multi-million* image collection of Flickr photos labelled by **sentiment-related ANPs** (adjective–noun pairs) discovered via Plutchik emotion keywords【74†L28-L36】. MVSO contains ~7.4 million images grouped by 3,911 English ANPs【74†L28-L36】 (e.g. *“beautiful landscape,” “sad eyes”*). In a follow-up, a subset of 11,733 of these images (3 images × 3 annotators per ANP) was human‑rated on a sentiment scale (−2 to +2)【74†L66-L74】. 

  *Task:* Concept detection / sentiment regression.  
  *Label schema:* ANP class (3,911 categories) + **continuous sentiment score** (via sum of 3 × {-2..+2})【74†L66-L74】.  
  *Granularity:* Image-level (one ANP per image, with optional metadata).  
  *Public Access:* Yes (image URLs and annotation available)【74†L66-L74】.  
  *Domain:* Crowdsourced Flickr images covering diverse content; ANPs drawn from 12 languages.  
  *Strengths:* Massive scale and variety; ANPs tie images to both objects and adjectives. The sentiment annotations let one analyze images directly in a polarity scale. Useful for studying how embedding geometry correlates with human-judged sentiment (the continuous scores).  
  *Weaknesses:* Most images only have noisy ANP tags, not explicit sentiment labels (except the annotated subset). ANPs conflate sentiment with the noun (“beautiful” vs “ugly” with “landscape” etc.), so class clusters may reflect object semantics as much as emotion. The annotated subset is relatively small (~12K) for robust geometry analysis.  
  *Suitability:* Good for **polarity analysis** (using the sentiment scores) and for exploring concept clusters (embedding of different ANP classes). Not ideal for fine emotions (only +/− scale). No object-level localization. Citations: dataset composition【74†L28-L36】【74†L66-L74】.

- **VSO/SentiBank**【79†L43-L46】 (Borth *et al.*, ACM MM’13): The Visual Sentiment Ontology provides a collection of Flickr images (CC‑licensed) organized by ANP. A subset (“SentiBank dataset”) has **1553 ANPs** with images【79†L43-L46】. In total, roughly tens of thousands of images are provided (e.g. ~50 images/ANP in SentiBank v1). Each image is labeled by one ANP. The full VSO covers 3,244 ANPs, but only a CC-licensed subset is directly downloadable【79†L43-L46】. 

  *Task:* Multi-label concept classification (1 ANP per image).  
  *Label schema:* 1,553 adjective–noun concepts (e.g. *“amazed cat”, “ecstatic crowd”*)【79†L43-L46】.  
  *Granularity:* Image-level (concept).  
  *Public Access:* Yes (images and labels publicly released under research license)【79†L43-L46】.  
  *Domain:* Creative Commons Flickr images spanning many scenes and objects.  
  *Strengths:* Access to images and labels is straightforward. ANPs cover a broad range of affectively descriptive phrases, enabling cluster analysis by these concepts. Many ANPs inherently encode sentiment (e.g. *“sad eyes”*).  
  *Weaknesses:* With 1,553 classes, clusters will be very fine-grained and sparse. Many ANPs are aesthetic or neutral rather than explicitly sentiment-laden. Distinguishing *positive* vs *negative* requires parsing the adjective (e.g. “adorable puppy” vs “afraid dog”) or grouping ANPs by valence, which is non-trivial. Object or scene cues may dominate over sentiment cues.  
  *Suitability:* Useful mainly as a large concept dataset; less useful directly for “polarity” geometry unless one aggregates ANPs by positive/negative adjectives. Objects (nouns) vary widely, so embeddings may cluster by object rather than sentiment. Citation: ANP count and license【79†L43-L46】.

- **EMOTIC**【32†L16-L20】 (Kosti *et al.*, PAMI’19): A context‑rich emotion dataset of 23,571 people in 18,316 images. Each *person* instance is labeled with up to 26 emotion categories (a Plutchik-derived set) and continuous Valence/Arousal/Dominance scores【32†L16-L20】. The images are drawn from COCO, ADE, and the Internet.  

  *Task:* Person-level emotion estimation.  
  *Label schema:* 26 discrete emotion labels (multi-label) + continuous V/A/D per person.  
  *Granularity:* **Object-level** (bounding boxes for each person).  
  *Public Access:* Yes (upon agreement for research)【33†L10-L18】.  
  *Domain:* “In-the-wild” everyday scenes with people.  
  *Strengths:* Perfect for analyzing **object/person-level affect** in images (embedding geometry of individuals’ emotions). Because context is included, one can study how context vs appearance relate to emotion clusters. Continuous V/A allows dimensional analysis.  
  *Weaknesses:* Each image often contains multiple people with different emotions (multi-label on the image), making “global” sentiment cluster analysis hard. Focus is on the person’s emotion, not the scene mood. The 26 labels include many subtle states (e.g. “engagement”, “distress”), so clusters may overlap.  
  *Suitability:* Good for **fine-grained emotion** and **object-level sentiment**. Less suited for simple positive/negative polarity since many labels are subtle (though valence scores are provided). Not cross-modal (no text). Citation: dataset size and labels【32†L16-L20】.

- **MVSA (Twitter)**【68†L1-L4】: A Twitter-based **multimodal sentiment** dataset. MVSA-Single has 5,129 tweets (image+text) each labeled (by 1 annotator) as positive/neutral/negative【68†L1-L4】. MVSA-Multiple has 19,600 such pairs labeled by 3 annotators.  

  *Task:* Sentiment classification of tweets (3-way).  
  *Label schema:* 3 classes (Pos/Neu/Neg).  
  *Granularity:* Pair-level (image+text together share one label).  
  *Domain:* Social media (tweets).  
  *Access:* Public (provided via links)【43†L69-L77】.  
  *Strengths:* Enables comparison of image vs text sentiment (clusters of images vs clusters of text). It has gold-standard sentiment labels.  
  *Weaknesses:* Very small (a few thousand items); images are often memes or social snapshots with text overlays. Not suited for fine-grained or object-level analysis. Clusters are coarse (only 3 labels).  
  *Suitability:* Useful if cross-modal alignment is needed; otherwise too small and coarse for robust embedding analysis. Citation: dataset size and classes【68†L1-L4】.

# Rejected or Weak Candidates

We also reviewed older or smaller datasets, but these were deemed **unsuitable** for robust geometric analysis due to their limitations:

- **IAPS/IAPSa (psychological)**【78†L159-L164】: Only ~395 images labeled with valence/arousal (Mikels 8 emotions) – far too small for embedding analysis.  
- **ArtPhoto**: ~806 images of art photography with single emotion labels – too small and biased (artistic domain).  
- **Abstract Paintings**: ~228 images with emotions – too small and abstract domain.  
- **Emotion6**【80†L1-L4】: 1,980 Flickr images in 6 categories – very small.  
- **GAPED (Geneva Affect Picture)**: ~730 stock images, with normative valence/arousal ratings. Limited size and only binary valence partition (neg/pos).  
- **Twitter I/II (Borth 2013)**: 603 and ~3,000 images labeled positive/negative – very small classes.  
- **TumEmo**: A Tumblr dataset (7 emotion labels, 195K posts)【55†L19-L27】, but labels derived from Tumblr context; not easily separable without text.  
- **Sentiment-140, SemEval image sets**: Primarily text-focused; images were incidental and not carefully labeled for sentiment.  
- **Aesthetic/Preference Datasets (AVA, AVP)**: These rate aesthetics or preference, not core sentiment or affect; irrelevant for emotion geometry.  

These datasets either lack scale, have overly coarse or highly confounded labels, or do not directly target image sentiment/emotion. We exclude them from the main recommendations.

# Final Recommendation

For embedding‑geometry experiments, the **best datasets** are those with *substantial size*, *meaningful affective labels*, and *few trivial visual cues*. Based on our analysis, we recommend:

1. **EmoSet-118K (Yang *et al.*, ICCV 2023)** – *Size/Balance:* At 118K images evenly split into 8 emotions【78†L115-L120】, this is by far the largest well-balanced emotion-labeled image set. Clusters of positive vs negative emotions should be clear, and centroids will be stable. The additional attribute labels enable deeper analysis of what factors drive cluster geometry.  
2. **You *et al.* “FI” dataset (AAAI 2016)** – *High-Quality Labels:* With 23K images and 8 emotion classes【26†L255-L258】, it offers a tested benchmark. Its multiple annotators per image yield relatively clean labels, facilitating clean separation of affect classes. This dataset has been widely used, making comparisons to prior work possible.  
3. **FindingEmo (Mertens et al. 2024)** – *Dimensional & Categorical Labels:* This dataset’s 25.9K images have both continuous valence/arousal and 24 emotion labels【30†L233-L240】. It allows analysis of embeddings in both categorical and continuous affect spaces. It also includes complex scenes (with multiple people), extending geometry analysis beyond single-subject images.  

These three provide **complementary benefits**: EmoSet for scale and balance, FI for data quality, and FindingEmo for richness of annotation. Together they cover binary sentiment, multi-class emotion, and dimensional affect, providing a robust basis for centroid/cluster/overlap studies. All are publicly accessible for research use【78†L96-L100】【26†L255-L258】【30†L233-L240】.

