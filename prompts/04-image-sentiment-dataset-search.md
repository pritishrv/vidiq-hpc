Identify candidate image datasets for sentiment analysis that could support embedding-geometry experiments.

Goal:
Find image datasets where the labels capture sentiment, affect, or closely related evaluative/emotional dimensions in a way that could plausibly support centroid / cluster / overlap analysis in embedding space.

What I need:
1. A shortlist of the best candidate datasets.
2. For each dataset, give:
   - dataset name
   - source / publisher
   - task type
   - label schema
   - number of images
   - whether labels are binary, multiclass, multilabel, or continuous
   - whether labels are image-level or region/object-level
   - whether the dataset is public and easy to access
   - licensing / usage constraints
   - likely image domain
   - likely strengths for embedding-geometry analysis
   - likely weaknesses / confounds
   - whether it is suitable for:
     - clean sentiment polarity
     - finer affect categories
     - object-level sentiment
     - cross-modality comparison with text experiments
3. Rank the datasets from most useful to least useful for this research goal.
4. Recommend the top 3 datasets and explain why.

Research objective:
We are not mainly trying to maximize benchmark accuracy. We want datasets that are useful for geometric analysis of embeddings:
- compactness of class clusters
- centroid distances
- overlap between sentiment classes
- comparison of raw images vs segmented objects
- possibly comparing image geometry with text sentiment/emotion geometry

Important selection criteria:
- labels should be meaningful and not too noisy
- dataset should be large enough for stable geometry analysis
- classes should not be too heavily confounded by domain or trivial visual cues
- image-level sentiment is useful, but also note any datasets with sentiment tied to objects, scenes, or attributes
- if a dataset has continuous valence/arousal annotations, discuss whether it can be discretized sensibly
- if a dataset is famous but poorly suited for geometry analysis, say so explicitly

Please search broadly across:
- visual sentiment datasets
- affective image datasets
- emotion-labeled image datasets
- aesthetic / preference datasets only if they are genuinely relevant to sentiment-style geometry
- multimodal datasets only if the image labels are usable without relying on text captions

Output format:
1. Executive summary
2. Ranked dataset table
3. Detailed notes on each strong candidate
4. Rejected / weak candidates and why
5. Final recommendation

Be concrete and critical. Do not just list datasets. Compare them for this specific research use case.
