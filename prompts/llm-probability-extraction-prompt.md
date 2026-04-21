## Prompt: Generate Confidence-Scored Text Dataset

### Goal
Produce a version of the text dataset where every sample carries a per-class confidence/probability vector instead of a single label. Use the same encoder that created the embeddings (`BAAI/bge-base-en-v1.5`) to report how likely each class is for every text row, so downstream analysis can work with soft assignments (e.g., a sentence can be half “sad” and half “angry”).

### Instructions
1. Load the target split (binary `SST-2` or multiclass `dair-ai/emotion`) from `/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/data/Text_Datasets`. Identify the label vocabulary for each split (positive/negative for SST-2; sadness, joy, love, anger, fear, surprise for the multiclass split).
2. Tokenize each sentence and feed it through `BAAI/bge-base-en-v1.5`. Instead of storing only the embedding, capture the logits (or softmax probabilities) for every class. If the model does not directly expose per-class logits, project the final hidden state onto simple prompt/label prototypes to yield a score for each label and normalize to probabilities.
3. Create a structured output (JSONL or CSV) per split that contains: `text`, `gold_label`, `confidences` (array of probabilities matching the label order above), and any metadata needed for traceability (e.g., split name, timestamp, model version).
4. Save the generated dataset under the experiment data folder (e.g., `experiments/text/<task>/<dataset>/data/confidence_scores.jsonl`) so it can be consumed by subsequent embedding analyses.

### Output format example (JSONL)
```
{"text": "I loved the movie", "gold_label": 1, "confidences": [0.03, 0.97]}
{"text": "The plot dragged on", "gold_label": 0, "confidences": [0.88, 0.12]}
```

### Notes
- This prompt is purely about dataset creation with per-class probabilities; do not describe broader validation experiments here.
- Keep the label vocabulary consistent with the datasets and store the output near the corresponding experiment so future steps can read it without rerunning inference.
