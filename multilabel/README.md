# Multilabel Probability Extraction

Scripts in this folder rebuild the text datasets under `data/Text_Datasets` into a stable location inside the repo and then compute per-label probability distributions over the binary and multiclass labels using `BAAI/bge-base-en-v1.5`. The outputs mimic the embedding pipeline (per-split metadata, consistent JSON lines, and explicit logits) so they can slot into later analyses.

1. `scripts/prepare_dataset.py` copies texts/labels from the provided source directory into `data/prepared/<dataset_name>`.
2. `label_config.json` defines the label names for each dataset to keep the prototypes aligned with the target taxonomy.
3. `scripts/run_probability_extraction.py` loads the prepared datasets, uses the encoder to compute dense vectors, projects them against the label prototypes, and writes pre-softmax scores for each label into `artifacts/probabilities/<dataset_name>.jsonl`.
4. Each dataset keeps metadata (`metadata.json`) describing the label names and counts for traceability.

Run the scripts in order before consuming the probability outputs in subsequent reports or prompts.
