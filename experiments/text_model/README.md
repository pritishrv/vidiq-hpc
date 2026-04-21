# `train_multiclass.py` Method README

This document records the multiclass text-model training pipeline used in the historical `vidiq-hpc` HPC experiments. It is intended as a paper-facing methods note for the Qwen-based emotion-classification runs launched from Hyperion SLURM jobs.

## Scope

The documented entrypoint is:

- `experiments/text_model/train_multiclass.py`

That script is not present on the current `main` checkout, but it exists in the repository history and was used to generate the multiclass training runs and saved embedding artifacts referenced elsewhere in this project. This README preserves the implementation details needed for the paper.

## Experiment Goal

The training pipeline was designed to fine-tune a causal language model backbone for multiclass emotion classification while also exporting internal representations for geometric analysis. The core paper-facing idea is:

- train a classifier on emotion labels
- extract a fixed embedding for each example from the backbone
- compare class separation in embedding space against downstream predictive performance

This makes the script relevant both as a classifier-training pipeline and as an embedding-extraction method.

## Backbone And Classifier Head

`train_multiclass.py` builds an `EmotionEmbeddingClassifier` with:

- backbone: `AutoModelForCausalLM.from_pretrained(...)`
- default model reference: `Qwen/Qwen3-1.7B`
- `output_hidden_states=True` so the model returns hidden states for all layers
- optional `--freeze-backbone` mode for head-only training

On top of the backbone, the script attaches a small projection head:

```python
nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.LayerNorm(hidden_size),
    nn.ReLU(inplace=True),
    nn.Linear(hidden_size, num_labels),
)
```

The projection head maps the selected hidden representation to class logits.

## Exact Point Where Embeddings Are Taken

This is the key detail for the paper.

During the forward pass, the script computes:

```python
outputs = self.transformer(...)
hidden = outputs.hidden_states[-1][:, -1, :].to(torch.float32)
logits = self.projection(hidden)
```

So the exported embedding is:

- the final transformer layer
- at the last token position in the sequence
- converted to `float32`
- taken before the classifier head

This means the saved embeddings are not pooled over tokens. They are single-vector sequence representations derived from the final-token hidden state of the Qwen backbone.

## Data Inputs

The script supports two input modes.

### 1. Processed dataset root

Pass `--data-root` pointing at a dataset directory. The script expects processed training data under:

- `data/processed/train/texts.jsonl`

Each JSONL row is expected to contain:

- `text`
- `label`

### 2. Raw CSV input

Pass `--csv-path` pointing at a CSV file. The CSV must contain:

- an `emotion` column for labels
- one of `cleaned_text`, `sentence`, or `text` for the input text

Rows with empty text or empty labels are skipped.

## Label Processing And Splitting

For CSV data:

- string labels are sorted
- labels are mapped to integer ids
- the mapping is saved into run metadata

For evaluation:

- the script builds a train/test split with `StratifiedShuffleSplit`
- default `test_size` is `0.2`
- default `seed` is `42`

This preserves class balance between train and evaluation splits.

## Tokenization And Batching

The tokenizer is loaded with:

```python
AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
```

Batch collation uses:

- right padding
- truncation
- `max_length` control
- `input_ids`
- `attention_mask`
- optional `token_type_ids`

The tokenizer pad token is set to the EOS token to support causal-language-model batching.

## Training Procedure

The training loop uses:

- optimizer: `AdamW`
- loss: cross-entropy over the classifier logits
- gradient clipping: `max_norm=2.0`
- shuffled training DataLoader
- separate evaluation DataLoader

At each epoch:

1. the model trains on the training split
2. the model is evaluated on the held-out split
3. epoch loss and accuracy are appended to `train_metrics.json`

The script prints one summary line per epoch with loss and accuracy.

## Evaluation Outputs

The evaluation stage collects:

- embeddings
- logits
- probabilities
- labels
- predictions
- accuracy

The embeddings and logits are accumulated batch-by-batch on the held-out split and then concatenated into full evaluation arrays.

## Saved Artifacts

For each run, the script creates a run directory under:

- `experiments/text_model/runs/<run_name>/`

It saves:

- `analysis/eval_embeddings.npy`
- `analysis/eval_logits.npy`
- `analysis/eval_metrics.json`
- `analysis/centroids.json`
- `train_metrics.json`
- `run_metadata.json`

It also saves model artifacts:

- transformer checkpoint under an archive directory
- classification head weights as `head.pt`
- tokenizer files in the archive area

The local run directory then exposes symlinks back to archived model and tokenizer directories.

## Centroid Computation

After final evaluation, the script computes one centroid per class:

- gather all evaluation embeddings for a class
- take the arithmetic mean across those vectors
- save the resulting centroid vectors to `analysis/centroids.json`

This centroid file is the main bridge from supervised classification to downstream embedding-geometry analysis.

## Run Metadata

`run_metadata.json` records:

- run name
- UTC timestamp
- git commit if available
- SLURM job name and job id
- resolved arguments
- dataset source and label mapping
- local artifact locations
- archive targets for model and tokenizer

This is important for reproducibility and for attributing results in the paper.

## Command-Line Interface

Important arguments supported by the script include:

- `--data-root`: processed dataset root
- `--csv-path`: raw CSV dataset path
- `--model-path`: local checkpoint path or model id, default `Qwen/Qwen3-1.7B`
- `--run-root`: local run-output root
- `--archive-root`: archive location for heavyweight artifacts
- `--run-name`: explicit run id
- `--batch-size`: default `8`
- `--num-epochs`: default `3`
- `--learning-rate`: default `5e-5`
- `--max-length`: default `256`
- `--test-size`: default `0.2`
- `--seed`: default `42`
- `--device`: defaults to `cuda` when available
- `--freeze-backbone`: disables backbone updates

## Historical HPC Entry Points

The multiclass jobs were launched from historical SLURM scripts in `hpc/`, including:

- `hpc/train_multiclass.slurm`
- `hpc/train_multiclass_frozen_backbone.slurm`
- `hpc/train_multiclass_balanced_10e.slurm`
- `hpc/train_multiclass_balanced_50e.slurm`
- `hpc/train_multiclass_balanced_100e.slurm`
- `hpc/train_multiclass_balanced_250e.slurm`
- `hpc/train_multiclass_balanced_500e.slurm`
- `hpc/train_multiclass_balanced_1000e.slurm`
- `hpc/train_multiclass_unbalanced_10e.slurm`
- `hpc/train_multiclass_unbalanced_50e.slurm`
- `hpc/train_multiclass_unbalanced_100e.slurm`
- `hpc/train_multiclass_unbalanced_250e.slurm`
- `hpc/train_multiclass_unbalanced_500e.slurm`
- `hpc/train_multiclass_unbalanced_1000e.slurm`

These jobs configured GPU resources, cache directories, runtime limits, and run names, then invoked `train_multiclass.py` with the relevant CSV path and epoch count.

## Dependencies

The historical `experiments/text_model/requirements.txt` listed:

- `torch`
- `transformers`
- `datasets`
- `numpy`
- `scikit-learn`
- `tqdm`

The script also relies on standard-library modules such as `argparse`, `csv`, `json`, `pathlib`, and `subprocess`.

## Interpretation Notes For The Paper

When describing this method in the paper, the most important implementation facts are:

- the backbone is a causal LM, not a sentence-transformer encoder
- the exported representation is the final-layer hidden state at the last token
- logits are produced by a learned projection head on top of that representation
- geometric analysis is performed on held-out evaluation embeddings
- class centroids are computed from evaluation embeddings after training

That framing accurately reflects what the code does and avoids implying token pooling, CLS-token extraction, or frozen off-the-shelf embedding usage when those were not the operative choices in this pipeline.
