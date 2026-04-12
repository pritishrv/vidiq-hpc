# Multiclass Embedding-Logit Alignment Experiment

This experiment centers on training a transformer-based classifier on the multiclass emotion dataset so we can inspect how the 768-dimensional embeddings relate to the downstream logits. Key steps:

1. **Data** – use the balanced version of the dair-ai/emotion split. Keep `texts.jsonl`/labels as-is for training, and reserve the existing validation set for inference.
2. **Model** – the transformer backbone produces a 768-dimensional representation for each text. A small fully connected head projects that vector to 5 logits (matching the dataset’s classes) before a softmax layer.
3. **Training loop** – cross-entropy loss on logits; checkpoint both the transformer+head weights and the intermediate 768D embedding output for the validation splits.
4. **Inference / probe** – run the trained model on the test set, record per-sample embeddings plus logits, then aggregate:
   * Average each emotion’s embeddings to derive centroid vectors.
   * Compare point-to-centroid distances with logits/softmax confidences for the predicted class.
5. **Evaluation** – visualize embeddings (PCA + radial distance) and report whether logits correlate with embedding distance to centroids.

Outputs (models, npy embeddings, logits, metrics) should live under `experiments/text_model/<run-id>` so downstream analysis can point back to a single run. Once you’ve picked the backbone (BGE, Qwen, etc.), I can help scaffold the training/inference scripts.

## Training Script

`train_multiclass.py` instantiates a Qwen3 transformer, trains a small projection head for the 5-class emotion task, and collects the 768-dimensional embeddings for the held-out split. It automatically stratifies the balanced dataset, saves checkpoints/metrics under the run directory, and exports the embeddings plus centroid summary so you can compare logits with embedding geometry after training.
