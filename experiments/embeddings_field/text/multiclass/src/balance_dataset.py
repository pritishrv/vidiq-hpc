from __future__ import annotations

import json
from pathlib import Path

import numpy as np


VIDIQ_ROOT = Path(__file__).resolve().parents[5]
ORIGINAL_ROOT = VIDIQ_ROOT / "experiments" / "text" / "multiclass" / "dair-ai-emotion"
BALANCED_ROOT = VIDIQ_ROOT / "experiments" / "embeddings_field" / "text" / "multiclass" / "balanced_dataset"
BALANCED_DATA_DIR = BALANCED_ROOT / "data" / "processed" / "train"
BALANCED_ARTIFACTS = BALANCED_ROOT / "artifacts" / "embeddings"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def balance():
    labels_path = ORIGINAL_ROOT / "data" / "processed" / "train" / "labels.npy"
    texts_path = ORIGINAL_ROOT / "data" / "processed" / "train" / "texts.jsonl"
    embeddings_path = ORIGINAL_ROOT / "artifacts" / "embeddings" / "dair_ai_emotion_train_bge-base-en-v1-5_raw.npy"

    labels = np.load(labels_path)
    texts = []
    with texts_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            texts.append(json.loads(line)["text"])

    unique, counts = np.unique(labels, return_counts=True)
    min_count = int(counts.min())
    rng = np.random.default_rng(42)
    selected_indices = []
    for label in unique:
        label_indices = np.where(labels == label)[0]
        selected = rng.choice(label_indices, size=min_count, replace=False)
        selected_indices.extend(selected.tolist())
    selected_indices = np.sort(selected_indices)

    balanced_labels = labels[selected_indices]
    balanced_texts = [texts[i] for i in selected_indices]
    balanced_embeddings = np.load(embeddings_path)[selected_indices]

    ensure_dir(BALANCED_DATA_DIR)
    ensure_dir(BALANCED_ARTIFACTS)

    np.save(BALANCED_DATA_DIR / "labels.npy", balanced_labels)
    with (BALANCED_DATA_DIR / "texts.jsonl").open("w", encoding="utf-8") as fh:
        for text, label in zip(balanced_texts, balanced_labels):
            fh.write(json.dumps({"text": text, "label": int(label)}, ensure_ascii=False) + "\n")
    metadata = {
        "min_count": min_count,
        "num_examples": len(balanced_labels),
        "class_counts": {str(k): int(min_count) for k in unique},
    }
    with (BALANCED_DATA_DIR / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    np.save(
        BALANCED_ARTIFACTS / "dair_ai_emotion_train_balanced_bge-base-en-v1-5_raw.npy",
        balanced_embeddings,
    )
    with (BALANCED_ARTIFACTS / "metadata.json").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"min_count": int(min_count)}, indent=2))

    print("Balanced dataset created:", BALANCED_ROOT)


if __name__ == "__main__":
    balance()
