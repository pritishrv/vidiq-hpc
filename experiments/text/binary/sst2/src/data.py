from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from io_utils import ensure_dir, write_json
from paths import DATA_PROCESSED_DIR, DATA_RAW_DIR


def load_sst2_dataset(config: dict[str, Any]):
    dataset_cfg = config["dataset"]
    path = dataset_cfg["path"]
    name = dataset_cfg.get("name")
    if dataset_cfg.get("loader") == "glue":
        return load_dataset(path, name)
    return load_dataset(path)


def _split_output_dir(split_name: str) -> Path:
    return ensure_dir(DATA_PROCESSED_DIR / split_name)


def persist_split(split_name: str, texts: list[str], labels: list[int]) -> dict[str, Any]:
    split_dir = _split_output_dir(split_name)
    jsonl_path = split_dir / "texts.jsonl"
    labels_path = split_dir / "labels.npy"
    metadata_path = split_dir / "metadata.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for text, label in zip(texts, labels):
            f.write(json.dumps({"text": text, "label": int(label)}, ensure_ascii=True) + "\n")

    np.save(labels_path, np.array(labels, dtype=np.int64))
    token_lengths = [len(text.split()) for text in texts]
    class_counts = {str(k): int(v) for k, v in sorted(Counter(labels).items())}

    metadata = {
        "split": split_name,
        "num_examples": len(texts),
        "class_counts": class_counts,
        "avg_whitespace_token_length": float(np.mean(token_lengths)) if token_lengths else 0.0,
        "median_whitespace_token_length": float(np.median(token_lengths)) if token_lengths else 0.0,
    }
    write_json(metadata_path, metadata)
    return metadata


def sample_train_subset(
    texts: list[str],
    labels: list[int],
    fraction: float,
    seed: int,
) -> dict[str, list[Any]]:
    indices = np.arange(len(texts))
    sampled_indices, _ = train_test_split(
        indices,
        train_size=fraction,
        stratify=np.array(labels, dtype=np.int64),
        random_state=seed,
    )
    sampled_indices = np.sort(sampled_indices)
    return {
        "texts": [texts[i] for i in sampled_indices],
        "labels": [labels[i] for i in sampled_indices],
    }


def prepare_dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset = load_sst2_dataset(config)
    dataset_cfg = config["dataset"]
    text_col = dataset_cfg["text_column"]
    label_col = dataset_cfg["label_column"]
    train_split = dataset_cfg["train_split"]
    val_split = dataset_cfg["validation_split"]

    raw_info = {
        "dataset_path": dataset_cfg["path"],
        "dataset_name": dataset_cfg.get("name"),
        "splits": list(dataset.keys()),
    }
    write_json(DATA_RAW_DIR / "dataset-source.json", raw_info)

    prepared: dict[str, Any] = {}
    for split_name in [train_split, val_split]:
        split = dataset[split_name]
        texts = [str(x).strip() for x in split[text_col] if str(x).strip()]
        labels = [int(x) for text, x in zip(split[text_col], split[label_col]) if str(text).strip()]
        prepared[split_name] = {
            "texts": texts,
            "labels": labels,
            "metadata": persist_split(split_name, texts, labels),
        }

    sample_fraction = dataset_cfg.get("train_sample_fraction_for_model_selection")
    sample_seed = int(dataset_cfg.get("train_sample_seed", 42))
    if sample_fraction is not None and 0 < float(sample_fraction) < 1:
        sampled = sample_train_subset(
            prepared[train_split]["texts"],
            prepared[train_split]["labels"],
            float(sample_fraction),
            sample_seed,
        )
        prepared["train_model_selection"] = {
            "texts": sampled["texts"],
            "labels": sampled["labels"],
            "metadata": persist_split("train_model_selection", sampled["texts"], sampled["labels"]),
        }
    return prepared
