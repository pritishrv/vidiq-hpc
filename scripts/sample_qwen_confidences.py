from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable
import argparse

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


def load_samples_from_csv(
    csv_path: Path,
    text_column: str,
    label_column: str,
    num_per_class: int,
) -> list[dict]:
    per_class = defaultdict(list)
    samples: list[dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row[label_column])
            if len(per_class[label]) >= num_per_class:
                continue
            text = row[text_column].strip()
            entry = {"text": text, "label": label}
            per_class[label].append(entry)
            samples.append(entry)
            if len(per_class[label]) >= num_per_class and all(
                len(per_class[k]) >= num_per_class for k in per_class
            ):
                break
    return samples


def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int = 64,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            outputs.append(pooled.cpu().numpy())
    return np.vstack(outputs)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def summarize(samples: list[dict], confidences: np.ndarray, label_names: Iterable[str]) -> None:
    label_names = list(label_names)
    for record, conf in zip(samples, confidences):
        label = int(record["label"])
        print(f"Text: {record['text']}")
        print(f"Gold label: {label} ({label_names[label]})")
        print(
            "Confidences:",
            ", ".join(f"{name}:{prob:.3f}" for name, prob in zip(label_names, conf)),
        )
        print("-" * 60)


def collect_samples(
    source_path: Path,
    text_column: str,
    label_column: str,
    label_names: list[str],
    prompts: list[str],
) -> list[dict]:
    samples = load_samples_from_csv(source_path, text_column, label_column, num_per_class=5)
    if not samples:
        return []
    texts = [s["text"] for s in samples]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
    prompt_embeddings = encode_texts(prompts, tokenizer, model, DEVICE)
    prompt_embeddings = normalize(prompt_embeddings)
    text_embeddings = encode_texts(texts, tokenizer, model, DEVICE)
    text_embeddings = normalize(text_embeddings)
    logits = text_embeddings @ prompt_embeddings.T
    confs = softmax(logits / TEMPERATURE)
    output = []
    for sample, conf, logit_row in zip(samples, confs, logits):
        entry = {
            "text": sample["text"],
            "label": sample["label"],
            "label_name": label_names[sample["label"]],
        }
        for name, score in zip(label_names, logit_row):
            entry[f"logit_{name}"] = f"{score:.6f}"
        for name, score in zip(label_names, conf):
            entry[f"conf_{name}"] = f"{score:.6f}"
        entry["dataset"] = source_path.name
        output.append(entry)
    return output


MODEL_PATH = "/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/models/qwen3-0.6B"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TEMPERATURE = 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample Qwen3 confidences and export to CSV.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/sample_qwen_confidences.csv"),
        help="Path to write the sample CSV.",
    )
    args = parser.parse_args()
    base = Path("..") / "data" / "Text_Datasets"
    binary_path = base / "SST-2" / "sst2_train.csv"
    multiclass_path = base / "DSAIRAI-EMOTION" / "emotion_train.csv"
    all_samples = []
    all_samples.extend(
        collect_samples(
            binary_path,
            text_column="sentence",
            label_column="label",
            label_names=["negative", "positive"],
            prompts=["negative sentiment", "positive sentiment"],
        )
    )
    all_samples.extend(
        collect_samples(
            multiclass_path,
            text_column="text",
            label_column="label",
            label_names=["sadness", "joy", "love", "anger", "fear", "surprise"],
            prompts=["sadness", "joy", "love", "anger", "fear", "surprise"],
        )
    )
    if not all_samples:
        print("No samples collected.")
        exit(1)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "text", "label", "label_name"]
    for name in ["negative", "positive", "sadness", "joy", "love", "anger", "fear", "surprise"]:
        fieldnames.append(f"logit_{name}")
    for name in ["negative", "positive", "sadness", "joy", "love", "anger", "fear", "surprise"]:
        fieldnames.append(f"conf_{name}")
    with args.output_csv.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_samples:
            writer.writerow(row)
    print(f"Wrote {len(all_samples)} rows to {args.output_csv}")
