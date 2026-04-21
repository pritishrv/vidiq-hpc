from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_label_texts(label_names: list[str]) -> list[str]:
    return [f"This sentence expresses {name}." for name in label_names]


def build_prototypes(tokenizer, model, label_texts: list[str], device: torch.device) -> torch.Tensor:
    inputs = tokenizer(
        label_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state
        pooled = mean_pool(outputs, inputs["attention_mask"])
    return torch.nn.functional.normalize(pooled, dim=1)


def compute_probs(
    texts: list[str],
    tokenizer,
    model,
    prototypes: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    batch_size = 32
    logits_list = []
    model.to(device)
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state
            pooled = mean_pool(outputs, inputs["attention_mask"])
        normalized = torch.nn.functional.normalize(pooled, dim=1)
        logits = normalized @ prototypes.t()
        logits_list.append(logits.cpu().numpy())
    return np.vstack(logits_list)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-label logits/probs.")
    parser.add_argument("--prepared-root", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/probabilities"))
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--label-config", type=Path, default=None)
    args = parser.parse_args()

    label_config = {}
    if args.label_config:
        with args.label_config.open("r", encoding="utf-8") as fh:
            label_config = json.load(fh)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    device = torch.device(args.device)

    ensure_dir(args.output_dir)
    dataset_dirs = []
    for dir_path in sorted(args.prepared_root.rglob("*")):
        if not dir_path.is_dir():
            continue
        texts_path = dir_path / "texts.jsonl"
        labels_path = dir_path / "labels.npy"
        if texts_path.exists() and labels_path.exists():
            dataset_dirs.append(dir_path)

    for dataset_dir in dataset_dirs:
        texts_path = dataset_dir / "texts.jsonl"
        labels_path = dataset_dir / "labels.npy"

        label_names = label_config.get(dataset_dir.name, label_config.get("default"))
        if not label_names:
            raise ValueError(f"No label names provided for {dataset_dir.name}")
        label_texts = load_label_texts(label_names)
        prototypes = build_prototypes(tokenizer, model, label_texts, device)

        texts = []
        with texts_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                texts.append(json.loads(line)["text"])
        logits = compute_probs(texts, tokenizer, model, prototypes, device)
        probs = softmax(logits)

        suffix = dataset_dir.relative_to(args.prepared_root).as_posix().replace("/", "_")
        output_path = ensure_dir(args.output_dir) / f"{suffix}-probabilities.jsonl"
        with output_path.open("w", encoding="utf-8") as fh:
            for text, label, logit_row, prob_row in zip(texts, np.load(labels_path), logits, probs):
                fh.write(
                    json.dumps(
                        {
                            "text": text,
                            "label": int(label),
                            "label_names": label_names,
                            "logits": logit_row.tolist(),
                            "probabilities": prob_row.tolist(),
                        }
                    )
                    + "\n"
                )
        print(f"Saved probabilities for {dataset_dir.name} -> {output_path}")


if __name__ == "__main__":
    main()
