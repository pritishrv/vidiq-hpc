from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_CSV_PATH = Path(
    "/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/data/Text_Datasets/20-emotions/"
    "balanced_emotions_6classes.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq-hpc/artifacts/embeddings/"
    "balanced-6-emotions"
)
DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEFAULT_LABELS = ["sadness", "happiness", "love", "anger", "fear", "surprise"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_rows(csv_path: Path, text_column: str, label_column: str, label_names: list[str]) -> list[dict[str, str]]:
    expected = set(label_names)
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {text_column, label_column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row_index, row in enumerate(reader):
            text = (row[text_column] or "").strip()
            label = (row[label_column] or "").strip()
            if not text:
                continue
            if label not in expected:
                raise ValueError(f"Unexpected label {label!r} at row {row_index}; expected {label_names}")
            rows.append({"row_index": str(row_index), "text": text, "label": label})
    return rows


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(
    texts: list[str],
    model_name: str,
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = choose_device()
    model.to(device)
    model.eval()

    outputs: list[np.ndarray] = []
    token_lengths: list[int] = []
    truncation_count = 0
    total_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), total=total_batches, unit="batch"):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_length=True,
            )
            lengths = [int(length) for length in encoded.pop("length").tolist()]
            token_lengths.extend(lengths)
            truncation_count += sum(1 for length in lengths if length >= max_length)

            encoded = {key: value.to(device) for key, value in encoded.items()}
            model_output = model(**encoded)
            pooled = mean_pool(model_output.last_hidden_state, encoded["attention_mask"])
            outputs.append(pooled.cpu().numpy())

    embeddings = np.vstack(outputs)
    metadata = {
        "model_name": model_name,
        "device": device,
        "embedding_kind": "raw_mean_pool",
        "max_length": max_length,
        "batch_size": batch_size,
        "num_examples": len(texts),
        "embedding_shape": list(embeddings.shape),
        "avg_tokenized_length": float(np.mean(token_lengths)) if token_lengths else 0.0,
        "median_tokenized_length": float(np.median(token_lengths)) if token_lengths else 0.0,
        "truncation_count": int(truncation_count),
        "truncation_rate": float(truncation_count / len(texts)) if texts else 0.0,
    }
    return embeddings, metadata


def save_per_emotion(
    rows: list[dict[str, str]],
    embeddings: np.ndarray,
    output_root: Path,
    label_names: list[str],
    run_metadata: dict[str, Any],
) -> None:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped_indices[row["label"]].append(idx)

    for label in label_names:
        emotion_dir = ensure_dir(output_root / label)
        indices = np.array(grouped_indices[label], dtype=np.int64)
        emotion_embeddings = embeddings[indices]
        np.save(emotion_dir / "raw_embeddings.npy", emotion_embeddings)
        np.save(emotion_dir / "source_indices.npy", indices)
        with (emotion_dir / "texts.jsonl").open("w", encoding="utf-8") as fh:
            for idx in indices:
                row = rows[int(idx)]
                fh.write(
                    json.dumps(
                        {
                            "source_index": int(idx),
                            "csv_row_index": int(row["row_index"]),
                            "label": label,
                            "text": row["text"],
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        write_json(
            emotion_dir / "metadata.json",
            {
                **run_metadata,
                "emotion": label,
                "num_examples": int(len(indices)),
                "embedding_shape": list(emotion_embeddings.shape),
                "embeddings_file": "raw_embeddings.npy",
                "source_indices_file": "source_indices.npy",
                "texts_file": "texts.jsonl",
            },
        )


def parse_labels(raw: str) -> list[str]:
    return [label.strip() for label in raw.split(",") if label.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw BGE embeddings for the balanced six-emotion CSV.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--text-column", default="cleaned_text")
    parser.add_argument("--label-column", default="emotion")
    parser.add_argument("--label-names", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    label_names = parse_labels(args.label_names)
    rows = load_rows(args.csv_path, args.text_column, args.label_column, label_names)
    texts = [row["text"] for row in rows]
    output_root = ensure_dir(args.output_root)

    embeddings, embedding_metadata = embed_texts(
        texts,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    run_metadata = {
        **embedding_metadata,
        "source_csv": str(args.csv_path),
        "text_column": args.text_column,
        "label_column": args.label_column,
        "label_names": label_names,
    }
    write_json(output_root / "metadata.json", run_metadata)
    save_per_emotion(rows, embeddings, output_root, label_names, run_metadata)
    print(f"Saved raw per-emotion embeddings to {output_root}")


if __name__ == "__main__":
    main()
