from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from io_utils import ensure_dir, slugify, write_json
from paths import EMBEDDINGS_DIR


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


def mean_center(vectors: np.ndarray) -> np.ndarray:
    return vectors - vectors.mean(axis=0, keepdims=True)


def embed_texts(
    texts: list[str],
    model_name: str,
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = _device()
    model.to(device)
    model.eval()

    outputs = []
    truncation_count = 0
    token_lengths = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_length=True,
            )
            lengths = encoded.pop("length").tolist()
            token_lengths.extend(int(x) for x in lengths)
            truncation_count += sum(1 for x in lengths if int(x) >= max_length)

            encoded = {k: v.to(device) for k, v in encoded.items()}
            model_output = model(**encoded)
            pooled = mean_pool(model_output.last_hidden_state, encoded["attention_mask"])
            outputs.append(pooled.cpu().numpy())

    raw = np.vstack(outputs)
    metadata = {
        "model_name": model_name,
        "device": device,
        "max_length": max_length,
        "batch_size": batch_size,
        "avg_tokenized_length": float(np.mean(token_lengths)) if token_lengths else 0.0,
        "median_tokenized_length": float(np.median(token_lengths)) if token_lengths else 0.0,
        "truncation_count": int(truncation_count),
        "truncation_rate": float(truncation_count / len(texts)) if texts else 0.0,
    }
    return raw, metadata


def save_embedding_variants(
    split_name: str,
    model_slug: str,
    vectors_raw: np.ndarray,
    model_metadata: dict[str, Any],
) -> dict[str, str]:
    ensure_dir(EMBEDDINGS_DIR)
    base = f"sst2_{split_name}_{slugify(model_slug)}"

    raw_path = EMBEDDINGS_DIR / f"{base}_raw.npy"
    l2_path = EMBEDDINGS_DIR / f"{base}_l2.npy"
    centered_l2_path = EMBEDDINGS_DIR / f"{base}_centered_l2.npy"
    metadata_path = EMBEDDINGS_DIR / f"{base}_metadata.json"

    vectors_l2 = l2_normalize(vectors_raw)
    vectors_centered_l2 = l2_normalize(mean_center(vectors_raw))

    np.save(raw_path, vectors_raw)
    np.save(l2_path, vectors_l2)
    np.save(centered_l2_path, vectors_centered_l2)
    write_json(metadata_path, model_metadata)

    return {
        "raw": str(raw_path),
        "l2": str(l2_path),
        "centered_l2": str(centered_l2_path),
        "metadata": str(metadata_path),
    }

