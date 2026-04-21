from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Any

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


def read_prompt_file(prompt_path: Path | None) -> str | None:
    if not prompt_path:
        return None
    if not prompt_path.exists():
        raise FileNotFoundError(f"prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def save_embeddings(
    split_name: str,
    embeddings: np.ndarray,
    output_dir: Path,
    dataset_slug: str,
    metadata: dict[str, Any],
) -> None:
    ensure_dir(output_dir)
    out_path = output_dir / f"{dataset_slug}_{split_name}_{metadata['model_slug']}_raw.npy"
    meta_path = output_dir / f"{dataset_slug}_{split_name}_{metadata['model_slug']}_metadata.json"
    np.save(out_path, embeddings.astype(np.float32))
    meta_record = {
        "split": split_name,
        "dataset_slug": dataset_slug,
        "model_name": metadata["model_name"],
        "model_slug": metadata["model_slug"],
        "tokenizer_max_length": metadata.get("max_length"),
        "batch_size": metadata.get("batch_size"),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        **{
            "label_prompts": metadata.get("label_prompts"),
            "prompt_context": metadata.get("prompt_context"),
        },
    }
    write_json(meta_path, meta_record)


if TRANSFORMERS_AVAILABLE:
    def _device() -> torch.device:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


    def embed_texts(
        tokenizer: AutoTokenizer,
        model: AutoModel,
        texts: list[str],
        batch_size: int,
        max_length: int,
        device: torch.device,
    ) -> np.ndarray:
        outputs = []
        order = range(0, len(texts), batch_size)
        with torch.no_grad():
            for start in tqdm(order, desc="Embedding texts", unit="batch"):
                batch = texts[start : start + batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_length=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                lengths = encoded.pop("length")
                model_output = model(**encoded)
                pooled = mean_pool(model_output.last_hidden_state, encoded["attention_mask"])
                outputs.append(pooled.cpu().numpy())
        if outputs:
            return np.vstack(outputs)
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)


def load_split_texts(split_dir: Path) -> tuple[list[str], list[int]]:
    texts_path = split_dir / "texts.jsonl"
    labels_path = split_dir / "labels.npy"
    if not texts_path.exists():
        raise FileNotFoundError(f"missing texts file {texts_path}")
    labels = np.load(labels_path) if labels_path.exists() else np.array([], dtype=np.int64)
    texts: list[str] = []
    label_list: list[int] = []
    with texts_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            texts.append(record.get("text", "").strip())
            if idx < len(labels):
                label_list.append(int(labels[idx]))
            else:
                label_list.append(record.get("label", -1))
    return texts, label_list


def gather_split_dirs(processed_dir: Path) -> list[Path]:
    splits: list[Path] = []
    for child in processed_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name == "confidence_scores":
            continue
        if (child / "texts.jsonl").exists() and (child / "labels.npy").exists():
            splits.append(child)
    return sorted(splits, key=lambda p: p.name)


def compute_centroids(
    embeddings: np.ndarray,
    labels: list[int],
    num_classes: int,
) -> np.ndarray:
    dim = embeddings.shape[1]
    centroids = np.zeros((num_classes, dim), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=int)
    for emb, label in zip(embeddings, labels):
        if 0 <= label < num_classes:
            centroids[label] += emb
            counts[label] += 1
    for idx in range(num_classes):
        if counts[idx]:
            centroids[idx] /= counts[idx]
    return centroids


def write_confidence_records(
    split_name: str,
    texts: list[str],
    labels: Iterable[int],
    logits: np.ndarray,
    confidences: np.ndarray,
    label_names: list[str],
    output_dir: Path,
    metadata_base: dict[str, Any],
) -> None:
    ensure_dir(output_dir)
    output_path = output_dir / f"{split_name}_confidence_scores.jsonl"
    metadata_path = output_dir / f"{split_name}_confidence_metadata.json"
    counts = Counter(labels)
    with output_path.open("w", encoding="utf-8") as out:
        for text, label, logit_row, conf_row in zip(texts, labels, logits, confidences):
            row = {
                "text": text,
                "gold_label": int(label),
                "logits": [float(x) for x in logit_row],
                "confidences": [float(x) for x in conf_row],
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "split": split_name,
        "num_examples": len(texts),
        "label_names": label_names,
        "label_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "generated_at": datetime.utcnow().isoformat() + "Z",
        **metadata_base,
    }
    write_json(metadata_path, metadata)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-class confidence datasets.")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="Path to the experiment root (e.g., experiments/text/binary/sst2).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Ordered label names that match the dataset classes.",
    )
    parser.add_argument(
        "--label-prompts",
        nargs="+",
        help="Optional prompt texts used to describe each label.",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        help="Slug used in the artifacts filenames (default: experiment directory name with hyphens replaced by underscores).",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Path to the prompt/context file describing how confidences should be generated.",
    )
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-base-en-v1.5",
        help="Transformers encoder to reuse for embeddings.",
    )
    parser.add_argument(
        "--model-slug",
        default="bge-base-en-v1-5",
        help="Short slug that matches the naming in the artifacts directory.",
    )
    parser.add_argument("--max-length", type=int, default=64, help="Tokenizer max length.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute outputs even if they already exist.",
    )
    return parser.parse_args()


def generate_from_model(
    args: argparse.Namespace,
    splits: list[Path],
    label_names: list[str],
    dataset_slug: str,
    prompt_text: str | None,
) -> None:
    prompts = args.label_prompts or label_names
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    label_embeddings = embed_texts(
        tokenizer,
        model,
        prompts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    label_embeddings = normalize_vectors(label_embeddings)
    confidence_dir = args.experiment_root / "data" / "confidence_scores"

    embedding_dir = args.experiment_root / "artifacts" / "embeddings"
    for split_dir in splits:
        split_name = split_dir.name
        output_path = confidence_dir / f"{split_name}_confidence_scores.jsonl"
        if output_path.exists() and not args.force:
            print(f"skip {split_name}, output already exists at {output_path}")
            continue
        texts, labels = load_split_texts(split_dir)
        if not texts:
            print(f"skip {split_name}, no texts found")
            continue
        text_embeddings = embed_texts(
            tokenizer,
            model,
            texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
        )
        text_embeddings = normalize_vectors(text_embeddings)
        logits = text_embeddings @ label_embeddings.T
        logits = logits.astype(np.float32)
        confidences = softmax(logits / args.temperature)
        metadata_base = {
            "model_name": args.model_name,
            "model_slug": args.model_slug,
            "label_prompts": prompts,
            "generation_method": "label_prompt_similarity",
            "temperature": args.temperature,
            "dataset_slug": dataset_slug,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
        }
        if prompt_text:
            metadata_base["prompt_context"] = prompt_text
        write_confidence_records(
            split_name,
            texts,
            labels,
            logits,
            confidences,
            label_names,
            confidence_dir,
            metadata_base,
        )
        save_embeddings(split_name, text_embeddings, embedding_dir, dataset_slug, metadata_base)
        print(f"generated confidences for {split_name} using model embeddings")


def generate_from_embeddings(
    args: argparse.Namespace,
    splits: list[Path],
    label_names: list[str],
    dataset_slug: str,
    prompt_text: str | None,
) -> None:
    confidence_dir = args.experiment_root / "data" / "confidence_scores"
    artifact_dir = args.experiment_root / "artifacts" / "embeddings"

    for split_dir in splits:
        split_name = split_dir.name
        output_path = confidence_dir / f"{split_name}_confidence_scores.jsonl"
        if output_path.exists() and not args.force:
            print(f"skip {split_name}, output already exists at {output_path}")
            continue
        texts, labels = load_split_texts(split_dir)
        if not texts:
            print(f"skip {split_name}, no texts found for embeddings fallback")
            continue
        embeddings_path = artifact_dir / f"{dataset_slug}_{split_name}_{args.model_slug}_raw.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"cannot find embedding file {embeddings_path}")
        embeddings = np.load(embeddings_path)
        if len(embeddings) != len(texts):
            raise ValueError(
                f"embedding count {len(embeddings)} does not match text count {len(texts)} for {split_name}"
            )
        centroids = compute_centroids(embeddings, labels, len(label_names))
        centroids = normalize_vectors(centroids)
        normalized_texts = normalize_vectors(embeddings)
        logits = normalized_texts @ centroids.T
        logits = logits.astype(np.float32)
        confidences = softmax(logits / args.temperature)
        metadata_base = {
            "model_name": args.model_name,
            "model_slug": args.model_slug,
            "generation_method": "centroid_similarity",
            "temperature": args.temperature,
            "dataset_slug": dataset_slug,
            "label_prompts": args.label_prompts or label_names,
        }
        if prompt_text:
            metadata_base["prompt_context"] = prompt_text
        write_confidence_records(
            split_name,
            texts,
            labels,
            logits,
            confidences,
            label_names,
            confidence_dir,
            metadata_base,
        )
        print(f"generated confidences for {split_name} using stored embeddings")


def main() -> None:
    args = parse_arguments()
    if args.label_prompts and len(args.label_prompts) != len(args.labels):
        raise ValueError("`--labels` and `--label-prompts` must have the same length.")
    processed_dir = args.experiment_root / "data" / "processed"
    confidence_dir = args.experiment_root / "data" / "confidence_scores"
    ensure_dir(confidence_dir)
    splits = gather_split_dirs(processed_dir)
    if not splits:
        raise FileNotFoundError(f"no processed splits found under {processed_dir}")
    dataset_slug = args.dataset_slug or args.experiment_root.name.replace("-", "_")
    label_names = args.labels
    prompt_text = read_prompt_file(args.prompt_file)

    if TRANSFORMERS_AVAILABLE:
        try:
            generate_from_model(args, splits, label_names, dataset_slug, prompt_text)
            return
        except Exception as exc:  # pragma: no cover
            print(f"huggingface mode failed: {exc}; falling back to stored embeddings")
    generate_from_embeddings(args, splits, label_names, dataset_slug, prompt_text)


if __name__ == "__main__":
    main()
