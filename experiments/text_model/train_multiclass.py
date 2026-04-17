from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class RunConfig:
    root: Path
    archive_root: Path
    model_dir: Path
    tokenizer_dir: Path
    archive_model_dir: Path
    archive_tokenizer_dir: Path
    analysis_dir: Path
    embeddings_dir: Path
    logits_dir: Path
    metrics_path: Path
    centroid_path: Path
    train_metrics_path: Path
    run_metadata_path: Path


@dataclass
class DatasetInfo:
    source_type: str
    source_path: Path
    text_column: str
    label_column: str
    label_to_id: dict[str, int]


class EmotionTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray) -> None:
        self.texts = texts
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], int(self.labels[idx])


def collate_fn_builder(tokenizer: AutoTokenizer, max_length: int):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def collate(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts, labels = zip(*batch)
        tokens = tokenizer(
            list(texts),
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_tensor = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if "token_type_ids" in tokens:
            batch_tensor["token_type_ids"] = tokens["token_type_ids"]
        return batch_tensor

    return collate


class EmotionEmbeddingClassifier(nn.Module):
    def __init__(self, backbone: str, num_labels: int, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.transformer = AutoModelForCausalLM.from_pretrained(
            backbone,
            trust_remote_code=True,
            output_hidden_states=True,
        )
        if freeze_backbone:
            for param in self.transformer.parameters():
                param.requires_grad = False
        hidden_size = self.transformer.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1][:, -1, :].to(torch.float32)
        logits = self.projection(hidden)
        return hidden, logits


def load_texts_and_labels_from_jsonl(base_path: Path) -> tuple[list[str], np.ndarray, DatasetInfo]:
    texts_path = base_path / "texts.jsonl"
    if not texts_path.exists():
        raise FileNotFoundError(f"Missing texts file {texts_path}")
    texts: list[str] = []
    labels: list[int] = []
    with texts_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            texts.append(payload["text"])
            labels.append(int(payload.get("label", -1)))
    label_ids = sorted(set(labels))
    dataset_info = DatasetInfo(
        source_type="jsonl",
        source_path=texts_path.resolve(),
        text_column="text",
        label_column="label",
        label_to_id={str(label): int(label) for label in label_ids},
    )
    return texts, np.array(labels, dtype=np.int64), dataset_info


def load_texts_and_labels_from_csv(csv_path: Path) -> tuple[list[str], np.ndarray, DatasetInfo]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file {csv_path}")
    texts: list[str] = []
    raw_labels: list[str] = []
    text_column: str | None = None
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} has no header row")
        if "emotion" not in reader.fieldnames:
            raise ValueError(f"CSV file {csv_path} must contain an 'emotion' column")
        candidate_text_columns = [column for column in ("cleaned_text", "sentence", "text") if column in reader.fieldnames]
        if not candidate_text_columns:
            raise ValueError(f"CSV file {csv_path} must contain one of: cleaned_text, sentence, text")
        for row in reader:
            chosen_text: str | None = None
            for column in candidate_text_columns:
                value = str(row.get(column, "")).strip()
                if value:
                    chosen_text = value
                    text_column = column
                    break
            if not chosen_text:
                continue
            label = str(row.get("emotion", "")).strip()
            if not label:
                continue
            texts.append(chosen_text)
            raw_labels.append(label)
    unique_labels = sorted(set(raw_labels))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    labels = np.array([label_to_id[label] for label in raw_labels], dtype=np.int64)
    dataset_info = DatasetInfo(
        source_type="csv",
        source_path=csv_path.resolve(),
        text_column=text_column or candidate_text_columns[0],
        label_column="emotion",
        label_to_id=label_to_id,
    )
    return texts, labels, dataset_info


def load_texts_and_labels(data_root: Path | None, csv_path: Path | None) -> tuple[list[str], np.ndarray, DatasetInfo]:
    if csv_path is not None:
        return load_texts_and_labels_from_csv(csv_path)
    if data_root is not None:
        return load_texts_and_labels_from_jsonl(data_root)
    raise ValueError("Either data_root or csv_path must be provided")


def build_splits(labels: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return train_idx, test_idx


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: EmotionEmbeddingClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in tqdm(dataloader, desc="train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        embeddings, logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        loss = F.cross_entropy(logits, batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def evaluate(
    model: EmotionEmbeddingClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    all_embeddings: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_labels: list[int] = []
    all_texts: list[str] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            embeddings, logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            all_embeddings.append(embeddings.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
    embeddings = np.concatenate(all_embeddings, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {
        "embeddings": embeddings,
        "logits": logits,
        "probabilities": probs,
        "labels": labels,
        "predictions": preds,
        "accuracy": acc,
    }


def save_evaluation_artifacts(run_config: RunConfig, eval_output: dict[str, object]) -> dict[str, object]:
    run_config.analysis_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_config.embeddings_dir / "eval_embeddings.npy", eval_output["embeddings"])
    np.save(run_config.logits_dir / "eval_logits.npy", eval_output["logits"])
    metrics = {
        "accuracy": float(eval_output["accuracy"]),
        "dataset_size": len(eval_output["labels"]),
    }
    with run_config.metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    centroids = []
    for cls in sorted(set(eval_output["labels"])):
        centroid = np.mean(eval_output["embeddings"][eval_output["labels"] == cls], axis=0)
        centroids.append({"class": int(cls), "centroid": centroid.tolist()})
    with run_config.centroid_path.open("w", encoding="utf-8") as fh:
        json.dump(centroids, fh, indent=2)
    return metrics


def create_run_config(base_root: Path, archive_root: Path, run_name: str) -> RunConfig:
    run_root = (base_root / run_name).resolve()
    archive_run_root = (archive_root / run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "analysis").mkdir(parents=True, exist_ok=True)
    archive_run_root.mkdir(parents=True, exist_ok=True)
    return RunConfig(
        root=run_root,
        archive_root=archive_run_root,
        model_dir=run_root / "model",
        tokenizer_dir=run_root / "tokenizer",
        archive_model_dir=archive_run_root / "model",
        archive_tokenizer_dir=archive_run_root / "tokenizer",
        analysis_dir=run_root / "analysis",
        embeddings_dir=run_root / "analysis",
        logits_dir=run_root / "analysis",
        metrics_path=run_root / "analysis" / "eval_metrics.json",
        centroid_path=run_root / "analysis" / "centroids.json",
        train_metrics_path=run_root / "train_metrics.json",
        run_metadata_path=run_root / "run_metadata.json",
    )


def save_checkpoint(model: EmotionEmbeddingClassifier, run_config: RunConfig) -> None:
    run_config.archive_model_dir.mkdir(parents=True, exist_ok=True)
    model.transformer.save_pretrained(run_config.archive_model_dir)
    torch.save(model.projection.state_dict(), run_config.archive_model_dir / "head.pt")


def create_directory_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"Cannot create symlink at {link_path}: path already exists and is not a symlink")
    link_path.symlink_to(target_path, target_is_directory=True)


def try_get_git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def save_run_metadata(run_config: RunConfig, args: argparse.Namespace, dataset_info: DatasetInfo) -> None:
    slurm_job_name = os.getenv("SLURM_JOB_NAME")
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    metadata = {
        "run_name": run_config.root.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": try_get_git_commit(),
        "slurm_job_name": slurm_job_name,
        "slurm_job_id": slurm_job_id,
        "args": jsonable_args(args),
        "dataset": {
            "source_type": dataset_info.source_type,
            "source_path": str(dataset_info.source_path),
            "text_column": dataset_info.text_column,
            "label_column": dataset_info.label_column,
            "num_labels": len(dataset_info.label_to_id),
            "label_to_id": dataset_info.label_to_id,
        },
        "local_run_root": str(run_config.root),
        "archive_run_root": str(run_config.archive_root),
        "analysis_dir": str(run_config.analysis_dir),
        "model_symlink": str(run_config.model_dir),
        "model_target": str(run_config.archive_model_dir),
        "tokenizer_symlink": str(run_config.tokenizer_dir),
        "tokenizer_target": str(run_config.archive_tokenizer_dir),
    }
    with run_config.run_metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def get_default_run_name(args: argparse.Namespace) -> str:
    slurm_job_name = os.getenv("SLURM_JOB_NAME")
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    if slurm_job_name and slurm_job_id:
        return f"{slurm_job_name}_{slurm_job_id}"
    model_tag = Path(str(args.model_path).rstrip("/")).name or "text-model"
    return f"{model_tag}-{args.num_epochs}e-{args.batch_size}b-{random.randint(1000,9999)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train qwen3-based emotion classifier and log 768D embeddings.")
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root containing processed text/labels.")
    parser.add_argument("--csv-path", type=Path, default=None, help="Raw CSV dataset path for direct training.")
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-1.7B",
        help="Local checkpoint path or Hugging Face model id for the Qwen backbone.",
    )
    parser.add_argument("--run-root", type=Path, default=Path("experiments/text_model/runs"))
    parser.add_argument("--archive-root", type=Path, default=Path("/users/aczd097/archive/vidiq-hpc/text_model"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_root is not None:
        args.data_root = args.data_root.expanduser()
    if args.csv_path is not None:
        args.csv_path = args.csv_path.expanduser()
    expanded_model_path = Path(args.model_path).expanduser()
    if expanded_model_path.exists():
        args.model_path = str(expanded_model_path)
    args.run_root = args.run_root.expanduser()
    args.archive_root = args.archive_root.expanduser()
    if args.data_root is None and args.csv_path is None:
        raise ValueError("Pass either --data-root or --csv-path")
    set_seed(args.seed)
    dataset_root = args.data_root / "data" / "processed" / "train" if args.data_root is not None else None
    texts, labels, dataset_info = load_texts_and_labels(dataset_root, args.csv_path)
    train_idx, test_idx = build_splits(labels, args.test_size, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    collate_fn = collate_fn_builder(tokenizer, args.max_length)
    train_texts = [texts[i] for i in train_idx]
    train_labels = labels[train_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = labels[test_idx]
    train_loader = DataLoader(
        EmotionTextDataset(train_texts, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        EmotionTextDataset(test_texts, test_labels),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    model = EmotionEmbeddingClassifier(
        str(args.model_path),
        num_labels=len(dataset_info.label_to_id),
        freeze_backbone=args.freeze_backbone,
    )
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    run_name = args.run_name or get_default_run_name(args)
    run_config = create_run_config(args.run_root, args.archive_root, run_name)
    metrics_history: list[dict[str, object]] = []
    for epoch in range(1, args.num_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        eval_output = evaluate(model, eval_loader, device)
        metrics_history.append({"epoch": epoch, "loss": loss, "accuracy": eval_output["accuracy"]})
        print(f"Epoch {epoch}: loss={loss:.4f} acc={eval_output['accuracy']:.4f}")
    with run_config.train_metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_history, fh, indent=2)
    save_checkpoint(model, run_config)
    final_eval = evaluate(model, eval_loader, device)
    metrics = save_evaluation_artifacts(run_config, final_eval)
    run_config.archive_tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(run_config.archive_tokenizer_dir)
    create_directory_symlink(run_config.model_dir, run_config.archive_model_dir)
    create_directory_symlink(run_config.tokenizer_dir, run_config.archive_tokenizer_dir)
    save_run_metadata(run_config, args, dataset_info)
    print(
        f"Run completed. metrics={metrics}. local artifacts stored under {run_config.root}; "
        f"archived model artifacts stored under {run_config.archive_root}"
    )


if __name__ == "__main__":
    main()
