from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

from data import prepare_dataset
from embeddings import embed_texts, save_embedding_variants
from io_utils import ensure_dir, read_json, write_json
from paths import RUNS_DIR, ensure_base_dirs


def _run_dir(run_id: str) -> Path:
    return ensure_dir(RUNS_DIR / run_id)


def _save_run_config(run_dir: Path, config_path: Path, config: dict) -> None:
    shutil.copy2(config_path, run_dir / "config.json")
    write_json(
        run_dir / "run-metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path),
            "experiment_name": config["experiment_name"],
        },
    )


def _log_stage(run_dir: Path, stage: str, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {stage}: {message}"
    print(line, flush=True)
    log_path = run_dir / "logs" / "stage-log.txt"
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BGE embeddings for dair-ai/emotion.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "bge-embedding-generation.json",
    )
    args = parser.parse_args()

    ensure_base_dirs()
    config = read_json(args.config)
    prepared = prepare_dataset(config)

    dataset_cfg = config["dataset"]
    train_split = dataset_cfg["train_split"]
    val_split = dataset_cfg["validation_split"]
    model_cfg = config["model"]
    embedding_cfg = config["embedding"]

    run_id = f"run-001-{model_cfg['slug']}-embedding-generation"
    run_dir = _run_dir(run_id)
    ensure_dir(run_dir / "logs")
    _save_run_config(run_dir, args.config, config)

    _log_stage(run_dir, "embed-start", f"embedding {train_split} split with {model_cfg['name']}")
    train_raw, train_meta = embed_texts(
        prepared[train_split]["texts"],
        model_cfg["name"],
        embedding_cfg["max_length"],
        embedding_cfg["batch_size"],
    )
    _log_stage(run_dir, "embed-done", f"finished {train_split} split")

    _log_stage(run_dir, "embed-start", f"embedding {val_split} split with {model_cfg['name']}")
    val_raw, val_meta = embed_texts(
        prepared[val_split]["texts"],
        model_cfg["name"],
        embedding_cfg["max_length"],
        embedding_cfg["batch_size"],
    )
    _log_stage(run_dir, "embed-done", f"finished {val_split} split")

    train_paths = save_embedding_variants(train_split, model_cfg["slug"], train_raw, train_meta)
    val_paths = save_embedding_variants(val_split, model_cfg["slug"], val_raw, val_meta)

    write_json(
        run_dir / "artifacts.json",
        {
            "train_split_used": train_split,
            "validation_split_used": val_split,
            "train_embeddings": train_paths,
            "validation_embeddings": val_paths,
        },
    )
    _log_stage(run_dir, "complete", "saved multiclass embedding artifacts")


if __name__ == "__main__":
    main()
