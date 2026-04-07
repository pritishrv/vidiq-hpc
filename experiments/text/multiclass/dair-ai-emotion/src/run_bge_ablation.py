from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from io_utils import ensure_dir, read_json, write_json
from metrics import (
    _safe_davies_bouldin,
    _safe_silhouette,
    centroid_metrics,
    confusion_matrix_payload,
    knn_probe,
    logistic_probe,
    pca_metrics,
)
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


def _write_progress(run_dir: Path, payload: dict) -> None:
    write_json(run_dir / "progress.json", payload)


def _load_array(path_str: str) -> np.ndarray:
    return np.load(path_str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BGE embedding ablations for dair-ai/emotion.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "bge-ablation-stage.json",
    )
    args = parser.parse_args()

    ensure_base_dirs()
    config = read_json(args.config)
    generation_artifacts = read_json(
        Path(__file__).resolve().parents[1]
        / "runs"
        / "run-001-bge-base-en-v1-5-embedding-generation"
        / "artifacts.json"
    )

    dataset_cfg = config["dataset"]
    train_labels = _load_array(
        str(Path(__file__).resolve().parents[1] / "data" / "processed" / dataset_cfg["train_split"] / "labels.npy")
    )
    val_labels = _load_array(
        str(Path(__file__).resolve().parents[1] / "data" / "processed" / dataset_cfg["validation_split"] / "labels.npy")
    )
    label_names = dataset_cfg["label_names"]
    unique_labels = sorted(int(x) for x in np.unique(val_labels))
    eval_cfg = config["evaluation"]
    model_cfg = config["model"]

    for idx, variant_cfg in enumerate(tqdm(config["variants"], desc="Multiclass BGE variants", unit="variant"), start=1):
        variant_name = variant_cfg["name"]
        embedding_key = variant_cfg["embedding_key"]
        run_id = f"run-{idx + 100:03d}-{model_cfg['slug']}-{variant_name}"
        run_dir = _run_dir(run_id)
        ensure_dir(run_dir / "metrics")
        ensure_dir(run_dir / "logs")
        _save_run_config(run_dir, args.config, config)
        _write_progress(
            run_dir,
            {
                "status": "started",
                "variant_name": variant_name,
                "embedding_key": embedding_key,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

        _log_stage(run_dir, "load", f"loading embedding variant '{embedding_key}'")
        train_x = _load_array(generation_artifacts["train_embeddings"][embedding_key])
        val_x = _load_array(generation_artifacts["validation_embeddings"][embedding_key])

        summary: dict[str, object] = {}

        _log_stage(run_dir, "metric-start", "running multinomial logistic regression")
        logistic_metrics, logistic_preds = logistic_probe(
            train_x,
            train_labels,
            val_x,
            val_labels,
            eval_cfg["logistic_max_iter"],
        )
        summary["logistic_regression"] = logistic_metrics
        write_json(run_dir / "metrics" / "summary.json", summary)
        _log_stage(run_dir, "metric-done", "finished multinomial logistic regression")

        _log_stage(run_dir, "metric-start", "running kNN probes")
        summary["knn"] = knn_probe(
            train_x,
            train_labels,
            val_x,
            val_labels,
            eval_cfg["knn_k_values"],
        )
        write_json(run_dir / "metrics" / "summary.json", summary)
        _log_stage(run_dir, "metric-done", "finished kNN probes")

        _log_stage(run_dir, "metric-start", "computing centroid metrics")
        summary["geometry"] = centroid_metrics(val_x, val_labels)
        write_json(run_dir / "metrics" / "summary.json", summary)
        _log_stage(run_dir, "metric-done", "finished centroid metrics")

        _log_stage(run_dir, "metric-start", "computing cluster diagnostics")
        summary["cluster"] = {
            "silhouette": _safe_silhouette(val_x, val_labels),
            "davies_bouldin": _safe_davies_bouldin(val_x, val_labels),
        }
        write_json(run_dir / "metrics" / "summary.json", summary)
        _log_stage(run_dir, "metric-done", "finished cluster diagnostics")

        _log_stage(run_dir, "metric-start", "computing PCA spectrum metrics")
        summary["pca"] = pca_metrics(val_x)
        write_json(run_dir / "metrics" / "summary.json", summary)
        _log_stage(run_dir, "metric-done", "finished PCA spectrum metrics")

        _log_stage(run_dir, "metric-start", "computing confusion matrix")
        confusion = confusion_matrix_payload(unique_labels, val_labels, logistic_preds, label_names)
        write_json(run_dir / "metrics" / "summary.json", summary)
        write_json(run_dir / "metrics" / "confusion-matrix.json", confusion)
        write_json(
            run_dir / "artifacts.json",
            {
                "train_split_used": dataset_cfg["train_split"],
                "validation_split_used": dataset_cfg["validation_split"],
                "embedding_variant": embedding_key,
                "train_embeddings": generation_artifacts["train_embeddings"],
                "validation_embeddings": generation_artifacts["validation_embeddings"],
            },
        )
        _write_progress(
            run_dir,
            {
                "status": "completed",
                "variant_name": variant_name,
                "embedding_key": embedding_key,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        _log_stage(run_dir, "complete", f"finished variant '{variant_name}'")


if __name__ == "__main__":
    main()
