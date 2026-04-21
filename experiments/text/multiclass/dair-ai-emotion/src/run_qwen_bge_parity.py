from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from embeddings import l2_normalize, mean_center
from io_utils import ensure_dir, read_json, write_json
from metrics import (
    _safe_davies_bouldin,
    _safe_silhouette,
    centroid_metrics,
    confusion_matrix_payload,
    pca_metrics,
)
from paths import EMBEDDINGS_DIR, METRICS_DIR, RUNS_DIR, ensure_base_dirs


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


def _resolve_bridge_artifact_path(bridge_run_dir: Path, rel_path: str) -> Path:
    return (bridge_run_dir / rel_path).resolve()


def _repo_root(experiment_root: Path) -> Path:
    return experiment_root.parents[3]


def _repo_relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def _label_names_from_metadata(source_run_metadata: dict) -> list[str]:
    label_to_id = source_run_metadata["dataset"]["label_to_id"]
    return [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]


def _csv_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_csv_path(repo_root: Path, source_run_metadata: dict) -> Path:
    args_csv = source_run_metadata["args"].get("csv_path")
    if args_csv:
        csv_path = Path(args_csv)
        if csv_path.exists():
            return csv_path.resolve()
        candidate = (repo_root / csv_path).resolve()
        if candidate.exists():
            return candidate
    dataset_path = source_run_metadata["dataset"].get("source_path")
    if dataset_path:
        csv_path = Path(dataset_path)
        if csv_path.exists():
            return csv_path.resolve()
        candidate = (repo_root / Path("experiments/text/multiclass/dair-ai-emotion/data/raw") / csv_path.name).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not resolve the balanced CSV path from source run metadata.")


def _load_filtered_csv_rows(csv_path: Path, label_to_id: dict[str, int]) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    raw_labels: list[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} has no header row")
        candidate_text_columns = [column for column in ("cleaned_text", "sentence", "text") if column in reader.fieldnames]
        if not candidate_text_columns:
            raise ValueError(f"CSV file {csv_path} must contain one of: cleaned_text, sentence, text")
        for row in reader:
            chosen_text = None
            for column in candidate_text_columns:
                value = str(row.get(column, "")).strip()
                if value:
                    chosen_text = value
                    break
            if not chosen_text:
                continue
            label = str(row.get("emotion", "")).strip()
            if not label:
                continue
            texts.append(chosen_text)
            raw_labels.append(label_to_id[label])
    return texts, np.array(raw_labels, dtype=np.int64)


def _reconstruct_eval_split(
    repo_root: Path,
    source_run_metadata: dict,
) -> tuple[np.ndarray, dict[str, object]]:
    args = source_run_metadata["args"]
    dataset = source_run_metadata["dataset"]
    csv_path = _resolve_csv_path(repo_root, source_run_metadata)
    label_to_id = {str(label): int(idx) for label, idx in dataset["label_to_id"].items()}
    texts, labels = _load_filtered_csv_rows(csv_path, label_to_id)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(args["test_size"]),
        random_state=int(args["seed"]),
    )
    _, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
    eval_labels = labels[test_idx]
    text_hash = hashlib.sha256()
    for idx in test_idx[:50]:
        text_hash.update(texts[int(idx)].encode("utf-8"))
        text_hash.update(b"\n")
    provenance = {
        "csv_path": str(csv_path),
        "csv_sha256": _csv_sha256(csv_path),
        "filtered_row_count": int(len(labels)),
        "eval_label_counts": [int(x) for x in np.bincount(eval_labels, minlength=len(label_to_id))],
        "eval_text_sample_sha256": text_hash.hexdigest(),
        "test_size": float(args["test_size"]),
        "seed": int(args["seed"]),
    }
    return eval_labels, provenance


def _save_embedding_variants(base_slug: str, vectors_raw: np.ndarray, model_metadata: dict) -> dict[str, str]:
    ensure_dir(EMBEDDINGS_DIR)
    raw_path = EMBEDDINGS_DIR / f"{base_slug}_raw.npy"
    l2_path = EMBEDDINGS_DIR / f"{base_slug}_l2.npy"
    centered_l2_path = EMBEDDINGS_DIR / f"{base_slug}_centered_l2.npy"
    metadata_path = EMBEDDINGS_DIR / f"{base_slug}_metadata.json"

    np.save(raw_path, vectors_raw)
    np.save(l2_path, l2_normalize(vectors_raw))
    np.save(centered_l2_path, l2_normalize(mean_center(vectors_raw)))
    write_json(metadata_path, model_metadata)

    return {
        "raw": str(raw_path),
        "l2": str(l2_path),
        "centered_l2": str(centered_l2_path),
        "metadata": str(metadata_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Qwen parity runs and embedding variants for dair-ai/emotion.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "qwen-bge-parity.json",
    )
    args = parser.parse_args()

    ensure_base_dirs()
    config = read_json(args.config)
    experiment_root = Path(__file__).resolve().parents[1]
    repo_root = _repo_root(experiment_root)

    bridge_run_dir = experiment_root / "runs" / config["bridge_run"]
    bridge_artifacts = read_json(bridge_run_dir / "artifacts.json")
    bridge_config = read_json(bridge_run_dir / "config.json")
    source_run_metadata = read_json(_resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["run_metadata"]))
    label_names = _label_names_from_metadata(source_run_metadata)

    eval_embeddings_path = _resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["analysis"]["eval_embeddings"])
    eval_logits_path = _resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["analysis"]["eval_logits"])
    eval_metrics_path = _resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["analysis"]["eval_metrics"])
    train_metrics_path = _resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["train_metrics"])

    eval_embeddings_raw = np.load(eval_embeddings_path)
    eval_logits = np.load(eval_logits_path)
    eval_metrics = read_json(eval_metrics_path)
    eval_labels, split_provenance = _reconstruct_eval_split(repo_root, source_run_metadata)
    if len(eval_labels) != len(eval_embeddings_raw):
        raise ValueError(
            f"Reconstructed eval labels length {len(eval_labels)} does not match embedding rows {len(eval_embeddings_raw)}."
        )

    base_slug = config["embedding_export"]["base_slug"]
    output_bundle_slug = config["embedding_export"]["output_bundle_slug"]
    model_metadata = {
        "artifact_split": "heldout_eval",
        "batching": {
            "padding_side": "right",
            "pad_token_policy": "eos_as_pad_token",
        },
        "bridge_run": config["bridge_run"],
        "embedding_extraction": config["model"]["embedding_extraction"],
        "label_names": label_names,
        "label_note": config["evaluation"]["label_mismatch_note"],
        "model_name": config["model"]["name"],
        "model_slug": config["model"]["slug"],
        "num_examples": int(eval_embeddings_raw.shape[0]),
        "source_eval_only": bool(config["evaluation"]["source_eval_only"]),
        "source_run_id": bridge_config["source_run_id"],
        "source_run_root": bridge_artifacts["source_run_root"],
        "split_provenance": split_provenance,
        "vectors_shape": [int(x) for x in eval_embeddings_raw.shape],
    }
    embedding_paths = _save_embedding_variants(base_slug, eval_embeddings_raw, model_metadata)

    predicted_labels = np.argmax(eval_logits, axis=1)
    prediction_summary = {
        "accuracy": float(accuracy_score(eval_labels, predicted_labels)),
        "macro_f1": float(f1_score(eval_labels, predicted_labels, average="macro")),
        "source_eval_metrics_accuracy": float(eval_metrics["accuracy"]),
    }

    bundle_summary = {
        "bridge_run": config["bridge_run"],
        "embedding_export": {
            "base_slug": base_slug,
            "paths": {key: _repo_relative(Path(path), repo_root) for key, path in embedding_paths.items()},
        },
        "generated_runs": [],
        "limits": {
            "source_eval_only": bool(config["evaluation"]["source_eval_only"]),
            "skipped_bge_style_probes": [
                "logistic_regression_probe",
                "knn_probe",
            ],
            "reason": "The tracked Qwen bridge/source artifacts expose held-out embeddings and logits, but not a train/validation embedding bundle comparable to the earlier BGE embedding-generation stage.",
            "label_mismatch_note": config["evaluation"]["label_mismatch_note"],
        },
        "source_artifacts": {
            "bridge_run": _repo_relative(bridge_run_dir, repo_root),
            "eval_embeddings": _repo_relative(eval_embeddings_path, repo_root),
            "eval_logits": _repo_relative(eval_logits_path, repo_root),
            "eval_metrics": _repo_relative(eval_metrics_path, repo_root),
            "train_metrics": _repo_relative(train_metrics_path, repo_root),
        },
    }

    variant_vectors = {
        "raw": eval_embeddings_raw,
        "l2": l2_normalize(eval_embeddings_raw),
        "centered_l2": l2_normalize(mean_center(eval_embeddings_raw)),
    }

    for offset, variant_cfg in enumerate(config["variants"]):
        variant_name = variant_cfg["name"]
        embedding_key = variant_cfg["embedding_key"]
        run_id = f"run-{config['run_start_index'] + offset:03d}-{config['model']['slug']}-bge-parity-{variant_name}"
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

        _log_stage(run_dir, "load", f"loading Qwen parity embedding variant '{embedding_key}'")
        vectors = variant_vectors[embedding_key]

        summary = {
            "source_model_prediction": prediction_summary,
            "comparability": {
                "bge_style_train_probe_available": False,
                "embedding_variant": embedding_key,
                "label_note": config["evaluation"]["label_mismatch_note"],
                "source_eval_only": bool(config["evaluation"]["source_eval_only"]),
            },
        }

        _log_stage(run_dir, "metric-start", "computing centroid metrics")
        summary["geometry"] = centroid_metrics(vectors, eval_labels)
        write_json(run_dir / "metrics" / "summary.json", summary)
        _log_stage(run_dir, "metric-done", "finished centroid metrics")

        _log_stage(run_dir, "metric-start", "computing cluster diagnostics")
        summary["cluster"] = {
            "silhouette": _safe_silhouette(vectors, eval_labels),
            "davies_bouldin": _safe_davies_bouldin(vectors, eval_labels),
        }
        write_json(run_dir / "metrics" / "summary.json", summary)
        _log_stage(run_dir, "metric-done", "finished cluster diagnostics")

        _log_stage(run_dir, "metric-start", "computing PCA spectrum metrics")
        summary["pca"] = pca_metrics(vectors)
        write_json(run_dir / "metrics" / "summary.json", summary)
        _log_stage(run_dir, "metric-done", "finished PCA spectrum metrics")

        _log_stage(run_dir, "metric-start", "writing model prediction confusion matrix")
        confusion = confusion_matrix_payload(
            list(range(len(label_names))),
            eval_labels,
            predicted_labels,
            label_names,
        )
        write_json(run_dir / "metrics" / "summary.json", summary)
        write_json(run_dir / "metrics" / "confusion-matrix.json", confusion)
        write_json(
            run_dir / "artifacts.json",
            {
                "embedding_variant": embedding_key,
                "eval_embeddings": {
                    "raw": embedding_paths["raw"],
                    "l2": embedding_paths["l2"],
                    "centered_l2": embedding_paths["centered_l2"],
                    "metadata": embedding_paths["metadata"],
                },
                "source_bridge_run": str(bridge_run_dir),
                "source_run_artifacts": {
                    "eval_embeddings": str(eval_embeddings_path),
                    "eval_logits": str(eval_logits_path),
                    "eval_metrics": str(eval_metrics_path),
                    "train_metrics": str(train_metrics_path),
                },
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
        _log_stage(run_dir, "complete", f"finished Qwen parity variant '{variant_name}'")
        bundle_summary["generated_runs"].append(
            {
                "run_id": run_id,
                "variant_name": variant_name,
                "summary_path": _repo_relative(run_dir / "metrics" / "summary.json", repo_root),
            }
        )

    write_json(METRICS_DIR / output_bundle_slug / "summary.json", bundle_summary)


if __name__ == "__main__":
    main()
