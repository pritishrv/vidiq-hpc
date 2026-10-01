from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from metrics import centroid_summary, ensure_dir, load_metadata, load_split_embeddings, load_split_labels, write_json


def remove_direction(vectors: np.ndarray, direction: np.ndarray) -> np.ndarray:
    unit = direction / (np.linalg.norm(direction) + 1e-12)
    projection = (vectors @ unit[:, np.newaxis]) @ unit[np.newaxis, :]
    return vectors - projection


def load_eval_setup(experiment_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    train_x = load_split_embeddings(experiment_root, "train")
    train_y = load_split_labels(experiment_root, "train")
    test_embeddings_path = experiment_root / "artifacts" / "embeddings" / "test_raw.npy"
    test_labels_path = experiment_root / "data" / "processed" / "test" / "labels.npy"

    if test_embeddings_path.exists() and test_labels_path.exists():
        test_x = np.load(test_embeddings_path)
        test_y = np.load(test_labels_path)
        provenance = {
            "evaluation_mode": "heldout_test_split",
            "train_examples": int(train_x.shape[0]),
            "test_examples": int(test_x.shape[0]),
        }
        return train_x, train_y, test_x, test_y, provenance

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, eval_idx = next(splitter.split(train_x, train_y))
    provenance = {
        "evaluation_mode": "stratified_holdout_from_train",
        "seed": 42,
        "test_size": 0.2,
        "train_examples": int(len(train_idx)),
        "eval_examples": int(len(eval_idx)),
    }
    return train_x[train_idx], train_y[train_idx], train_x[eval_idx], train_y[eval_idx], provenance


def stratified_subsample(vectors: np.ndarray, labels: np.ndarray, max_examples: int | None) -> tuple[np.ndarray, np.ndarray, dict | None]:
    if max_examples is None or len(vectors) <= max_examples:
        return vectors, labels, None

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_examples, random_state=42)
    keep_idx, _ = next(splitter.split(vectors, labels))
    payload = {
        "original_examples": int(len(vectors)),
        "sampled_examples": int(len(keep_idx)),
        "seed": 42,
    }
    return vectors[keep_idx], labels[keep_idx], payload


def run_erasure(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    label_names: list[str],
    n_steps: int,
    checkpoints: set[int],
    max_iter: int,
) -> tuple[dict, dict]:
    x_train = train_x.copy()
    x_eval = eval_x.copy()
    results = {"accuracies": [], "weights": []}
    checkpoint_rdms: dict[str, dict] = {}

    clf = LogisticRegression(max_iter=max_iter, random_state=42, solver="liblinear")
    clf.fit(x_train, train_y)
    baseline_acc = float(accuracy_score(eval_y, clf.predict(x_eval)))
    results["accuracies"].append(baseline_acc)
    checkpoint_rdms["0"] = centroid_summary(x_eval, eval_y, label_names)

    for step in range(1, n_steps + 1):
        if step == 1 or step % 5 == 0 or step == n_steps:
            print(f"Erasure step {step}/{n_steps}", flush=True)
        clf.fit(x_train, train_y)
        _, singular_values, vh = svd(clf.coef_, full_matrices=False)
        top_direction = vh[0]
        results["weights"].append(float(singular_values[0]))
        x_train = remove_direction(x_train, top_direction)
        x_eval = remove_direction(x_eval, top_direction)

        clf_next = LogisticRegression(max_iter=max_iter, random_state=42, solver="liblinear")
        clf_next.fit(x_train, train_y)
        acc = float(accuracy_score(eval_y, clf_next.predict(x_eval)))
        results["accuracies"].append(acc)

        if step in checkpoints:
            checkpoint_rdms[str(step)] = centroid_summary(x_eval, eval_y, label_names)

    return results, checkpoint_rdms


def plot_curves(accuracies: list[float], output_root: Path, chance_level: float) -> None:
    top_k = min(25, len(accuracies) - 1)

    plt.figure(figsize=(10, 6))
    plt.plot(range(top_k + 1), accuracies[: top_k + 1], marker="o", linewidth=2)
    plt.axhline(chance_level, color="black", linestyle=":", label="chance")
    plt.title("Image Signal Cliff: First 25 Directions Removed")
    plt.xlabel("Directions removed")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "cliff_zoom_top25.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(accuracies)), accuracies, linewidth=2)
    plt.axhline(chance_level, color="black", linestyle=":", label="chance")
    plt.title("Image Directional Signal Erasure")
    plt.xlabel("Directions removed")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "full_erasure.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run image embedding directional erasure analysis.")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("experiments/embeddings_field/image/emotions/fer2013_balanced_no_disgust_clip_vitl14"),
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-train-examples", type=int, default=6000)
    parser.add_argument("--max-eval-examples", type=int, default=2000)
    parser.add_argument("--logistic-max-iter", type=int, default=300)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    output_root = ensure_dir(args.output_root or experiment_root / "reports" / "phase3")
    metadata = load_metadata(experiment_root)
    label_names = metadata["classes"]

    train_x, train_y, eval_x, eval_y, provenance = load_eval_setup(experiment_root)
    train_x, train_y, train_sample = stratified_subsample(train_x, train_y, args.max_train_examples)
    eval_x, eval_y, eval_sample = stratified_subsample(eval_x, eval_y, args.max_eval_examples)
    if train_sample:
        provenance["train_subsample"] = train_sample
    if eval_sample:
        provenance["eval_subsample"] = eval_sample
    checkpoints = {0, 1, 2, 5, 10, 20, 50, args.steps}
    results, checkpoint_rdms = run_erasure(
        train_x,
        train_y,
        eval_x,
        eval_y,
        label_names,
        args.steps,
        checkpoints,
        args.logistic_max_iter,
    )

    chance_level = 1.0 / max(len(np.unique(eval_y)), 1)
    plot_curves(results["accuracies"], output_root, chance_level)

    write_json(
        output_root / "retention_metrics.json",
        {
            "provenance": provenance,
            "steps": int(args.steps),
            "chance_level": chance_level,
            **results,
        },
    )
    write_json(output_root / "rdm_checkpoints.json", checkpoint_rdms)
    print(f"Saved erasure analysis to {output_root}")


if __name__ == "__main__":
    main()
