from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_CLASSES = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
DEFAULT_EXCLUDED = ["disgust"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def gather_split_files(split_root: Path, included_classes: list[str]) -> dict[str, list[Path]]:
    per_class: dict[str, list[Path]] = {}
    for class_name in included_classes:
        class_dir = split_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        per_class[class_name] = sorted(
            path for path in class_dir.iterdir() if path.is_file() and not path.name.startswith(".")
        )
    return per_class


def choose_balanced_subset(
    per_class_files: dict[str, list[Path]],
    seed: int,
) -> tuple[dict[str, list[Path]], int]:
    target = min(len(paths) for paths in per_class_files.values())
    rng = np.random.default_rng(seed)
    balanced: dict[str, list[Path]] = {}
    for class_name, paths in per_class_files.items():
        if len(paths) == target:
            selected = list(paths)
        else:
            indices = np.sort(rng.choice(len(paths), size=target, replace=False))
            selected = [paths[idx] for idx in indices]
        balanced[class_name] = selected
    return balanced, target


def write_split_manifest(
    dataset_root: Path,
    split_name: str,
    split_files: dict[str, list[Path]],
    label_to_id: dict[str, int],
    output_root: Path,
) -> dict[str, object]:
    split_dir = ensure_dir(output_root / "data" / "processed" / split_name)
    labels: list[int] = []
    manifest_path = split_dir / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as fh:
        for class_name in sorted(split_files.keys(), key=lambda name: label_to_id[name]):
            label_id = label_to_id[class_name]
            for image_path in split_files[class_name]:
                rel_path = image_path.relative_to(dataset_root)
                payload = {
                    "source_path": str(image_path.resolve()),
                    "relative_path": str(rel_path),
                    "split": split_name,
                    "label": class_name,
                    "label_id": label_id,
                    "filename": image_path.name,
                }
                fh.write(json.dumps(payload) + "\n")
                labels.append(label_id)

    labels_arr = np.asarray(labels, dtype=np.int64)
    np.save(split_dir / "labels.npy", labels_arr)
    split_metadata = {
        "split": split_name,
        "count": int(len(labels_arr)),
        "per_class_counts": {
            class_name: int(len(split_files[class_name]))
            for class_name in sorted(split_files.keys(), key=lambda name: label_to_id[name])
        },
        "manifest": str(manifest_path),
        "labels_path": str(split_dir / "labels.npy"),
    }
    (split_dir / "metadata.json").write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")
    return split_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a balanced FER2013 dataset without the disgust class.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/data/Image_Dataset/FER2013"),
        help="FER2013 dataset root containing train/ and test/ class folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/embeddings_field/image/emotions/fer2013_balanced_no_disgust_clip_vitl14"),
        help="Repo-local output root for processed manifests and metadata.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Class names to retain, in the desired label-id order.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=DEFAULT_EXCLUDED,
        help="Class names to exclude explicitly from discovery/documentation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = ensure_dir(args.output_root.expanduser().resolve())

    included_classes = [name for name in args.classes if name not in set(args.exclude)]
    label_to_id = {name: idx for idx, name in enumerate(included_classes)}

    split_summaries: dict[str, dict[str, object]] = {}
    for split_name in ("train", "test"):
        split_root = dataset_root / split_name
        per_class_files = gather_split_files(split_root, included_classes)
        balanced_files, target_count = choose_balanced_subset(per_class_files, seed=args.seed)
        split_metadata = write_split_manifest(dataset_root, split_name, balanced_files, label_to_id, output_root)
        split_metadata["balanced_target_per_class"] = int(target_count)
        split_summaries[split_name] = split_metadata

    metadata = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "seed": int(args.seed),
        "excluded_classes": list(args.exclude),
        "classes": included_classes,
        "label_to_id": label_to_id,
        "splits": split_summaries,
    }
    (output_root / "data" / "processed" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared balanced FER2013 dataset under {output_root}")


if __name__ == "__main__":
    main()
