from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def gather_class_files(dataset_root: Path) -> dict[str, list[Path]]:
    # The animal dataset seems to have folders directly under 'animals'
    animals_dir = dataset_root / "animals" / "animals"
    if not animals_dir.exists():
        # Try alternative structure if it was nested differently
        animals_dir = dataset_root / "animals"
        
    if not animals_dir.exists():
        raise FileNotFoundError(f"Missing animals directory: {animals_dir}")
        
    per_class: dict[str, list[Path]] = {}
    # Get all subdirectories (classes)
    class_dirs = [d for d in animals_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        per_class[class_name] = sorted(
            path for path in class_dir.iterdir() if path.is_file() and not path.name.startswith(".")
            and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
    return per_class


def write_manifest(
    dataset_root: Path,
    all_files: dict[str, list[Path]],
    label_to_id: dict[str, int],
    output_root: Path,
    split_name: str = "full",
) -> dict[str, object]:
    split_dir = ensure_dir(output_root / "data" / "processed" / split_name)
    labels: list[int] = []
    manifest_path = split_dir / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as fh:
        for class_name in sorted(all_files.keys(), key=lambda name: label_to_id.get(name, 999)):
            if class_name not in label_to_id:
                continue
            label_id = label_to_id[class_name]
            for image_path in all_files[class_name]:
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
            class_name: int(len(all_files[class_name]))
            for class_name in sorted(all_files.keys()) if class_name in label_to_id
        },
        "manifest": str(manifest_path),
        "labels_path": str(split_dir / "labels.npy"),
    }
    (split_dir / "metadata.json").write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")
    return split_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Animal Dataset manifests.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/data/Image_Dataset/animal_dataset"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/embeddings_field/image/animals"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = ensure_dir(args.output_root.expanduser().resolve())

    per_class_files = gather_class_files(dataset_root)
    classes = sorted(per_class_files.keys())
    label_to_id = {name: idx for idx, name in enumerate(classes)}

    split_metadata = write_manifest(dataset_root, per_class_files, label_to_id, output_root)

    metadata = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "classes": classes,
        "label_to_id": label_to_id,
        "splits": {"full": split_metadata},
    }
    (output_root / "data" / "processed" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared Animal dataset under {output_root}")


if __name__ == "__main__":
    main()
