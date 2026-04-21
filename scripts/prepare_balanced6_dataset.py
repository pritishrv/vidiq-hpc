from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LABEL_NAMES = ["sadness", "happiness", "love", "anger", "fear", "surprise"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_texts(emotion_dir: Path) -> list[str]:
    file_path = emotion_dir / "texts.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    lines = []
    with file_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            lines.append(json.loads(line)["text"])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine per-emotion raw embeddings into one dataset.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/embeddings/balanced-6-emotions"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/embeddings_field/text/multiclass/balanced-6-emotions"),
    )
    parser.add_argument("--embeddings-name", default="balanced-6-emotions_raw.npy")
    args = parser.parse_args()

    combined = []
    labels = []
    texts = []
    counts: dict[str, int] = {}

    for idx, label in enumerate(LABEL_NAMES):
        folder = args.source_root / label
        raw_path = folder / "raw_embeddings.npy"
        if not raw_path.exists():
            raise FileNotFoundError(f"missing embeddings for {label}")
        arr = np.load(raw_path)
        label_texts = load_texts(folder)
        if arr.shape[0] != len(label_texts):
            raise ValueError(f"{label}: embeddings {arr.shape[0]} vs texts {len(label_texts)}")
        combined.append(arr)
        texts.extend(label_texts)
        labels.extend([idx] * arr.shape[0])
        counts[label] = arr.shape[0]

    embeddings = np.vstack(combined)
    label_array = np.array(labels, dtype=np.int64)

    data_dir = ensure_dir(args.output_root / "data" / "processed" / "train")
    artifacts_dir = ensure_dir(args.output_root / "artifacts" / "embeddings")

    np.save(artifacts_dir / args.embeddings_name, embeddings)
    np.save(data_dir / "labels.npy", label_array)
    with (data_dir / "texts.jsonl").open("w", encoding="utf-8") as fh:
        for text, label in zip(texts, labels):
            fh.write(json.dumps({"text": text, "label": int(label)}, ensure_ascii=True) + "\n")

    metadata = {
        "label_names": LABEL_NAMES,
        "counts": counts,
        "num_examples": int(label_array.size),
        "embeddings_shape": list(embeddings.shape),
        "source_root": str(args.source_root),
    }
    write_json(data_dir / "metadata.json", metadata)
    write_json(args.output_root / "metadata.json", metadata)

    print("Combined dataset ready at", args.output_root)


if __name__ == "__main__":
    main()
