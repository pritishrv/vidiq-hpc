from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


TEXT_CANDIDATES = ["text", "sentence", "review"]
LABEL_CANDIDATES = ["label", "labels"]

def detect_column(fieldnames: list[str], candidates: list[str]) -> str:
    for cand in candidates:
        if cand in fieldnames:
            return cand
    return fieldnames[-1]


def ingest_csv(source: Path, target: Path, text_col: str | None, label_col: str | None) -> None:
    reader = csv.DictReader(source.open("r", encoding="utf-8"))
    fieldnames = reader.fieldnames or []
    if not text_col:
        text_col = detect_column(fieldnames, TEXT_CANDIDATES)
    if not label_col:
        label_col = detect_column(fieldnames, LABEL_CANDIDATES)
    texts = []
    labels = []
    for row in reader:
        texts.append(str(row[text_col]).strip())
        labels.append(int(row[label_col]))
    sorted_texts = texts
    with (target / "texts.jsonl").open("w", encoding="utf-8") as fh:
        for text, label in zip(sorted_texts, labels):
            fh.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
    np_labels = np.array(labels, dtype=np.int64)
    np.save(target / "labels.npy", np_labels)
    with (target / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "source": str(source),
                "count": len(labels),
                "text_column": text_col,
                "label_column": label_col,
            },
            fh,
            indent=2,
        )


def prepare_subset(source: Path, target_root: Path, text_column: str | None, label_column: str | None) -> None:
    target = ensure_dir(target_root / source.parent.name / source.stem)
    ingest_csv(source, target, text_column, label_column)
    print(f"Prepared {source.parent.name}/{source.stem} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Text_Datasets into repo.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, default=Path("data/prepared"))
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    args = parser.parse_args()

    for subset in args.source.iterdir():
        if not subset.is_dir():
            continue
        for csv_file in subset.glob("*.csv"):
            prepare_subset(csv_file, args.target_root, args.text_column, args.label_column)


if __name__ == "__main__":
    import numpy as np

    main()
