from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_manifest(manifest_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batched(seq: list[dict[str, object]], batch_size: int) -> list[list[dict[str, object]]]:
    return [seq[idx : idx + batch_size] for idx in range(0, len(seq), batch_size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CLIP ViT-L/14 image embeddings for Animal dataset.")
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("experiments/embeddings_field/image/animals"),
    )
    parser.add_argument("--split", default="full")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--model-id",
        default="openai/clip-vit-large-patch14",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    prepared_root = args.prepared_root.expanduser().resolve()
    split_root = prepared_root / "data" / "processed" / args.split
    manifest_path = split_root / "manifest.jsonl"
    labels_path = split_root / "labels.npy"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels: {labels_path}")

    rows = read_manifest(manifest_path)
    labels = np.load(labels_path)
    if len(rows) != len(labels):
        raise ValueError(f"Manifest rows {len(rows)} do not match labels length {len(labels)}")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Loading CLIP model {args.model_id} on {device}...")
    processor = CLIPProcessor.from_pretrained(args.model_id)
    model = CLIPModel.from_pretrained(args.model_id)
    model.eval()
    model.to(device)

    artifact_root = ensure_dir(prepared_root / "artifacts" / "embeddings")
    embeddings_batches: list[np.ndarray] = []

    print(f"Generating embeddings for {len(rows)} images...")
    with torch.no_grad():
        for i, batch_rows in enumerate(batched(rows, args.batch_size)):
            images = []
            for row in batch_rows:
                image_path = Path(str(row["source_path"]))
                try:
                    image = Image.open(image_path).convert("RGB")
                    images.append(image)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
                    # In a real scenario, we'd need to handle this to maintain alignment
                    # For now, we'll just skip or raise
                    raise
            
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled = vision_outputs.pooler_output
            image_features = model.visual_projection(pooled)
            embeddings_batches.append(image_features.detach().cpu().to(torch.float32).numpy())
            
            if (i + 1) % 10 == 0:
                print(f"Processed {min((i + 1) * args.batch_size, len(rows))}/{len(rows)} images")

    embeddings = np.concatenate(embeddings_batches, axis=0)
    if embeddings.shape[0] != len(rows):
        raise ValueError(f"Embedding rows {embeddings.shape[0]} do not match manifest rows {len(rows)}")

    output_path = artifact_root / f"{args.split}_raw.npy"
    np.save(output_path, embeddings)

    metadata = {
        "prepared_root": str(prepared_root),
        "split": args.split,
        "model_id": args.model_id,
        "batch_size": int(args.batch_size),
        "device": str(device),
        "manifest_path": str(manifest_path),
        "labels_path": str(labels_path),
        "output_path": str(output_path),
        "num_examples": int(embeddings.shape[0]),
        "embedding_dimensionality": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
    }
    (artifact_root / f"{args.split}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.split} embeddings to {output_path} with shape {embeddings.shape}")


if __name__ == "__main__":
    main()
