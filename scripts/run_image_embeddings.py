#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def _bootstrap_repo_src() -> None:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    repo_src_str = str(repo_src)
    if repo_src_str not in sys.path:
        sys.path.insert(0, repo_src_str)

_bootstrap_repo_src()

from image_experiments.config import load_config
from image_experiments.io_utils import ensure_dir, write_json, get_timestamp, format_duration
from image_experiments.datasets import EmoSetDataset
from image_experiments.embeddings import ImageEmbedder
from image_experiments.geometry import calculate_centroids, calculate_radial_density


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image embedding geometry experiments.")
    parser.add_argument("--config", required=True, help="Path to the experiment config JSON file.")
    return parser.parse_args()


def main() -> None:
    start_time = time.time()
    start_timestamp = get_timestamp()
    
    args = parse_args()
    config = load_config(args.config)
    
    ensure_dir(config.run_dir)
    ensure_dir(config.log_dir)
    ensure_dir(config.artifact_dir)

    print(f"{'='*60}")
    print(f"EXPERIMENT: {config.run_name}")
    print(f"STARTED:    {start_timestamp}")
    print(f"BACKBONE:   {config.backbone}")
    print(f"DEVICE:     {config.device}")
    # Infrastructure TBC
    print(f"INFRA:      TBC (Node: {os.uname().nodename if hasattr(os, 'uname') else 'unknown'})")
    print(f"{'='*60}")

    # 1. Dataset
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Using EmoSet as default for Phase 1
    dataset = EmoSetDataset(config.data_root, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)

    # 2. Embeddings
    embedder = ImageEmbedder(config.backbone, device=config.device)
    embeddings, labels = embedder.generate_embeddings(dataloader)

    # 3. Geometry
    centroids = calculate_centroids(embeddings, labels)
    
    density_results = {}
    for label_id, centroid in centroids.items():
        class_embeddings = embeddings[labels == label_id]
        density_results[str(label_id)] = calculate_radial_density(class_embeddings, centroid)

    # 4. Save Artifacts
    torch.save(embeddings, config.artifact_dir / "embeddings.pt")
    torch.save(labels, config.artifact_dir / "labels.pt")
    
    centroid_serializable = {k: v.tolist() for k, v in centroids.items()}
    write_json(config.artifact_dir / "centroids.json", centroid_serializable)
    write_json(config.artifact_dir / "density.json", density_results)

    # 5. Final Log
    end_time = time.time()
    duration = end_time - start_time
    
    metadata = {
        "run_name": config.run_name,
        "start_timestamp": start_timestamp,
        "end_timestamp": get_timestamp(),
        "duration_seconds": duration,
        "duration_formatted": format_duration(duration),
        "backbone": config.backbone,
        "config": vars(config),
        "infra": "TBC"
    }
    write_json(config.log_dir / "run_metadata.json", metadata)

    print(f"{'='*60}")
    print(f"COMPLETED:  {metadata['end_timestamp']}")
    print(f"DURATION:   {metadata['duration_formatted']}")
    print(f"RESULTS:    {config.run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import os
    main()
