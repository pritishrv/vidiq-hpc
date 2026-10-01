from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare balanced FER2013 no-disgust manifests and generate CLIP ViT-L/14 embeddings.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/data/Image_Dataset/FER2013"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/embeddings_field/image/emotions/fer2013_balanced_no_disgust_clip_vitl14"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    if not args.skip_prepare:
        prep_cmd = [
            sys.executable,
            str(script_dir / "prepare_fer2013_dataset.py"),
            "--dataset-root",
            str(args.dataset_root),
            "--output-root",
            str(args.output_root),
            "--seed",
            str(args.seed),
        ]
        run(prep_cmd)

    for split in args.splits:
        embed_cmd = [
            sys.executable,
            str(script_dir / "generate_clip_image_embeddings.py"),
            "--prepared-root",
            str(args.output_root),
            "--split",
            split,
            "--batch-size",
            str(args.batch_size),
        ]
        if args.device:
            embed_cmd.extend(["--device", args.device])
        run(embed_cmd)


if __name__ == "__main__":
    main()
