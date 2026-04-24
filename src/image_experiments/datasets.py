from __future__ import annotations

import csv
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from image_experiments.io_utils import ensure_dir


class EmoSetDataset(Dataset):
    """
    EmoSet-118K Dataset implementation.
    Reference: https://vcc.tech/EmoSet
    """
    EMOTIONS = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]

    def __init__(
        self,
        data_root: Path,
        transform=None,
        download: bool = False,
        dataset_id: str = "Woleek/EmoSet-118K",
        split: str = "train",
        hf_cache_dir: Path | None = None,
    ):
        self.data_root = data_root
        self.transform = transform
        self.image_dir = data_root / "images"
        self.annotation_file = data_root / "metadata.csv"
        self.dataset_id = dataset_id
        self.split = split
        self.hf_cache_dir = hf_cache_dir
        self.source_manifest = data_root / "source_manifest.json"
        self.backend = "local"
        self.hf_dataset = None

        self.label_to_id = {emotion: i for i, emotion in enumerate(self.EMOTIONS)}
        self.samples = []

        if self.annotation_file.exists():
            self._load_local_samples()
        elif download:
            self._download()
        else:
            raise FileNotFoundError(self._missing_data_message())

        if self.backend == "local" and not self.samples:
            raise RuntimeError(f"No EmoSet samples were loaded from {self.annotation_file}.")

    def _load_local_samples(self) -> None:
        self.backend = "local"
        with open(self.annotation_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

    def _load_hf_samples(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "The `datasets` package is required for Hugging Face-backed EmoSet loading."
            ) from exc

        cache_dir = str(self.hf_cache_dir) if self.hf_cache_dir is not None else None
        self.hf_dataset = load_dataset(self.dataset_id, split=self.split, cache_dir=cache_dir)
        self.backend = "huggingface"
        ensure_dir(self.data_root)
        with self.source_manifest.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_id": self.dataset_id,
                    "split": self.split,
                    "backend": self.backend,
                    "data_root": str(self.data_root),
                    "hf_cache_dir": cache_dir,
                    "note": "Images and metadata are provided by the Hugging Face dataset cache rather than repo-local files.",
                },
                f,
                indent=2,
            )

    def _missing_data_message(self, hf_error: Exception | None = None) -> str:
        lines = [
            f"EmoSet staged metadata was not found at {self.annotation_file}.",
            f"Expected staged image directory: {self.image_dir}",
            "The image pipeline now supports two valid sources:",
            "1. A staged local dataset root with `metadata.csv` and an `images/` tree.",
            f"2. A Hugging Face-backed dataset load from `{self.dataset_id}` split `{self.split}`.",
        ]
        if self.hf_cache_dir is not None:
            lines.append(f"Hugging Face cache directory: {self.hf_cache_dir}")
        lines.extend(
            [
                f"Configured data root: {self.data_root}",
                "Recommended Hyperion storage:",
                "- /users/aczd097/archive/vidiq-hpc/data/image/emoset for staged dataset manifests or copied assets",
                "- /users/aczd097/sharedscratch/huggingface/datasets for Hugging Face cache data",
            ]
        )
        if hf_error is not None:
            lines.extend(
                [
                    "The fallback Hugging Face load also failed.",
                    f"Underlying error: {hf_error}",
                ]
            )
        lines.extend(
            [
                "If you already have EmoSet assets elsewhere, point `data_root` at that staged location.",
                "If you intend to rely on the public Hugging Face mirror, ensure the `datasets` package is installed and outbound access is available, then retry.",
            ]
        )
        return "\n".join(lines)

    def _download(self):
        ensure_dir(self.data_root)

        if self.annotation_file.exists():
            self._load_local_samples()
            return

        print("Attempting Hugging Face-backed EmoSet load...")
        try:
            self._load_hf_samples()
        except Exception as exc:
            raise FileNotFoundError(self._missing_data_message(exc)) from exc

    def __len__(self):
        if self.backend == "huggingface":
            return len(self.hf_dataset)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.backend == "huggingface":
            row = self.hf_dataset[idx]
            image = row["image"]
            if not isinstance(image, Image.Image):
                if isinstance(image, dict):
                    image_path = image.get("path")
                    if image_path:
                        image = Image.open(image_path).convert("RGB")
                    else:
                        raise RuntimeError("Hugging Face EmoSet row did not provide a usable image payload.")
                else:
                    raise RuntimeError("Unsupported Hugging Face image payload type.")
            else:
                image = image.convert("RGB")

            if "label" in row and isinstance(row["label"], int):
                label = int(row["label"])
            else:
                label = self.label_to_id[str(row["emotion"])]

            if self.transform:
                image = self.transform(image)

            return image, label

        row = self.samples[idx]
        img_path = self.image_dir / row["image_path"]
        if not img_path.exists():
            raise FileNotFoundError(
                f"EmoSet staged image is missing: {img_path}. "
                f"Check that `data_root` points at the correct staged dataset root."
            )
        image = Image.open(img_path).convert("RGB")
        label = self.label_to_id[row["emotion"]]

        if self.transform:
            image = self.transform(image)

        return image, label


class EmoVerseDataset(Dataset):
    """
    EmoVerse Dataset implementation with Background-Attribute-Subject (B-A-S) triplets.
    Reference: https://arxiv.org/html/2511.12554v1
    """
    def __init__(self, data_root: Path, transform=None, download: bool = False, mode: str = "full"):
        self.data_root = data_root
        self.transform = transform
        self.mode = mode # 'full', 'subject', 'background'
        
        if download:
            self._download()
            
        self.samples = [] 

    def _download(self):
        print("Downloading EmoVerse B-A-S data...")
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert("RGB")
        
        if self.mode == "subject":
            pass
        elif self.mode == "background":
            pass
            
        if self.transform:
            image = self.transform(image)
            
        return image, sample['label']
