from __future__ import annotations

import os
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class EmoSetDataset(Dataset):
    """
    EmoSet-118K Dataset implementation.
    Reference: https://vcc.tech/EmoSet
    """
    EMOTIONS = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]

    def __init__(self, data_root: Path, transform=None, download: bool = False):
        self.data_root = data_root
        self.transform = transform
        self.image_dir = data_root / "images"
        self.annotation_file = data_root / "metadata.csv"

        if download:
            self._download()

        if not self.annotation_file.exists():
            raise FileNotFoundError(f"EmoSet metadata not found at {self.annotation_file}. Please ensure download=True or data is placed correctly.")

        self.df = pd.read_csv(self.annotation_file)
        self.label_to_id = {emotion: i for i, emotion in enumerate(self.EMOTIONS)}

    def _download(self):
        # Implementation for downloading EmoSet-118K from Dropbox/URLs
        # This would involve downloading the CSV metadata and then fetching images via URLs
        if not self.data_root.exists():
            self.data_root.mkdir(parents=True, exist_ok=True)
        
        if not self.annotation_file.exists():
            print("Downloading EmoSet-118K metadata...")
            # Placeholder for actual download logic
            # subprocess.run(["wget", ...])
            pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row['image_path']
        image = Image.open(img_path).convert("RGB")
        label = self.label_to_id[row['emotion']]

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
            
        # Placeholder for EmoVerse manifest loading
        self.samples = [] 

    def _download(self):
        print("Downloading EmoVerse B-A-S data...")
        # Placeholder for EmoVerse download logic
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Implementation for loading B-A-S and applying SAM masks
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert("RGB")
        
        if self.mode == "subject":
            # Apply SAM mask for subject
            pass
        elif self.mode == "background":
            # Apply inverse SAM mask for background
            pass
            
        if self.transform:
            image = self.transform(image)
            
        return image, sample['label']
