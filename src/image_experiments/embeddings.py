from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


class ImageEmbedder:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()

    def generate_embeddings(self, dataloader) -> tuple[torch.Tensor, torch.Tensor]:
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Generating embeddings"):
                # Note: Images are PIL if transform is not applied in Dataset, 
                # but standard practice is to handle processing in the loader.
                # If images are already tensors from Dataset:
                inputs = images.to(self.device)
                
                if hasattr(self.model, "get_image_features"):
                    features = self.model.get_image_features(pixel_values=inputs)
                else:
                    outputs = self.model(pixel_values=inputs)
                    features = outputs.pooler_output
                
                all_embeddings.append(features.cpu())
                all_labels.append(labels)

        return torch.cat(all_embeddings), torch.cat(all_labels)
