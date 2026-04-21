from __future__ import annotations

from pathlib import Path
import argparse

import torch
from transformers import AutoModel, AutoTokenizer


MODEL = Path("/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/models/qwen3-0.6B")
PROMPT_FILE = Path("/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq-hpc/prompts/llm-probability-extraction-prompt.md")
LABEL_PROMPTS = ["negative sentiment", "positive sentiment"]
TEMPERATURE = 1.0


def mean_pool(last_hidden_state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).expand_as(last_hidden_state).float()
    summed = last_hidden_state * mask
    summed = summed.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def softmax(logits: torch.Tensor) -> torch.Tensor:
    shifted = logits - logits.max()
    exp = shifted.exp()
    return exp / exp.sum(dim=-1, keepdim=True)


def encode(texts: list[str], tokenizer: AutoTokenizer, model: AutoModel, device: torch.device) -> torch.Tensor:
    embeddings = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            pooled = mean_pool(hidden, encoded["attention_mask"])
            embeddings.append(pooled.cpu())
    return torch.vstack(embeddings)


def main() -> None:
    prompt_context = PROMPT_FILE.read_text(encoding="utf-8").strip()
    print("Prompt context:")
    print(prompt_context)
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=False, help="Sentence to score.")
    parser.add_argument("--text-file", type=Path, help="Path to a text file (UTF-8).")
    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    text_value = args.text
    if args.text_file:
        text_value = args.text_file.read_text(encoding="utf-8").strip()
    if not text_value:
        raise ValueError("No text provided")
    label_emb = encode(LABEL_PROMPTS, tokenizer, model, device)
    text_emb = encode([text_value], tokenizer, model, device)
    label_emb = label_emb / label_emb.norm(dim=1, keepdim=True).clamp(min=1e-12)
    text_emb = text_emb / text_emb.norm(dim=1, keepdim=True).clamp(min=1e-12)
    logits = (text_emb @ label_emb.T).squeeze(0)
    confs = softmax(logits / TEMPERATURE)

    print("Sentence:", text_value)
    print("Pre-softmax logits:", logits.tolist())
    print("Softmax confidences:", confs.tolist())


if __name__ == "__main__":
    main()
