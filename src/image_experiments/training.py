from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class LinearProbeConfig:
    test_size: float = 0.2
    seed: int = 42
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


def _stratified_split(labels: torch.Tensor, test_size: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []

    for label in torch.unique(labels):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        perm = indices[torch.randperm(len(indices), generator=generator)]
        n_test = max(1, int(round(len(indices) * test_size)))
        if n_test >= len(indices):
            n_test = max(1, len(indices) - 1)
        test_parts.append(perm[:n_test])
        train_parts.append(perm[n_test:])

    train_idx = torch.cat(train_parts)
    test_idx = torch.cat(test_parts)
    train_idx = train_idx[torch.randperm(len(train_idx), generator=generator)]
    test_idx = test_idx[torch.randperm(len(test_idx), generator=generator)]
    return train_idx, test_idx


def _confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for true_label, pred_label in zip(labels.tolist(), preds.tolist()):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def _macro_f1(confusion: torch.Tensor) -> float:
    f1_scores: list[float] = []
    for idx in range(confusion.shape[0]):
        tp = float(confusion[idx, idx].item())
        fp = float(confusion[:, idx].sum().item() - tp)
        fn = float(confusion[idx, :].sum().item() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (precision + recall) > 0:
            f1_scores.append((2.0 * precision * recall) / (precision + recall))
        else:
            f1_scores.append(0.0)
    return float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0


def run_linear_probe(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    config: LinearProbeConfig,
    device: str,
) -> dict:
    embeddings = embeddings.detach().to(torch.float32).cpu()
    labels = labels.detach().to(torch.long).cpu()

    train_idx, test_idx = _stratified_split(labels, config.test_size, config.seed)
    train_x = embeddings[train_idx]
    train_y = labels[train_idx]
    test_x = embeddings[test_idx]
    test_y = labels[test_idx]

    num_classes = int(torch.max(labels).item()) + 1
    model = LinearProbe(input_dim=embeddings.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(config.batch_size, len(train_x)),
        shuffle=True,
    )
    eval_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=min(config.batch_size, len(test_x)),
        shuffle=False,
    )

    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * len(batch_x)
            preds = torch.argmax(logits, dim=-1)
            running_correct += int((preds == batch_y).sum().item())
            seen += len(batch_x)

        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(seen, 1),
                "train_accuracy": running_correct / max(seen, 1),
            }
        )

    model.eval()
    eval_loss = 0.0
    eval_seen = 0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            eval_loss += float(loss.item()) * len(batch_x)
            eval_seen += len(batch_x)
            all_preds.append(torch.argmax(logits, dim=-1).cpu())
            all_labels.append(batch_y.cpu())

    preds = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
    truth = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)
    confusion = _confusion_matrix(preds, truth, num_classes)
    accuracy = float((preds == truth).to(torch.float32).mean().item()) if len(truth) else 0.0
    macro_f1 = _macro_f1(confusion)

    return {
        "splits": {
            "num_train": int(len(train_idx)),
            "num_eval": int(len(test_idx)),
            "test_size": float(config.test_size),
            "seed": int(config.seed),
        },
        "training": {
            "epochs": int(config.epochs),
            "learning_rate": float(config.learning_rate),
            "weight_decay": float(config.weight_decay),
            "batch_size": int(config.batch_size),
            "history": history,
        },
        "evaluation": {
            "loss": eval_loss / max(eval_seen, 1),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        },
        "confusion_matrix": confusion.tolist(),
        "num_classes": num_classes,
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
    }
