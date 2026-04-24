from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageExperimentConfig:
    data_root: Path
    run_root: Path
    run_name: str
    backbone: str
    batch_size: int
    num_workers: int
    image_size: int
    device: str = "cuda"
    hf_dataset_id: str = "Woleek/EmoSet-118K"
    dataset_split: str = "train"
    hf_cache_dir: Path | None = None
    allow_hf_download: bool = True
    enable_eval: bool = True
    eval_test_size: float = 0.2
    eval_seed: int = 42
    eval_epochs: int = 20
    eval_learning_rate: float = 1e-3
    eval_weight_decay: float = 0.0
    eval_batch_size: int = 128

    @property
    def run_dir(self) -> Path:
        return self.run_root / self.run_name

    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def artifact_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def metrics_dir(self) -> Path:
        return self.artifact_dir / "metrics"

    @property
    def model_dir(self) -> Path:
        return self.artifact_dir / "models"


def _expand_path(value: str | None) -> Path | None:
    if value is None:
        return None
    expanded = os.path.expanduser(value)

    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[:-]-(.*?)\}")

    def replace_default(match: re.Match[str]) -> str:
        env_name = match.group(1)
        default_value = match.group(2)
        return os.environ.get(env_name, default_value)

    expanded = pattern.sub(replace_default, expanded)
    expanded = os.path.expandvars(expanded)
    return Path(expanded)


def load_config(config_path: str | Path) -> ImageExperimentConfig:
    with open(config_path, "r") as f:
        data = json.load(f)

    # Convert paths
    data["data_root"] = _expand_path(data["data_root"])
    data["run_root"] = _expand_path(data["run_root"])
    data["hf_cache_dir"] = _expand_path(data.get("hf_cache_dir"))

    return ImageExperimentConfig(**data)
