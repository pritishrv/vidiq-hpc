from __future__ import annotations

import json
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

    @property
    def run_dir(self) -> Path:
        return self.run_root / self.run_name

    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def artifact_dir(self) -> Path:
        return self.run_dir / "artifacts"


def load_config(config_path: str | Path) -> ImageExperimentConfig:
    with open(config_path, "r") as f:
        data = json.load(f)

    # Convert paths
    data["data_root"] = Path(data["data_root"])
    data["run_root"] = Path(data["run_root"])

    return ImageExperimentConfig(**data)
