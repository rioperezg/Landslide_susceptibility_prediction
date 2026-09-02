from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Dataset


@dataclass
class TrainingConfig:
    root_dir: Path
    data_dir: Path
    output_dir: Path
    split_file: Path
    split_extra_file: Path | None
    model: str
    selected_variables: list[str]
    training: dict[str, Any]
    model_params: dict[str, Any]
    evaluation: dict[str, Any]

    @classmethod
    def load(cls, config_path: Path, root_dir: Path | None = None) -> TrainingConfig:
        root = (root_dir or config_path.parent.parent).resolve()
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        def resolve(value: str | None) -> Path | None:
            if value is None:
                return None
            path = Path(value)
            return path if path.is_absolute() else (root / path).resolve()

        return cls(
            root_dir=root,
            data_dir=resolve(raw["data_dir"]),  # type: ignore[arg-type]
            output_dir=resolve(raw["output_dir"]),  # type: ignore[arg-type]
            split_file=resolve(raw["split_file"]),  # type: ignore[arg-type]
            split_extra_file=resolve(raw.get("split_extra_file")),
            model=str(raw["model"]).lower(),
            selected_variables=list(raw["selected_variables"]),
            training=dict(raw.get("training", {})),
            model_params=dict(raw.get("model_params", {})),
            evaluation=dict(raw.get("evaluation", {})),
        )

    @property
    def in_channels(self) -> int:
        return len(self.selected_variables)

    @property
    def num_classes(self) -> int:
        if self.training.get("target_mode", "ce") == "bce":
            return 1
        return int(self.training.get("num_classes", 2))

    @property
    def target_mode(self) -> str:
        return str(self.training.get("target_mode", "ce"))

    @property
    def stats_path(self) -> Path:
        return self.output_dir / "normalization_stats.pt"

    @property
    def checkpoint_path(self) -> Path:
        name = str(self.training.get("checkpoint_name", "best_model.pth"))
        return self.output_dir / name

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / "metrics.json"


class LandslideTensorDataset(Dataset):
    """Load enriched .pt tensors and adapt them to Sen12Landslides model input."""

    def __init__(
        self,
        pt_dirs: str | Path | list[str | Path],
        patch_ids: list[str],
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        selected_variables: list[str] | None = None,
        target_mode: str = "ce",
    ) -> None:
        if isinstance(pt_dirs, (str, Path)):
            pt_dirs = [pt_dirs]
        self.pt_dirs = [Path(d) for d in pt_dirs]
        self.patch_ids = patch_ids
        self.mean = mean
        self.std = std
        self.selected_variables = selected_variables
        self.target_mode = target_mode

    def __len__(self) -> int:
        return len(self.patch_ids)

    def _resolve_path(self, patch_id: str) -> Path:
        for pt_dir in self.pt_dirs:
            if patch_id.endswith(".pt"):
                candidate = pt_dir / patch_id
            else:
                candidate = pt_dir / patch_id.replace(".nc", ".pt")
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Patch '{patch_id}' not found in {self.pt_dirs}")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        found_path = self._resolve_path(self.patch_ids[idx])
        item = torch.load(found_path, map_location="cpu", weights_only=False)

        x = item["x"].float()  # (C, T, H, W)

        if "variable_names" not in item:
            raise KeyError(f"Missing variable_names in {found_path}")

        variable_names = list(item["variable_names"])
        mean = self.mean
        std = self.std

        if self.selected_variables is not None:
            missing = [v for v in self.selected_variables if v not in variable_names]
            if missing:
                raise KeyError(f"Missing variables in {found_path.name}: {missing}")
            indices = [variable_names.index(v) for v in self.selected_variables]
            x = x[indices]
            if mean is not None and std is not None:
                mean = mean[indices]
                std = std[indices]

        if mean is not None and std is not None:
            x = (x - mean[:, None, None, None]) / (std[:, None, None, None] + 1e-6)

        # Sen12Landslides models expect (T, C, H, W) per sample
        x = x.permute(1, 0, 2, 3)

        if self.target_mode == "ce":
            y = item["y"].long()
        elif self.target_mode == "bce":
            y = item["y"].float().unsqueeze(0)
        else:
            raise ValueError("target_mode must be 'ce' or 'bce'")

        return x, y


def load_split_ids(split_file: Path, extra_split_file: Path | None = None) -> dict[str, list[str]]:
    with open(split_file, encoding="utf-8") as f:
        split = json.load(f)

    if extra_split_file is None:
        return {key: list(value) for key, value in split.items()}

    with open(extra_split_file, encoding="utf-8") as f:
        extra = json.load(f)

    merged: dict[str, list[str]] = {}
    for key in ("train", "val", "test"):
        merged[key] = list(split.get(key, [])) + list(extra.get(key, []))
    return merged


def compute_channel_stats(
    pt_dirs: list[Path],
    patch_ids: list[str],
    selected_variables: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel mean/std over the training split."""
    sums = None
    sq_sums = None
    counts = 0

    for patch_id in patch_ids:
        dataset = LandslideTensorDataset(
            pt_dirs=pt_dirs,
            patch_ids=[patch_id],
            selected_variables=selected_variables,
            target_mode="ce",
        )
        x, _ = dataset[0]  # (T, C, H, W)
        x = x.permute(1, 0, 2, 3)  # back to (C, T, H, W)

        if sums is None:
            sums = torch.zeros(x.shape[0], dtype=torch.float64)
            sq_sums = torch.zeros(x.shape[0], dtype=torch.float64)

        flat = x.reshape(x.shape[0], -1)
        sums += flat.sum(dim=1).double()
        sq_sums += (flat ** 2).sum(dim=1).double()
        counts += flat.shape[1]

    assert sums is not None and sq_sums is not None
    mean = (sums / counts).float()
    var = (sq_sums / counts) - mean.double() ** 2
    std = torch.sqrt(torch.clamp(var, min=0.0)).float()
    return mean, std
