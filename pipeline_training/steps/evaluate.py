from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from pipeline_training.dataset import LandslideTensorDataset, TrainingConfig, load_split_ids
from pipeline_training.metrics import logits_to_probs, scan_thresholds
from pipeline_training.models import build_model
from pipeline_training.utils import write_json

logger = logging.getLogger(__name__)


def run_evaluate(cfg: TrainingConfig) -> dict:
    if not cfg.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")

    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    selected_variables = checkpoint.get("selected_variables", cfg.selected_variables)
    model_name = checkpoint.get("model", cfg.model)
    model_params = checkpoint.get("model_params", cfg.model_params)
    num_classes = int(checkpoint.get("num_classes", cfg.num_classes))
    target_mode = checkpoint.get("target_mode", cfg.target_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name,
        in_channels=len(selected_variables),
        num_classes=num_classes,
        params=model_params,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    splits = load_split_ids(cfg.split_file, cfg.split_extra_file)
    split_name = str(cfg.evaluation.get("split", "val"))
    patch_ids = splits[split_name]

    dataset = LandslideTensorDataset(
        pt_dirs=cfg.data_dir,
        patch_ids=patch_ids,
        mean=mean,
        std=std,
        selected_variables=selected_variables,
        target_mode=target_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(cfg.training.get("num_workers", 0)),
    )

    probs_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x).cpu()
            probs = logits_to_probs(logits, target_mode)
            probs_list.append(probs)
            targets_list.append(y)

    metrics = scan_thresholds(probs_list, targets_list)
    metrics["split"] = split_name
    metrics["n_patches"] = len(patch_ids)
    metrics["model"] = model_name

    write_json(cfg.metrics_path, metrics)
    logger.info("Evaluation metrics: %s", metrics)
    logger.info("Saved metrics to %s", cfg.metrics_path)
    return metrics
