from __future__ import annotations

import logging
from pathlib import Path

import torch

from pipeline_training.dataset import TrainingConfig, compute_channel_stats, load_split_ids
from pipeline_training.utils import write_json

logger = logging.getLogger(__name__)


def run_compute_stats(cfg: TrainingConfig) -> tuple[torch.Tensor, torch.Tensor]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.stats_path.exists():
        logger.info("Loading cached normalization stats from %s", cfg.stats_path)
        payload = torch.load(cfg.stats_path, map_location="cpu", weights_only=False)
        return payload["mean"], payload["std"]

    splits = load_split_ids(cfg.split_file, cfg.split_extra_file)
    logger.info("Computing normalization stats on %d training patches", len(splits["train"]))
    mean, std = compute_channel_stats(
        pt_dirs=[cfg.data_dir],
        patch_ids=splits["train"],
        selected_variables=cfg.selected_variables,
    )
    torch.save(
        {
            "mean": mean,
            "std": std,
            "selected_variables": cfg.selected_variables,
        },
        cfg.stats_path,
    )
    write_json(
        cfg.output_dir / "normalization_stats.json",
        {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "selected_variables": cfg.selected_variables,
        },
    )
    logger.info("Saved normalization stats to %s", cfg.stats_path)
    return mean, std
