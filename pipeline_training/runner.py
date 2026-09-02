from __future__ import annotations

import logging
import time

from pipeline_training.dataset import TrainingConfig
from pipeline_training.steps import run_compute_stats, run_evaluate, run_train

logger = logging.getLogger(__name__)

STEP_ORDER = ["compute_stats", "train", "evaluate"]


def run_training_pipeline(cfg: TrainingConfig, steps: list[str] | None = None) -> None:
    selected = steps or STEP_ORDER
    unknown = set(selected) - set(STEP_ORDER)
    if unknown:
        raise ValueError(f"Unknown steps: {sorted(unknown)}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    mean = std = None

    if "compute_stats" in selected or "train" in selected:
        mean, std = run_compute_stats(cfg)

    if "train" in selected:
        logger.info("=== Step: train ===")
        assert mean is not None and std is not None
        run_train(cfg, mean, std)

    if "evaluate" in selected:
        logger.info("=== Step: evaluate ===")
        run_evaluate(cfg)

    elapsed = time.time() - start
    logger.info("Training pipeline finished in %.1f seconds", elapsed)
    logger.info("Artifacts directory: %s", cfg.output_dir)
