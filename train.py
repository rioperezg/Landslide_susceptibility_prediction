#!/usr/bin/env python3
"""
Reproducible deep-learning training pipeline for landslide susceptibility.

Usage:
    python train.py
    python train.py --model utae --config config/training_config.yaml
    python train.py --steps compute_stats train
    python train.py --steps evaluate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_training.dataset import TrainingConfig
from pipeline_training.models import MODEL_NAMES
from pipeline_training.runner import STEP_ORDER, run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate landslide susceptibility models on enriched .pt tensors."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "training_config.yaml",
        help="Path to YAML training configuration.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override enriched tensor directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override training artifacts directory.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default=None,
        help="Override model architecture.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEP_ORDER,
        default=None,
        help=f"Run selected steps. Default: all ({', '.join(STEP_ORDER)}).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    cfg = TrainingConfig.load(args.config, root_dir=ROOT)
    if args.data_dir is not None:
        cfg.data_dir = (
            args.data_dir.resolve()
            if args.data_dir.is_absolute()
            else (ROOT / args.data_dir).resolve()
        )
    if args.output_dir is not None:
        cfg.output_dir = (
            args.output_dir.resolve()
            if args.output_dir.is_absolute()
            else (ROOT / args.output_dir).resolve()
        )
    if args.model is not None:
        cfg.model = args.model

    logging.info("Model: %s", cfg.model)
    logging.info("Data:  %s", cfg.data_dir)
    logging.info("Out:   %s", cfg.output_dir)

    run_training_pipeline(cfg, steps=args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
