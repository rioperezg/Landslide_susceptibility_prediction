#!/usr/bin/env python3
"""
Reproducible landslide dataset pipeline.

Usage:
    python main.py
    python main.py --input matching_files --output Enriched_files_pt
    python main.py --steps validate temporal
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import PipelineConfig
from pipeline.runner import STEP_ORDER, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Enriched_files_pt from matching_files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Override input directory (matching_files).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output directory (Enriched_files_pt).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Override intermediate artifacts directory.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEP_ORDER,
        default=None,
        help=f"Run only selected steps. Default: all ({', '.join(STEP_ORDER)}).",
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

    cfg = PipelineConfig.load(args.config, root_dir=ROOT)
    if args.input is not None:
        cfg.input_dir = (ROOT / args.input).resolve() if not args.input.is_absolute() else args.input
    if args.output is not None:
        cfg.output_dir = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output
    if args.work_dir is not None:
        cfg.work_dir = (ROOT / args.work_dir).resolve() if not args.work_dir.is_absolute() else args.work_dir

    logging.info("Input:  %s", cfg.input_dir)
    logging.info("Output: %s", cfg.output_dir)
    logging.info("Work:   %s", cfg.work_dir)

    run_pipeline(cfg, steps=args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
