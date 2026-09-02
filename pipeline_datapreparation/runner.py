from __future__ import annotations

import logging
import time
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.steps import (
    run_dem_drainage,
    run_enrich_pt,
    run_nc_to_pt,
    run_temporal,
    run_validate,
)

logger = logging.getLogger(__name__)

STEP_ORDER = ["validate", "temporal", "dem_drainage", "nc_to_pt", "enrich_pt"]


def run_pipeline(cfg: PipelineConfig, steps: list[str] | None = None) -> None:
    selected = steps or STEP_ORDER
    unknown = set(selected) - set(STEP_ORDER)
    if unknown:
        raise ValueError(f"Unknown steps: {sorted(unknown)}")

    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    common_numbers: set[str] | None = None
    start = time.time()

    if "validate" in selected:
        logger.info("=== Step: validate ===")
        common_numbers = run_validate(cfg)

    if any(step in selected for step in ("temporal", "dem_drainage", "nc_to_pt", "enrich_pt")):
        if common_numbers is None:
            from pipeline.utils import common_patch_numbers, read_json

            surviving_path = cfg.work_dir / "metadata" / "surviving_patch_numbers.json"
            if surviving_path.exists():
                common_numbers = set(read_json(surviving_path))
            else:
                variant_dirs = {name: cfg.variant_input(name) for name in cfg.variants}
                common_numbers = common_patch_numbers(variant_dirs)

    if "temporal" in selected:
        logger.info("=== Step: temporal ===")
        assert common_numbers is not None
        common_numbers = run_temporal(cfg, common_numbers)

    if "dem_drainage" in selected:
        logger.info("=== Step: dem_drainage ===")
        run_dem_drainage(cfg)

    if "nc_to_pt" in selected:
        logger.info("=== Step: nc_to_pt ===")
        run_nc_to_pt(cfg)

    if "enrich_pt" in selected:
        logger.info("=== Step: enrich_pt ===")
        run_enrich_pt(cfg)

    elapsed = time.time() - start
    logger.info("Pipeline finished in %.1f seconds", elapsed)
    logger.info("Output directory: %s", cfg.output_dir)
