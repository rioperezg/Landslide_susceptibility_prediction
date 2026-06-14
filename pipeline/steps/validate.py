from __future__ import annotations

import logging

from pipeline.config import PipelineConfig
from pipeline.utils import common_patch_numbers, list_nc_files, write_json

logger = logging.getLogger(__name__)


def run_validate(cfg: PipelineConfig) -> set[str]:
    """Ensure all variants share the same patch IDs."""
    variant_dirs = {name: cfg.variant_input(name) for name in cfg.variants}
    for name, directory in variant_dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing input directory for {name}: {directory}")

    counts = {name: len(list_nc_files(directory)) for name, directory in variant_dirs.items()}
    common = common_patch_numbers(variant_dirs)

    report = {
        "variant_counts": counts,
        "common_patch_count": len(common),
        "common_patch_numbers": sorted(common, key=lambda x: int(x)),
    }
    write_json(cfg.work_dir / "validate_report.json", report)

    logger.info("Variant file counts: %s", counts)
    logger.info("Common patches across variants: %d", len(common))

    if not common:
        raise ValueError("No common patch IDs found across asc, dsc, and Sen2.")

    return common
