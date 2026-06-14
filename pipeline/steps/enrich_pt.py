from __future__ import annotations

import logging
import shutil
from pathlib import Path

from Helpers_fase2.feature_functions_pt import (
    add_aspect_to_pt_folder,
    add_distance_to_drainage_to_pt_folder,
    add_ls_to_pt_folder,
    add_nbr_to_pt_folder,
    add_ndvi_to_pt_folder,
    add_profile_curvature_to_pt_folder,
    add_slope_to_pt_folder,
    add_spi_to_pt_folder,
    add_twi_to_pt_folder,
)
from pipeline.config import PipelineConfig
from pipeline.utils import list_pt_files, write_json

logger = logging.getLogger(__name__)


def _stage_dir(cfg: PipelineConfig, variant: str, stage: str) -> Path:
    return cfg.work_dir / "04_pt_stages" / variant / stage


def _enrich_variant(cfg: PipelineConfig, variant: str) -> dict:
    profile = cfg.variable_profile(variant)
    base_names = profile["dynamic"] + profile["static"]
    final_order = profile["final_order"]
    resolution = float(cfg.topography["resolution"])
    log10_threshold = float(cfg.topography["distance_to_drainage_log10_threshold"])

    src = cfg.variant_work("03_pt_base", variant)
    out = cfg.variant_output(variant)
    out.mkdir(parents=True, exist_ok=True)

    stages = [
        ("slope", lambda inp, tmp: add_slope_to_pt_folder(
            input_dir=str(inp),
            output_dir=str(tmp),
            final_order=final_order,
            base_variable_names=base_names,
            overwrite=True,
        )),
        ("aspect", lambda inp, tmp: add_aspect_to_pt_folder(
            input_dir=str(inp),
            output_dir=str(tmp),
            final_order=final_order,
            base_variable_names=base_names,
            resolution=resolution,
            overwrite=True,
        )),
        ("profile_curvature", lambda inp, tmp: add_profile_curvature_to_pt_folder(
            input_dir=str(inp),
            output_dir=str(tmp),
            final_order=final_order,
            base_variable_names=base_names,
            resolution=resolution,
            overwrite=True,
        )),
        ("ls", lambda inp, tmp: add_ls_to_pt_folder(
            input_dir=str(inp),
            output_dir=str(tmp),
            final_order=final_order,
            overwrite=True,
        )),
        ("spi", lambda inp, tmp: add_spi_to_pt_folder(
            input_dir=str(inp),
            output_dir=str(tmp),
            final_order=final_order,
            overwrite=True,
        )),
        ("twi", lambda inp, tmp: add_twi_to_pt_folder(
            input_dir=str(inp),
            output_dir=str(tmp),
            final_order=final_order,
            overwrite=True,
        )),
        ("distance_to_drainage", lambda inp, tmp: add_distance_to_drainage_to_pt_folder(
            input_dir=str(inp),
            output_dir=str(tmp),
            log10_threshold=log10_threshold,
            cellsize_x=resolution,
            cellsize_y=resolution,
            overwrite=True,
        )),
    ]

    if cfg.is_sen2(variant):
        stages.extend([
            ("ndvi", lambda inp, tmp: add_ndvi_to_pt_folder(
                input_dir=str(inp),
                output_dir=str(tmp),
                overwrite=True,
            )),
            ("nbr", lambda inp, tmp: add_nbr_to_pt_folder(
                input_dir=str(inp),
                output_dir=str(tmp),
                add_delta=False,
                overwrite=True,
            )),
        ])

    current = src
    stage_reports: dict[str, dict] = {}

    for stage_name, fn in stages:
        stage_dir = _stage_dir(cfg, variant, stage_name)
        stage_dir.mkdir(parents=True, exist_ok=True)
        failed = fn(current, stage_dir)
        stage_reports[stage_name] = {
            "failed_count": len(failed),
            "failed": [(Path(path).name, err) for path, err in failed[:20]],
        }
        current = stage_dir
        logger.info("%s %s: failed=%d", variant, stage_name, len(failed))

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(current, out)

    return {
        "output_dir": str(out),
        "n_files": len(list_pt_files(out)),
        "stages": stage_reports,
    }


def run_enrich_pt(cfg: PipelineConfig) -> None:
    """Add topographic and (for Sen2) spectral indices to tensor files."""
    summary = {variant: _enrich_variant(cfg, variant) for variant in cfg.variants}
    write_json(cfg.work_dir / "04_enrich_pt_summary.json", summary)
    logger.info("Final enriched tensors written to %s", cfg.output_dir)
