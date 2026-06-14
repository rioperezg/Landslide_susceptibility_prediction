from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from tqdm import tqdm

from Helpers_fase2.feature_functions_pt import LandslideDataset
from pipeline.config import PipelineConfig
from pipeline.utils import list_nc_files, write_json

logger = logging.getLogger(__name__)


def _pt_output_name(nc_name: str) -> str:
    stem = Path(nc_name).stem
    if stem.endswith("_enriched"):
        return f"{stem}.pt"
    return f"{stem}_enriched.pt"


def run_nc_to_pt(cfg: PipelineConfig) -> None:
    """Convert enriched NetCDF patches to base PyTorch tensor files."""
    summary: dict[str, dict] = {}

    for variant in cfg.variants:
        profile = cfg.variable_profile(variant)
        dynamic_vars = profile["dynamic"]
        static_vars = profile["static"]
        variable_names = dynamic_vars + static_vars

        nc_dir = cfg.variant_work("02_enriched_nc", variant)
        pt_dir = cfg.variant_work("03_pt_base", variant)
        pt_dir.mkdir(parents=True, exist_ok=True)

        patch_ids = [path.name for path in list_nc_files(nc_dir)]
        dataset = LandslideDataset(
            patches_dir=str(nc_dir),
            patch_ids=patch_ids,
            dynamic_vars=dynamic_vars,
            static_vars=static_vars,
        )

        ok = 0
        failed: list[tuple[str, str]] = []
        for index, patch_id in enumerate(tqdm(patch_ids, desc=f"nc->pt {variant}")):
            try:
                x, y = dataset[index]
                out_path = pt_dir / _pt_output_name(patch_id)
                torch.save(
                    {
                        "x": x,
                        "y": y,
                        "patch_id": Path(patch_id).stem,
                        "variable_names": variable_names,
                    },
                    out_path,
                )
                ok += 1
            except Exception as exc:
                failed.append((patch_id, str(exc)))
                logger.error("Failed nc->pt for %s: %s", patch_id, exc)

        summary[variant] = {"ok": ok, "failed": failed}
        logger.info("%s nc->pt: ok=%d failed=%d", variant, ok, len(failed))

    write_json(cfg.work_dir / "03_pt_base_summary.json", summary)
