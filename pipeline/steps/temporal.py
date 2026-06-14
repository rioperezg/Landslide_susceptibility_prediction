from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from Procesamiento.feature_functions import add_patch_days_to_json, full_patch_id
from pipeline.config import PipelineConfig
from pipeline.utils import (
    filter_files_by_numbers,
    list_nc_files,
    patch_number,
    read_event_date,
    write_json,
)

logger = logging.getLogger(__name__)


def _trim_patch_to_last_n_before_event(
    src: Path,
    dst: Path,
    n_timesteps: int,
    event_attrs: list[str],
) -> dict:
    with xr.open_dataset(src) as ds:
        event_date = read_event_date(ds, event_attrs)
        if event_date is None:
            return {"status": "skipped", "reason": "missing_event_date"}

        reference = np.datetime64(pd.Timestamp(event_date).normalize())
        time_vals = ds["time"].values.astype("datetime64[ns]")
        pre_idx = np.where(time_vals < reference)[0]

        if len(pre_idx) < n_timesteps:
            return {
                "status": "discarded",
                "reason": f"only_{len(pre_idx)}_timesteps_before_event",
                "event_date": str(reference),
            }

        last_idx = pre_idx[-n_timesteps:]
        ds_out = ds.isel(time=last_idx).load()
        ds_out.attrs["reference_date_used"] = str(reference)
        ds_out.attrs["n_pre_reference_kept"] = n_timesteps
        ds_out.attrs["selected_time_indices_original"] = ",".join(map(str, last_idx))
        ds_out.attrs["selected_times"] = ",".join(str(t) for t in ds_out["time"].values)

        dst.parent.mkdir(parents=True, exist_ok=True)
        ds_out.to_netcdf(dst)

    return {
        "status": "kept",
        "event_date": str(reference),
        "selected_times": ds_out.attrs["selected_times"],
    }


def run_temporal(cfg: PipelineConfig, common_numbers: set[str]) -> set[str]:
    """Record dates and keep the last N timestamps before each landslide event."""
    n_timesteps = int(cfg.temporal["n_timesteps"])
    event_attrs = list(cfg.temporal["event_date_attrs"])
    event_dates: dict[str, str] = {}
    summary: dict[str, dict] = {}

    for variant in cfg.variants:
        in_dir = cfg.variant_input(variant)
        out_dir = cfg.variant_work("01_trimmed", variant)
        out_dir.mkdir(parents=True, exist_ok=True)

        patch_days_path = cfg.work_dir / "metadata" / f"patch_days_{variant}.json"
        add_patch_days_to_json(str(in_dir), str(patch_days_path))

        variant_summary = {"kept": [], "discarded": [], "skipped": []}
        for src in filter_files_by_numbers(list_nc_files(in_dir), common_numbers):
            number = patch_number(src.name)
            dst = out_dir / src.name
            result = _trim_patch_to_last_n_before_event(
                src=src,
                dst=dst,
                n_timesteps=n_timesteps,
                event_attrs=event_attrs,
            )
            variant_summary[result["status"]].append(src.name)

            if result["status"] == "kept":
                patch_id = full_patch_id(str(dst))
                event_dates[patch_id] = result["event_date"]

        summary[variant] = {
            key: {"count": len(value), "files": value}
            for key, value in variant_summary.items()
        }
        logger.info(
            "%s temporal trim: kept=%d discarded=%d skipped=%d",
            variant,
            len(variant_summary["kept"]),
            len(variant_summary["discarded"]),
            len(variant_summary["skipped"]),
        )

    # Reconcile: keep only patches that survived trimming in every variant.
    surviving: set[str] | None = None
    for variant in cfg.variants:
        out_dir = cfg.variant_work("01_trimmed", variant)
        numbers = {patch_number(path.name) for path in list_nc_files(out_dir)}
        surviving = numbers if surviving is None else surviving & numbers

    assert surviving is not None
    for variant in cfg.variants:
        out_dir = cfg.variant_work("01_trimmed", variant)
        for path in list_nc_files(out_dir):
            if patch_number(path.name) not in surviving:
                path.unlink()

    write_json(cfg.work_dir / "metadata" / "event_dates.json", event_dates)
    write_json(cfg.work_dir / "temporal_summary.json", summary)
    write_json(
        cfg.work_dir / "metadata" / "surviving_patch_numbers.json",
        sorted(surviving, key=lambda x: int(x)),
    )

    logger.info("Patches surviving temporal filter in all variants: %d", len(surviving))
    return surviving
