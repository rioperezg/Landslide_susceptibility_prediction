from __future__ import annotations

import logging
import shutil

from pyproj import Transformer

from Procesamiento.feature_functions import export_dem_patches_from_nc_folder
from Helpers_fase2.feature_functions_pt import (
    add_area_drainage_to_patches_by_sector,
    lonlat_to_utm,
    process_area_drainage_by_sectors,
    sector_bounds_from_lonlat,
)
from pipeline.config import PipelineConfig
from pipeline.utils import write_json

logger = logging.getLogger(__name__)


def run_dem_drainage(cfg: PipelineConfig) -> None:
    """Build regional DEM mosaics, compute drainage, and enrich trimmed NetCDF patches."""
    dem_tif_dir = cfg.work_dir / "02_dem" / "patch_tifs"
    hydrology_dir = cfg.work_dir / "02_dem" / "hydrology"
    dem_tif_dir.mkdir(parents=True, exist_ok=True)
    hydrology_dir.mkdir(parents=True, exist_ok=True)

    # Use Sen2 trimmed patches as DEM source (DEM is static and shared).
    dem_source = cfg.variant_work("01_trimmed", "Sen2")
    export_dem_patches_from_nc_folder(
        nc_dir=str(dem_source),
        out_tif_dir=str(dem_tif_dir),
        dem_var=cfg.dem["dem_var"],
        crs_fallback=cfg.dem["crs_fallback"],
    )

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    sector_points_lonlat = cfg.load_sectors_lonlat()
    sectors = {
        name: sector_bounds_from_lonlat(points, transformer)
        for name, points in sector_points_lonlat.items()
    }

    sector_outputs = process_area_drainage_by_sectors(
        dem_tif_dir=str(dem_tif_dir),
        sectors=sectors,
        out_dir=str(hydrology_dir),
        method=cfg.dem["hydrology_method"],
        pixel_size=float(cfg.dem["pixel_size"]),
    )
    write_json(cfg.work_dir / "02_dem" / "sector_outputs.json", {
        name: {key: value for key, value in info.items() if key != "patch_tifs"}
        for name, info in sector_outputs.items()
    })

    for variant in cfg.variants:
        trimmed_dir = cfg.variant_work("01_trimmed", variant)
        enriched_dir = cfg.variant_work("02_enriched_nc", variant)
        enriched_dir.mkdir(parents=True, exist_ok=True)

        for nc_file in trimmed_dir.glob("*.nc"):
            shutil.copy2(nc_file, enriched_dir / nc_file.name)

        result = add_area_drainage_to_patches_by_sector(
            patches_dir=str(enriched_dir),
            sector_outputs=sector_outputs,
            area_var_name="area_drainage",
            repeat_in_time=True,
        )
        write_json(
            cfg.work_dir / "02_dem" / f"area_drainage_report_{variant}.json",
            {
                "ok": result["ok"],
                "skipped": [Path(p).name for p in result["skipped"]],
                "failed": [(Path(p).name, sector, err) for p, sector, err in result["failed"]],
            },
        )
        logger.info(
            "%s area_drainage assignment: ok=%d skipped=%d failed=%d",
            variant,
            result["ok"],
            len(result["skipped"]),
            len(result["failed"]),
        )
