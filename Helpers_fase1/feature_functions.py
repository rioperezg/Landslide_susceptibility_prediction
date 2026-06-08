import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs

import os
import glob
from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd


import shutil

import ast
from collections import Counter
import intake

import json
from pathlib import Path
# import mlcast_datasets

from pyproj import CRS, Transformer
import tempfile
import fiona
    

import re

import rasterio
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt


def save_last_8_before_reference(in_dir, out_dir, reference_date):
    os.makedirs(out_dir, exist_ok=True)
    files = glob.glob(os.path.join(in_dir, "*.nc"))

    kept = 0
    discarded = 0

    for f in files:
        with xr.open_dataset(f) as ds:
            time_vals = ds["time"].values.astype("datetime64[ns]")

            # índices de timestamps anteriores a la fecha de referencia
            pre_idx = np.where(time_vals < reference_date)[0]

            if len(pre_idx) < 8:
                discarded += 1
                continue

            last8_idx = pre_idx[-8:]
            ds_out = ds.isel(time=last8_idx).load()

            # guardar metadatos útiles
            ds_out.attrs["reference_date_used"] = str(reference_date)
            ds_out.attrs["n_pre_reference_kept"] = 8
            ds_out.attrs["selected_time_indices_original"] = ",".join(map(str, last8_idx))
            ds_out.attrs["selected_times"] = ",".join(str(t) for t in ds_out["time"].values)

            out_path = os.path.join(out_dir, os.path.basename(f))
            ds_out.to_netcdf(out_path)
            kept += 1

    print(f"\nCarpeta origen: {in_dir}")
    print(f"Carpeta destino: {out_dir}")
    print(f"Archivos guardados: {kept}")
    print(f"Archivos descartados (<8 timestamps antes de {reference_date}): {discarded}")


# ----------------------------------------

def full_patch_id(filepath):
    filename = os.path.basename(filepath)
    stem = os.path.splitext(filename)[0]

    m = re.match(r"^(italy)_(s2|s1asc|s1dsc)_(\d+)$", stem, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"No se pudo extraer patch_id de: {filename}")

    country = m.group(1).lower()
    sensor = m.group(2).lower()
    number = m.group(3)

    return f"{country}_{sensor}_{number}"


def add_patch_days_to_json(patches_dir, out_json):
    files = glob.glob(os.path.join(patches_dir, "*.nc"))

    # Si ya existe el JSON, lo cargamos
    if os.path.exists(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            patch_days = json.load(f)
    else:
        patch_days = {}

    added = 0
    overwritten = 0

    for f in files:
        patch_id = full_patch_id(f)

        with xr.open_dataset(f) as ds:
            times = ds["time"].values

            # Convertimos cada timestamp a fecha YYYY-MM-DD
            days = [
                str(pd.Timestamp(t).normalize().date())
                for t in times
            ]

        if patch_id in patch_days:
            overwritten += 1
            print(f"Aviso: {patch_id} ya existía en el JSON y se va a actualizar.")
        else:
            added += 1

        patch_days[patch_id] = days

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(patch_days, f, indent=2, ensure_ascii=False)

    print(f"Entradas nuevas añadidas: {added}")
    print(f"Entradas actualizadas: {overwritten}")
    print(f"Total de entradas en el JSON: {len(patch_days)}")

    return patch_days


def unique_days_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        patch_days = json.load(f)

    unique_days = sorted({
        day
        for days in patch_days.values()
        for day in days
    })

    return np.array(unique_days, dtype="datetime64[ns]")

# -----------------------------------------


def add_slope_to_patch(ds, dem_var="DEM", resolution=10.0, repeat_in_time=True):
    dem = ds[dem_var]

    # Si el DEM tiene dimensión temporal, usamos la primera capa
    if "time" in dem.dims:
        dem2d = dem.isel(time=0)
    else:
        dem2d = dem

    # Gradientes espaciales
    dz_dy, dz_dx = np.gradient(dem2d.values, resolution, resolution)

    # Pendiente en radianes
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

    # Pendiente en grados
    slope_deg = np.degrees(slope_rad)

    # DataArrays base 2D
    slope_rad_da = xr.DataArray(
        slope_rad,
        coords=dem2d.coords,
        dims=dem2d.dims,
        name="slope_rad"
    )

    slope_deg_da = xr.DataArray(
        slope_deg,
        coords=dem2d.coords,
        dims=dem2d.dims,
        name="slope"
    )

    slope_rad_da.attrs = {
        "units": "radians",
        "description": "Slope computed from DEM"
    }
    slope_deg_da.attrs = {
        "units": "degrees",
        "description": "Slope computed from DEM"
    }

    ds_out = ds.copy()

    if repeat_in_time and "time" in ds.coords:
        slope_rad_da = slope_rad_da.expand_dims(time=ds["time"]).transpose("time", "y", "x")
        slope_deg_da = slope_deg_da.expand_dims(time=ds["time"]).transpose("time", "y", "x")

    ds_out["slope_rad"] = slope_rad_da
    ds_out["slope"] = slope_deg_da

    return ds_out



def add_slope_inplace_folder(
    patches_dir,
    dem_var="DEM",
    resolution=10.0
):
    patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.nc")))

    ok = 0
    failed = 0
    failed_files = []

    for patch_file in patch_files:
        try:
            # 1) abrir, cargar todo en memoria y cerrar
            ds = xr.open_dataset(patch_file)
            ds.load()
            ds.close()

            # 2) crear dataset enriquecido
            ds_out = add_slope_to_patch(ds, dem_var=dem_var, resolution=resolution)

            # 3) guardar primero en temporal
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc")
            os.close(tmp_fd)

            ds_out.to_netcdf(tmp_path, mode="w")

            # 4) reemplazar original
            os.replace(tmp_path, patch_file)

            ok += 1
            print(f"OK: {os.path.basename(patch_file)}")

        except Exception as e:
            failed += 1
            failed_files.append((patch_file, str(e)))
            print(f"ERROR en {os.path.basename(patch_file)}: {e}")

            # limpieza por si el temporal quedó creado
            try:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except:
                pass

    print("\n--- RESUMEN ---")
    print(f"OK: {ok}")
    print(f"FAILED: {failed}")

    if failed_files:
        print("\nArchivos con error:")
        for f, err in failed_files:
            print(f"- {os.path.basename(f)} -> {err}")

    return {
        "ok": ok,
        "failed": failed,
        "failed_files": failed_files
    }


# -----------------------------------------



def add_aspect_to_patch(ds, dem_var="DEM", resolution=10.0, repeat_in_time=True):
    dem = ds[dem_var]

    # Si el DEM tiene tiempo, usamos una sola capa
    if "time" in dem.dims:
        dem2d = dem.isel(time=0)
    else:
        dem2d = dem

    # Gradientes espaciales
    dz_dy, dz_dx = np.gradient(dem2d.values, resolution, resolution)

    # Aspecto en radianes
    aspect_rad = np.arctan2(-dz_dx, dz_dy)

    # Pasar de [-pi, pi] a [0, 2pi)
    aspect_rad = np.mod(aspect_rad, 2 * np.pi)

    # Aspecto en grados
    aspect_deg = np.degrees(aspect_rad)

    # Opcional: en zonas casi planas, el aspecto no está bien definido
    # slope_mag = np.sqrt(dz_dx**2 + dz_dy**2)
    # flat_mask = slope_mag < 1e-6

    # aspect_rad = np.where(flat_mask, np.nan, aspect_rad)
    # aspect_deg = np.where(flat_mask, np.nan, aspect_deg)

    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)

    # DataArrays base
    aspect_rad_da = xr.DataArray(
        aspect_rad,
        coords=dem2d.coords,
        dims=dem2d.dims,
        name="aspect_rad"
    )

    aspect_deg_da = xr.DataArray(
        aspect_deg,
        coords=dem2d.coords,
        dims=dem2d.dims,
        name="aspect"
    )

    aspect_sin_da = xr.DataArray(
        aspect_sin,
        coords=dem2d.coords,
        dims=dem2d.dims,
        name="aspect_sin"
    )

    aspect_cos_da = xr.DataArray(
        aspect_cos,
        coords=dem2d.coords,
        dims=dem2d.dims,
        name="aspect_cos"
    )

    aspect_rad_da.attrs = {
        "units": "radians",
        "description": "Aspect computed from DEM"
    }
    aspect_deg_da.attrs = {
        "units": "degrees",
        "description": "Aspect computed from DEM, clockwise from North"
    }
    aspect_sin_da.attrs = {
        "description": "Sine transform of aspect"
    }
    aspect_cos_da.attrs = {
        "description": "Cosine transform of aspect"
    }

    ds_out = ds.copy()

    if repeat_in_time and "time" in ds.coords:
        aspect_rad_da = aspect_rad_da.expand_dims(time=ds["time"]).transpose("time", "y", "x")
        aspect_deg_da = aspect_deg_da.expand_dims(time=ds["time"]).transpose("time", "y", "x")
        aspect_sin_da = aspect_sin_da.expand_dims(time=ds["time"]).transpose("time", "y", "x")
        aspect_cos_da = aspect_cos_da.expand_dims(time=ds["time"]).transpose("time", "y", "x")

    ds_out["aspect_rad"] = aspect_rad_da
    ds_out["aspect"] = aspect_deg_da
    ds_out["aspect_sin"] = aspect_sin_da
    ds_out["aspect_cos"] = aspect_cos_da

    return ds_out

def add_aspect_inplace_folder(
    patches_dir,
    dem_var="DEM",
    resolution=10.0
):
    patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.nc")))

    ok = 0
    failed = 0

    for patch_file in patch_files:
        try:
            # abrir, cargar y cerrar
            ds = xr.open_dataset(patch_file)
            ds.load()
            ds.close()

            # añadir aspect
            ds_out = add_aspect_to_patch(ds, dem_var=dem_var, resolution=resolution)

            # guardar en archivo temporal
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc")
            os.close(tmp_fd)

            ds_out.to_netcdf(tmp_path, mode="w")

            # reemplazar original
            os.replace(tmp_path, patch_file)

            ok += 1
            print(f"OK: {os.path.basename(patch_file)}")

        except Exception as e:
            failed += 1
            print(f"ERROR en {os.path.basename(patch_file)}: {e}")

    print("\n--- RESUMEN ---")
    print(f"OK: {ok}")
    print(f"FAILED: {failed}")


# -----------------------------------------





def add_profile_curvature_to_patch(ds, dem_var="DEM", resolution=10.0, repeat_in_time=True):
    dem = ds[dem_var]

    # Si el DEM tiene tiempo, usamos una sola capa
    if "time" in dem.dims:
        dem2d = dem.isel(time=0)
    else:
        dem2d = dem

    z = dem2d.values

    # Primeras derivadas
    dz_dy, dz_dx = np.gradient(z, resolution, resolution)

    # Segundas derivadas
    d2z_dy2, d2z_dydx = np.gradient(dz_dy, resolution, resolution)
    d2z_dxdy, d2z_dx2 = np.gradient(dz_dx, resolution, resolution)

    p = dz_dx
    q = dz_dy
    r = d2z_dx2
    s = 0.5 * (d2z_dxdy + d2z_dydx)   # derivada mixta simétrica
    t = d2z_dy2

    grad2 = p**2 + q**2

    # Evitar divisiones por cero en zonas planas
    eps = 1e-12
    denom = np.maximum(grad2, eps) * np.power(1 + grad2, 1.5)

    profile_curvature = - (r * p**2 + 2 * s * p * q + t * q**2) / denom

    # # En zonas casi planas, la curvatura de perfil no es muy estable
    # flat_mask = grad2 < 1e-20
    # profile_curvature = np.where(flat_mask, np.nan, profile_curvature)

    pc_da = xr.DataArray(
        profile_curvature,
        coords=dem2d.coords,
        dims=dem2d.dims,
        name="profile_curvature"
    )

    pc_da.attrs = {
        "description": "Profile curvature computed from DEM",
        "units": "1/m"
    }

    ds_out = ds.copy()

    if repeat_in_time and "time" in ds.coords:
        pc_da = pc_da.expand_dims(time=ds["time"]).transpose("time", "y", "x")

    ds_out["profile_curvature"] = pc_da

    return ds_out

def add_profile_curvature_inplace_folder(
    patches_dir,
    dem_var="DEM",
    resolution=10.0
):
    patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.nc")))

    ok = 0
    failed = 0
    failed_files = []

    for patch_file in patch_files:
        tmp_path = None
        try:
            # Abrir, cargar en memoria y cerrar
            ds = xr.open_dataset(patch_file)
            ds.load()
            ds.close()

            # Añadir curvatura de perfil
            ds_out = add_profile_curvature_to_patch(
                ds,
                dem_var=dem_var,
                resolution=resolution,
                repeat_in_time=True
            )

            # Guardar en temporal y reemplazar original
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc")
            os.close(tmp_fd)

            ds_out.to_netcdf(tmp_path, mode="w")
            os.replace(tmp_path, patch_file)

            ok += 1
            print(f"OK: {os.path.basename(patch_file)}")

        except Exception as e:
            failed += 1
            failed_files.append((patch_file, str(e)))
            print(f"ERROR en {os.path.basename(patch_file)}: {e}")

            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass

    print("\n--- RESUMEN ---")
    print(f"Carpeta origen: {patches_dir}")
    print(f"Procesados correctamente: {ok}")
    print(f"Con error: {failed}")

    if failed_files:
        print("\nArchivos con error:")
        for f, err in failed_files:
            print(f"- {os.path.basename(f)} -> {err}")

    return {
        "ok": ok,
        "failed": failed,
        "failed_files": failed_files
    }


# -----------------------------------------

# FUNCIONES PARA CREAR MOSAICO DEM

def export_dem_patches_from_nc_folder(
    nc_dir,
    out_tif_dir,
    dem_var="DEM",
    crs_fallback=None
):
    """
    Recorre todos los .nc de una carpeta, extrae la variable DEM
    y la guarda como GeoTIFF en otra carpeta.

    Parámetros
    ----------
    nc_dir : str
        Carpeta con archivos .nc
    out_tif_dir : str
        Carpeta de salida para los .tif
    dem_var : str
        Nombre de la variable DEM dentro del .nc
    crs_fallback : str o None
        CRS de respaldo, por ejemplo 'EPSG:32632', si no se puede leer del archivo
    """
    os.makedirs(out_tif_dir, exist_ok=True)

    nc_files = sorted(glob.glob(os.path.join(nc_dir, "*.nc")))

    ok = 0
    failed = 0
    failed_files = []

    for nc_file in nc_files:
        try:
            filename = os.path.basename(nc_file)
            stem = os.path.splitext(filename)[0]
            out_tif = os.path.join(out_tif_dir, f"{stem}.tif")

            ds = xr.open_dataset(nc_file)
            ds.load()
            ds.close()

            if dem_var not in ds.data_vars:
                raise KeyError(f"La variable '{dem_var}' no existe en {filename}")

            dem = ds[dem_var]

            # Si el DEM viene repetido en time, cogemos una sola capa
            if "time" in dem.dims:
                dem2d = dem.isel(time=0)
            else:
                dem2d = dem

            z = dem2d.values

            if z.ndim != 2:
                raise ValueError(f"El DEM de {filename} no es 2D después de seleccionar time")

            x = ds["x"].values
            y = ds["y"].values

            if x.ndim != 1 or y.ndim != 1:
                raise ValueError(f"Las coordenadas x/y de {filename} no son 1D")

            if len(x) < 2 or len(y) < 2:
                raise ValueError(f"No hay suficientes coordenadas x/y en {filename}")

            # Resoluciones
            dx = float(np.mean(np.diff(x)))
            dy = float(np.mean(np.diff(y)))

            # GeoTransform tipo rasterio
            # x[0] e y[0] son centros de píxel, así que desplazamos media celda
            transform = Affine.translation(x[0] - dx / 2, y[0] - dy / 2) * Affine.scale(dx, dy)

            # CRS: intentamos leerlo de spatial_ref
            crs = None
            if "spatial_ref" in ds:
                spatial_ref = ds["spatial_ref"]
                if "crs_wkt" in spatial_ref.attrs:
                    crs = spatial_ref.attrs["crs_wkt"]
                elif "spatial_ref" in spatial_ref.attrs:
                    crs = spatial_ref.attrs["spatial_ref"]

            if crs is None:
                crs = crs_fallback

            if crs is None:
                raise ValueError(
                    f"No se pudo determinar el CRS para {filename}. "
                    f"Usa crs_fallback='EPSG:32632' o similar."
                )

            with rasterio.open(
                out_tif,
                "w",
                driver="GTiff",
                height=z.shape[0],
                width=z.shape[1],
                count=1,
                dtype=z.dtype,
                crs=crs,
                transform=transform,
                nodata=np.nan if np.issubdtype(z.dtype, np.floating) else None
            ) as dst:
                dst.write(z, 1)

            ok += 1
            print(f"OK: {filename} -> {os.path.basename(out_tif)}")

        except Exception as e:
            failed += 1
            failed_files.append((nc_file, str(e)))
            print(f"ERROR en {os.path.basename(nc_file)}: {e}")

    print("\n--- RESUMEN ---")
    print(f"Carpeta origen: {nc_dir}")
    print(f"Carpeta salida: {out_tif_dir}")
    print(f"Convertidos correctamente: {ok}")
    print(f"Con error: {failed}")

    if failed_files:
        print("\nArchivos con error:")
        for f, err in failed_files:
            print(f"- {os.path.basename(f)} -> {err}")

    return {
        "ok": ok,
        "failed": failed,
        "failed_files": failed_files
    }


def create_dem_mosaic(
    dem_patches_dir,
    output_tif,
    dem_band_name=None
):
    """
    Crea un mosaico DEM a partir de múltiples archivos raster.

    Parámetros:
    - dem_patches_dir: carpeta con los DEM (.tif)
    - output_tif: ruta de salida del mosaico
    - dem_band_name: opcional, por si quieres filtrar archivos
    """

    # Buscar archivos
    tif_files = sorted(glob.glob(os.path.join(dem_patches_dir, "*.tif")))

    if len(tif_files) == 0:
        raise ValueError("No se encontraron archivos .tif en la carpeta")

    print(f"Encontrados {len(tif_files)} patches DEM")

    srcs = []
    for fp in tif_files:
        try:
            src = rasterio.open(fp)
            srcs.append(src)
        except Exception as e:
            print(f"Error abriendo {fp}: {e}")

    print("Creando mosaico...")

    # Merge
    mosaic, out_transform = merge(srcs)

    # Metadata base
    meta = srcs[0].meta.copy()
    meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })

    # Guardar mosaico
    with rasterio.open(output_tif, "w", **meta) as dst:
        dst.write(mosaic)

    print(f"Mosaico guardado en: {output_tif}")

    # Cerrar archivos
    for s in srcs:
        s.close()

    print("Archivos cerrados correctamente")

    return output_tif





def compute_filled_dem_and_flow_direction(
    dem_tif,
    out_filled_dem,
    out_flow_direction,
    working_dir=None,
    method="breach"
):
    """
    A partir de un DEM en GeoTIFF:
      1) corrige depresiones (fill o breach)
      2) calcula flow direction D8

    Parámetros
    ----------
    dem_tif : str
        Ruta al DEM de entrada (.tif)
    out_filled_dem : str
        Ruta al DEM corregido de salida (.tif)
    out_flow_direction : str
        Ruta al raster de flow direction D8 (.tif)
    working_dir : str or None
        Carpeta de trabajo para WhiteboxTools. Si es None, usa la carpeta del DEM.
    method : str
        "breach" o "fill"
        - breach: suele preservar mejor la morfología
        - fill: rellena depresiones
    """

    if not os.path.exists(dem_tif):
        raise FileNotFoundError(f"No existe el DEM de entrada: {dem_tif}")

    if working_dir is None:
        working_dir = os.path.dirname(os.path.abspath(dem_tif))

    os.makedirs(os.path.dirname(out_filled_dem), exist_ok=True)
    os.makedirs(os.path.dirname(out_flow_direction), exist_ok=True)

    wbt = WhiteboxTools()
    wbt.set_working_dir(working_dir)

    print("Whitebox working dir:", working_dir)
    print("DEM de entrada:", dem_tif)

    # 1) Corregir depresiones
    if method.lower() == "breach":
        print("Corrigiendo depresiones con breach_depressions_least_cost...")
        wbt.breach_depressions_least_cost(
            dem=dem_tif,
            output=out_filled_dem, dist = 50
        )
    elif method.lower() == "fill":
        print("Corrigiendo depresiones con fill_depressions...")
        wbt.fill_depressions(
            dem=dem_tif,
            output=out_filled_dem
        )
    else:
        raise ValueError("method debe ser 'breach' o 'fill'")

    print("DEM corregido guardado en:", out_filled_dem)

    # 2) Flow direction D8
    print("Calculando flow direction D8...")
    wbt.d8_pointer(
        dem=out_filled_dem,
        output=out_flow_direction
    )

    print("Flow direction guardado en:", out_flow_direction)

    return {
        "filled_dem": out_filled_dem,
        "flow_direction": out_flow_direction
    }


# -----------------------------------------





def compute_filled_dem_and_flow_direction(
    dem_tif,
    out_filled_dem,
    out_flow_direction,
    working_dir=None,
    method="breach"
):
    """
    A partir de un DEM en GeoTIFF:
      1) corrige depresiones (fill o breach)
      2) calcula flow direction D8

    Parámetros
    ----------
    dem_tif : str
        Ruta al DEM de entrada (.tif)
    out_filled_dem : str
        Ruta al DEM corregido de salida (.tif)
    out_flow_direction : str
        Ruta al raster de flow direction D8 (.tif)
    working_dir : str or None
        Carpeta de trabajo para WhiteboxTools. Si es None, usa la carpeta del DEM.
    method : str
        "breach" o "fill"
        - breach: suele preservar mejor la morfología
        - fill: rellena depresiones
    """

    if not os.path.exists(dem_tif):
        raise FileNotFoundError(f"No existe el DEM de entrada: {dem_tif}")

    if working_dir is None:
        working_dir = os.path.dirname(os.path.abspath(dem_tif))

    os.makedirs(os.path.dirname(out_filled_dem), exist_ok=True)
    os.makedirs(os.path.dirname(out_flow_direction), exist_ok=True)

    wbt = WhiteboxTools()
    wbt.set_working_dir(working_dir)

    print("Whitebox working dir:", working_dir)
    print("DEM de entrada:", dem_tif)

    # 1) Corregir depresiones
    if method.lower() == "breach":
        print("Corrigiendo depresiones con breach_depressions_least_cost...")
        wbt.breach_depressions_least_cost(
            dem=dem_tif,
            output=out_filled_dem, dist = 50
        )
    elif method.lower() == "fill":
        print("Corrigiendo depresiones con fill_depressions...")
        wbt.fill_depressions(
            dem=dem_tif,
            output=out_filled_dem
        )
    else:
        raise ValueError("method debe ser 'breach' o 'fill'")

    print("DEM corregido guardado en:", out_filled_dem)

    # 2) Flow direction D8
    print("Calculando flow direction D8...")
    wbt.d8_pointer(
        dem=out_filled_dem,
        output=out_flow_direction
    )

    print("Flow direction guardado en:", out_flow_direction)

    return {
        "filled_dem": out_filled_dem,
        "flow_direction": out_flow_direction
    }


def compute_flow_accumulation(
    filled_dem_tif,
    out_flow_accumulation_tif,
    working_dir=None,
    out_type="cells",
    log_transform=False,
    clip=False
):
    """
    Calcula flow accumulation D8 a partir de un DEM ya corregido.

    Parámetros
    ----------
    filled_dem_tif : str
        Ruta al DEM corregido (fill o breach) en GeoTIFF.
    out_flow_accumulation_tif : str
        Ruta de salida para el raster de flow accumulation.
    working_dir : str or None
        Carpeta de trabajo para WhiteboxTools. Si es None, usa la carpeta del DEM.
    out_type : str
        Tipo de salida. Whitebox acepta típicamente:
        - "cells"   -> número de celdas acumuladas
        - "catchment area"
        - "specific contributing area"
        Para tu caso recomiendo empezar con "cells".
    log_transform : bool
        Si True, aplica logaritmo a la salida.
    clip : bool
        Si True, recorta valores extremos para visualización.
    """

    if not os.path.exists(filled_dem_tif):
        raise FileNotFoundError(f"No existe el DEM corregido de entrada: {filled_dem_tif}")

    if working_dir is None:
        working_dir = os.path.dirname(os.path.abspath(filled_dem_tif))

    os.makedirs(os.path.dirname(out_flow_accumulation_tif), exist_ok=True)

    wbt = WhiteboxTools()
    wbt.set_working_dir(working_dir)

    print("Whitebox working dir:", working_dir)
    print("DEM corregido de entrada:", filled_dem_tif)
    print("Calculando flow accumulation D8...")

    wbt.d8_flow_accumulation(
        i=filled_dem_tif,
        output=out_flow_accumulation_tif,
        out_type=out_type,
        log=log_transform,
        clip=clip
    )

    print("Flow accumulation guardado en:", out_flow_accumulation_tif)

    return {
        "flow_accumulation": out_flow_accumulation_tif
    }


def compute_area_drainage(
    flow_accum_tif,
    out_area_drainage_tif,
    pixel_size=10.0
):
    """
    Calcula el área de drenaje a partir de un raster de flow accumulation.

    Parámetros
    ----------
    flow_accum_tif : str
        Ruta al raster de flow accumulation (en número de celdas).
    out_area_drainage_tif : str
        Ruta de salida del raster de área de drenaje.
    pixel_size : float
        Tamaño del píxel en metros. Por defecto 10 m.

    Retorna
    -------
    dict
        Diccionario con la ruta del archivo generado.
    """

    if not os.path.exists(flow_accum_tif):
        raise FileNotFoundError(f"No existe el raster de flow accumulation: {flow_accum_tif}")

    out_dir = os.path.dirname(out_area_drainage_tif)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pixel_area = pixel_size * pixel_size  # m²

    with rasterio.open(flow_accum_tif) as src:
        acc = src.read(1).astype("float32")
        meta = src.meta.copy()
        nodata = src.nodata

    # Calcular área de drenaje
    area = acc * pixel_area

    # Respetar nodata si existe
    if nodata is not None:
        mask = acc == nodata
        area[mask] = nodata

    meta.update(dtype="float32")

    with rasterio.open(out_area_drainage_tif, "w", **meta) as dst:
        dst.write(area.astype("float32"), 1)

    print(f"Área de drenaje guardada en: {out_area_drainage_tif}")

    return {
        "area_drainage": out_area_drainage_tif
    }



# -----------------------------------------


def add_area_drainage_to_patch(
    patch_file,
    area_drainage_tif,
    out_file,
    area_var_name="area_drainage",
    repeat_in_time=True
):
    """
    Añade area_drainage a un solo patch .nc a partir de un raster global .tif.

    Parámetros
    ----------
    patch_file : str
        Ruta al patch .nc de entrada.
    area_drainage_tif : str
        Ruta al raster global de area_drainage (.tif).
    out_file : str
        Ruta de salida del nuevo .nc enriquecido.
    area_var_name : str
        Nombre de la variable a añadir.
    repeat_in_time : bool
        Si True y el patch tiene dimensión time, repite la capa en todos los timestamps.
    """

    ds = xr.open_dataset(patch_file)
    ds.load()
    ds.close()

    x = ds["x"].values
    y = ds["y"].values

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Las coordenadas x e y del patch deben ser 1D")

    if len(x) < 2 or len(y) < 2:
        raise ValueError("No hay suficientes coordenadas x/y para definir resolución espacial")

    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))

    # Bounding box a partir de centros de píxel
    xmin = float(x.min() - dx / 2)
    xmax = float(x.max() + dx / 2)
    ymin = float(y.min() - abs(dy) / 2)
    ymax = float(y.max() + abs(dy) / 2)

    with rasterio.open(area_drainage_tif) as src:
        transform = src.transform
        nodata = src.nodata

        window = from_bounds(xmin, ymin, xmax, ymax, transform=transform)
        window = window.round_offsets().round_lengths()

        area_patch = src.read(1, window=window).astype("float32")

    expected_shape = (len(y), len(x))
    if area_patch.shape != expected_shape:
        raise ValueError(
            f"Shape recortada {area_patch.shape} no coincide con la esperada {expected_shape}"
        )

    if nodata is not None:
        area_patch[area_patch == nodata] = np.nan

    area_da = xr.DataArray(
        area_patch,
        coords={"y": ds["y"], "x": ds["x"]},
        dims=("y", "x"),
        name=area_var_name
    )

    area_da.attrs = {
        "units": "m2",
        "description": "Drainage area extracted from global raster"
    }

    if repeat_in_time and "time" in ds.coords:
        area_da = area_da.expand_dims(time=ds["time"]).transpose("time", "y", "x")

    ds_out = ds.copy()
    ds_out[area_var_name] = area_da

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc")
    os.close(tmp_fd)

    try:
        ds_out.to_netcdf(tmp_path, mode="w")
        os.replace(tmp_path, out_file)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return ds_out


def add_area_drainage_inplace_folder(
    patches_dir,
    area_drainage_tif,
    area_var_name="area_drainage",
    repeat_in_time=True
):
    patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.nc")))

    ok = 0
    failed = 0
    failed_files = []

    for patch_file in patch_files:
        try:
            # sobrescribir el mismo archivo
            add_area_drainage_to_patch(
                patch_file=patch_file,
                area_drainage_tif=area_drainage_tif,
                out_file=patch_file,
                area_var_name=area_var_name,
                repeat_in_time=repeat_in_time
            )

            ok += 1
            print(f"OK: {os.path.basename(patch_file)}")

        except Exception as e:
            failed += 1
            failed_files.append((patch_file, str(e)))
            print(f"ERROR en {os.path.basename(patch_file)}: {e}")

    print("\n--- RESUMEN ---")
    print(f"Carpeta: {patches_dir}")
    print(f"Procesados correctamente: {ok}")
    print(f"Con error: {failed}")

    if failed_files:
        print("\nArchivos con error:")
        for f, err in failed_files:
            print(f"- {os.path.basename(f)} -> {err}")

    return {
        "ok": ok,
        "failed": failed,
        "failed_files": failed_files
    }


# -----------------------------------------



def add_ls_to_patch(
    patch_file,
    out_file,
    slope_var="slope_rad",
    area_var="area_drainage",
    ls_var_name="ls",
    cellsize=10.0,
    m=0.4,
    n=1.3,
    eps=1e-6
):
    """
    Añade LS usando slope en radianes directamente.
    """

    ds = xr.open_dataset(patch_file)
    ds.load()
    ds.close()

    if slope_var not in ds:
        raise KeyError(f"No existe la variable '{slope_var}'")

    if area_var not in ds:
        raise KeyError(f"No existe la variable '{area_var}'")

    slope_rad = ds[slope_var].astype("float32")
    area_drainage = ds[area_var].astype("float32")

    # ---- área -> longitud equivalente ----
    area_spec = area_drainage / float(cellsize)

    area_spec = xr.where(area_spec > 0, area_spec, np.nan)

    # ---- pendiente ----
    sin_beta = np.sin(slope_rad)
    sin_beta = xr.where(sin_beta > eps, sin_beta, eps)

    # ---- LS ----
    ls = ((area_spec / 22.13) ** m) * ((sin_beta / 0.0896) ** n)
    ls = ls.astype("float32")
    ls.name = ls_var_name

    ls.attrs = {
        "units": "dimensionless",
        "description": "LS factor",
        "slope_units": "radians"
    }

    ds_out = ds.copy()
    ds_out[ls_var_name] = ls

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc")
    os.close(tmp_fd)

    try:
        ds_out.to_netcdf(tmp_path, mode="w")
        os.replace(tmp_path, out_file)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return ds_out


def add_ls_inplace_folder(
    patches_dir,
    slope_var="slope_rad",
    area_var="area_drainage",
    ls_var_name="ls",
    cellsize=10.0,
    m=0.4,
    n=1.3,
    eps=1e-6
):
    """
    Calcula LS para todos los patches .nc de una carpeta (inplace).

    Requiere que cada patch ya tenga:
    - slope_rad
    - area_drainage
    """

    patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.nc")))

    ok = 0
    failed = 0
    failed_files = []

    for patch_file in patch_files:
        try:
            add_ls_to_patch(
                patch_file=patch_file,
                out_file=patch_file,
                slope_var=slope_var,
                area_var=area_var,
                ls_var_name=ls_var_name,
                cellsize=cellsize,
                m=m,
                n=n,
                eps=eps
            )

            ok += 1
            print(f"OK: {os.path.basename(patch_file)}")

        except Exception as e:
            failed += 1
            failed_files.append((patch_file, str(e)))
            print(f"ERROR en {os.path.basename(patch_file)}: {e}")

    print("\n--- RESUMEN ---")
    print(f"Carpeta: {patches_dir}")
    print(f"Procesados correctamente: {ok}")
    print(f"Con error: {failed}")

    if failed_files:
        print("\nArchivos con error:")
        for f, err in failed_files:
            print(f"- {os.path.basename(f)} -> {err}")

    return {
        "ok": ok,
        "failed": failed,
        "failed_files": failed_files
    }

# -----------------------------------------


def add_spi_to_patch(
    patch_file,
    out_file,
    slope_var="slope_rad",
    area_var="area_drainage",
    spi_var_name="spi",
    eps=1e-6
):
    """
    Añade SPI (Stream Power Index) a un patch .nc.

    Fórmula:
        SPI = As * tan(beta)

    donde:
    - As = área de drenaje (m²)
    - beta = pendiente en radianes
    """

    ds = xr.open_dataset(patch_file)
    ds.load()
    ds.close()

    if slope_var not in ds:
        raise KeyError(f"No existe la variable '{slope_var}'")

    if area_var not in ds:
        raise KeyError(f"No existe la variable '{area_var}'")

    slope_rad = ds[slope_var].astype("float32")
    area_drainage = ds[area_var].astype("float32")

    # evitar áreas no válidas
    area_drainage = xr.where(area_drainage > 0, area_drainage, np.nan)

    tan_beta = np.tan(slope_rad)
    tan_beta = xr.where(np.isfinite(tan_beta), tan_beta, np.nan)
    tan_beta = xr.where(tan_beta > eps, tan_beta, eps)

    spi = area_drainage * tan_beta
    spi = spi.astype("float32")
    spi.name = spi_var_name

    spi.attrs = {
        "units": "m2",
        "description": "Stream Power Index",
        "formula": "SPI = As * tan(beta)",
        "As_units": "m2",
        "slope_units": "radians"
    }

    ds_out = ds.copy()
    ds_out[spi_var_name] = spi

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc")
    os.close(tmp_fd)

    try:
        ds_out.to_netcdf(tmp_path, mode="w")
        os.replace(tmp_path, out_file)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return ds_out


def add_spi_inplace_folder(
    patches_dir,
    slope_var="slope_rad",
    area_var="area_drainage",
    spi_var_name="spi",
    eps=1e-6
):
    """
    Calcula SPI para todos los patches .nc de una carpeta (inplace).

    Requiere que cada patch ya tenga:
    - slope_rad
    - area_drainage
    """

    patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.nc")))

    ok = 0
    failed = 0
    failed_files = []

    for patch_file in patch_files:
        try:
            add_spi_to_patch(
                patch_file=patch_file,
                out_file=patch_file,
                slope_var=slope_var,
                area_var=area_var,
                spi_var_name=spi_var_name,
                eps=eps
            )

            ok += 1
            print(f"OK: {os.path.basename(patch_file)}")

        except Exception as e:
            failed += 1
            failed_files.append((patch_file, str(e)))
            print(f"ERROR en {os.path.basename(patch_file)}: {e}")

    print("\n--- RESUMEN ---")
    print(f"Carpeta: {patches_dir}")
    print(f"Procesados correctamente: {ok}")
    print(f"Con error: {failed}")

    if failed_files:
        print("\nArchivos con error:")
        for f, err in failed_files:
            print(f"- {os.path.basename(f)} -> {err}")

    return {
        "ok": ok,
        "failed": failed,
        "failed_files": failed_files
    }


# -----------------------------------------


def add_twi_to_patch(
    patch_file,
    out_file,
    slope_var="slope_rad",
    area_var="area_drainage",
    twi_var_name="twi",
    eps=1e-6
):
    """
    Añade TWI (Topographic Wetness Index) a un patch .nc.

    Fórmula:
        TWI = ln(As / tan(beta))

    donde:
    - As = área de drenaje (m²)
    - beta = pendiente en radianes
    """

    ds = xr.open_dataset(patch_file)
    ds.load()
    ds.close()

    if slope_var not in ds:
        raise KeyError(f"No existe la variable '{slope_var}'")

    if area_var not in ds:
        raise KeyError(f"No existe la variable '{area_var}'")

    slope_rad = ds[slope_var].astype("float32")
    area_drainage = ds[area_var].astype("float32")

    area_drainage = xr.where(area_drainage > eps, area_drainage, np.nan)

    tan_beta = np.tan(slope_rad)
    tan_beta = xr.where(np.isfinite(tan_beta), tan_beta, np.nan)
    tan_beta = xr.where(tan_beta > eps, tan_beta, eps)

    twi = np.log(area_drainage / tan_beta)
    twi = xr.where(np.isfinite(twi), twi, np.nan)

    twi = twi.astype("float32")
    twi.name = twi_var_name

    twi.attrs = {
        "units": "dimensionless",
        "description": "Topographic Wetness Index",
        "formula": "TWI = ln(As / tan(beta))",
        "As_units": "m2",
        "slope_units": "radians",
        "notes": "Computed with epsilon protection for numerical stability"
    }

    ds_out = ds.copy()
    ds_out[twi_var_name] = twi

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc")
    os.close(tmp_fd)

    try:
        ds_out.to_netcdf(tmp_path, mode="w")
        os.replace(tmp_path, out_file)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return ds_out


def add_twi_inplace_folder(
    patches_dir,
    slope_var="slope_rad",
    area_var="area_drainage",
    twi_var_name="twi",
    eps=1e-6
):
    """
    Calcula TWI para todos los patches .nc de una carpeta (inplace).

    Requiere que cada patch ya tenga:
    - slope_rad
    - area_drainage
    """

    patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.nc")))

    ok = 0
    failed = 0
    failed_files = []

    for patch_file in patch_files:
        try:
            add_twi_to_patch(
                patch_file=patch_file,
                out_file=patch_file,
                slope_var=slope_var,
                area_var=area_var,
                twi_var_name=twi_var_name,
                eps=eps
            )

            ok += 1
            print(f"OK: {os.path.basename(patch_file)}")

        except Exception as e:
            failed += 1
            failed_files.append((patch_file, str(e)))
            print(f"ERROR en {os.path.basename(patch_file)}: {e}")

    print("\n--- RESUMEN ---")
    print(f"Carpeta: {patches_dir}")
    print(f"Procesados correctamente: {ok}")
    print(f"Con error: {failed}")

    if failed_files:
        print("\nArchivos con error:")
        for f, err in failed_files:
            print(f"- {os.path.basename(f)} -> {err}")

    return {
        "ok": ok,
        "failed": failed,
        "failed_files": failed_files
    }


# -----------------------------------------




def add_distance_to_drainage_to_patch(
    patch_file,
    area_var="area_drainage",
    dist_var_name="distance_to_drainage",
    log10_threshold=5.0,
    repeat_in_time=True
):
    """
    Calcula la distancia a la red de drenaje para un patch y devuelve ds_out.

    La red de drenaje se define como:
        log10(area_drainage) > log10_threshold
    """

    ds = xr.open_dataset(patch_file)
    ds.load()
    ds.close()

    x = ds["x"].values
    y = ds["y"].values

    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))

    cellsize_x = abs(dx)
    cellsize_y = abs(dy)

    area = ds[area_var]

    # coger 2D
    if "time" in area.dims:
        area2d = area.isel(time=0).astype("float32")
    else:
        area2d = area.astype("float32")

    area_vals = area2d.values

    valid = np.isfinite(area_vals) & (area_vals > 0)

    # red de drenaje
    drainage = valid & (np.log10(area_vals) > log10_threshold)

    if not np.any(drainage):
        raise ValueError(
            f"No se detectaron píxeles de drenaje con log10(area) > {log10_threshold}"
        )

    # distancia mínima al curso más cercano
    non_drainage = ~drainage

    dist = distance_transform_edt(
        non_drainage,
        sampling=(cellsize_y, cellsize_x)
    ).astype("float32")

    dist[~valid] = np.nan

    dist_da = xr.DataArray(
        dist,
        coords={"y": ds["y"], "x": ds["x"]},
        dims=("y", "x"),
        name=dist_var_name
    )

    dist_da.attrs = {
        "units": "m",
        "description": "Euclidean distance to nearest drainage cell",
        "drainage_definition": f"log10({area_var}) > {log10_threshold}",
        "threshold_area_m2": float(10 ** log10_threshold)
    }

    if repeat_in_time and "time" in ds.coords:
        dist_da = dist_da.expand_dims(time=ds["time"]).transpose("time", "y", "x")

    ds_out = ds.copy()
    ds_out[dist_var_name] = dist_da

    return ds_out



def add_distance_to_drainage_inplace_folder(
    patches_dir,
    area_var="area_drainage",
    dist_var_name="distance_to_drainage",
    log10_threshold=5.0,
    repeat_in_time=True
):
    """
    Calcula distance_to_drainage para todos los patches .nc de una carpeta (inplace).

    Requiere que cada patch ya tenga:
    - area_drainage

    La red de drenaje se define como:
        log10(area_drainage) > log10_threshold
    """

    patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.nc")))

    ok = 0
    failed = 0
    failed_files = []

    for patch_file in patch_files:
        try:
            ds_out = add_distance_to_drainage_to_patch(
                patch_file=patch_file,
                area_var=area_var,
                dist_var_name=dist_var_name,
                log10_threshold=log10_threshold,
                repeat_in_time=repeat_in_time
            )

            # sobrescribir el mismo archivo
            ds_out.to_netcdf(patch_file, mode="w")

            ok += 1
            print(f"OK: {os.path.basename(patch_file)}")

        except Exception as e:
            failed += 1
            failed_files.append((patch_file, str(e)))
            print(f"ERROR en {os.path.basename(patch_file)}: {e}")

    print("\n--- RESUMEN ---")
    print(f"Carpeta: {patches_dir}")
    print(f"Procesados correctamente: {ok}")
    print(f"Con error: {failed}")

    if failed_files:
        print("\nArchivos con error:")
        for f, err in failed_files:
            print(f"- {os.path.basename(f)} -> {err}")

    return {
        "ok": ok,
        "failed": failed,
        "failed_files": failed_files
    }


# -----------------------------------------


def add_ndvi_to_patch(
    patch_file,
    b8_name="B08",
    b4_name="B04",
    ndvi_var_name="NDVI"
):
    ds = xr.open_dataset(patch_file)
    ds.load()
    ds.close()

    if b8_name not in ds:
        raise KeyError(f"La variable '{b8_name}' no existe en el dataset.")
    if b4_name not in ds:
        raise KeyError(f"La variable '{b4_name}' no existe en el dataset.")

    b8 = ds[b8_name]
    b4 = ds[b4_name]
    den = b8 + b4

    ndvi = xr.where(
        den != 0,
        (b8 - b4) / den,
        np.nan
    )

    ndvi.name = ndvi_var_name

    ds_out = ds.copy()
    ds_out[ndvi_var_name] = ndvi

    return ds_out

def add_ndvi_inplace_folder(
    patches_dir,
    b8_name="B08",
    b4_name="B04",
    ndvi_var_name="NDVI"
):
    """
    Añade NDVI a todos los patches .nc de una carpeta (inplace).

    Parámetros
    ----------
    patches_dir : str
        Carpeta con los patches .nc
    b8_name : str
        Nombre de la banda NIR (por defecto B08)
    b4_name : str
        Nombre de la banda roja (por defecto B04)
    ndvi_var_name : str
        Nombre de la variable NDVI a añadir
    """

    patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.nc")))

    ok = 0
    failed = 0
    failed_files = []

    for patch_file in patch_files:
        try:
            ds_out = add_ndvi_to_patch(
                patch_file=patch_file,
                b8_name=b8_name,
                b4_name=b4_name,
                ndvi_var_name=ndvi_var_name
            )

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc")
            os.close(tmp_fd)

            try:
                ds_out.to_netcdf(tmp_path, mode="w")
                os.replace(tmp_path, patch_file)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

            ok += 1
            print(f"OK: {os.path.basename(patch_file)}")

        except Exception as e:
            failed += 1
            failed_files.append((patch_file, str(e)))
            print(f"ERROR en {os.path.basename(patch_file)}: {e}")

    print("\n--- RESUMEN ---")
    print(f"Carpeta: {patches_dir}")
    print(f"Procesados correctamente: {ok}")
    print(f"Con error: {failed}")

    if failed_files:
        print("\nArchivos con error:")
        for f, err in failed_files:
            print(f"- {os.path.basename(f)} -> {err}")

    return {
        "ok": ok,
        "failed": failed,
        "failed_files": failed_files
    }


# -----------------------------------------


def add_nbr_to_patch(
    patch_file,
    b8_name="B08",
    b12_name="B12",
    nbr_var_name="NBR",
    delta_nbr_var_name="delta_NBR"
):
    """
    Añade NBR y delta_NBR a un patch .nc y devuelve el dataset modificado
    sin guardarlo en disco.

    NBR = (B08 - B12) / (B08 + B12)

    La variación temporal se define como:
    - delta_NBR(t0) = NBR(t0)
    - delta_NBR(t) = NBR(t) - NBR(t-1), para t >= 1
    """

    ds = xr.open_dataset(patch_file)
    ds.load()
    ds.close()

    if b8_name not in ds:
        raise KeyError(f"La variable '{b8_name}' no existe en el dataset.")
    if b12_name not in ds:
        raise KeyError(f"La variable '{b12_name}' no existe en el dataset.")
    if "time" not in ds.dims:
        raise ValueError("El dataset no tiene dimensión 'time'.")

    b8 = ds[b8_name]
    b12 = ds[b12_name]

    den = b8 + b12

    nbr = xr.where(
        den != 0,
        (b8 - b12) / den,
        np.nan
    )
    nbr.name = nbr_var_name

    # delta_NBR:
    # t0 = NBR(t0)
    # t>=1 = NBR(t) - NBR(t-1)
    nbr_values = nbr.values
    delta_nbr_values = np.empty_like(nbr_values, dtype=np.float32)

    delta_nbr_values[0] = nbr_values[0]
    delta_nbr_values[1:] = nbr_values[1:] - nbr_values[:-1]

    delta_nbr = xr.DataArray(
        delta_nbr_values,
        coords=nbr.coords,
        dims=nbr.dims,
        name=delta_nbr_var_name
    )

    ds_out = ds.copy()
    ds_out[nbr_var_name] = nbr
    ds_out[delta_nbr_var_name] = delta_nbr

    return ds_out


def add_nbr_inplace_folder(
    patches_dir,
    b8_name="B08",
    b12_name="B12",
    nbr_var_name="NBR",
    delta_nbr_var_name="delta_NBR"
):
    """
    Añade NBR y delta_NBR a todos los patches .nc de una carpeta (inplace).

    Parámetros
    ----------
    patches_dir : str
        Carpeta con los patches .nc
    b8_name : str
        Nombre de la banda NIR (por defecto B08)
    b12_name : str
        Nombre de la banda SWIR2 (por defecto B12)
    nbr_var_name : str
        Nombre de la variable NBR a añadir
    delta_nbr_var_name : str
        Nombre de la variable delta_NBR a añadir
    """

    patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.nc")))

    ok = 0
    failed = 0
    failed_files = []

    for patch_file in patch_files:
        try:
            ds_out = add_nbr_to_patch(
                patch_file=patch_file,
                b8_name=b8_name,
                b12_name=b12_name,
                nbr_var_name=nbr_var_name,
                delta_nbr_var_name=delta_nbr_var_name
            )

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc")
            os.close(tmp_fd)

            try:
                ds_out.to_netcdf(tmp_path, mode="w")
                os.replace(tmp_path, patch_file)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

            ok += 1
            print(f"OK: {os.path.basename(patch_file)}")

        except Exception as e:
            failed += 1
            failed_files.append((patch_file, str(e)))
            print(f"ERROR en {os.path.basename(patch_file)}: {e}")

    print("\n--- RESUMEN ---")
    print(f"Carpeta: {patches_dir}")
    print(f"Procesados correctamente: {ok}")
    print(f"Con error: {failed}")

    if failed_files:
        print("\nArchivos con error:")
        for f, err in failed_files:
            print(f"- {os.path.basename(f)} -> {err}")

    return {
        "ok": ok,
        "failed": failed,
        "failed_files": failed_files
    }


# -----------------------------------------
# FUNCIONES NECESARIAS PARA INTEGRAR DATOS DE LLUVIA 
# CON FUNCIÓN IN FOLDER ACTUALIZADA PARA SOBREESCIBIRSE SOBRE LA MISMA CARPETA




def full_patch_id2(filepath):
    filename = os.path.basename(filepath)
    stem = os.path.splitext(filename)[0]

    m = re.match(r"^(italy)_(s2|s1asc|s1dsc)_(\d+)(?:_.*)?$", stem, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"No se pudo extraer patch_id de: {filename}")

    country = m.group(1).lower()
    sensor = m.group(2).lower()
    number = m.group(3)

    return f"{country}_{sensor}_{number}"

def extract_rain_for_patch(
    patch_file,
    rain_zarr_path,
    patch_days_json,
    patch_indices_json
):
    # 1) Abrir dataset de lluvia
    ds_rain = xr.open_zarr(rain_zarr_path)

    # 2) Cargar JSONs
    with open(patch_days_json, "r", encoding="utf-8") as f:
        patch_days = json.load(f)

    with open(patch_indices_json, "r", encoding="utf-8") as f:
        patch_indices = json.load(f)

    # 3) Obtener patch_id específico: italy_s2_250 / italy_s1asc_250 / italy_s1dsc_250
    patch_id = full_patch_id(patch_file)

    # 4) Obtener patch_id universal: italy_250
    parts = patch_id.split("_")
    patch_id_universal = f"{parts[0]}_{parts[2]}"

    # 5) Fechas del patch
    if patch_id not in patch_days:
        raise KeyError(f"{patch_id} no está en {patch_days_json}")

    days = np.array(patch_days[patch_id], dtype="datetime64[ns]")

    # 6) Índices espaciales del patch
    if patch_id_universal not in patch_indices:
        raise KeyError(f"{patch_id_universal} no está en {patch_indices_json}")

    info = patch_indices[patch_id_universal]

    y0, y1 = info["iy_min"], info["iy_max"]
    x0, x1 = info["ix_min"], info["ix_max"]

    # 7) Extraer lluvia para esas fechas y ese bloque espacial
    ds_patch_rain = ds_rain.sel(time=days).isel(
        y=slice(y0, y1 + 1),
        x=slice(x0, x1 + 1)
    )

    return ds_patch_rain

def enrich_patch_with_rain(
    patch_file,
    rain_zarr_path,
    patch_days_json,
    patch_indices_json,
    out_file
):
    # Patch original
    with xr.open_dataset(patch_file) as ds_patch:
        ds_patch = ds_patch.load()

    # Lluvia asociada
    ds_patch_rain = extract_rain_for_patch(
        patch_file=patch_file,
        rain_zarr_path=rain_zarr_path,
        patch_days_json=patch_days_json,
        patch_indices_json=patch_indices_json
    ).load()

    # Renombrar dims espaciales de lluvia para no chocar con las de Sentinel
    ds_patch_rain = ds_patch_rain.rename({
        "y": "rain_y",
        "x": "rain_x"
    })

    # Integrar variables de lluvia
    ds_out = ds_patch.copy()

    ds_out["prec7"] = ds_patch_rain["prec7"]
    ds_out["prec20"] = ds_patch_rain["prec20"]
    ds_out["max2d_7"] = ds_patch_rain["max2d_7"]

    # Añadir coords de lluvia
    ds_out = ds_out.assign_coords({
        "rain_y": ds_patch_rain["rain_y"],
        "rain_x": ds_patch_rain["rain_x"],
        "rain_lat": ds_patch_rain["lat"],
        "rain_lon": ds_patch_rain["lon"],
    })

    # Metadatos
    ds_out.attrs["rain_features_integrated"] = "prec7, prec20, max2d_7"
    ds_out.attrs["rain_patch_id"] = full_patch_id(patch_file)

    # Guardar
    ds_out.to_netcdf(out_file)

    return ds_out


def enrich_all_patches_inplace_folder(
    patches_dir,
    rain_zarr_path,
    patch_days_json,
    patch_indices_json,
    suffix="_enriched.nc"
):
    """
    Enriquece todos los patches .nc de una carpeta y sobrescribe
    los archivos en la misma carpeta de origen de forma segura.

    Parámetros
    ----------
    patches_dir : str
        Carpeta con los patches .nc
    rain_zarr_path : str
        Ruta al dataset global de precipitación en formato zarr
    patch_days_json : str
        JSON con fechas por patch
    patch_indices_json : str
        JSON con índices/bounding boxes por patch
    suffix : str
        Se mantiene por compatibilidad, pero no se usa para crear nuevos archivos
    """

    patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.nc")))

    ok = 0
    failed = 0
    failed_files = []

    for patch_file in patch_files:
        try:
            filename = os.path.basename(patch_file)

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc")
            os.close(tmp_fd)

            try:
                enrich_patch_with_rain(
                    patch_file=patch_file,
                    rain_zarr_path=rain_zarr_path,
                    patch_days_json=patch_days_json,
                    patch_indices_json=patch_indices_json,
                    out_file=tmp_path
                )

                os.replace(tmp_path, patch_file)

            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

            ok += 1
            print(f"OK: {filename}")

        except Exception as e:
            failed += 1
            failed_files.append((patch_file, str(e)))
            print(f"ERROR en {os.path.basename(patch_file)}: {e}")

    print("\n--- RESUMEN ---")
    print(f"Carpeta origen: {patches_dir}")
    print(f"Procesados correctamente: {ok}")
    print(f"Con error: {failed}")

    if failed_files:
        print("\nArchivos con error:")
        for f, err in failed_files:
            print(f"- {os.path.basename(patch_file)}: {e}")

    print("\n--- RESUMEN ---")
    print(f"Carpeta origen: {patches_dir}")
    print(f"Procesados correctamente: {ok}")
    print(f"Con error: {failed}")

    if failed_files:
        print("\nArchivos con error:")
        for f, err in failed_files:
            print(f"- {os.path.basename(f)} -> {err}")

    return {
        "ok": ok,
        "failed": failed,
        "failed_files": failed_files
    }



# ------------------------------------------
# FUNCIÓN PARA AÑADIR LITOLOGÍA


def add_lithology_to_patch(
    patch_file,
    gdf_lith,
    lith_var_name="lithology_class",
    repeat_in_time=True
):
    ds = xr.open_dataset(patch_file)
    ds.load()
    ds.close()

    x = ds["x"].values
    y = ds["y"].values

    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))

    xmin = float(x.min() - dx / 2)
    xmax = float(x.max() + dx / 2)
    ymin = float(y.min() - abs(dy) / 2)
    ymax = float(y.max() + abs(dy) / 2)

    height = len(y)
    width = len(x)

    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    shapes = (
        (geom, value)
        for geom, value in zip(gdf_lith.geometry, gdf_lith["cat"])
    )

    lith_raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="int16"
    )

    lith_da = xr.DataArray(
        lith_raster,
        coords={"y": ds["y"], "x": ds["x"]},
        dims=("y", "x"),
        name=lith_var_name
    )

    if repeat_in_time and "time" in ds.coords:
        lith_da = lith_da.expand_dims(time=ds["time"]).transpose("time", "y", "x")

    ds_out = ds.copy()
    ds_out[lith_var_name] = lith_da

    return ds_out


def add_lithology_inplace_folder(
    patches_dir,
    gdf_lith,
    lith_var_name="lithology_class",
    cat_col="cat",
    repeat_in_time=True,
    all_touched=True
):
    """
    Añade litología a todos los patches .nc de una carpeta (inplace).

    Requiere que:
    - los patches estén en UTM 32N
    - gdf_lith ya esté reproyectado al mismo CRS que los patches
      (por ejemplo EPSG:32632)

    Parámetros
    ----------
    patches_dir : str
        Carpeta con los patches .nc
    gdf_lith : GeoDataFrame
        Litología ya cargada y reproyectada al CRS del patch
    lith_var_name : str
        Nombre de la variable a añadir
    cat_col : str
        Columna numérica de litología a rasterizar
    repeat_in_time : bool
        Si True, repite la capa en todos los timestamps
    all_touched : bool
        Si True, rasteriza toda celda tocada por el polígono
    """

    patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.nc")))

    ok = 0
    failed = 0
    failed_files = []

    for patch_file in patch_files:
        try:
            ds_out = add_lithology_to_patch(
                patch_file=patch_file,
                gdf_lith=gdf_lith,
                lith_var_name=lith_var_name,
                repeat_in_time=repeat_in_time,
            )

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc")
            os.close(tmp_fd)

            try:
                ds_out.to_netcdf(tmp_path, mode="w")
                os.replace(tmp_path, patch_file)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

            ok += 1
            print(f"OK: {os.path.basename(patch_file)}")

        except Exception as e:
            failed += 1
            failed_files.append((patch_file, str(e)))
            print(f"ERROR en {os.path.basename(patch_file)}: {e}")

    print("\n--- RESUMEN ---")
    print(f"Carpeta: {patches_dir}")
    print(f"Procesados correctamente: {ok}")
    print(f"Con error: {failed}")

    if failed_files:
        print("\nArchivos con error:")
        for f, err in failed_files:
            print(f"- {os.path.basename(f)} -> {err}")

    return {
        "ok": ok,
        "failed": failed,
        "failed_files": failed_files
    }


# ------------------------------------------
# FUNCIÓN PARA ELIMINAR VARIABLES

def remove_rain_variables_inplace_folder(
    patches_dir,
    vars_to_remove=("prec7", "prec20", "max2d_7")
):
    """
    Elimina variables de precipitación de todos los .nc de una carpeta
    sobrescribiendo los archivos inplace.

    Parámetros
    ----------
    patches_dir : str
        Carpeta con los patches .nc
    vars_to_remove : tuple/list
        Variables a eliminar
    """

    patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.nc")))

    ok = 0
    failed = 0
    skipped = 0
    failed_files = []

    for patch_file in patch_files:
        try:
            filename = os.path.basename(patch_file)

            ds = xr.open_dataset(patch_file)
            ds.load()
            ds.close()

            existing_vars = [v for v in vars_to_remove if v in ds]

            if not existing_vars:
                skipped += 1
                print(f"SKIP: {filename} (ninguna variable encontrada)")
                continue

            ds_out = ds.drop_vars(existing_vars)

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc")
            os.close(tmp_fd)

            try:
                ds_out.to_netcdf(tmp_path, mode="w")
                os.replace(tmp_path, patch_file)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

            ok += 1
            print(f"OK: {filename} -> eliminadas {existing_vars}")

        except Exception as e:
            failed += 1
            failed_files.append((patch_file, str(e)))
            print(f"ERROR en {filename}: {e}")

    print("\n--- RESUMEN ---")
    print(f"Carpeta: {patches_dir}")
    print(f"Procesados correctamente: {ok}")
    print(f"Saltados: {skipped}")
    print(f"Con error: {failed}")

    if failed_files:
        print("\nArchivos con error:")
        for f, err in failed_files:
            print(f"- {os.path.basename(f)} -> {err}")

    return {
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "failed_files": failed_files
    }                


# ------------------------------------------RESAMPLEAR VARIABLES DE LLUVIA


def resample_rain_variable_to_patch_grid(ds, var_name):
    """
    Remuestrea una variable de precipitación al grid 128x128 del patch
    usando solo el tamaño de la matriz, no las coordenadas rain_x/rain_y.

    Se asume que la matriz de lluvia ya corresponde espacialmente
    al área completa del patch.
    """

    if var_name not in ds:
        raise KeyError(f"La variable '{var_name}' no existe en el dataset.")

    da = ds[var_name]

    if "rain_y" not in da.dims or "rain_x" not in da.dims:
        raise ValueError(
            f"La variable '{var_name}' no tiene dimensiones ('rain_y', 'rain_x')."
        )

    target_y = ds["y"]
    target_x = ds["x"]

    target_height = len(target_y)
    target_width = len(target_x)

    out_list = []

    has_time = "time" in da.dims

    if has_time:
        n_time = da.sizes["time"]
        iterator = range(n_time)
    else:
        iterator = [None]

    for t in iterator:
        if has_time:
            src = da.isel(time=t).values
            time_value = da["time"].isel(time=t).item()
        else:
            src = da.values

        src = np.asarray(src, dtype=np.float32)

        src_height, src_width = src.shape
        dst = np.empty((target_height, target_width), dtype=np.float32)

        # Transformaciones "ficticias" pero coherentes:
        # ambas matrices se consideran cubriendo el mismo extent [0,1]x[0,1]
        src_transform = from_bounds(0, 0, 1, 1, src_width, src_height)
        dst_transform = from_bounds(0, 0, 1, 1, target_width, target_height)

        reproject(
            source=src,
            destination=dst,
            src_transform=src_transform,
            src_crs="EPSG:4326",
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest
        )

        da_out = xr.DataArray(
            dst,
            coords={"y": target_y, "x": target_x},
            dims=("y", "x"),
            name=var_name
        )

        if has_time:
            da_out = da_out.expand_dims(time=[time_value])

        out_list.append(da_out)

    if has_time:
        da_resampled = xr.concat(out_list, dim="time")
    else:
        da_resampled = out_list[0]

    da_resampled.name = var_name

    return da_resampled



def add_resampled_rain_to_patch2(
    patch_file,
    rain_vars=("prec7", "prec20", "max2d_7"),
    overwrite_original=True,
    suffix="_resampled"
):
    ds = xr.open_dataset(patch_file)
    ds.load()
    ds.close()

    ds_out = ds.copy()

    for var_name in rain_vars:
        if var_name not in ds:
            raise KeyError(f"La variable '{var_name}' no existe en {patch_file}")

        da_resampled = resample_rain_variable_to_patch_grid(ds, var_name)

        # Limpiar coords conflictivas y forzar coords del patch
        if "time" in da_resampled.dims:
            da_clean = xr.DataArray(
                da_resampled.values,
                coords={
                    "time": ds["time"],
                    "y": ds["y"],
                    "x": ds["x"]
                },
                dims=("time", "y", "x"),
                name=var_name if overwrite_original else f"{var_name}{suffix}"
            )
        else:
            da_clean = xr.DataArray(
                da_resampled.values,
                coords={
                    "y": ds["y"],
                    "x": ds["x"]
                },
                dims=("y", "x"),
                name=var_name if overwrite_original else f"{var_name}{suffix}"
            )

        if overwrite_original:
            ds_out[var_name] = da_clean
        else:
            ds_out[f"{var_name}{suffix}"] = da_clean

    return ds_out

def add_resampled_rain_inplace_folder(
    patches_dir,
    rain_vars=("prec7", "prec20", "max2d_7"),
    overwrite_original=True,
    suffix="_resampled"
):
    """
    Remuestrea las variables de precipitación en todos los patches .nc
    de una carpeta y sobrescribe inplace de forma segura.

    Parámetros
    ----------
    patches_dir : str
        Carpeta con los patches .nc
    rain_vars : tuple/list
        Variables de precipitación a remuestrear
    overwrite_original : bool
        Si True, sustituye las variables originales
        Si False, añade nuevas variables con sufijo
    suffix : str
        Sufijo a usar si overwrite_original=False
    """

    patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.nc")))

    ok = 0
    failed = 0
    failed_files = []

    for patch_file in patch_files:
        try:
            ds_out = add_resampled_rain_to_patch2(
                patch_file=patch_file,
                rain_vars=rain_vars,
                overwrite_original=overwrite_original,
                suffix=suffix
            )
            prec7 = ds_out['prec7'].isel(time = 0)
            print(f"Esta es la matriz: {prec7.values}")

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc")
            os.close(tmp_fd)

            try:
                ds_out.to_netcdf(tmp_path, mode="w")
                os.replace(tmp_path, patch_file)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

            ok += 1
            print(f"OK: {os.path.basename(patch_file)}")

        except Exception as e:
            failed += 1
            failed_files.append((patch_file, str(e)))
            print(f"ERROR en {os.path.basename(patch_file)}: {e}")

    print("\n--- RESUMEN ---")
    print(f"Carpeta: {patches_dir}")
    print(f"Procesados correctamente: {ok}")
    print(f"Con error: {failed}")

    if failed_files:
        print("\nArchivos con error:")
        for f, err in failed_files:
            print(f"- {os.path.basename(f)} -> {err}")

    return {
        "ok": ok,
        "failed": failed,
        "failed_files": failed_files
    }