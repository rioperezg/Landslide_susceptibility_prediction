# VARIABLES QUE SE AÑADIRÁN EN UN FUTURO CON LOS ARCHIVOS PT, CON EL FIN DE AGILIZAR EL PROCESO DE LA FASE 2:
# AÑADIR PENDIENTE, ASPECTO, CURVATURA DE PERFIL, LS, SPI, TWI, Distance to drainage, NDVI, NBR, DELTA_NBR
# LAS QUE NO SE PUEDEN: LITOLOGÍA, PRECIPITACIONES, AREA DE DRENAJE
# ESO SÍ FALTA AÑADIR 

# ------------------------------------------
# FUNCIONES PARA CONFECCIONAR DEM REGIONAL

# lonlat_to_utm
# sector_bounds_from_lonlat
# raster_center_in_bounds
# create_dem_mosaic_from_files
# process_area_drainage_by_sectors
# nc_center_in_bounds
# add_area_drainage_to_patches_by_sector
# Cargamos librerías
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
import torch

import rasterio
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import train_test_split
# from PIL import Image

from joblib import Parallel, delayed
import string
import numpy as np


import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import time

# from whitebox import WhiteboxTools


# así que primero hay que transformarlas a UTM 32N.
def lonlat_to_utm(lon, lat, transformer):
    x, y = transformer.transform(lon, lat)
    return x, y

# Y ahora generamos bounds UTM automáticamente
def sector_bounds_from_lonlat(points_dict, transformer):
    xs = []
    ys = []

    for lon, lat in points_dict.values():
        x, y = lonlat_to_utm(lon, lat, transformer)
        xs.append(x)
        ys.append(y)

    return min(xs), min(ys), max(xs), max(ys)

# Usamos el centro/bounds del .tif de cada patch

def raster_center_in_bounds(tif_path, bounds):
    xmin, ymin, xmax, ymax = bounds

    with rasterio.open(tif_path) as src:
        b = src.bounds
        cx = (b.left + b.right) / 2
        cy = (b.bottom + b.top) / 2

    return (xmin <= cx <= xmax) and (ymin <= cy <= ymax)


# Esta es una versión sectorizada de create_dem_mosaic:

def create_dem_mosaic_from_files(tif_files, output_tif):
    if len(tif_files) == 0:
        raise ValueError("No hay archivos .tif para crear el mosaico")

    srcs = []

    try:
        for fp in tif_files:
            srcs.append(rasterio.open(fp))

        mosaic, out_transform = merge(srcs)

        meta = srcs[0].meta.copy()
        meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform
        })

        os.makedirs(os.path.dirname(output_tif), exist_ok=True)

        with rasterio.open(output_tif, "w", **meta) as dst:
            dst.write(mosaic)

    finally:
        for s in srcs:
            s.close()

    print(f"Mosaico sector guardado en: {output_tif}")
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



def process_area_drainage_by_sectors(
    dem_tif_dir,
    sectors,
    out_dir,
    method="breach",
    pixel_size=10.0
):
    os.makedirs(out_dir, exist_ok=True)

    all_tifs = sorted(glob.glob(os.path.join(dem_tif_dir, "*.tif")))

    sector_outputs = {}

    for sector_name, bounds in sectors.items():

        print(f"\n==============================")
        print(f"Procesando {sector_name}")
        print(f"Bounds: {bounds}")
        print(f"==============================")

        sector_tifs = [
            fp for fp in all_tifs
            if raster_center_in_bounds(fp, bounds)
        ]

        print(f"Patches DEM en sector: {len(sector_tifs)}")

        if len(sector_tifs) == 0:
            print("Sector vacío, se omite.")
            continue

        sector_dir = os.path.join(out_dir, sector_name)
        os.makedirs(sector_dir, exist_ok=True)

        dem_mosaic = os.path.join(sector_dir, f"{sector_name}_dem_mosaic.tif")
        filled_dem = os.path.join(sector_dir, f"{sector_name}_filled_dem.tif")
        flow_dir = os.path.join(sector_dir, f"{sector_name}_flow_direction.tif")
        flow_acc = os.path.join(sector_dir, f"{sector_name}_flow_accumulation.tif")
        area_drainage = os.path.join(sector_dir, f"{sector_name}_area_drainage.tif")

        create_dem_mosaic_from_files(
            tif_files=sector_tifs,
            output_tif=dem_mosaic
        )

        compute_filled_dem_and_flow_direction(
            dem_tif=dem_mosaic,
            out_filled_dem=filled_dem,
            out_flow_direction=flow_dir,
            working_dir=sector_dir,
            method=method
        )

        compute_flow_accumulation(
            filled_dem_tif=filled_dem,
            out_flow_accumulation_tif=flow_acc,
            working_dir=sector_dir,
            out_type="cells",
            log_transform=False,
            clip=False
        )

        compute_area_drainage(
            flow_accum_tif=flow_acc,
            out_area_drainage_tif=area_drainage,
            pixel_size=pixel_size
        )

        sector_outputs[sector_name] = {
            "bounds": bounds,
            "dem_mosaic": dem_mosaic,
            "filled_dem": filled_dem,
            "flow_direction": flow_dir,
            "flow_accumulation": flow_acc,
            "area_drainage": area_drainage,
            "n_patches": len(sector_tifs),
            "patch_tifs": sector_tifs
        }

    return sector_outputs

# Necesitamos asignar cada .nc al sector según su centro
def nc_center_in_bounds(nc_file, bounds):
    xmin, ymin, xmax, ymax = bounds

    ds = xr.open_dataset(nc_file)
    ds.load()
    ds.close()

    x = ds["x"].values
    y = ds["y"].values

    cx = float((x.min() + x.max()) / 2)
    cy = float((y.min() + y.max()) / 2)

    return (xmin <= cx <= xmax) and (ymin <= cy <= ymax)

# Y aplicar tu función antigua add_area_drainage_to_patch:
def add_area_drainage_to_patches_by_sector(
    patches_dir,
    sector_outputs,
    area_var_name="area_drainage",
    repeat_in_time=True
):
    patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.nc")))

    ok = 0
    failed = []
    skipped = []

    for patch_file in patch_files:

        matched_sector = None

        for sector_name, info in sector_outputs.items():
            if nc_center_in_bounds(patch_file, info["bounds"]):
                matched_sector = sector_name
                break

        if matched_sector is None:
            skipped.append(patch_file)
            print(f"SKIP sin sector: {os.path.basename(patch_file)}")
            continue

        area_tif = sector_outputs[matched_sector]["area_drainage"]

        try:
            add_area_drainage_to_patch(
                patch_file=patch_file,
                area_drainage_tif=area_tif,
                out_file=patch_file,
                area_var_name=area_var_name,
                repeat_in_time=repeat_in_time
            )

            ok += 1
            print(f"OK {os.path.basename(patch_file)} -> {matched_sector}")

        except Exception as e:
            failed.append((patch_file, matched_sector, str(e)))
            print(f"ERROR {os.path.basename(patch_file)} en {matched_sector}: {e}")

    print("\n--- RESUMEN ---")
    print("OK:", ok)
    print("SKIPPED:", len(skipped))
    print("FAILED:", len(failed))

    return {
        "ok": ok,
        "skipped": skipped,
        "failed": failed
    }


# ------------------------------------------
# FUNCIONES PARA DIVIDIR EL CONJUNTO


def create_random_split(
    patches_dir,
    output_path,
    train_ratio=0.79,
    val_ratio=0.13,
    test_ratio=0.08,
    seed=42
):
    """
    Crea split train/val/test aleatorio sin estratificar.
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Los ratios deben sumar 1."

    patch_files = sorted(glob.glob(os.path.join(patches_dir, "*.nc")))

    patch_ids = np.array([
        os.path.basename(f)
        for f in patch_files
    ])

    print(f"Total patches encontrados: {len(patch_ids)}")

    # ---------- PRIMER SPLIT: TRAIN vs TEMP ----------
    temp_ratio = val_ratio + test_ratio

    train_ids, temp_ids = train_test_split(
        patch_ids,
        test_size=temp_ratio,
        random_state=seed,
        shuffle=True
    )

    # ---------- SEGUNDO SPLIT: VAL vs TEST ----------
    relative_test_ratio = test_ratio / temp_ratio

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=relative_test_ratio,
        random_state=seed,
        shuffle=True
    )

    split_dict = {
        "train": train_ids.tolist(),
        "val": val_ids.tolist(),
        "test": test_ids.tolist()
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2)

    print("\n===== SPLIT =====")
    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")
    print(f"Total: {len(train_ids) + len(val_ids) + len(test_ids)}")

    print(f"\nSplit guardado en: {output_path}")

    return split_dict


# ------------------------------------------
# ESTRUCTURA DEL DATASET NC


class LandslideDataset(Dataset):
    def __init__(self, patches_dir, patch_ids, dynamic_vars, static_vars, mean=None, std=None):
        self.patches_dir = patches_dir
        self.patch_ids = patch_ids
        self.dynamic_vars = dynamic_vars
        self.static_vars = static_vars
        self.mean = mean
        self.std = std
    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]
        path = os.path.join(self.patches_dir, patch_id)

        ds = xr.open_dataset(path)

        # ---------- DINÁMICAS ----------
        dyn_list = []
        for var in self.dynamic_vars:
            arr = ds[var].values.astype(np.float32)  # (time, H, W)
            dyn_list.append(arr)

        # ---------- ESTÁTICAS ----------
        static_list = []
        for var in self.static_vars:
            arr = ds[var].values.astype(np.float32)

            if "time" in ds[var].dims:
                arr = arr[0]  # quitar dimensión time

            # repetir en el tiempo
            time_len = dyn_list[0].shape[0]
            arr = np.repeat(arr[None, :, :], time_len, axis=0)

            static_list.append(arr)

        # ---------- COMBINAR ----------
        all_vars = dyn_list + static_list  # lista de (time, H, W)

        # (channels, time, H, W)
        x = np.stack(all_vars, axis=0)

        # ---------- MASK ----------
        mask = ds["MASK"].values

        if "time" in ds["MASK"].dims:
            mask = mask[0]

        y = mask.astype(np.int64)  # para CrossEntropy

        ds.close()

        # convertir a tensor
        x = torch.from_numpy(x)  # float32
        y = torch.from_numpy(y)

        # ---------- NORMALIZACIÓN ----------
        if self.mean is not None and self.std is not None:
            x = (x - self.mean[:, None, None, None]) / (self.std[:, None, None, None] + 1e-6)

        return x, y
    
# ------------------------------------------
# FUNCIONES PARA INSERTAR CANALES EN EL ORDEN CORRECTO


def insert_or_replace_channel(x, variable_names, new_channel, new_name, final_order=None):
    """
    x: (C, T, H, W)
    new_channel: (T, H, W)
    """

    variable_names = list(variable_names)

    if new_channel.ndim != 3:
        raise ValueError(f"{new_name} debe tener shape (T,H,W)")

    # Si existe, sustituir
    if new_name in variable_names:
        idx = variable_names.index(new_name)
        x[idx] = new_channel
        return x, variable_names

    # Si no existe, añadir al final temporalmente
    x = torch.cat([x, new_channel.unsqueeze(0)], dim=0)
    variable_names.append(new_name)

    # Si hay orden final, reordenar
    if final_order is not None:
        existing_final_order = [v for v in final_order if v in variable_names]

        indices = [variable_names.index(v) for v in existing_final_order]

        x = x[indices]
        variable_names = existing_final_order

    return x, variable_names
    
# ------------------------------------------
# FUNCIONES PARA AÑADIR PENDIENTE PT

# FUNCIONES PARA AÑADIR PENDIENTE PT

def add_slope_to_pt_item(
    item,
    base_variable_names=None,
    dem_var="DEM",
    resolution=10.0,
    final_order=None
):
    x = item["x"].float()

    if "variable_names" in item:
        variable_names = list(item["variable_names"])
    else:
        if base_variable_names is None:
            raise KeyError("El .pt no tiene variable_names y no se pasó base_variable_names")
        variable_names = list(base_variable_names)

    dem_idx = variable_names.index(dem_var)

    dem2d = x[dem_idx, 0].cpu().numpy().astype(np.float32)

    dz_dy, dz_dx = np.gradient(dem2d, resolution, resolution)

    slope_rad = np.arctan(
        np.sqrt(dz_dx**2 + dz_dy**2)
    ).astype(np.float32)

    slope = np.degrees(slope_rad).astype(np.float32)

    T = x.shape[1]

    slope_rad_t = torch.from_numpy(slope_rad).unsqueeze(0).repeat(T, 1, 1)
    slope_t = torch.from_numpy(slope).unsqueeze(0).repeat(T, 1, 1)

    x, variable_names = insert_or_replace_channel(
        x, variable_names, slope_t, "slope", final_order
    )

    x, variable_names = insert_or_replace_channel(
        x, variable_names, slope_rad_t, "slope_rad", final_order
    )

    item["x"] = x
    item["variable_names"] = variable_names

    return item



def add_slope_to_pt_folder(
    input_dir,
    output_dir,
    final_order,
    base_variable_names=None,
    overwrite=True
):
    os.makedirs(output_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))

    ok = 0
    failed = []

    for path in tqdm(pt_files):
        try:
            item = torch.load(path, map_location="cpu")

            item = add_slope_to_pt_item(
                item=item,
                base_variable_names=base_variable_names,
                dem_var="DEM",
                resolution=10.0,
                final_order=final_order
            )

            out_path = os.path.join(output_dir, os.path.basename(path))

            if os.path.exists(out_path) and not overwrite:
                continue

            torch.save(item, out_path)
            ok += 1

        except Exception as e:
            failed.append((path, str(e)))
            print(f"ERROR {os.path.basename(path)}: {e}")

    print("\n===== RESUMEN =====")
    print("OK:", ok)
    print("FAILED:", len(failed))

    if failed:
        print("\nPrimeros errores:")
        for fp, err in failed[:10]:
            print(os.path.basename(fp), "->", err)

    return failed



# ------------------------------------------
# FUNCIONES PARA AÑADIR ASPECTO PT

def add_aspect_to_pt_item(
    item,
    base_variable_names,
    dem_var="DEM",
    resolution=10.0,
    final_order=None,
    keep_aux=True
):
    """
    Añade aspect_sin, aspect_cos y opcionalmente aspect_rad/aspect a un item .pt.

    item["x"]: (C, T, H, W)
    item["variable_names"]: lista de nombres de canales
    """

    x = item["x"].float()
    if "variable_names" in item:
        variable_names = list(item["variable_names"])
    else:
        if base_variable_names is None:
            raise KeyError(
            "El .pt no tiene variable_names "
            "y no se pasó base_variable_names"
        )

    variable_names = list(base_variable_names)

    dem_idx = variable_names.index(dem_var)

    # DEM estático: usamos timestamp 0
    dem2d = x[dem_idx, 0].cpu().numpy().astype(np.float32)

    # Gradientes espaciales
    dz_dy, dz_dx = np.gradient(dem2d, resolution, resolution)

    # Aspecto en radianes
    aspect_rad = np.arctan2(-dz_dx, dz_dy)

    # Pasar de [-pi, pi] a [0, 2pi)
    aspect_rad = np.mod(aspect_rad, 2 * np.pi).astype(np.float32)

    # Aspecto en grados
    aspect_deg = np.degrees(aspect_rad).astype(np.float32)

    # Transformación circular
    aspect_sin = np.sin(aspect_rad).astype(np.float32)
    aspect_cos = np.cos(aspect_rad).astype(np.float32)

    T = x.shape[1]

    def repeat_2d(arr2d):
        return torch.from_numpy(arr2d).unsqueeze(0).repeat(T, 1, 1)

    # Variables que sí usarías como input
    channels_to_add = [
        ("aspect_sin", repeat_2d(aspect_sin)),
        ("aspect_cos", repeat_2d(aspect_cos)),
    ]

    # Variables auxiliares opcionales
    if keep_aux:
        channels_to_add += [
            ("aspect_rad", repeat_2d(aspect_rad)),
            ("aspect", repeat_2d(aspect_deg)),
        ]

    for name, channel in channels_to_add:
        x, variable_names = insert_or_replace_channel(
            x=x,
            variable_names=variable_names,
            new_channel=channel,
            new_name=name,
            final_order=final_order
        )

    item["x"] = x
    item["variable_names"] = variable_names

    return item


def add_aspect_to_pt_folder(
    input_dir,
    output_dir,
    final_order=None,
    base_variable_names=None,
    dem_var="DEM",
    resolution=10.0,
    keep_aux=True,
    overwrite=False
):
    os.makedirs(output_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))

    ok = 0
    failed = []

    for path in tqdm(pt_files):
        try:
            item = torch.load(path, map_location="cpu")

            item = add_aspect_to_pt_item(
                item=item,
                base_variable_names=base_variable_names,
                dem_var=dem_var,
                resolution=resolution,
                final_order=final_order,
                keep_aux=keep_aux
            )

            out_path = os.path.join(output_dir, os.path.basename(path))

            if os.path.exists(out_path) and not overwrite:
                continue

            torch.save(item, out_path)
            ok += 1

        except Exception as e:
            failed.append((path, str(e)))
            print(f"ERROR {os.path.basename(path)}: {e}")

    print("\n===== RESUMEN =====")
    print("OK:", ok)
    print("FAILED:", len(failed))

    if failed:
        print("\nPrimeros errores:")
        for fp, err in failed[:10]:
            print(os.path.basename(fp), "->", err)

    return failed


# ------------------------------------------
# FUNCIONES PARA AÑADIR CURVATURA DE PERFIL PT

def add_profile_curvature_to_pt_item(
    item,
    base_variable_names=None,
    dem_var="DEM",
    resolution=10.0,
    final_order=None
):
    """
    Añade profile_curvature a un item .pt.

    item["x"]: (C, T, H, W)
    item["variable_names"]: lista de nombres de canales
    """

    x = item["x"].float()
    if "variable_names" in item:
        variable_names = list(item["variable_names"])
    else:
        if base_variable_names is None:
            raise KeyError(
                "El .pt no tiene variable_names "
                "y no se pasó base_variable_names"
            )
        variable_names = list(base_variable_names)

    if dem_var not in variable_names:
        raise KeyError(f"No existe {dem_var} en variable_names")

    dem_idx = variable_names.index(dem_var)

    # DEM estático: usamos timestamp 0
    z = x[dem_idx, 0].cpu().numpy().astype(np.float32)

    # Primeras derivadas
    dz_dy, dz_dx = np.gradient(z, resolution, resolution)

    # Segundas derivadas
    d2z_dy2, d2z_dydx = np.gradient(dz_dy, resolution, resolution)
    d2z_dxdy, d2z_dx2 = np.gradient(dz_dx, resolution, resolution)

    p = dz_dx
    q = dz_dy
    r = d2z_dx2
    s = 0.5 * (d2z_dxdy + d2z_dydx)
    t = d2z_dy2

    grad2 = p**2 + q**2

    eps = 1e-12
    denom = np.maximum(grad2, eps) * np.power(1 + grad2, 1.5)

    profile_curvature = -(
        r * p**2 + 2 * s * p * q + t * q**2
    ) / denom

    profile_curvature = profile_curvature.astype(np.float32)

    T = x.shape[1]

    profile_curvature_t = (
        torch.from_numpy(profile_curvature)
        .unsqueeze(0)
        .repeat(T, 1, 1)
    )

    x, variable_names = insert_or_replace_channel(
        x=x,
        variable_names=variable_names,
        new_channel=profile_curvature_t,
        new_name="profile_curvature",
        final_order=final_order
    )

    item["x"] = x
    item["variable_names"] = variable_names

    return item


def add_profile_curvature_to_pt_folder(
    input_dir,
    output_dir,
    final_order,
    base_variable_names=None,
    dem_var="DEM",
    resolution=10.0,
    overwrite=False
):
    os.makedirs(output_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))

    ok = 0
    failed = []

    for path in tqdm(pt_files):
        try:
            item = torch.load(path, map_location="cpu")

            item = add_profile_curvature_to_pt_item(
                item=item,
                base_variable_names=base_variable_names,
                dem_var=dem_var,
                resolution=resolution,
                final_order=final_order
            )

            out_path = os.path.join(output_dir, os.path.basename(path))

            if os.path.exists(out_path) and not overwrite:
                continue

            torch.save(item, out_path)
            ok += 1

        except Exception as e:
            failed.append((path, str(e)))
            print(f"ERROR {os.path.basename(path)}: {e}")

    print("\n===== RESUMEN =====")
    print("OK:", ok)
    print("FAILED:", len(failed))

    if failed:
        print("\nPrimeros errores:")
        for fp, err in failed[:10]:
            print(os.path.basename(fp), "->", err)

    return failed

# ------------------------------------------
# FUNCIONES PARA AÑADIR LS PT

def add_ls_to_pt_item(
    item,
    slope_var="slope_rad",
    area_var="area_drainage",
    ls_var_name="ls",
    cellsize=10.0,
    m=0.4,
    n=1.3,
    eps=1e-6,
    final_order=None
):
    """
    Añade LS a un item .pt.

    Requiere:
    - slope_rad
    - area_drainage
    """

    x = item["x"].float()
    variable_names = list(item["variable_names"])

    if slope_var not in variable_names:
        raise KeyError(f"No existe {slope_var} en variable_names")

    if area_var not in variable_names:
        raise KeyError(f"No existe {area_var} en variable_names")

    slope_idx = variable_names.index(slope_var)
    area_idx = variable_names.index(area_var)

    slope_rad = x[slope_idx]       # (T,H,W)
    area_drainage = x[area_idx]    # (T,H,W)

    # área -> longitud equivalente
    area_spec = area_drainage / float(cellsize)
    area_spec = torch.where(
        area_spec > 0,
        area_spec,
        torch.tensor(float("nan"), device=area_spec.device)
    )

    # pendiente
    sin_beta = torch.sin(slope_rad)
    sin_beta = torch.where(
        sin_beta > eps,
        sin_beta,
        torch.tensor(eps, device=sin_beta.device)
    )

    # LS
    ls = ((area_spec / 22.13) ** m) * ((sin_beta / 0.0896) ** n)
    ls = ls.float()  # (T,H,W)

    x, variable_names = insert_or_replace_channel(
        x=x,
        variable_names=variable_names,
        new_channel=ls,
        new_name=ls_var_name,
        final_order=final_order
    )

    item["x"] = x
    item["variable_names"] = variable_names
    print("print en add_ls_to_pt_item:", variable_names)
    print(x.shape)

    return item


def add_ls_to_pt_folder(
    input_dir,
    output_dir,
    final_order=None,
    slope_var="slope_rad",
    area_var="area_drainage",
    ls_var_name="ls",
    cellsize=10.0,
    m=0.4,
    n=1.3,
    eps=1e-6,
    overwrite=False
):
    os.makedirs(output_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))

    ok = 0
    failed = []

    for path in tqdm(pt_files):
        try:
            item = torch.load(path, map_location="cpu")

            item = add_ls_to_pt_item(
                item=item,
                slope_var=slope_var,
                area_var=area_var,
                ls_var_name=ls_var_name,
                cellsize=cellsize,
                m=m,
                n=n,
                eps=eps,
                final_order=final_order
            )

            out_path = os.path.join(output_dir, os.path.basename(path))

            if os.path.exists(out_path) and not overwrite:
                continue
            print("print en add_ls_to_pt_folder:", item["variable_names"])

            torch.save(item, out_path)
            ok += 1

        except Exception as e:
            failed.append((path, str(e)))
            print(f"ERROR {os.path.basename(path)}: {e}")

    print("\n===== RESUMEN =====")
    print("OK:", ok)
    print("FAILED:", len(failed))

    if failed:
        print("\nPrimeros errores:")
        for fp, err in failed[:10]:
            print(os.path.basename(fp), "->", err)

    return failed



# ------------------------------------------
# FUNCIONES PARA AÑADIR SPI PT


def add_spi_to_pt_item(
    item,
    slope_var="slope_rad",
    area_var="area_drainage",
    spi_var_name="spi",
    eps=1e-6
):
    """
    Añade SPI a un item .pt.

    SPI = area_drainage * tan(slope_rad)

    Requiere:
    - slope_rad
    - area_drainage
    """

    x = item["x"].float()
    variable_names = list(item["variable_names"])

    if slope_var not in variable_names:
        raise KeyError(f"No existe {slope_var} en variable_names")

    if area_var not in variable_names:
        raise KeyError(f"No existe {area_var} en variable_names")

    slope_idx = variable_names.index(slope_var)
    area_idx = variable_names.index(area_var)

    slope_rad = x[slope_idx]       # (T,H,W)
    area_drainage = x[area_idx]    # (T,H,W)

    # área válida
    area_drainage = torch.where(
        area_drainage > 0,
        area_drainage,
        torch.tensor(float("nan"), device=area_drainage.device)
    )

    # tan(beta)
    tan_beta = torch.tan(slope_rad)
    tan_beta = torch.where(
        torch.isfinite(tan_beta),
        tan_beta,
        torch.tensor(float("nan"), device=tan_beta.device)
    )

    tan_beta = torch.where(
        tan_beta > eps,
        tan_beta,
        torch.tensor(eps, device=tan_beta.device)
    )

    spi = area_drainage * tan_beta
    spi = spi.float()

    x, variable_names = insert_or_replace_channel(
        x=x,
        variable_names=variable_names,
        new_channel=spi,
        new_name=spi_var_name,
        final_order=None
    )

    item["x"] = x
    item["variable_names"] = variable_names

    return item


def add_spi_to_pt_folder(
    input_dir,
    output_dir,
    slope_var="slope_rad",
    area_var="area_drainage",
    spi_var_name="spi",
    eps=1e-6,
    overwrite=True
):
    os.makedirs(output_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))

    ok = 0
    failed = []

    for path in tqdm(pt_files):
        try:
            item = torch.load(path, map_location="cpu")

            item = add_spi_to_pt_item(
                item=item,
                slope_var=slope_var,
                area_var=area_var,
                spi_var_name=spi_var_name,
                eps=eps
            )

            out_path = os.path.join(output_dir, os.path.basename(path))

            if os.path.exists(out_path) and not overwrite:
                continue

            torch.save(item, out_path)
            ok += 1

        except Exception as e:
            failed.append((path, str(e)))
            print(f"ERROR {os.path.basename(path)}: {e}")

    print("\n===== RESUMEN =====")
    print("OK:", ok)
    print("FAILED:", len(failed))

    if failed:
        print("\nPrimeros errores:")
        for fp, err in failed[:10]:
            print(os.path.basename(fp), "->", err)

    return failed


# ------------------------------------------
# FUNCIONES PARA AÑADIR TWI PT


def add_twi_to_pt_item(
    item,
    slope_var="slope_rad",
    area_var="area_drainage",
    twi_var_name="twi",
    eps=1e-6
):
    """
    Añade TWI a un item .pt.

    TWI = ln(area_drainage / tan(slope_rad))

    Requiere:
    - slope_rad
    - area_drainage
    """

    x = item["x"].float()
    variable_names = list(item["variable_names"])

    if slope_var not in variable_names:
        raise KeyError(f"No existe {slope_var} en variable_names")

    if area_var not in variable_names:
        raise KeyError(f"No existe {area_var} en variable_names")

    slope_idx = variable_names.index(slope_var)
    area_idx = variable_names.index(area_var)

    slope_rad = x[slope_idx]       # (T,H,W)
    area_drainage = x[area_idx]    # (T,H,W)

    # área válida
    area_drainage = torch.where(
        area_drainage > eps,
        area_drainage,
        torch.tensor(float("nan"), device=area_drainage.device)
    )

    # tan(beta)
    tan_beta = torch.tan(slope_rad)

    tan_beta = torch.where(
        torch.isfinite(tan_beta),
        tan_beta,
        torch.tensor(float("nan"), device=tan_beta.device)
    )

    tan_beta = torch.where(
        tan_beta > eps,
        tan_beta,
        torch.tensor(eps, device=tan_beta.device)
    )

    # TWI
    twi = torch.log(area_drainage / tan_beta)

    twi = torch.where(
        torch.isfinite(twi),
        twi,
        torch.tensor(float("nan"), device=twi.device)
    )

    twi = twi.float()

    x, variable_names = insert_or_replace_channel(
        x=x,
        variable_names=variable_names,
        new_channel=twi,
        new_name=twi_var_name,
        final_order=None
    )

    item["x"] = x
    item["variable_names"] = variable_names

    return item


def add_twi_to_pt_folder(
    input_dir,
    output_dir,
    slope_var="slope_rad",
    area_var="area_drainage",
    twi_var_name="twi",
    eps=1e-6,
    overwrite=True
):
    os.makedirs(output_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))

    ok = 0
    failed = []

    for path in tqdm(pt_files):
        try:
            item = torch.load(path, map_location="cpu")

            item = add_twi_to_pt_item(
                item=item,
                slope_var=slope_var,
                area_var=area_var,
                twi_var_name=twi_var_name,
                eps=eps
            )

            out_path = os.path.join(output_dir, os.path.basename(path))

            if os.path.exists(out_path) and not overwrite:
                continue

            torch.save(item, out_path)
            ok += 1

        except Exception as e:
            failed.append((path, str(e)))
            print(f"ERROR {os.path.basename(path)}: {e}")

    print("\n===== RESUMEN =====")
    print("OK:", ok)
    print("FAILED:", len(failed))

    if failed:
        print("\nPrimeros errores:")
        for fp, err in failed[:10]:
            print(os.path.basename(fp), "->", err)

    return failed


# ------------------------------------------
# FUNCIONES PARA AÑADIR DISTANCIA A DRENAJE PT
def add_distance_to_drainage_to_pt_item(
    item,
    area_var="area_drainage",
    dist_var_name="distance_to_drainage",
    log10_threshold=5.0,
    cellsize_x=10.0,
    cellsize_y=10.0
):
    """
    Añade distance_to_drainage a un item .pt.

    La red de drenaje se define como:
        log10(area_drainage) > log10_threshold

    Requiere:
    - area_drainage
    """

    x = item["x"].float()
    variable_names = list(item["variable_names"])

    if area_var not in variable_names:
        raise KeyError(f"No existe {area_var} en variable_names")

    area_idx = variable_names.index(area_var)

    # area_drainage estático: usamos timestamp 0
    area = x[area_idx, 0].cpu().numpy().astype("float32")

    valid = np.isfinite(area) & (area > 0)

    drainage = valid & (np.log10(area) > log10_threshold)

    if not np.any(drainage):
        raise ValueError(
            f"No se detectaron píxeles de drenaje con log10(area) > {log10_threshold}"
        )

    non_drainage = ~drainage

    dist = distance_transform_edt(
        non_drainage,
        sampling=(cellsize_y, cellsize_x)
    ).astype("float32")

    dist[~valid] = np.nan

    T = x.shape[1]

    dist_t = (
        torch.from_numpy(dist)
        .unsqueeze(0)
        .repeat(T, 1, 1)
    )

    x, variable_names = insert_or_replace_channel(
        x=x,
        variable_names=variable_names,
        new_channel=dist_t,
        new_name=dist_var_name,
        final_order=None
    )

    item["x"] = x
    item["variable_names"] = variable_names

    return item

def add_distance_to_drainage_to_pt_folder(
    input_dir,
    output_dir,
    area_var="area_drainage",
    dist_var_name="distance_to_drainage",
    log10_threshold=5.0,
    cellsize_x=10.0,
    cellsize_y=10.0,
    overwrite=True
):
    os.makedirs(output_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))

    ok = 0
    failed = []

    for path in tqdm(pt_files):
        try:
            item = torch.load(path, map_location="cpu")

            item = add_distance_to_drainage_to_pt_item(
                item=item,
                area_var=area_var,
                dist_var_name=dist_var_name,
                log10_threshold=log10_threshold,
                cellsize_x=cellsize_x,
                cellsize_y=cellsize_y
            )

            out_path = os.path.join(output_dir, os.path.basename(path))

            if os.path.exists(out_path) and not overwrite:
                continue

            torch.save(item, out_path)
            ok += 1

        except Exception as e:
            failed.append((path, str(e)))
            print(f"ERROR {os.path.basename(path)}: {e}")

    print("\n===== RESUMEN =====")
    print("OK:", ok)
    print("FAILED:", len(failed))

    if failed:
        print("\nPrimeros errores:")
        for fp, err in failed[:10]:
            print(os.path.basename(fp), "->", err)

    return failed

# ------------------------------------------
# FUNCIONES PARA AÑADIR NDVI A  PT

def add_ndvi_to_pt_item(
    item,
    b8_name="B08",
    b4_name="B04",
    ndvi_var_name="NDVI",
    eps=1e-8
):
    """
    Añade NDVI a un item .pt.

    NDVI = (B08 - B04) / (B08 + B04)

    A diferencia de las topográficas, NDVI es dinámico:
    se calcula para cada timestamp.

    Requiere:
    - B08
    - B04
    """

    x = item["x"].float()
    variable_names = list(item["variable_names"])

    if b8_name not in variable_names:
        raise KeyError(f"No existe {b8_name} en variable_names")

    if b4_name not in variable_names:
        raise KeyError(f"No existe {b4_name} en variable_names")

    b8_idx = variable_names.index(b8_name)
    b4_idx = variable_names.index(b4_name)

    b8 = x[b8_idx]  # (T,H,W)
    b4 = x[b4_idx]  # (T,H,W)

    den = b8 + b4

    ndvi = torch.where(
        torch.abs(den) > eps,
        (b8 - b4) / den,
        torch.tensor(float("nan"), device=den.device)
    )

    ndvi = ndvi.float()

    x, variable_names = insert_or_replace_channel(
        x=x,
        variable_names=variable_names,
        new_channel=ndvi,
        new_name=ndvi_var_name,
        final_order=None
    )

    item["x"] = x
    item["variable_names"] = variable_names

    return item


def add_ndvi_to_pt_folder(
    input_dir,
    output_dir,
    b8_name="B08",
    b4_name="B04",
    ndvi_var_name="NDVI",
    eps=1e-8,
    overwrite=True
):
    os.makedirs(output_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))

    ok = 0
    failed = []

    for path in tqdm(pt_files):
        try:
            item = torch.load(path, map_location="cpu")

            item = add_ndvi_to_pt_item(
                item=item,
                b8_name=b8_name,
                b4_name=b4_name,
                ndvi_var_name=ndvi_var_name,
                eps=eps
            )

            out_path = os.path.join(output_dir, os.path.basename(path))

            if os.path.exists(out_path) and not overwrite:
                continue

            torch.save(item, out_path)
            ok += 1

        except Exception as e:
            failed.append((path, str(e)))
            print(f"ERROR {os.path.basename(path)}: {e}")

    print("\n===== RESUMEN =====")
    print("OK:", ok)
    print("FAILED:", len(failed))

    if failed:
        print("\nPrimeros errores:")
        for fp, err in failed[:10]:
            print(os.path.basename(fp), "->", err)

    return failed

# ------------------------------------------
# FUNCIONES PARA AÑADIR NBR A  PT

def add_nbr_to_pt_item(
    item,
    b8_name="B08",
    b12_name="B12",
    nbr_var_name="NBR",
    delta_nbr_var_name="delta_NBR",
    eps=1e-8,
    add_delta=True
):
    """
    Añade NBR y opcionalmente delta_NBR a un item .pt.

    NBR = (B08 - B12) / (B08 + B12)

    delta_NBR:
    - delta_NBR(t0) = NBR(t0)
    - delta_NBR(t) = NBR(t) - NBR(t-1)
    """

    x = item["x"].float()
    variable_names = list(item["variable_names"])

    if b8_name not in variable_names:
        raise KeyError(f"No existe {b8_name} en variable_names")

    if b12_name not in variable_names:
        raise KeyError(f"No existe {b12_name} en variable_names")

    b8_idx = variable_names.index(b8_name)
    b12_idx = variable_names.index(b12_name)

    b8 = x[b8_idx]      # (T,H,W)
    b12 = x[b12_idx]    # (T,H,W)

    den = b8 + b12

    nbr = torch.where(
        torch.abs(den) > eps,
        (b8 - b12) / den,
        torch.tensor(float("nan"), device=den.device)
    )

    nbr = nbr.float()

    x, variable_names = insert_or_replace_channel(
        x=x,
        variable_names=variable_names,
        new_channel=nbr,
        new_name=nbr_var_name,
        final_order=None
    )

    if add_delta:
        delta_nbr = torch.empty_like(nbr)
        delta_nbr[0] = nbr[0]
        delta_nbr[1:] = nbr[1:] - nbr[:-1]

        delta_nbr = delta_nbr.float()

        x, variable_names = insert_or_replace_channel(
            x=x,
            variable_names=variable_names,
            new_channel=delta_nbr,
            new_name=delta_nbr_var_name,
            final_order=None
        )

    item["x"] = x
    item["variable_names"] = variable_names

    return item

def add_nbr_to_pt_folder(
    input_dir,
    output_dir,
    b8_name="B08",
    b12_name="B12",
    nbr_var_name="NBR",
    delta_nbr_var_name="delta_NBR",
    eps=1e-8,
    add_delta=True,
    overwrite=True
):
    os.makedirs(output_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))

    ok = 0
    failed = []

    for path in tqdm(pt_files):
        try:
            item = torch.load(path, map_location="cpu")

            item = add_nbr_to_pt_item(
                item=item,
                b8_name=b8_name,
                b12_name=b12_name,
                nbr_var_name=nbr_var_name,
                delta_nbr_var_name=delta_nbr_var_name,
                eps=eps,
                add_delta=add_delta
            )

            out_path = os.path.join(output_dir, os.path.basename(path))

            if os.path.exists(out_path) and not overwrite:
                continue

            torch.save(item, out_path)
            ok += 1

        except Exception as e:
            failed.append((path, str(e)))
            print(f"ERROR {os.path.basename(path)}: {e}")

    print("\n===== RESUMEN =====")
    print("OK:", ok)
    print("FAILED:", len(failed))

    if failed:
        print("\nPrimeros errores:")
        for fp, err in failed[:10]:
            print(os.path.basename(fp), "->", err)

    return failed
