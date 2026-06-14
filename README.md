# Landslide Susceptibility Prediction using Deep Learning and Multi-Source Remote Sensing

## Overview

This repository contains the code developed for a Bachelor's Thesis on **landslide susceptibility prediction using deep learning and multi-source geospatial data**.

The objective is to study whether modern deep learning architectures can identify areas susceptible to landslides by combining **temporal satellite imagery** with **static and dynamic environmental variables**.

Unlike traditional susceptibility mapping approaches that rely mostly on static predictors, this work integrates **multi-temporal Sentinel-1 SAR, Sentinel-2 optical data, topography, lithology, and precipitation** into a unified spatio-temporal framework for semantic segmentation.

The repository includes:

- A **reproducible preprocessing pipeline** (`main.py`) that converts aligned NetCDF patches into enriched PyTorch tensors
- Feature engineering utilities for NetCDF and tensor data
- Exploratory notebooks used during dataset design
- Model training and evaluation experiments

---

## Quick Start

### Prerequisites

- Python 3.10+
- [WhiteboxTools](https://www.whiteboxgeo.com/manual/wbt_book/intro.html) installed and available on your system (required for DEM hydrology)
- Sufficient disk space for intermediate artifacts and final tensor files

### Installation

```bash
pip install -r requirements.txt
```

### Input layout

The pipeline expects aligned patch samples across three sensor variants:

```
matching_files/
├── asc/          # Sentinel-1 ascending  (*.nc)
├── dsc/          # Sentinel-1 descending  (*.nc)
└── Sen2/         # Sentinel-2            (*.nc)
```

Each folder must contain the **same patch IDs** (e.g. `italy_s1asc_250.nc`, `italy_s1dsc_250.nc`, `italy_s2_250.nc`). The raw full dataset is not included in this repository (see [Dataset Availability](#dataset-availability)).

### Run the pipeline

From the repository root:

```bash
python main.py
```

Custom paths:

```bash
python main.py --input matching_files --output Enriched_files_pt --work-dir pipeline_artifacts
```

Run selected steps only:

```bash
python main.py --steps validate temporal dem_drainage nc_to_pt enrich_pt
```

### Output

```
Enriched_files_pt/
├── asc/          # e.g. italy_s1asc_250_enriched.pt
├── dsc/
└── Sen2/
```

Intermediate logs and artifacts are written to `pipeline_artifacts/`.

---

## Project Pipeline

The workflow has two stages: an **external patch extraction step** (outside this pipeline) and a **reproducible in-repo pipeline** driven by `main.py`.

```
External source database
        │
        ▼
Patch extraction (manual / upstream)
        │
        ▼
matching_files/  {asc, dsc, Sen2}
        │
        ▼
┌───────────────────────────────────────────────────┐
│  python main.py                                   │
│                                                   │
│  1. validate      → common patch IDs              │
│  2. temporal      → event dates, last 8 samples   │
│  3. dem_drainage  → mosaic, flow acc, drainage    │
│  4. nc_to_pt      → base tensor conversion        │
│  5. enrich_pt     → derived variables on .pt      │
└───────────────────────────────────────────────────┘
        │
        ▼
Enriched_files_pt/  {asc, dsc, Sen2}
        │
        ▼
Train / val / test splits  →  Deep learning models  →  Evaluation
```

### Pipeline steps

| Step | Description |
|------|-------------|
| **validate** | Checks that `asc`, `dsc`, and `Sen2` contain the same patch IDs |
| **temporal** | Records dates per patch; keeps the 8 timestamps immediately before the landslide event (`date_event` or `event_date` attribute) |
| **dem_drainage** | Exports patch DEMs, builds regional mosaics by sector, computes flow accumulation and area drainage, assigns `area_drainage` to each NetCDF patch |
| **nc_to_pt** | Converts enriched NetCDF files to base PyTorch tensors with `variable_names` metadata |
| **enrich_pt** | Adds derived topographic variables on tensors; for Sentinel-2 also computes NDVI and NBR |

---

## Repository Structure

```
.
├── main.py                          # Pipeline entry point
├── requirements.txt
├── config/
│   ├── default_config.yaml          # Paths, variables, temporal window, DEM settings
│   └── sectors_italy.json         # Regional sectors for DEM / hydrology
├── pipeline/
│   ├── runner.py                    # Orchestrates all steps
│   ├── config.py
│   └── steps/
│       ├── validate.py
│       ├── temporal.py
│       ├── dem_drainage.py
│       ├── nc_to_pt.py
│       └── enrich_pt.py
├── Procesamiento/
│   ├── feature_functions.py         # NetCDF feature engineering (DEM, drainage, etc.)
│   └── Machine_learn.ipynb          # Model training experiments (3D U-Net)
├── Helpers_fase2/
│   └── feature_functions_pt.py      # Tensor enrichment and dataset utilities
├── Preprocesamiento/                # Exploratory / legacy notebook workflow
├── Helpers_fase1/                   # Phase-1 zone selection and EDA helpers
├── Confección_fase2.ipynb           # Legacy phase-2 dataset assembly notebook
└── GRAPHS/                          # Precomputed analysis outputs (JSON)
```

**Notebooks** (`Preprocesamiento/`, `Confección_fase2.ipynb`) document the iterative research process used to design the dataset. The **reproducible path** for building `Enriched_files_pt` is `main.py`.

---

## Configuration

### `config/default_config.yaml`

Main settings:

| Key | Purpose |
|-----|---------|
| `input_dir` | Input folder (`matching_files`) |
| `output_dir` | Final tensor output (`Enriched_files_pt`) |
| `work_dir` | Intermediate artifacts (`pipeline_artifacts`) |
| `temporal.n_timesteps` | Number of pre-event timestamps to keep (default: 8) |
| `dem.sectors_config` | Sector definitions for regional DEM processing |
| `variables.sar` / `variables.sen2` | Dynamic, static, and final channel order per variant |

### `config/sectors_italy.json`

Defines geographic sectors used to build regional DEM mosaics and compute consistent flow accumulation / area drainage for the Italian study area. Edit this file if the spatial extent changes.

### CLI overrides

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML config file |
| `--input` | Override input directory |
| `--output` | Override output directory |
| `--work-dir` | Override intermediate artifacts directory |
| `--steps` | Run only selected steps |
| `--log-level` | Logging verbosity |

---

## Data Sources

The models combine information from multiple remote sensing and environmental products.

### Sentinel-1 SAR (`asc`, `dsc`)

Temporal synthetic aperture radar observations.

- **Variables:** VV, VH
- **Relevance:** surface roughness, moisture, vegetation structure

### Sentinel-2 Optical (`Sen2`)

Multi-temporal optical observations.

- **Variables:** B04, B05, B06, B07, B08, B11, B12 (plus derived NDVI, NBR)
- **Relevance:** vegetation condition, burn severity, surface moisture

### Topographic Variables

Derived from the Digital Elevation Model (DEM):

- DEM, slope, aspect (sin/cos), profile curvature
- Area drainage, LS factor, SPI, TWI
- Distance to drainage

These describe terrain morphology and hydrological behaviour. Topographic derivatives on tensors are computed in the `enrich_pt` step; `area_drainage` is computed from a **regional DEM mosaic** and assigned at the NetCDF stage.

### Lithology

Geological information describing underlying rock and soil materials (`lithology_class`), included as a static categorical channel.

### Precipitation

Rainfall accumulation variables from external products:

- `prec7` — 7-day accumulation
- `prec20` — 20-day accumulation
- `max2d_7` — maximum 2-day accumulation within the previous week

These aim to capture rainfall-triggering mechanisms.

---

## Dataset Construction

Each sample corresponds to a **spatial patch** shared across the three sensor variants.

### Tensor format

```
x → (Channels, Time, Height, Width)    # Time = 8 pre-event timestamps
y → (Height, Width)                    # Landslide mask (MASK)
```

The mask is stored separately from the input tensor to avoid information leakage during supervised learning.

### Each `.pt` file stores

- `x` — input tensor
- `y` — target mask
- `patch_id` — patch identifier
- `variable_names` — ordered list of channel names (supports dynamic feature selection at training time)

### Temporal selection

For each patch, the pipeline reads the landslide date from NetCDF attributes (`date_event` or `event_date`) and retains only the **8 timestamps immediately before** that date. Patches with fewer than 8 valid pre-event observations are discarded consistently across all three variants.

---

## Feature Engineering

### NetCDF stage (`dem_drainage`)

- Export per-patch DEM GeoTIFFs
- Build regional DEM mosaics by sector
- Compute filled DEM, flow direction, flow accumulation, and area drainage
- Assign `area_drainage` to each patch

Implemented in `Procesamiento/feature_functions.py` and orchestrated by `pipeline/steps/dem_drainage.py`.

### Tensor stage (`enrich_pt`)

Added to all variants:

- Slope, aspect, profile curvature
- LS, SPI, TWI
- Distance to drainage

Added to Sentinel-2 only:

- NDVI, NBR

Implemented in `Helpers_fase2/feature_functions_pt.py`.

### At training time

- Variable normalization
- Static variable replication across time (handled during `nc_to_pt`)
- Dynamic feature selection from stored `variable_names` without rebuilding the dataset

---

## Feature Selection

The framework supports **dynamic feature selection during training**. Because each tensor stores `variable_names`, subsets of channels can be selected directly from `.pt` files without regenerating the dataset.

This enables efficient experimentation with combinations of:

- SAR variables
- Optical variables
- Topographic / hydrological variables
- Lithology
- Precipitation

---

## Model Training

Training follows the standard PyTorch workflow:

```
Dataset → DataLoader → Forward pass → Loss → Backpropagation → Optimizer step
```

### Implemented in this repository

- **Simple 3D U-Net** — baseline spatio-temporal segmentation model (`Procesamiento/Machine_learn.ipynb`)

### Described as part of the broader thesis framework

The following architectures were explored or planned as extensions of the baseline:

- **ConvGRU U-Net** — spatial encoder + ConvGRU temporal module + decoder
- **ConvLSTM U-Net** — ConvLSTM-based temporal memory
- **U-TAE** — Temporal Attention Encoder for satellite time series

Refer to the thesis document and notebook experiments for details on which architectures were fully evaluated.

### Loss functions

Supported strategies include:

- Cross Entropy
- Weighted Cross Entropy
- BCEWithLogitsLoss
- BCE + Dice Loss

These address severe class imbalance typical of landslide mapping.

---

## Evaluation

Models are evaluated with pixel-wise semantic segmentation metrics:

- Recall, Precision, F1-score, IoU
- Loss evolution over epochs
- Optional threshold and class-weighting analysis

---

## Dataset Availability

The **full raw dataset is not included** in this repository. It occupies on the order of **100 GB** due to multi-temporal satellite imagery, engineered variables, and tensor representations.

However, preprocessing is **reproducible** given aligned input patches:

1. Place your extracted patches in `matching_files/{asc,dsc,Sen2}`
2. Run `python main.py`
3. Obtain `Enriched_files_pt/{asc,dsc,Sen2}`

Work is underway to prepare a public distribution strategy (compressed releases or external storage). Once available, download instructions will be added.

---

## Motivation

Landslides are among the most destructive natural hazards worldwide. Many existing susceptibility maps rely exclusively on static variables and do not explicitly model the temporal evolution of environmental conditions before an event.

This project investigates whether incorporating **multi-temporal satellite observations** together with topography, lithology, and precipitation can improve landslide susceptibility estimation through deep learning.

---

## Future Work

Potential extensions include:

- Additional temporal attention architectures (ConvGRU, ConvLSTM, U-TAE)
- Transformer-based models
- Self-supervised pretraining on satellite time series
- Multi-scale fusion and multi-region generalization
- Domain adaptation across geographic areas
- Probabilistic susceptibility estimation

---

## Author

Developed as part of a Bachelor's Thesis on deep learning for landslide susceptibility prediction.

The repository is intended as a research framework for experimentation with spatio-temporal remote sensing data and semantic segmentation models.
