# Landslide Susceptibility Prediction using Deep Learning and Multi-Source Remote Sensing

## Overview

This repository contains the code developed for a Bachelor's Thesis on **landslide susceptibility prediction using deep learning and multi-source geospatial data**.

The objective is to study whether modern deep learning architectures can identify areas susceptible to landslides by combining **temporal satellite imagery** with **static and dynamic environmental variables**.

Unlike traditional susceptibility mapping approaches that rely mostly on static predictors, this work integrates **multi-temporal Sentinel-1 SAR, Sentinel-2 optical data, topography, lithology, and precipitation** into a unified spatio-temporal framework for semantic segmentation.

The repository includes:

- A **reproducible data-preparation pipeline** (`main.py` → `pipeline_datapreparation/`) that converts aligned NetCDF patches into enriched PyTorch tensors
- A **reproducible training pipeline** (`train.py` → `pipeline_training/`) that trains and evaluates deep learning models on `.pt` files
- Feature engineering utilities for NetCDF and tensor data
- Exploratory Colab notebooks used during dataset design and raw model experiments (`Landslide_susceptibility_prediction-main/Procesamiento/`)

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

### Run the data-preparation pipeline

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

### Run the training pipeline

Once `Enriched_files_pt/` is available, train a model from the repository root:

```bash
python train.py
```

Select architecture:

```bash
python train.py --model unet3d
python train.py --model utae
python train.py --model convgru
python train.py --model fcn_crnn
```

Run selected steps only:

```bash
python train.py --steps compute_stats train
python train.py --steps evaluate
```

Custom paths:

```bash
python train.py --data-dir Enriched_files_pt/asc --output-dir training_artifacts/unet3d_asc
```

Training configuration lives in `config/training_config.yaml` (data paths, feature selection, hyperparameters, model-specific settings). Checkpoints and metrics are written to the configured `output_dir`.

### Output

```
Enriched_files_pt/
├── asc/          # e.g. italy_s1asc_250_enriched.pt
├── dsc/
└── Sen2/
```

Intermediate logs and artifacts are written to `pipeline_artifacts/`.

Training artifacts (checkpoints, normalization stats, metrics) are written to `training_artifacts/` by default.

---

## Project Pipeline

The workflow has three layers: an **external patch extraction step**, a **reproducible data-preparation pipeline**, and a **reproducible training pipeline**.

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
│  python main.py  (pipeline_datapreparation)       │
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
┌───────────────────────────────────────────────────┐
│  python train.py  (pipeline_training)             │
│                                                   │
│  1. compute_stats → channel mean/std (train)      │
│  2. train         → fit model, save checkpoint    │
│  3. evaluate      → F1, IoU, threshold search     │
└───────────────────────────────────────────────────┘
        │
        ▼
training_artifacts/  (checkpoints, metrics)
```

### Data-preparation steps

| Step | Description |
|------|-------------|
| **validate** | Checks that `asc`, `dsc`, and `Sen2` contain the same patch IDs |
| **temporal** | Records dates per patch; keeps the 8 timestamps immediately before the landslide event (`date_event` or `event_date` attribute) |
| **dem_drainage** | Exports patch DEMs, builds regional mosaics by sector, computes flow accumulation and area drainage, assigns `area_drainage` to each NetCDF patch |
| **nc_to_pt** | Converts enriched NetCDF files to base PyTorch tensors with `variable_names` metadata |
| **enrich_pt** | Adds derived topographic variables on tensors; for Sentinel-2 also computes NDVI and NBR |

### Training steps

| Step | Description |
|------|-------------|
| **compute_stats** | Computes per-channel mean and std on the training split for normalization |
| **train** | Trains the selected model; saves the best checkpoint (weights, mean, std, feature list) |
| **evaluate** | Runs pixel-wise evaluation with threshold search (Precision, Recall, F1, IoU) |

---

## Repository Structure

```
.
├── main.py                          # Data-preparation entry point
├── train.py                         # Training / evaluation entry point
├── requirements.txt
├── config/
│   ├── default_config.yaml          # Data-prep paths, variables, DEM settings
│   ├── training_config.yaml         # Training paths, model, hyperparameters
│   └── sectors_italy.json           # Regional sectors for DEM / hydrology
├── pipeline_datapreparation/        # Reproducible NetCDF → .pt pipeline
│   ├── runner.py
│   ├── config.py
│   └── steps/
│       ├── validate.py
│       ├── temporal.py
│       ├── dem_drainage.py
│       ├── nc_to_pt.py
│       └── enrich_pt.py
├── pipeline_training/               # Reproducible .pt → model pipeline
│   ├── runner.py
│   ├── config.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── models/                      # Sen12Landslides architectures + wrappers
│   │   ├── unet3d.py
│   │   ├── utae.py
│   │   ├── convgru.py
│   │   ├── fcn_crnn.py
│   │   ├── wrappers.py
│   │   └── factory.py
│   └── steps/
│       ├── compute_stats.py
│       ├── train.py
│       └── evaluate.py
└── Landslide_susceptibility_prediction-main/
    ├── Procesamiento/               # Raw Colab research notebooks
    │   ├── U_NET.ipynb
    │   ├── U_GRU.ipynb
    │   ├── C_LSTM.ipynb
    │   ├── U_TAE.ipynb
    │   └── split_*.json             # Train/val/test patch IDs
    ├── Preprocesamiento/            # Exploratory dataset design
    ├── Helpers_fase1/
    └── Helpers_fase2/
```

**Notebooks** under `Landslide_susceptibility_prediction-main/` document the iterative Colab research process (paths reference Google Drive; not intended for local re-run). The **reproducible paths** are `main.py` for `Enriched_files_pt/` and `train.py` for model training.

---

## Configuration

### `config/default_config.yaml` (data preparation)

Main settings:

| Key | Purpose |
|-----|---------|
| `input_dir` | Input folder (`matching_files`) |
| `output_dir` | Final tensor output (`Enriched_files_pt`) |
| `work_dir` | Intermediate artifacts (`pipeline_artifacts`) |
| `temporal.n_timesteps` | Number of pre-event timestamps to keep (default: 8) |
| `dem.sectors_config` | Sector definitions for regional DEM processing |
| `variables.sar` / `variables.sen2` | Dynamic, static, and final channel order per variant |

### `config/training_config.yaml` (model training)

Main settings:

| Key | Purpose |
|-----|---------|
| `data_dir` | Enriched `.pt` folder for one sensor variant (`Enriched_files_pt/asc`, etc.) |
| `split_file` / `split_extra_file` | Train/val/test patch ID lists (JSON) |
| `model` | Architecture: `unet3d`, `utae`, `convgru`, or `fcn_crnn` |
| `selected_variables` | Channel subset loaded via `variable_names` metadata |
| `training.*` | Batch size, epochs, learning rate, loss, class weights |
| `model_params.*` | Architecture-specific settings (`img_res`, `hidden_dim`, etc.) |
| `output_dir` | Checkpoints, normalization stats, and metrics |

### `config/sectors_italy.json`

Defines geographic sectors used to build regional DEM mosaics and compute consistent flow accumulation / area drainage for the Italian study area. Edit this file if the spatial extent changes.

### CLI overrides (`main.py`)

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML config file |
| `--input` | Override input directory |
| `--output` | Override output directory |
| `--work-dir` | Override intermediate artifacts directory |
| `--steps` | Run only selected steps |
| `--log-level` | Logging verbosity |

### CLI overrides (`train.py`)

| Flag | Description |
|------|-------------|
| `--config` | Path to training YAML config |
| `--data-dir` | Override enriched tensor directory |
| `--output-dir` | Override training artifacts directory |
| `--model` | Override model (`unet3d`, `utae`, `convgru`, `fcn_crnn`) |
| `--steps` | Run only selected steps (`compute_stats`, `train`, `evaluate`) |
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

Implemented in `Procesamiento/feature_functions.py` and orchestrated by `pipeline_datapreparation/steps/dem_drainage.py`.

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

The reproducible entry point is `train.py`, which loads enriched `.pt` tensors, normalizes them, and trains one of four architectures. Model definitions are adapted from the [Sen12Landslides](https://github.com/PaulH97/Sen12Landslides) repository (satellite data source project), with thin wrappers to match this thesis tensor format.

### Implemented architectures (`pipeline_training/models/`)

| Model | Source | Description |
|-------|--------|-------------|
| **UNet3D** | `unet3d.py` | 3D U-Net baseline for spatio-temporal segmentation |
| **UTAE** | `utae.py` | U-TAE with temporal attention encoder |
| **ConvGRU** | `convgru.py` | ConvGRU segmentation (`ConvGRU_Seg`) |
| **FCN-CRNN** | `fcn_crnn.py` | U-Net encoder + ConvLSTM bottleneck (`FCN_CRNN`) |

Raw Colab experiments for each architecture are preserved in `Landslide_susceptibility_prediction-main/Procesamiento/` (`U_NET.ipynb`, `U_TAE.ipynb`, `U_GRU.ipynb`, `C_LSTM.ipynb`).

### Input / output contract

All models share the Sen12Landslides interface:

- **Input:** `[Batch, Time, Channels, Height, Width]` (8 pre-event timestamps)
- **Output:** `[Batch, num_classes, Height, Width]` landslide segmentation logits

The dataset adapter permutes stored tensors from `(C, T, H, W)` to `(T, C, H, W)` and supports dynamic feature selection via `variable_names`.

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

- Recall, Precision, F1-score, IoU, Accuracy
- Loss evolution over epochs (saved in `training_history.pt`)
- Threshold search on validation split (saved in `metrics.json`)

The `evaluate` step in `train.py` reproduces the threshold-analysis workflow from the Colab notebooks.

---

## Dataset Availability

The **full raw dataset is not included** in this repository. It occupies on the order of **100 GB** due to multi-temporal satellite imagery, engineered variables, and tensor representations.

However, preprocessing and training are **reproducible** given aligned input patches and split JSON files:

1. Place your extracted patches in `matching_files/{asc,dsc,Sen2}`
2. Run `python main.py` → obtain `Enriched_files_pt/{asc,dsc,Sen2}`
3. Configure `config/training_config.yaml` and run `python train.py`

Work is underway to prepare a public distribution strategy (compressed releases or external storage). Once available, download instructions will be added.

---

## Motivation

Landslides are among the most destructive natural hazards worldwide. Many existing susceptibility maps rely exclusively on static variables and do not explicitly model the temporal evolution of environmental conditions before an event.

This project investigates whether incorporating **multi-temporal satellite observations** together with topography, lithology, and precipitation can improve landslide susceptibility estimation through deep learning.

---

## Future Work

Potential extensions include:

- Transformer-based models
- Self-supervised pretraining on satellite time series
- Multi-scale fusion and multi-region generalization
- Domain adaptation across geographic areas
- Probabilistic susceptibility estimation
- Per-variant training configs and automated experiment sweeps

---

## Author

Developed as part of a Bachelor's Thesis on deep learning for landslide susceptibility prediction.

The repository is intended as a research framework for experimentation with spatio-temporal remote sensing data and semantic segmentation models.
