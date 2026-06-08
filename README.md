# Landslide Susceptibility Prediction using Deep Learning and Multi-Source Remote Sensing

## Overview

This repository contains the code developed for the Bachelor's Thesis focused on **landslide susceptibility prediction using deep learning techniques and multi-source geospatial data**.

The objective of the project is to study the capability of modern deep learning architectures to identify areas susceptible to landslides by combining temporal satellite imagery with static environmental variables.

Unlike traditional susceptibility mapping approaches, this work integrates **multi-temporal Sentinel imagery, topographic information, hydrological variables, lithology and precipitation products** into a unified spatio-temporal framework.

The repository includes the complete preprocessing, feature engineering, tensor generation, model training and evaluation pipeline.

---

# Project Pipeline

The complete workflow implemented in this repository is summarized below:

```
Raw Geospatial Data
        │
        ▼
Data Acquisition
        │
        ▼
NetCDF Generation
        │
        ▼
Feature Engineering
        │
        ▼
Enriched NetCDF Files
        │
        ▼
Tensor Conversion (.pt)
        │
        ▼
Feature Selection
        │
        ▼
Deep Learning Models
        │
        ▼
Prediction
        │
        ▼
Evaluation & Analysis
```

The entire pipeline has been designed to facilitate experimentation with different variables, architectures and loss functions while maintaining reproducibility.

---

# Motivation

Landslides constitute one of the most destructive natural hazards worldwide.

Although numerous susceptibility maps exist, many approaches rely exclusively on static variables and do not explicitly model the temporal evolution of environmental conditions.

This project explores whether incorporating temporal information from satellite observations can improve susceptibility estimation through deep learning.

---

# Data Sources

The models combine information from multiple remote sensing products.

## Sentinel-1 SAR

Temporal synthetic aperture radar observations.

Typical variables include:

* VV
* VH

These variables provide information about:

* Surface roughness
* Moisture conditions
* Vegetation structure

---

## Sentinel-2 Optical

Multi-temporal optical observations.

Common variables include:

* Red Edge bands
* SWIR bands

These bands contain valuable information related to vegetation condition and surface moisture.

---

## Topographic Variables

Derived from Digital Elevation Models (DEM):

* DEM
* Slope
* Aspect
* Profile Curvature
* Drainage Area
* LS Factor
* SPI
* TWI
* Distance to Drainage

These variables describe terrain morphology and hydrological behavior.

---

## Lithology

Geological information describing the underlying rock and soil materials.

Lithology plays an important role in slope stability and landslide occurrence.

---

## Precipitation

Accumulated rainfall variables generated from external precipitation products.

Examples include:

* 7-day accumulation
* 20-day accumulation
* Maximum 2-day accumulation within the previous week

These variables aim to capture triggering mechanisms associated with intense rainfall events.

---

# Dataset Construction

The preprocessing pipeline generates a unified dataset where each sample corresponds to a spatial patch.

Input tensors follow the structure:

```
x → (Channels, Time, Height, Width)
```

while target masks are stored independently:

```
y → (Height, Width)
```

This design prevents information leakage while maintaining compatibility with supervised learning.

Each tensor file stores:

* Input variables
* Target mask
* Patch identifier
* Variable names
* Metadata required for feature selection

---

# Feature Engineering

Several preprocessing stages are implemented before training.

These include:

* Variable normalization
* Missing value handling
* Temporal median imputation
* Static variable replication across time
* Tensor generation
* Patch extraction

The modular design allows easy incorporation of additional variables.

---

# Feature Selection

The framework supports dynamic feature selection during training.

Instead of regenerating datasets, subsets of variables can be selected directly from stored tensors.

This enables efficient experimentation with different combinations of:

* SAR variables
* Optical variables
* Topographic variables
* Hydrological variables
* Lithology
* Precipitation

without rebuilding the dataset.

---

# Available Architectures

The repository includes implementations and experiments with several deep learning architectures for spatio-temporal semantic segmentation.

## 3D U-Net

Baseline architecture based on 3D convolutions.

Temporal information is processed jointly with spatial dimensions.

---

## ConvGRU U-Net

Hybrid architecture combining:

* Spatial encoder
* ConvGRU temporal module
* Spatial decoder

allowing explicit temporal modeling.

---

## ConvLSTM U-Net

Extension of ConvGRU replacing recurrent units by ConvLSTM cells to improve long-term temporal memory.

---

## U-TAE

Temporal Attention Encoder architecture specifically designed for satellite image time series.

Temporal attention mechanisms enable adaptive weighting of observations across time.

---

# Loss Functions

The framework supports different optimization strategies:

* Cross Entropy Loss
* Weighted Cross Entropy
* BCEWithLogitsLoss
* BCE + Dice Loss

allowing experimentation under severe class imbalance conditions.

---

# Training Pipeline

Training follows the standard PyTorch workflow:

```
Dataset
    │
    ▼
DataLoader
    │
    ▼
Forward Pass
    │
    ▼
Loss Computation
    │
    ▼
Backpropagation
    │
    ▼
Optimizer Step
```

The modular implementation allows swapping architectures while keeping the remainder of the pipeline unchanged.

---

# Evaluation

Models are evaluated using pixel-wise semantic segmentation metrics.

Typical metrics include:

* Recall
* Precision
* F1-score
* IoU
* Loss evolution

Different classification thresholds and class weighting strategies can also be explored.

---

# Visualization

The repository contains utilities for visualizing:

* Individual variables
* Temporal evolution
* Tensor channels
* NetCDF variables
* Lithology maps
* Precipitation layers
* Prediction masks

These tools facilitate exploratory analysis and model interpretation.

---

# Repository Philosophy

The project has been developed with modularity as a primary objective.

Individual components such as:

* datasets
* preprocessing
* feature selection
* architectures
* losses
* evaluation
* visualization

can be modified independently, making the framework suitable for future research and experimentation.

---

# Dataset Availability

The dataset used throughout this project is **not currently included in the repository**.

The complete database occupies several hundreds of gigabytes due to the storage of multi-temporal satellite imagery, engineered variables and tensor representations.

Work is currently underway to prepare a public distribution strategy, either through compressed releases or external storage services.

Once available, detailed download instructions and preprocessing scripts will be provided to allow full reproduction of the experiments.

---

# Future Work

Potential future developments include:

* Additional temporal attention architectures
* Transformer-based models
* Self-supervised pretraining
* Multi-scale fusion
* Multi-region generalization
* Domain adaptation
* Probabilistic susceptibility estimation

---

# Author

Developed as part of a Bachelor's Thesis on deep learning for landslide susceptibility prediction.

The repository is intended as a research framework for experimentation with spatio-temporal remote sensing data and semantic segmentation models.
