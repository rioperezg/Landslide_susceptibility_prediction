# Landslide Susceptibility Prediction

## Overview

## Motivation

## Methodology

## Repository Structure

├── data_generation/
├── preprocessing/
├── datasets/
├── models/
├── training/
├── inference/
├── utils/
├── visualization/
├── notebooks/

(explicación detallada de cada carpeta)

## Dataset generation pipeline

Sentinel-1
Sentinel-2
DEM
Hydrology
Lithology
Precipitation
↓

NetCDF

↓

Enriched NetCDF

↓

PT tensors

↓

Training

## Available architectures

- U-Net 3D
- ConvGRU
- ConvLSTM
- U-TAE

## Feature Selection

## Loss functions

- CrossEntropy
- Weighted CrossEntropy
- BCE
- BCE + Dice

## Experiments

## Reproducibility

## Future work

## Dataset availability

Due to the large size of the generated dataset (hundreds of gigabytes), it is currently not included in this repository.

A public release is under preparation. The repository already contains the complete code required to reproduce the preprocessing pipeline and training procedure. Once the dataset distribution strategy is finalized, download instructions will be added.

## Citation

## License