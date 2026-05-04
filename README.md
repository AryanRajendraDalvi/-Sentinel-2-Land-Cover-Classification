# 🛰️ Sentinel-2 Land Cover Classification

**EuroSAT Benchmark · Lightweight CNN · Band Ablation Study · MLflow Tracking**

A full end-to-end pipeline for land cover classification using Sentinel-2 multispectral satellite imagery, built on the [EuroSAT](https://github.com/phelber/EuroSAT) dataset. Trains a lightweight CNN to classify 10 land-use categories from 13-band GeoTIFF patches and performs an ablation study comparing RGB-only vs. all-band input.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Model Architecture](#model-architecture)
- [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
- [Ablation Study](#ablation-study)
- [Results](#results)
- [Running on Google Colab](#running-on-google-colab)

---

## Overview

This project classifies Sentinel-2 satellite patches into 10 land-use / land-cover (LULC) categories using a custom CNN trained from scratch. The notebook covers:

1. Downloading and inspecting the EuroSAT multispectral GeoTIFF dataset
2. Band-wise preprocessing — cirrus cloud masking, per-channel normalization
3. Building a lightweight 4-block CNN classifier in PyTorch
4. Training with MLflow experiment tracking (metrics, hyperparameters, model artifacts)
5. **Ablation study**: 13-band model vs. RGB-only model, same architecture
6. Evaluation: confusion matrices, per-class accuracy, visual predictions on test patches

---

## Dataset

**EuroSAT Multispectral** — 27,000 labelled Sentinel-2 patches (64×64 px), 10 classes, 13 spectral bands.

| Class | Description |
|---|---|
| AnnualCrop | Agricultural annual crops |
| Forest | Deciduous & coniferous forest |
| HerbaceousVegetation | Natural grassland & shrub |
| Highway | Roads and motorways |
| Industrial | Industrial buildings & warehouses |
| Pasture | Permanent grassland |
| PermanentCrop | Vineyards, orchards, etc. |
| Residential | Urban housing areas |
| River | Rivers and streams |
| SeaLake | Sea, lakes, reservoirs |

**Bands used (all 13 Sentinel-2 L1C bands):**
`B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12`

RGB subset: `B4 (Red) · B3 (Green) · B2 (Blue)`

Download mirrors (handled automatically by the notebook):
- `https://madm.dfki.de/files/sentinel/EuroSATallBands.zip`
- `https://zenodo.org/record/7711810/files/EuroSATallBands.zip` *(Zenodo backup)*

---

## Project Structure

```
.
├── sentinel2_land_cover_mlflow.ipynb   # Main notebook
├── mlruns/                             # MLflow tracking data (auto-created)
├── best_eurosat_all_bands.pt           # Best checkpoint — all-band model
├── best_eurosat_rgb_bands.pt           # Best checkpoint — RGB model
├── eurosat_cnn_all_bands_final.pt      # Final weights — all-band model
├── eurosat_cnn_rgb_final.pt            # Final weights — RGB model
├── ablation_results.txt                # Classification reports (both models)
├── band_overview.png                   # All 13 bands visualised for one patch
├── class_distribution.png             # EuroSAT class balance chart
├── cm_all_bands.png                    # Confusion matrix — 13-band model
├── cm_rgb.png                          # Confusion matrix — RGB model
└── per_class_accuracy.png             # Side-by-side per-class accuracy bar chart
```

---

## Setup & Installation

### Google Colab (recommended)

Open `sentinel2_land_cover_mlflow.ipynb` in Colab and run **Cell 0** — it installs everything:

```bash
!pip install rasterio GDAL geopandas torch torchvision mlflow \
             numpy matplotlib scikit-learn tqdm Pillow
```

### Local / Conda

```bash
conda create -n sentinel2 python=3.10
conda activate sentinel2
pip install rasterio GDAL geopandas torch torchvision mlflow \
            numpy matplotlib scikit-learn tqdm Pillow seaborn requests
```

> **GDAL note:** On Linux/macOS, installing GDAL via conda (`conda install -c conda-forge gdal`) is more reliable than pip.

---

## Pipeline Walkthrough

### 1. Data Download
The notebook auto-downloads EuroSATallBands.zip (~2.8 GB) from one of two mirrors, extracts it, and resolves the folder name automatically. A diagnostic cell helps confirm the correct `data_root` path on Colab.

### 2. Data Exploration
- Inspects CRS, transform, band shape, and value ranges of a sample GeoTIFF using `rasterio`
- Renders all 13 bands side-by-side
- Plots the class distribution

### 3. Preprocessing
- **`load_tif(path, band_indices)`** — reads selected bands as float32 `(C, H, W)`
- **`cloud_mask(arr, cirrus_band, threshold)`** — masks cirrus clouds using Band 10 (B10 > 0.004 TOA reflectance)
- **`compute_band_stats(file_paths, band_indices)`** — computes per-band mean & std from 600 random files for normalization
- DN values are scaled to TOA reflectance (÷ 10,000)

### 4. Dataset & Splits

| Split | Fraction | Approx. samples |
|---|---|---|
| Train | 75% | ~20,250 |
| Validation | 15% | ~4,050 |
| Test | 10% | ~2,700 |

`EuroSATDataset` applies random horizontal/vertical flips and 90° rotations during training.

### 5. Model Training
Two experiments are run back-to-back via `run_experiment('all')` and `run_experiment('rgb')`, logging everything to MLflow.

**Optimizer config:**

| Param | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | Cosine Annealing |
| Epochs | 30 |
| Batch size | 64 |
| Label smoothing | 0.1 |

---

## Model Architecture

`LandCoverCNN` — a lightweight 4-block CNN with Global Average Pooling.

```
Input  (B, C, 64, 64)       C = 3 (RGB) or 13 (all bands)
  │
  ├─ ConvBlock(C → 64)  × 2
  ├─ MaxPool2d(2)              → (B, 64, 32, 32)
  │
  ├─ ConvBlock(64 → 128) × 2
  ├─ MaxPool2d(2)              → (B, 128, 16, 16)
  │
  ├─ ConvBlock(128 → 256) × 2
  ├─ MaxPool2d(2)              → (B, 256, 8, 8)
  │
  ├─ ConvBlock(256 → 256)
  ├─ MaxPool2d(2)              → (B, 256, 4, 4)
  │
  ├─ AdaptiveAvgPool2d(1)      → (B, 256)
  │
  └─ Linear(256) → ReLU → Dropout(0.4) → Linear(10)

ConvBlock = Conv2d (3×3) → BatchNorm2d → ReLU
Total parameters (13-band): ~2.4M
```

---

## Experiment Tracking with MLflow

MLflow is used instead of Weights & Biases — **no account or API key required**.

Runs are stored locally in `./mlruns/` under the experiment `eurosat_land_cover`.

**What gets logged:**

| Type | Items |
|---|---|
| Parameters | All `CFG` keys, `band_mode`, `in_channels` |
| Metrics (per epoch) | `tr_loss`, `tr_acc`, `va_loss`, `va_acc`, `lr` |
| Metrics (test) | `test_acc`, `test_loss` |
| Artifacts | Best model checkpoint (`.pt`) |

**To view the MLflow UI on Colab:**

```python
# In a new Colab cell:
!pip install mlflow
!mlflow ui --port 5000 &

# Then use ngrok or colab-xterm to access the UI,
# or simply inspect run metrics via:
import mlflow
client = mlflow.tracking.MlflowClient()
runs = client.search_runs("1")  # experiment id
for r in runs:
    print(r.info.run_name, r.data.metrics)
```

**Locally:**
```bash
mlflow ui
# Open http://localhost:5000
```

---

## Ablation Study

The core experiment compares two identical models trained under the same conditions, differing only in input channels:

| Model | Input channels | Bands |
|---|---|---|
| **All-band** | 13 | B1–B12 (full Sentinel-2 L1C) |
| **RGB-only** | 3 | B4, B3, B2 |

This quantifies how much information the non-visible bands (NIR, SWIR, Red-Edge, Cirrus) contribute to land cover discrimination — particularly relevant for vegetation and water body classes.

---

## Results

Results will vary by hardware and random seed. Typical expected ranges:

| Model | Test Accuracy |
|---|---|
| All 13 bands | ~88–92% |
| RGB only | ~82–87% |
| Δ (all − RGB) | ~+4–6% |

The largest gains from extra bands are usually on `HerbaceousVegetation`, `PermanentCrop`, and `Pasture` — classes that look similar in RGB but differ spectrally in NIR/SWIR.

---

## Running on Google Colab

1. Upload `sentinel2_land_cover_mlflow.ipynb` to Colab
2. Set runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`)
3. Run all cells in order — the dataset downloads automatically (~2.8 GB, ~5–10 min)
4. If the data lands in an unexpected folder, the diagnostic cell (Section 2) will show you the correct path; update `CFG['data_root']` in the following cell accordingly
5. Full training (both experiments, 30 epochs each) takes approximately **60–90 minutes on a T4 GPU**

---

*Built with PyTorch · rasterio · MLflow · EuroSAT*
