# Pre-Processing Aided Divide and Conquer for Brain Tumor Segmentation: The Case of Enhancing Tumor Using U-Net

A two-stage divide-and-conquer pipeline for brain tumor segmentation on the BraTS 2020 dataset, with a specific focus on **Enhancing Tumor (ET)** segmentation using 2D U-Net models in PyTorch.

## Background

### Brain Tumour Segmentation (BraTS) Challenge
The BraTS challenge is an annual competition where researchers from around the world compete in creating the best machine learning system for segmenting brain tumour sub-regions from 3D MRI scans [[3]](#3).

### Data
This project uses the BraTS 2020 dataset. The original dataset consists of 369 patients where each patient record contains four MRI modalities (T1, T1ce, T2, FLAIR) plus a ground-truth segmentation label. The truth label contains four pixel values representing sub-regions: 0 for background, 1 for necrotic and non-enhancing tumour (NCR/NET), 2 for peritumoral edema (PE), and 4 for enhancing tumour (ET) [[4,5,6]](#4).

The data has been pre-processed into individual 2D HDF5 slices (`.h5` files), each containing an `image` array of shape (240, 240, 4) and a `mask` array of shape (240, 240, 3):
```
BraTS2020_training_data/content/data/
├── volume_1_slice_0.h5
├── volume_1_slice_1.h5
├── ...
├── meta_data.csv
├── name_mapping.csv
└── survival_info.csv
```

## Pipeline Architecture

The approach uses a **divide-and-conquer** strategy with two cascaded stages:

```
Input MRI Slice (4 modalities)
        │
        ▼
┌───────────────────────┐
│  Stage A: WT Localization  │  ResNet-18 U-Net (4ch → 1)
│  Input: T1, T1ce, T2, FLAIR │  Output: Binary WT mask
└───────────┬───────────┘
            │ Bounding-box ROI
            ▼
┌───────────────────────┐
│  Stage B: ET Segmentation  │  ResNet-18 U-Net (1ch → 1)
│  Input: T1ce (ROI crop)     │  Output: Binary ET mask
└───────────────────────┘
```

| Stage | Task | Input | Output | Encoder |
|-------|------|-------|--------|---------|
| **Stage A** | Whole-Tumor (WT) localization | 4-modality MRI (128×128) | Coarse WT mask → bounding-box ROI | ResNet-18 U-Net (ImageNet) |
| **Stage B** | Enhancing Tumor (ET) segmentation | T1ce ROI crop (160×160) | Fine ET binary mask | ResNet-18 U-Net (scratch) |

## Pre-Processing

| Step | Description |
|------|-------------|
| Center cropping | 240×240 → 128×128 (Stage A) / 160×160 (Stage B ROI) |
| Z-score normalization | Per-modality, computed over brain mask (non-zero voxels) with percentile clipping (0.5–99.5%) |
| Tumor-slice filtering | Background-only slices excluded from Stage-B training |
| Patient-level splitting | Volume-based 60/20/20 split prevents data leakage |
| Balanced sampling | 70% ET-positive / 30% WT-only oversampling for Stage-B |

## Training

### Configuration

| Parameter | Value |
|-----------|-------|
| Seed | 42 |
| Batch size | 8 |
| Learning rate | 1e-3 |
| Epochs (Stage A) | 10 |
| Epochs (Stage B) | 10 |
| Loss function | Dice + BCEWithLogits (both stages) |
| Optimizer | Adam |
| LR scheduler | ReduceLROnPlateau (Stage B, factor=0.5, patience=3) |

### Augmentation
Images are augmented using albumentations [[2]](#2):
- Elastic transformation, grid distortion, optical distortion (p=0.8)
- Random brightness/contrast (p=0.8)
- Applied consistently across all modalities

### Data Splitting
Data is split at the **patient (volume) level** to prevent data leakage:
- **60%** training, **20%** validation, **20%** testing
- Stage A uses all slices (including background)
- Stage B uses only tumor-containing slices

## Evaluation

### Metrics
- **Dice coefficient** (ET, volume-level)
- **Sensitivity / Recall** (ET)
- **Precision** (ET)
- **Specificity** (ET)
- **Hausdorff Distance 95th percentile** (HD95)
- **ET detection recall** (volume-level presence detection)

### Ablation Study
The notebook includes a systematic ablation over:
- **ROI cropping**: with vs. without Stage-A bounding box
- **Normalization**: z-score vs. raw intensities
- **Input modalities**: T1ce only vs. T1ce+FLAIR vs. all 4
- **Loss functions**: Dice+BCE vs. Dice+FocalBCE vs. Focal Tversky

## Project Structure

```
├── brats2020unet2D.ipynb          # Main notebook (full pipeline)
├── BraTS2020_training_data/       # Dataset (H5 slices)
│   └── content/data/
├── model_wt.pth                   # Saved Stage-A model (after training)
├── model_et.pth                   # Saved Stage-B model (after training)
├── et_volume_metrics.csv          # Per-volume evaluation results
├── ablation_results.csv           # Ablation study results
├── images/                        # Figures
└── README.md
```

## Requirements

```
torch
segmentation-models-pytorch
albumentations
numpy
pandas
matplotlib
h5py
scipy
tqdm
```

Install via:
```bash
pip install segmentation-models-pytorch albumentations nibabel pandas matplotlib tqdm scipy h5py
```

## References

<a id="1">[1]</a>
P. Yakubovskiy. Segmentation models pytorch. https://github.com/qubvel/segmentation_models.pytorch, 2020.

<a id="2">[2]</a>
A. Buslaev, V. I. Iglovikov, E. Khvedchenya, A. Parinov, M. Druzhinin, and A. A. Kalinin. Albumentations: Fast and flexible image augmentations. Information, 11(2), 2020.

<a id="3">[3]</a>
Brain tumor segmentation (brats) challenge 2020: Scope — CBICA — Perelman School of Medicine at the University of Pennsylvania. https://www.med.upenn.edu/cbica/brats2020/.

<a id="4">[4]</a>
B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

<a id="5">[5]</a>
S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

<a id="6">[6]</a>
S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)

<a id="7">[7]</a>
O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015.
