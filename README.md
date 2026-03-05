# BraTS2020-2D-PyTorch
2D U-Net implementation in PyTorch for segmenting brain-tumour sub-regions based on data provided by BraTS 2020.

## Background
### Brain Tumour Segmentation (BraTS) Challenge
The BraTS challenge is an annual competition where researchers from around the world compete in creating the best machine learning system for segmenting brain tumour sub-regions from 3D MRI scans [[3]](#3).

### Data
This project uses the BraTS 2020 dataset. The original dataset consists of 369 patients where each patient record contains four MRI modalities (T1, T1ce, T2, FLAIR) plus a ground-truth segmentation label. The truth label contains four pixel values representing sub-regions: 0 for background, 1 for necrotic and non-enhancing tumour (NCR/NET), 2 for peritumoral edema (PE), and 4 for enhancing tumour (ET) [[4,5,6]](#4).

The data used in this project has been pre-processed into individual 2D HDF5 slices (`.h5` files), each containing an `image` array of shape (240, 240, 4) and a `mask` array of shape (240, 240, 3):
```bash
BraTS2020_training_data/content/data/
├── volume_1_slice_0.h5
├── volume_1_slice_1.h5
├── ...
├── meta_data.csv
├── name_mapping.csv
└── survival_info.csv
```

## Design
### Preprocessing
All 2D slices that contain at least one tumour pixel are selected for training, validation, and testing. Each slice is centre-cropped to 128×128. The four MRI modalities (T1, T1ce, T2, FLAIR) are stacked as input channels and the three mask channels plus a computed background channel form the target.

### Model
The model is a U-Net [[7]](#7) with a ResNet-50 encoder (depth 5, decoder channels [1024, 512, 256, 128, 64]) with pre-trained ImageNet weights. It is imported from qubvel/segmentation_models.pytorch [[1]](#1).

### Training
60% of the data is allocated for training, 20% for validation, and 20% for testing. The model accepts a tensor of [batch_size, 4, 128, 128] as input (T1, T1ce, T2, FLAIR) and outputs [batch_size, 4, 128, 128] (background + 3 tumour sub-regions).

The loss is calculated using the multiclass Dice loss (`smp.losses.DiceLoss`), and Adam is used as the optimizer (lr=0.0001). At each validation step, the Dice score (F-score) and IoU (Jaccard Index) are computed with the background channel ignored. The best model checkpoint is saved based on the validation Dice score.

To improve generalisation, images are augmented using albumentations [[2]](#2) with elastic transformation, grid distortion, optical distortion, and random brightness/contrast.

### Evaluation
The model is evaluated on:
- The held-out 20% test set with per-subregion Dice scores (overall, NCR/NET, PE, ET)
- Individual patient volumes (single scan evaluation)
- Per-patient Dice scores across multiple patients

### Model Summary
|                       |              |
| -------------         |:-------------|
| Architecture          | U-Net|
| Encoder               | ResNet-50|
| Pre-trained weights   | ImageNet|
| Depth                 | 5|
| Decoder channels      | [1024, 512, 256, 128, 64]|
| Input                 | batch × 4 × 128 × 128|
| Output                | batch × 4 × 128 × 128|
| Loss Function         | Multiclass Dice|
| Optimizer             | Adam (lr=0.0001)|
| Augmentation          | Elastic Transformation<br>Grid Distortion<br>Optical Distortion<br>Random Brightness Contrast|
| Epochs                | 50|

## Results
Dice score (F-score) is used as the metric for evaluating performance.

![loss](./images/loss.png)![fscore](./images/fscore.png)

|View             | Overall Accuracy | NCR/NET | PE | ET |
|:---                   |:---:|:--:|:--:|:--:|
|Axial (test set)       | 0.75|0.76|0.73|0.80|
|Coronal (single scan)  | 0.61|0.72|0.62|0.70|
|Sagittal (single scan) | 0.40|0.62|0.40|0.43|

<br>

|View| Predicted         | Truth Label    |
|:--| :-------------: |:-------------:| 
|Axial| ![predicted](./images/predicted.gif) | ![actual](./images/actual.gif) |
|Coronal| ![predicted](./images/coronal_pred.gif) | ![actual](./images/coronal_label.gif) |
|Sagittal| ![predicted](./images/sagittal_pred.gif) | ![actual](./images/sagittal_label.gif) |

## Experiments Attempted
- Normalised images between 0-1
- Used only images that have at least 500 pixels of each class
- Trained model with just 800 images
- Tried different number of decoder channels
- Reduced learning rate after the 40th epoch
- Replaced softmax with sigmoid
- Experimented with a variety of augmentation combinations

## Further Work
- Experiment with Cross Entropy Loss as the loss function
- More carefully select the images instead of selecting all the images that have more than one segmentation pixel
- Use more data for training/validating/testing instead of just 2000 images
- Use a different backbone such as a DenseNet

## References
<a id="1">[1]</a> 
P. Yakubovskiy. Segmentation models pytorch. https://github.com/qubvel/segmentation_models.pytorch, 2020.

<a id="2">[2]</a> 
A. Buslaev, V. I. Iglovikov, E. Khvedchenya, A. Parinov, M. Druzhinin, and A. A. Kalinin. Albumentations: Fast and flexible image augmentations. Information, 11(2), 2020

<a id="3">[3]</a> 
Brain  tumor  segmentation  (brats)  challenge  2020:   Scope  —  cbica—  perelman  school  of  medicine  at  the  university  of  pennsylvania.https://www.med.upenn.edu/cbica/brats2020/. (Accessed on 11/25/2020).

<a id="4">[4]</a> 
B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

<a id="5">[5]</a> 
S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

<a id="6">[6]</a> 
S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)

<a id="7">[7]</a> 
O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for
biomedical image segmentation. In International Conference on Medical image
computing and computer-assisted intervention, pages 234–241. Springer, 2015.
