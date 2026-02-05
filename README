# PRMS Assignment — Hippocampus Segmentation 

This repository contains a small end-to-end pipeline for **3D medical image segmentation** using the **Hippocampus dataset** from the Medical Segmentation Decathlon.

The goal of the project is to load the dataset, apply basic preprocessing, train a simple segmentation model, and evaluate it using the Dice score.

The focus is on correctness and clarity rather than state-of-the-art performance.

---

## Dataset

- Dataset: Hippocampus (Medical Segmentation Decathlon)
- Modality: MRI
- Task: segment hippocampus structures
- Format: 3D NIfTI (`.nii.gz`)

dataset structure:

```
data/Task04_Hippocampus/
├── imagesTr/
├── labelsTr/
└── imagesTs/
```

The project trains only on `imagesTr/` and `labelsTr/`.  
The test set (`imagesTs/`) does not include ground truth labels, so it is not used for Dice evaluation.

---

## Environment Setup

This project was tested with Python 3.10 (conda environment).

``` 
conda create -n prms python=3.10
conda activate prms
```

Install dependencies:
pip install nibabel numpy torch scipy matplotlib


---

## Preprocessing

The following preprocessing steps are applied:

### 1) Intensity normalization
- Z-score normalization is applied to the MRI volume.
- The mean and standard deviation are computed on **non-zero voxels** to avoid background bias.

### 2) Resampling / voxel spacing handling
Before choosing preprocessing, voxel spacing was checked across the dataset:
- All volumes had spacing **(1.0, 1.0, 1.0)**  
Therefore **resampling was not required**.

### 3) Image dimension handling (padding/cropping)
The dataset volumes had many different shapes.
To enable batching and stable training, all volumes are padded/cropped symmetrically to a fixed shape:

- **Target shape: (43, 59, 47)**  
(this corresponds to the maximum shape found in the dataset)

---

## Train / Validation Split

The original training set is split into:

- 80% training
- 20% validation

A fixed random seed is used for reproducibility.

To generate the split:

```
PYTHONPATH=. python split_data.py
```

This creates:

```
data/splits/train.txt
data/splits/val.txt
```

---

## Model

The model is a small custom **3D U-Net** implemented in PyTorch.

- Input: `(B, 1, 43, 59, 47)`
- Output: `(B, 3, 43, 59, 47)`

The output has 3 classes:
- 0 = background
- 1 = hippocampus label 1
- 2 = hippocampus label 2

---

## Training

To train the model:

```
PYTHONPATH=. python train.py
```

Training details:
- Loss: `CrossEntropyLoss`
- Optimizer: Adam
- Validation metric: Dice score (computed on validation set, excluding background)

The best model is saved automatically to:

```
checkpoints/best_model.pt
```

---

## Results

Training was done on CPU.

Best validation Dice score:

Dice ≈ 0.86
- The model reaches a reasonable Dice score on the validation set.
- Qualitative visualizations show good overlap in most cases, with small errors mainly on boundary regions.

---

## Visualization

To generate qualitative results on a random validation case:

```
PYTHONPATH=. python visualization.py
```

Saved an image in:

results/

The saved image contains:
- MRI slice
- MRI + ground truth overlay
- MRI + prediction overlay

---

## Limitations / Possible Improvements

- No data augmentation was used (random flips/rotations)
- Only CrossEntropy loss was used (other loss could be added)
- The model is intentionally small (a deeper U-Net could improve results)
- Only a small number of qualitative examples were generated
