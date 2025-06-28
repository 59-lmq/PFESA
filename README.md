# PFESA: FFT-Based Parameter-Free Edge and Structure Attention for Medical Image Segmentation

### Introduction

This repository contains the official implementation of the paper "FFT-Based Parameter-Free Edge and Structure Attention for Medical Image Segmentation" (MICCAI 2025).

### News

```
<28.06.2025> We released the codes;
```

### Requirements

This repository is based on PyTorch 2.5.1, CUDA 12.4 and Python 3.10, using a single NVIDIA GeForce RTX 4090 GPU.

### Usage

1. Clone the repo.;

```
git clone https://github.com/59-lmq/PFESA.git
```

2. Put the data in '../dataset_name'; For example, the LA dataset should be placed in '../LA';

3. Train and test the model;

```
cd PFESA
# e.g., use PFESA on transunet on the ISIC-2017 dataset
python train_2d.py --dataset_name ISIC-2017 --network transunet --attention PFESA
```

### Acknowledgements:

Our code is adapted from [XNet](https://github.com/Yanfeng-Zhou/XNet), [MC-Net](https://github.com/ycwu1997/MC-Net/tree/main). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.



