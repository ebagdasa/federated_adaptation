## Introduction

This is the repository for the paper: "Salvaging Federated Learning by Local Adaptation" [https://arxiv.org/abs/2002.04758](https://arxiv.org/abs/2002.04758).

## Instructions

1. Requires Python >=3.7, PyTorch >=1.2
2. Configure training parameters using `utils/params.yaml`
3. Start training with `python training.py --params utils/params.yaml`

## Datasets

We use two datasets: 

1. CIFAR-10 through torchvision datasets
2. Reddit data:
  * Corpus parsed dataset: https://drive.google.com/file/d/1qTfiZP4g2ZPS5zlxU51G-GDCGGr23nvt/view?usp=sharing 
  * Whole dataset: https://drive.google.com/file/d/1yAmEbx7ZCeL45hYj5iEOvNv7k9UoX3vp/view?usp=sharing
  * Dictionary: https://drive.google.com/file/d/1gnS5CO5fGXKAGfHSzV3h-2TsjZXQXe39/view?usp=sharing


