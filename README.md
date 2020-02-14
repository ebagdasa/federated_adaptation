## Salvaging Federated Learning by Local Adaptation

#### Authors:
* [Tao Yu](http://www.cs.cornell.edu/~tyu/)
* [Eugene Bagdasaryan](https://ebagdasa.github.io)
* [Vitaly Shmatikov](http://www.cs.cornell.edu/~shmat/)

### Introduction
This repository contains official code and models for the paper, [Salvaging Federated Learning by Local Adaptation](https://arxiv.org/abs/2002.04758).

### Dependencies
Our implementation works with Python >=3.7 and PyTorch>=1.2.0. Install other dependencies: `$ pip install -r requirement.txt`

### Datasets
We use two datasets in the paper:

- CIFAR-10 through torchvision datasets
- Reddit data:
  * Corpus parsed dataset: https://drive.google.com/file/d/1qTfiZP4g2ZPS5zlxU51G-GDCGGr23nvt/view?usp=sharing 
  * Whole dataset: https://drive.google.com/file/d/1yAmEbx7ZCeL45hYj5iEOvNv7k9UoX3vp/view?usp=sharing
  * Dictionary: https://drive.google.com/file/d/1gnS5CO5fGXKAGfHSzV3h-2TsjZXQXe39/view?usp=sharing

### Usage
1. For the federated learning model, configure the parameters using `utils/params.yaml`, to train a federated learning model on the Reddit Corpus, run:
```
$ python training.py --name text --params utils/params.yaml
```

2. For the adaptation of the federated learning model, configure the parameters using `utils/adapt_text.yaml` or `utils/adapt_image.yaml`, to adapt a federated learning model on the Reddit Corpus, run:
```
$ python adapt.py --name text --params utils/adapt_text.yaml
```

Similarly, change `text` into `image` to train and adapt the federated learning model on CIFAR.

### Citation
If you use our code or wish to refer to our results, please use the following BibTex entry:
```
@misc{yu2020salvaging,
    title={Salvaging Federated Learning by Local Adaptation},
    author={Tao Yu and Eugene Bagdasaryan and Vitaly Shmatikov},
    year={2020},
    eprint={2002.04758},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```