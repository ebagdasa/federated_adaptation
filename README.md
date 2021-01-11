## Salvaging Federated Learning by Local Adaptation

### Dependencies
Our implementation works with Python >=3.7 and PyTorch>=1.2.0. Install other dependencies: `$ pip install -r requirement.txt`

### Datasets
We use two datasets in the paper:

- CIFAR-10 through torchvision datasets
- Reddit data, fetch following shared data and unzip files correspondingly,
  * test_data.json: https://drive.google.com/file/d/1X10JcpVGuRYqhUiwMPRCBJ6k-g9xhL3p/view?usp=sharing
  * Whole dataset: https://drive.google.com/file/d/1yAmEbx7ZCeL45hYj5iEOvNv7k9UoX3vp/view?usp=sharing
  * Dictionary: https://drive.google.com/file/d/1gnS5CO5fGXKAGfHSzV3h-2TsjZXQXe39/view?usp=sharing

### Usage
1. For the federated learning model, configure the parameters using `utils/params.yaml`, to train a federated learning model on the Reddit Corpus, run:
```
$ python training.py --name text --params utils/words.yaml
```

2. For the adaptation of the federated learning model, configure the parameters using `utils/adapt_text.yaml` or `utils/adapt_image.yaml`, to adapt a federated learning model on the Reddit Corpus, run:
```
$ python adapt.py --name text --params utils/adapt_text.yaml
```

Similarly, change `text`, `words.yaml` and `adapt_text.yaml` into `image`, `params.yaml` and `adapt_text.yaml` to train and adapt the federated learning model on CIFAR.