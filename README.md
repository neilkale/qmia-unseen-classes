# Membership Inference Attacks for Unseen Classes

This repository contains the official implementation of our paper titled "Membership Inference Attacks for Unseen Classes" (https://arxiv.org/abs/2506.06488).

## Setup

### Environment

Create and activate the conda environment using:

```bash
conda env create -f environment.yml
conda activate qmiaenv
```

### Data Preparation

#### CINIC-10
- Download into `data` from: https://datashare.ed.ac.uk/handle/10283/3192
- Combine the train and validation splits into a single folder named `trainval`
- Rename the directory to `data/cinic10`

#### CIFAR-100
- Automatically downloaded by TorchVision within the dataloader

#### ImageNet-1K
- Download by running the `download_imagenet_folder()` function provided in `data_utils.py`

#### Purchase
- Download into `data` from: https://github.com/privacytrustlab/datasets/blob/master/dataset_purchase.tgz
- Run `python convert_purchase_dataset.py`

#### Texas
- Download into `data` from: https://github.com/privacytrustlab/datasets/blob/master/dataset_texas.tgz
- Run `python convert_texas_dataset.py`

#### AG-News
- Automatically downloaded by HuggingFace within the dataloader.

#### 20 Newsgroups
- Run `python convert_20news_dataset.py`

## Experiments

### CIFAR-10 Class Dropout (Figure 3a)

Run `test_dropout.sh` with the following variables:
```bash
BASE_ARCHITECTURE=cifar-resnet-50
QMIA_ARCHITECTURE=facebook/convnext-tiny-224
BASE_DATASET=cinic10/0_16
ATTACK_DATASET=cinic10/0_16
DROPPED_CLASSES=(0)  # Change to (1), (2), ... (9) for other classes
```

Results will be stored in:
```
models/mia/base_cinic10/0_16/cifar-resnet-50/attack_cinic10/0_16/facebook/convnext-tiny-224/score_fn_top_two_margin/loss_fn_gaussian/cls_drop_###/predictions/plots
```

### CIFAR-100 Class Dropout (Figure 3b)

Run `test_dropout.sh` with the following variables:
```bash
BASE_ARCHITECTURE=cifar-resnet-50
QMIA_ARCHITECTURE=facebook/convnext-tiny-224
BASE_DATASET=cifar20/0_16
ATTACK_DATASET=cifar20/0_16
DROPPED_CLASSES=(0)  # Change to (1), (2), ... (19) for other classes
```

**Note:** CIFAR-20 refers to CIFAR-100 superclasses.

Results will be stored in:
```
models/mia/base_cifar20/0_16/cifar-resnet-50/attack_cifar20/0_16/facebook/convnext-tiny-224/score_fn_top_two_margin/loss_fn_gaussian/cls_drop_###/predictions/plots
```

### ImageNet Class Dropout (Figure 4a, 4b)

Run `test_dropout.sh` with the following variables:
```bash
BASE_ARCHITECTURE=resnet-50
QMIA_ARCHITECTURE=facebook/convnext-tiny-224
BASE_DATASET=imagenet-1k/0_16
ATTACK_DATASET=imagenet-1k/0_16
DROPPED_CLASSES=("0-10")  # Change to "0-30", "0-50", ... "0-990" for other ranges
```

Results will be stored in:
```
models/mia/base_imagenet-1k/0_16/resnet-50/attack_imagenet-1k/0_16/facebook/convnext-tiny-224/score_fn_top_two_margin/loss_fn_gaussian/cls_drop_0to###/predictions/plots
```

### ImageNet Data Scarcity (Figure 4c)

Run `test_scarcity.sh` with the following variables:
```bash
BASE_ARCHITECTURE=resnet-50
QMIA_ARCHITECTURE=facebook/convnext-tiny-224
BASE_DATASET=imagenet-1k/0_16
ATTACK_DATASET=imagenet-1k/0_16
CLS_SAMPLES=1  # Change to 5, 10, ... 200 for different sample sizes
```

Results will be stored in:
```
models/mia/base_imagenet-1k/0_16/resnet-50/attack_imagenet-1k/0_16/facebook/convnext-tiny-224/score_fn_top_two_margin/loss_fn_gaussian/cls_drop_none_samples_###/predictions/plots
```

### Texas Class Dropout (Figure 5a)

Run `test_dropout_tabular.sh` with the following variables:
```bash
BASE_ARCHITECTURE=mlp-texas-small
QMIA_ARCHITECTURE=mlp-texas-small
BASE_DATASET=texas/0_16
ATTACK_DATASET=texas/0_16
```

### 20 Newsgroups Dropout (Figure 5b)

Run `test_dropout_text.sh` with the following variables:
```bash
BASE_ARCHITECTURE=gpt2-seqcls-lora
QMIA_ARCHITECTURE=text-mlp-small #text-distilgpt2
BASE_DATASET=20newsgroups/0_16
ATTACK_DATASET=20newsgroups/0_16
```

### Gaussian Mixture Models (GMMs) for Last-Layer Embeddings (Figure 5)

Navigate to the notebooks directory and run the following Jupyter notebooks:
```bash
cd notebooks/
jupyter notebook explore_embeddings.ipynb
jupyter notebook explore_embeddings_c20.ipynb
jupyter notebook explore_embeddings_inet.ipynb
```
