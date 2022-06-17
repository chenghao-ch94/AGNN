# Attentive Graph Neural Networks for Few-Shot Learning

This repository contains the code for [Attentive Graph Neural Networks for Few-Shot Learning]().


## Running the code

### Preliminaries

**Environment**
- Python 3.7.3
- Pytorch 1.2.0
- tensorboardX

**Datasets**
- [miniImageNet](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) (courtesy of [Spyros Gidaris](https://github.com/gidariss/FewShotWithoutForgetting))
- [tieredImageNet](https://drive.google.com/open?id=1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG) (courtesy of [Kwonjoon Lee](https://github.com/kjunelee/MetaOptNet))

Download the datasets and link the folders into `materials/` with names `mini-imagenet`, `tiered-imagenet` and `imagenet`.
Note `imagenet` refers to ILSVRC-2012 1K dataset with two directories `train` and `val` with class folders.

When running python programs, use `--gpu` to specify the GPUs for running the code (e.g. `--gpu 0,1`).
For Classifier-Baseline, we train with 4 GPUs on miniImageNet and tieredImageNet and with 8 GPUs on ImageNet-800. Meta-Baseline uses half of the GPUs correspondingly.

In following we take miniImageNet as an example. For other datasets, replace `mini` with `tiered` or `im800`.
By default it is 1-shot, modify `shot` in config file for other shots. Models are saved in `save/`.

*The models on *miniImageNet* and *tieredImageNet* use ConvNet-4 as backbone, the channels in each block are **64-96-128-256**.

### 1. Training Classifier-Baseline
```
python train_classifier.py --config configs/train_classifier_mini.yaml
```
(The pretrained Classifier-Baselines can be downloaded [here]())

### 2. Training and Testing AGNN
```
python train_meta.py --config configs/train_meta_mini.yaml
```

### Citation
```
@article{cheng2020attentive,
  title={Attentive graph neural networks for few-shot learning},
  author={Cheng, Hao and Zhou, Joey Tianyi and Tay, Wee Peng and Wen, Bihan},
  journal={arXiv preprint arXiv:2007.06878},
  year={2020}
}
```

## Acknowledgment
We thank the following repos providing helpful components/functions in our work.
- [Few-shot GNN](https://github.com/vgsatorras/few-shot-gnn)

- [Transductive Propagation Network](https://github.com/csyanbin/TPN)

- [Few-shot Meta-Baseline](https://github.com/yinboc/few-shot-meta-baseline)

