# Attentive Graph Neural Networks for Few-Shot Learning

This repository contains the code for [Attentive Graph Neural Networks for Few-Shot Learning]().


## Running the code

### Preliminaries

**Environment**
- Python 3.7.3
- Pytorch 1.7.1
- tensorboardX

**Datasets**
- miniImageNet: the [Google Drive](https://drive.google.com/drive/folders/1sXJgi9pXo8i3Jj1nk08Sxo6x7dAQjf9u?usp=sharing) or [Baidu Drive (uk3o)](https://pan.baidu.com/s/17hbnrRhM1acpcjR41P3J0A) for downloading (courtesy of [DeepEMD](https://github.com/icoz69/DeepEMD))
- [tieredImageNet](https://drive.google.com/open?id=1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG) (courtesy of [Kwonjoon Lee](https://github.com/kjunelee/MetaOptNet))

Download the datasets and link the folders into `materials/` with names `mini-imagenet`, `tiered-imagenet` and `imagenet`.
Note `imagenet` refers to ILSVRC-2012 1K dataset with two directories `train` and `val` with class folders.

When running python programs, use `--gpu` to specify the GPUs for running the code (e.g. `--gpu 0,1`).
For Classifier-Baseline, we train with 4 GPUs on miniImageNet and tieredImageNet and with 8 GPUs on ImageNet-800. Meta-Baseline uses half of the GPUs correspondingly.

In following we take miniImageNet as an example. For other datasets, replace `mini` with `tiered`.
By default it is 1-shot, modify `shot` in config file for other shots. Models are saved in `save/`.

*The models on *miniImageNet* and *tieredImageNet* use ConvNet-4 as backbone, the channels in each block are **64-96-128-256**.

### 1. Pretraining the ConvNet4 Backbone
```
python train_classifier.py --config configs/train_classifier_mini.yaml
```
The pretrained Classifier-Baselines can be downloaded from Google Drive ([Mini-ImageNet](https://drive.google.com/file/d/16kl3I6gCeKYFE-aIP67b-VW1g8XAA6-M/view?usp=sharing) , [Tiered-ImageNet](https://drive.google.com/file/d/1M0xQaFl6Q5IBF9hmZveD9PmazFtOswDw/view?usp=sharing)).

You can unzip and place the foder under the 'save' folder.

### 2. Training and Testing AGNN
```
python train_meta.py --config configs/train_meta_mini.yaml
```

### Citation
```
@inproceedings{cheng2022attentive,
  title={Attentive graph neural networks for few-shot learning},
  author={Cheng, Hao and Zhou, Joey Tianyi and Tay, Wee Peng and Wen, Bihan},
  booktitle={2022 IEEE 5th International Conference on Multimedia Information Processing and Retrieval (MIPR)},
  pages={152--157},
  year={2022},
  organization={IEEE}
}
```
### Extension is accepted by IEEE Transactions on Multimedia, Update Soon

```
@article{cheng2023graph,
  title={Graph Neural Networks With Triple Attention for Few-Shot Learning},
  author={Cheng, Hao and Zhou, Joey Tianyi and Tay, Wee Peng and Wen, Bihan},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgment
We thank the following repos providing helpful components/functions in our work.
- [Few-shot GNN](https://github.com/vgsatorras/few-shot-gnn)

- [Transductive Propagation Network](https://github.com/csyanbin/TPN)

- [Few-shot Meta-Baseline](https://github.com/yinboc/few-shot-meta-baseline)

- [DPGN: Distribution Propagation Graph Network for Few-shot Learning](https://github.com/megvii-research/DPGN)
