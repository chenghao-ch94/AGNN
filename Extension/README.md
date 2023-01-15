# Graph Neural Networks With Triple Attention for Few-Shot Learning

This repository is the official implementation of [Graph Neural Networks With Triple Attention for Few-Shot Learning] (TMM 2023). 

## Environment

- Python 3.7.3
- Pytorch 1.7.1
- tensorboardX

## Dataset
For your convenience, you can download the datasets directly from links on the left, or you can make them from scratch following the original splits on the right.

|    Dataset    | Original Split |
| :-----------: |:----------------:|
|  [Mini-ImageNet](https://drive.google.com/open?id=15WuREBvhEbSWo4fTr1r-vMY0C_6QWv4w)  |  [Matching Networks](https://arxiv.org/pdf/1606.04080.pdf)  | 
|    [Tiered-ImageNet](https://drive.google.com/file/d/1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG)   |   [SSL](https://arxiv.org/abs/1803.00676)   |
|      [CUB-200-2011](https://github.com/wyharveychen/CloserLookFewShot/tree/master/filelists/CUB)     |   [Closer Look](https://arxiv.org/pdf/1904.04232.pdf)   |
|        [Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)     |   [COMET](https://arxiv.org/pdf/2007.07375.pdf)   |


The dataset directory should look like this:
```bash
├── dataset
    ├── mini-imagenet
        ├── mini_imagenet_test.pickle   
        ├── mini_imagenet_train.pickle  
        ├── mini_imagenet_val.pickle
    ├── tiered-imagenet
        ├── class_names.txt   
        ├── synsets.txt  
        ├── test_images.npz
        ├── test_labels.pkl   
        ├── train_images.npz  
        ├── train_labels.pkl
        ├── val_images.npz
        ├── val_labels.pkl
    ├── cub-200-2011
        ├── attributes   
        ├── bounding_boxes.txt 
        ├── classes.txt
        ├── image   
        ├── image_class_labels.txt 
        ├── images
        ├── images.txt   
        ├── parts
        ├── README
        ├── split
        ├── train_test_split.txt
    ├── flowers
        ├── dataset_split
            ├── test
            ├── train
            ├── val 
        ├── jpg
        ├── imagelabels.mat
        ├── setid.mat 
        ├── test.csv
        ├── train.csv
        ├── val.csv   
```

## Training & Evaluation

To train the model(s) in the paper, run:

```
python3 main_gnn.py --dataset_root dataset --config config/5way_1shot_convnet_mini-imagenet.py --num_gpu 1 --mode train

python3 main_gnn.py --dataset_root dataset --config config/5way_1shot_convnet_tiered-imagenet.py --num_gpu 1 --mode train
```


### Citation
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

- [FEAT](https://github.com/Sha-Lab/FEAT)
