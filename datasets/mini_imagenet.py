import os
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register
import numpy as np

import os.path as osp


@register('mini-imagenet')
class MiniImageNet(Dataset):

    def __init__(self, root_path, split='train', **kwargs):

        IMAGE_PATH = '/home/localstorage/few-shot-gnn-master/datasets/mini_imagenet/images'
        SPLIT_PATH = '/home/localstorage/few-shot-gnn-master/datasets/mini_imagenet'

        csv_path = osp.join(SPLIT_PATH, split + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data  # data path of all data
        self.label = label  # label of all data
        self.n_classes = len(set(label))   

        if split == 'val' or split == 'test':
    
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif split == 'train':

            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

        # split_tag = split
        # if split == 'train':
        #     split_tag = 'train_phase_train'
        # split_file = 'miniImageNet_category_split_{}.pickle'.format(split_tag)
        # with open(os.path.join(root_path, split_file), 'rb') as f:
        #     pack = pickle.load(f, encoding='latin1')
        # data = pack['data']
        # label = pack['labels']

        # image_size = 84
        # data = [Image.fromarray(x) for x in data]

        # min_label = min(label)
        # label = [x - min_label for x in label]
        
        # self.data = data
        # self.label = label
        # self.n_classes = max(self.label) + 1

        # self.default_transform = transforms.Compose([
        #     transforms.Resize([92,92]),
        #     transforms.CenterCrop(image_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
        #                 np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
        # ])
        # augment = kwargs.get('augment')
        # if augment == 'resize':
        #     self.transform = transforms.Compose([
        #         transforms.RandomResizedCrop(image_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         # normalize,
        #         transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
        #                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
        #     ])
        # elif augment == 'crop':
        #     self.transform = transforms.Compose([
        #         transforms.RandomResizedCrop(image_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
        #                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
        #     ])
        # elif augment is None:
        #     self.transform = self.default_transform

        # def convert_raw(x):
        #     mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
        #     std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
        #     return x * std + mean
        # self.convert_raw = convert_raw

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, i):
    #     return self.transform(self.data[i]), self.label[i]

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
