import os
import pickle
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register

def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data


@register('tiered-imagenet')
class TieredImageNet(Dataset):

    def __init__(self, root_path, split='train', mini=False, **kwargs):
        split_tag = split

        labels = load_data(os.path.join(
               root_path, '{}_labels.pkl'.format(split_tag)))['labels']
        self.data = np.load(os.path.join(
               root_path, '{}_images.npz'.format(split_tag)))['images']
        label = []
        lb = -1
        self.wnids = []
        for wnid in labels:
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            label.append(lb)
        
        self.label = label
        self.n_classes = len(set(label))
        # print(self.n_classes)
        image_size = 84

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            transforms.Resize([92, 92]),
            transforms.CenterCrop(image_size),
            # transforms.Resize(image_size),
            transforms.ToTensor(),
            # normalize,
            transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                 np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))  
        ])
        augment = kwargs.get('augment')
        if augment == 'resize':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
                # normalize,
            ])
        elif augment == 'crop':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment is None:
            self.transform = self.default_transform

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i], self.label[i]
        img = self.transform(Image.fromarray(img))
        return img, label
        # return self.transform(self.data[i]), self.label[i]