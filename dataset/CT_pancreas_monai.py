"""
This code is for the 3D pancreas CT dataset reading
"""
from curses import KEY_UNDO
import os
import sys
from monai import transforms
import numpy as np
import random
import json

import torch
from torch.utils.data import Dataset

from monai.data.dataset import PersistentDataset, CacheDataset


class CachePanDataset(CacheDataset):
    def __init__(self, root, depth_size, 
                 num_samples:int=12, ids:list=None, 
                 cache_num: int = sys.maxsize, cache_rate: float = 1):
        self.root = root
        self.depth_size = depth_size
        self.num_samples = num_samples
        self.image_crop = 512
        # self.image_crop = 320
        self.keys = ('image', 'label')
        self.low_clip = -96
        self.high_clip = 215
        self.mean = 77.99
        self.std = 75.4

        self.transform = self.get_transform()
        self.data = self.get_data(ids=ids)
        super().__init__(self.data, self.transform, cache_num, cache_rate, num_workers=8)

    def get_transform(self):
        transform = transforms.Compose([
                transforms.LoadImaged(keys=self.keys),
                transforms.AddChanneld(keys=self.keys),
                transforms.ScaleIntensityRanged(keys=self.keys[0],
                                                a_min=self.low_clip,
                                                a_max=self.high_clip,
                                                b_min=(self.low_clip-self.mean)/self.std,
                                                b_max=(self.high_clip-self.mean)/self.std,
                                                clip=True),
                transforms.Spacingd(self.keys, pixdim=(0.5, 0.5, 2.), mode = ("bilinear", "nearest")),
                transforms.Orientationd(self.keys, axcodes = 'RAS'),
                transforms.RandCropByPosNegLabeld(keys = self.keys,
                                                  label_key=self.keys[1], 
                                                  spatial_size = (self.image_crop, self.image_crop, self.depth_size),
                                                  pos = 0.7,
                                                  neg = 0.3),
                transforms.RandFlipd(self.keys, prob = 0.5, spatial_axis=0),
                transforms.RandRotate90d(self.keys, prob=0.5, spatial_axes=(0,1)),    
                transforms.ToTensord(self.keys)
            ])
        return transform

    def get_data(self, ids):
        full_img_path = sorted(os.listdir(os.path.join(self.root, 'imagesTr')))
        full_label_path = sorted(os.listdir(os.path.join(self.root, 'labelsTr')))
        self.img_path = [full_img_path[id] for id in ids]
        self.label_path = [full_label_path[id] for id in ids]
        data = [{'image': os.path.join(self.root, 'imagesTr', image_path),
                 'label': os.path.join(self.root, 'labelsTr', label_path),}
                 for image_path, label_path in zip(self.img_path, self.label_path)]
        return data


class EvaPanDataset(CacheDataset):
    def __init__(self, root, depth_size, 
                 num_samples:int=12, ids:list=None,  
                 cache_num: int = sys.maxsize, cache_rate: float = 1):
        self.root = root
        self.depth_size = depth_size
        self.num_samples = num_samples
        self.image_crop = 512
        # self.image_crop = 320
        self.keys = ('image', 'label')
        self.low_clip = -96
        self.high_clip = 215
        self.mean = 77.99
        self.std = 75.4

        self.transform = self.get_transform()
        self.data = self.get_data(ids=ids)

        super().__init__(self.data, self.transform, cache_num, cache_rate, num_workers=8)
    
    def get_transform(self):
        transform = transforms.Compose([
                transforms.LoadImaged(keys=self.keys),
                transforms.AddChanneld(keys=self.keys),
                transforms.ScaleIntensityRanged(keys=self.keys[0],
                                                a_min=self.low_clip,
                                                a_max=self.high_clip,
                                                b_min=(self.low_clip-self.mean)/self.std,
                                                b_max=(self.high_clip-self.mean)/self.std,
                                                clip=True),
                transforms.Spacingd(self.keys, pixdim=(0.5, 0.5, 2.), mode = ("bilinear", "nearest")),
                transforms.Orientationd(self.keys, axcodes = 'RAS'),
                transforms.ToTensord(self.keys)
            ])
        return transform

    def get_data(self, ids):
        full_img_path = sorted(os.listdir(os.path.join(self.root, 'imagesTr')))
        full_label_path = sorted(os.listdir(os.path.join(self.root, 'labelsTr')))
        self.img_path = [full_img_path[id] for id in ids]
        self.label_path = [full_label_path[id] for id in ids]
        data = [{'image': os.path.join(self.root, 'imagesTr', image_path),
                 'label': os.path.join(self.root, 'labelsTr', label_path),}
                 for image_path, label_path in zip(self.img_path, self.label_path)]
        return data
