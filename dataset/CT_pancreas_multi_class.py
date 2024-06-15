"""
This code is for the 3D pancreas CT dataset reading
"""
import os
from monai import transforms
import numpy as np
import random

import torch
from torch.utils.data import Dataset


class PanCTDataset(Dataset):
    def __init__(self, root, depth_size, 
                 num_samples:int=12, is_transform:bool=True):
        super().__init__()
        self.root = root
        self.is_transform = is_transform
        self.depth_size = depth_size
        self.num_samples = num_samples
        self.image_crop = 512
        self.full_img_path = sorted(os.listdir(os.path.join(self.root, 'image')))
        self.full_label_path = sorted(os.listdir(os.path.join(self.root, 'label')))
        
        self.keys = ('image', 'label')
        # self.low_clip = -150
        self.low_clip = -91
        self.high_clip = 250
        self.mean = 86.9
        self.std = 39.4

        self.prob = 0.4
        self.positive = 0.8

        self.transform = transforms.Compose([
                transforms.AddChanneld(keys=self.keys),
                transforms.RandCropByPosNegLabeld(keys=self.keys,
                                                  label_key=self.keys[1],
                                                  spatial_size = (self.image_crop, 
                                                                  self.image_crop,
                                                                  self.depth_size,),
                                                  pos = 0.7,
                                                  neg = 0.3,
                                                  num_samples=self.num_samples),
                transforms.RandRotated(keys=self.keys, 
                                       range_x=np.pi/9,
                                       range_y=np.pi/9, 
                                       range_z=np.pi/9,
                                       mode=('bilinear', 'bilinear'),
                                       align_corners=True),
                transforms.RandAdjustContrastd(keys='image', prob = self.prob),
                transforms.RandZoomd(keys=self.keys, prob=self.prob,
                                     min_zoom=0.7, max_zoom=1.3,
                                     mode=('trilinear', 'trilinear'),
                                     align_corners=True),
                transforms.RandFlipd(keys=self.keys, prob=self.prob, spatial_axis=(0, 1)),
                transforms.ToTensord(keys=self.keys),
            ])

    def __len__(self) -> int:
        return len(self.full_img_path)
    
    def __str__(self) -> str:
        return "CT pancreas dataset"

    def __getitem__(self, index):
        temp_img_path = self.full_img_path[index]
        temp_label_path = self.full_label_path[index]

        img = np.load(os.path.join(self.root, 'data', temp_img_path))
        label = np.load(os.path.join(self.root, 'label', temp_label_path))

        img[img < self.low_clip] = self.low_clip
        img[img > self.high_clip] = self.high_clip
        img = (img - self.mean) / self.std
        img = img.transpose((1, 2, 0))
        label = label.transpose((1, 2, 0))
        img = img.astype(np.float32)
        label = label.astype(np.uint8)
        data_dict = {'image': img,
                     'label': label,}
        data_dict = self.transform(data_dict)

        img = torch.stack([data_dict[i]['image'] for i in range(self.num_samples)], dim=0)
        label = torch.stack([data_dict[i]['label'].to(torch.uint8) for i in range(self.num_samples)], dim=0)
        return img, label


class IdPosPanCTDataset(Dataset):
    def __init__(self, root, depth_size, 
                 num_samples:int=12, is_transform:bool=True, ids:list=None):
        super().__init__()
        self.root = root
        self.is_transform = is_transform
        self.depth_size = depth_size
        self.num_samples = num_samples
        self.image_crop = 512
        self.full_img_path = sorted(os.listdir(os.path.join(self.root, 'image')))
        self.full_label_path = sorted(os.listdir(os.path.join(self.root, 'label')))
        self.img_path = [self.full_img_path[id] for id in ids]
        self.label_path = [self.full_label_path[id] for id in ids]
        self.keys = ('image', 'label')
        # self.low_clip = -150
        
        '''
        self.low_clip = -91
        self.high_clip = 250
        self.mean = 86.9
        self.std = 39.4
        '''
        self.low_clip = -96
        self.high_clip = 215
        self.mean = 77.99
        self.std = 75.4
        self.prob = 0.4
        self.positive = 0.8

        self.transform = transforms.Compose([
                transforms.AddChanneld(keys=self.keys),
                transforms.RandCropByPosNegLabeld(keys=self.keys,
                                                  label_key=self.keys[1],
                                                  spatial_size = (self.image_crop, 
                                                                  self.image_crop,
                                                                  self.depth_size,),
                                                  pos = 0.7,
                                                  neg = 0.3,
                                                  num_samples=self.num_samples),
                transforms.RandRotated(keys=self.keys, 
                                       range_x=np.pi/9,
                                       range_y=np.pi/9, 
                                       range_z=np.pi/9,
                                       mode=('bilinear', 'bilinear'),
                                       align_corners=True),
                transforms.RandAdjustContrastd(keys='image', prob = self.prob),
                transforms.RandZoomd(keys=self.keys, prob=self.prob,
                                     min_zoom=0.7, max_zoom=1.3,
                                     mode=('trilinear', 'trilinear'),
                                     align_corners=True),
                transforms.RandFlipd(keys=self.keys, prob=self.prob, spatial_axis=(0, 1)),
                transforms.ToTensord(keys=self.keys),
            ])

    def __len__(self) -> int:
        return len(self.img_path)
    
    def __str__(self) -> str:
        return "CT pancreas dataset"

    def __getitem__(self, index):
        temp_img_path = self.img_path[index]
        temp_label_path = self.label_path[index]

        img = np.load(os.path.join(self.root, 'image', temp_img_path))
        label = np.load(os.path.join(self.root, 'label', temp_label_path))

        img[img < self.low_clip] = self.low_clip
        img[img > self.high_clip] = self.high_clip
        img = (img - self.mean) / self.std
        img = img.transpose((1, 2, 0))
        label = label.transpose((1, 2, 0))
        img = img.astype(np.float32)
        label = label.astype(np.float32)
        # print('org', label.dtype)
        data_dict = {'image': img,
                     'label': label,}
        data_dict = self.transform(data_dict)

        img = torch.stack([data_dict[i]['image'] for i in range(self.num_samples)], dim=0)
        label = torch.stack([data_dict[i]['label'].to(torch.long) for i in range(self.num_samples)], dim=0)
        # print('after', label.dtype)
        return img, label


class EvaPanCTDataset(Dataset):
    def __init__(self, root, depth_size, ids:list=None):
        super().__init__()
        self.root = root
        self.depth_size = depth_size

        self.full_img_path = sorted(os.listdir(os.path.join(self.root, 'image')))
        self.full_label_path = sorted(os.listdir(os.path.join(self.root, 'label')))
        self.img_path = [self.full_img_path[id] for id in ids]
        self.label_path = [self.full_label_path[id] for id in ids]
        self.keys = ('image', 'label')
        # self.low_clip = -150
        # for 1354 experiment
        '''
        self.low_clip = -96
        self.high_clip = 215
        '''
        '''
        self.low_clip = -91
        self.high_clip = 250
        self.mean = 86.9
        self.std = 39.4
        '''
        self.low_clip = -96
        self.high_clip = 215
        self.mean = 77.99
        self.std = 75.4

        self.image_crop = 256

        self.transform = transforms.Compose([
            transforms.ToTensord(keys=self.keys),
            transforms.AddChanneld(keys=self.keys),
            # transforms.Resized(keys=self.keys,
            #                       spatial_size=(-1, self.image_crop, self.image_crop)),
            # transforms.ScaleIntensityd(keys='image', minv=0, maxv=1),
            ])

    def __len__(self) -> int:
        return len(self.img_path)
    
    def __str__(self) -> str:
        return "MRI pancreas dataset"

    def __getitem__(self, index):
        temp_img_path = self.img_path[index]
        temp_label_path = self.label_path[index]

        img = np.load(os.path.join(self.root, 'image', temp_img_path))
        label = np.load(os.path.join(self.root, 'label', temp_label_path))

        img[img < self.low_clip] = self.low_clip
        img[img > self.high_clip] = self.high_clip
        img = (img - self.mean) / self.std
        img = img.astype(np.float32)
        label = label.astype(np.int64)
        '''
        pos_index = np.sum(label, axis=(1, 2), keepdims=False)>0
        index = np.where(pos_index)
        min_index = np.min(index)
        max_index = np.max(index)
        if (max_index - min_index) < self.depth_size:
            center = min_index + max_index
            min_index = center - self.depth_size//2
            max_index = center + self.depth_size//2
            if min_index < 0:
                min_index = 0
                max_index = self.depth_size
            if max_index >= img.shape[0]:
                min_index = img.shape[0]-self.depth_size
                max_index = img.shape[0]
        # min_index = max(min_index//2, min_index-self.depth_size)
        img = img[min_index:max_index]
        label = label[min_index:max_index]
        '''
        data_dict = {'image': img,
                     'label': label}
        
        data_dict = self.transform(data_dict)

        img, label = data_dict['image'].permute(0, 2, 3, 1), data_dict['label'].permute(0, 2, 3, 1).to(torch.long)
        return img, label
