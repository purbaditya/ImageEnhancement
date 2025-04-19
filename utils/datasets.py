import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

import os
import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2ycbcr
import random

class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        labels (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, labels, sigma):
        super(DenoisingDataset, self).__init__()
        self.labels = labels
        self.sigma = sigma

    def __getitem__(self, index):
        label = self.labels[index]
        noise = torch.randn(label.size()).mul_(self.sigma/255.0)
        image = label + noise
        return image, label

    def __len__(self):
        return self.labels.size(0)
    

class SuperresolutionDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        labels (Tensor): HR image patches
        scale: sr factor, e.g., 2
    """
    def __init__(self, labels, scale):
        super(SuperresolutionDataset, self).__init__()
        self.labels = labels
        self.scale = scale

    def __getitem__(self, index):
        label = self.labels[index]
        image = torch.nn.functional.interpolate(label.unsqueeze(0), scale_factor=1/self.scale,mode='bicubic',antialias=True)
        image = torch.nn.functional.interpolate(image, scale_factor=self.scale,mode='bicubic',antialias=True).squeeze(0)
        return image, label

    def __len__(self):
        return self.labels.size(0)
    
class MultiChannelSuperresolutionDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        labels (Tensor): HR image patches
        scale: sr factor, e.g., 2
    """
    def __init__(self, labels, scale):
        super(MultiChannelSuperresolutionDataset, self).__init__()
        self.labels = labels
        self.scale = scale

    def __getitem__(self, index):
        label = self.labels[index]
        imageB = torch.nn.functional.interpolate(label.unsqueeze(0), scale_factor=1/self.scale,mode='bicubic',antialias=True).squeeze(0)
        imageN = torch.nn.functional.interpolate(label.unsqueeze(0), scale_factor=1/self.scale, mode='nearest').squeeze(0)
        image = torch.cat((imageB,imageN),dim=0)
        return image, label

    def __len__(self):
        return self.labels.size(0)
    
class DeblockingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        labels (Tensor): clean image patches
    """
    def __init__(self, labels):
        super(DeblockingDataset, self).__init__()
        self.labels = labels

    def __getitem__(self, index):
        data = self.labels[index]
        image = data[0,:,:].unsqueeze(0)
        label = data[1,:,:].unsqueeze(0)
        return image, label

    def __len__(self):
        return self.labels.size(0)