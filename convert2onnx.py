import argparse
import glob
import logging
import math
import os
import platform
import random
import time
from pathlib import Path
from threading import Thread

import numpy as np
from PIL import Image

from torchvision import datasets
import torchvision.transforms as transforms

import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tdata
import yaml
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import lpips
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.io import imread, imsave
from skimage.color import rgb2gray, rgb2ycbcr, ycbcr2rgb

import cv2
from tqdm import trange
import numpy as np
import pandas as pd
from utils.model import *
from utils.datasets import *
from utils.loss import *
from utils.general import *
from utils.torch_utils import *
from utils.metrics import *
from utils.utils import *

import onnx

def convert2onnx(args, device):
    task, factor, stddev, quality = args.task, args.factor, args.stddev, args.quality
    
    if task == 'sisr' or task == 'misr':
        value = factor
    elif task == 'dn':
        value = stddev
    elif task == 'car':
        value = quality
    
    # Load model
    #modelpath = f'runs/train/{task}/{str(value)+'_gray'}/weights/best.pt'
    modelpath = f'runs/train/{task}/{str(value)}/weights/best.pt'
    model = torch.load(modelpath,weights_only=False,map_location=device)['model']
    inputs = (torch.randn(1, 1, 256, 256).to(device), )                            # Fixed Size
    torch.onnx.export(model, inputs, task+'_'+str(value)+'.onnx')
    onnx_model = onnx.load(task+'_'+str(value)+'.onnx')
    onnx.checker.check_model(onnx_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',type=str,default='dn',help='multiimage(misr),singleimage(sisr),denoising(dn),compressionartefactremoval(car)')
    parser.add_argument('--factor',type=int,default=2,help='select superresolution factor')
    parser.add_argument('--stddev',type=int,default=25,help='select denoising sigma')
    parser.add_argument('--quality',type=int,default=20,help='select quality factor compression artefact removal')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or cpu, multi gpu support or DDP not implemented')       
    args = parser.parse_args()

    device = select_device(args.device, batch_size=1)
    convert2onnx(args, device)