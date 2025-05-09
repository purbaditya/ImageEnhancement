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

# torch.serialization.add_safe_globals((CustomSimpleRegressionNet, CustomRegressionNetAtt))

def degrade_image(image, imagepath, task, value):
    # Create low quality image
    if task == 'sisr':
        imagelq = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=1/value, mode='bicubic', antialias=True)
        imagelq = torch.nn.functional.interpolate(imagelq, scale_factor=value, mode='bicubic', antialias=True)
    elif task == 'misr':
        imageB = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=1/value, mode='bicubic', antialias=True)
        imageN = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=1/value, mode='nearest')
        imagelq = torch.cat((imageB, imageN), dim=1)
    elif task == 'dn':
        imagelq = torch.clip(image + (value/255)*torch.randn(image.shape), min=0, max=1)
    elif task == 'car':
        image = Image.open(imagepath)
        if len(image.split()) == 3:
            imagelq, _, _ = image.convert('YCbCr').split()
        else:
            imagelq = image
        imagelq.save('temp/temp.jpeg', format='jpeg', quality=value,keep_rgb=False)
        imagelq = np.asarray(Image.open('temp/temp.jpeg'))
        imagelq = torch.from_numpy(imagelq.astype(np.float32)/255).view(1, -1, imagelq.shape[0], imagelq.shape[1])
    # Add dimension for batch
    if len(imagelq.shape)==3:
        imagelq = imagelq.unsqueeze(0)        
    return imagelq

def test(args, device):
    data, task, factor, stddev, quality, batchsize = args.data, args.task, args.factor, args.stddev, \
        args.quality, args.batchsize
    
    if task == 'sisr' or task == 'misr':
        value = factor
    elif task == 'dn':
        value = stddev
    elif task == 'car':
        value = quality
    
    # Get test dataset path
    with open(args.data) as f:
        datadict = yaml.load(f, Loader=yaml.SafeLoader)

    testdir = os.listdir(datadict[args.testset])
    
    # Load model
    #modelpath = f'runs/train/{task}/{str(value)}/weights/best.pt'
    modelpath = f'runs/train/{task}/{str(value)}/weights/best.pt'
    model = torch.load(modelpath,weights_only=False,map_location=device)['model']
    
    # Get global stats for batchnorm 
    # activate_batchnorm_running(model)
    
    # Evaluate
    testpsnr_hq, testssim_hq, testpsnr_lq, testssim_lq = 0.0 ,0.0, 0.0, 0.0
    model.eval()
    for index in range(len(testdir)):
        imagepath = os.path.join(datadict[args.testset],testdir[index])
        label = np.array(imread(imagepath))
        if len(label.shape) == 3:
            if args.gray:
                Y = rgb2gray(label)
                Y = torch.from_numpy(Y.astype(np.float32)).view(1, -1, Y.shape[0], Y.shape[1])
            else:
                image = rgb2ycbcr(label)
                Y = torch.from_numpy(image[:,:,0].astype(np.float32)/255).view(1, -1, image.shape[0], image.shape[1])
                Cb = torch.from_numpy(image[:,:,1].astype(np.float32)/255).view(1, -1, image.shape[0], image.shape[1])
                Cr = torch.from_numpy(image[:,:,2].astype(np.float32)/255).view(1, -1, image.shape[0], image.shape[1])
        else:
            Y = torch.from_numpy(label.astype(np.float32)/255).view(1, -1, label.shape[0], label.shape[1])

        # Create low and high quality data
        Y = modcrop(Y.squeeze(0), value) if (task=='sisr' or task=='misr') else Y
        X = degrade_image(Y, imagepath, task, value)
        X, Y= X.to(device), Y.to(device)
        if not args.gray and len(label.shape) > 2:
            Cb = modcrop(Cb.squeeze(0), value) if (task=='sisr' or task=='misr') else Cb
            Cr = modcrop(Cr.squeeze(0), value) if (task=='sisr' or task=='misr') else Cr

        # Evaluate
        Y_p = model(X)
        if type(Y_p) is tuple:
            Y_pred = Y_p[1]
        else:
            Y_pred = Y_p

        X = upscale(X,Y)
        Y_pred = Y_pred.squeeze(0).squeeze(0).cpu().detach().numpy().astype(np.float32)
        X = X.squeeze(0).squeeze(0).cpu().detach().numpy().astype(np.float32)
        Y = Y.squeeze(0).squeeze(0).cpu().detach().numpy().astype(np.float32)

        testpsnr_hq += compare_psnr(Y_pred, Y, data_range=1)
        testssim_hq += compare_ssim(Y_pred, Y, data_range=1)
        testpsnr_lq += compare_psnr(X, Y, data_range=1)
        testssim_lq += compare_ssim(X, Y, data_range=1)
    
    testpsnr_hq = testpsnr_hq / len(testdir)
    testssim_hq = testssim_hq / len(testdir)
    testpsnr_lq = testpsnr_lq / len(testdir)
    testssim_lq = testssim_lq / len(testdir)
    
    print('HQ: ',testpsnr_hq,' LQ: ',testpsnr_lq)
    print('HQ: ',testssim_hq,' LQ: ',testssim_lq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str,default='data/testdataset.yaml',help='select the ml file for the appropriate test dataset')
    parser.add_argument('--testset',type=str,default='classic5',help='set5,set14,bsd100,sr119 (sr), bsd68,set12 (dn), live1,classic5 (car)')
    parser.add_argument('--task',type=str,default='car',help='multiimage(misr),singleimage(sisr),denoising(dn),compressionartefactremoval(car)')
    parser.add_argument('--gray',type=bool,default=False,help='gray or color image')
    parser.add_argument('--factor',type=int,default=2,help='select superresolution factor')
    parser.add_argument('--stddev',type=int,default=25,help='select denoising sigma')
    parser.add_argument('--quality',type=int,default=10,help='select quality factor compression artefact removal')
    parser.add_argument('--batchsize', type=int, default=2, help='total batch size for all CPUs/GPUs')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or cpu, multi gpu support or DDP not implemented')        
    args = parser.parse_args()

    device = select_device(args.device, batch_size=args.batchsize)
    test(args, device)