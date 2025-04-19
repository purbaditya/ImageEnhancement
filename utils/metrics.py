import numpy as np
import torch
import pytorch_msssim as pm
import lpips

"""
Model validation metrices - psnr, ssim
Common metrices with torch tensors as input
"""

def psnr_(x,y):
    if x.dtype != torch.float32:
        x,y = x.to(torch.float32), y.to(torch.float32)
    mse = torch.mean(0.5*(x-y)**2).item()
    maxval = 255 if y.max().item() > 1 else 1
    return 10*np.log10((maxval**2+1e-16)/(mse+1e-16))

def ssim_(x,y):
    if x.dtype != torch.float32:
        x,y = x.to(torch.float32), y.to(torch.float32)
    maxval = y.max()
    if len(x.shape)==3:
        x = x.unsqueeze(0)
    if len(y.shape)==3:
        y = y.unsqueeze(0)
    return pm.ssim(x,y,maxval).item()

def lpips(x,y):
    fn = lpips.LPIPS(net='alex')
    return ((fn(x)-fn(y))**2).mean().item()