# Regression Loss functions
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, vgg16
from torchmetrics.functional.image import total_variation
import numpy as np
import pytorch_msssim as pm
from torch.nn.modules.loss import _Loss

class CustomRegressionLoss(nn.Module):
    # General Regression Loss.
    def __init__(self, p=2, scale=0.0, alpha=1e-5):
        super(CustomRegressionLoss, self).__init__()
        self.loss_fcn = nn.L1Loss(reduction='none')
        self.exponent = p
        self.alpha = alpha
        self.scale = scale

    def forward(self, pred, true):
        loss = (self.scale*self.loss_fcn(pred, true) + (1/self.exponent)*torch.pow(self.loss_fcn(pred, true),self.exponent))/(1+self.scale) #+ self.alpha*torch.pow(pred,2)
        return loss.sum()

class DFTLoss(nn.Module):
    # DFT Loss.
    def __init__(self):
        super(DFTLoss, self).__init__()

    def forward(self, pred, true):
        pred_mag = torch.fft.fft2(pred).abs()
        true_mag = torch.fft.fft2(true).abs()
        loss = F.mse_loss(true_mag, pred_mag,size_average=None,reduce=None,reduction='sum')
        return 1e-4*loss

class SSIMLoss(nn.Module):
    # General Regression Loss.
    def __init__(self, scale=2e-1, datarange=1.0, numchannels=1):
        super(SSIMLoss, self).__init__()
        self.loss_fcn = pm.SSIM(data_range=datarange, channel=numchannels)
        self.scale = scale

    def forward(self, pred, true):
        loss = 1-self.loss_fcn(pred, true)
        return self.scale*loss.sum()
    
class SSELoss(_Loss):  # PyTorch 0.4.1
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(SSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        lossmse = F.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)
        lossmae = F.l1_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)
        losstv = total_variation(input, reduction="sum")
        return 0.2*lossmse+0.8*lossmae+1e-7*losstv