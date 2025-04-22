import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import orthogonal_
from torchvision.ops import DeformConv2d

from utils.common import *

class CustomRegressionNet(nn.Module):
    def __init__(self, numlayers=12, init=orthogonal_, gain=1):
        super(CustomRegressionNet,self).__init__()
        layers = []
        layers.append(ConvAct(1, 64, 3, 1, 1, init=init, gain=gain, bias=True))
        for _ in range(numlayers):
            layers.append(Conv(64, 64, 3, 1, 1, init=init, gain=gain))
        layers.append(ConvAct(64, 1, 3, 1, 1, act=False, init=init, gain=gain))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x_in = x
        y = self.net(x)
        return x_in - y

class CustomRegressionNetAtt(nn.Module):
    def __init__(self, numlayers=12, init=orthogonal_, gain=1, BN=True):
        super(CustomRegressionNetAtt,self).__init__()
        layers = []
        layers.append(ConvAct(1, 64, 3, 1, 1, init=init, gain=gain, bias=False))
        layers.append(DilConv(64, (32,16,8,8), 3, 1, (1,2,3,5), (1,2,3,5), init=init, gain=gain, BN=BN))
        #layers.append(ConvAct(64, 64, 3, 1, 1, init=init, gain=gain))
        for _ in range(numlayers):
            layers.append(IncpAtt(64, (32,16,8,8), 64, 3, 1, (1,2,3,5), (1,2,3,5), 2, init=init, gain=gain, BN=BN))
        layers.append(ConvAct(64, 1, 3, 1, 1, act=False, init=init, gain=gain))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        xin = x
        y = self.net(x)
        return xin - y     

class MultChannelRegressionNetAtt(nn.Module):
    def __init__(self, scale=2, numlayers=6, numreslayers=4, init=orthogonal_, gain=1, BN=True):
        super(MultChannelRegressionNetAtt,self).__init__()
        layers_a, layers_b = [], []
        layers_a.append(ConvAct(2, 64, 3, 1, 1, init=init, gain=gain, bias=True))
        for _ in range(numlayers):
            layers_a.append(RBA(64, 64, 3, 1, 1, 2))
            layers_a.append(ConvAct(64, 64, 3, 1, 1, init=init, gain=gain))
        layers_a.append(ConvAct(64, scale**2, 3, 1, 1, act=False, init=init, gain=gain))
        layers_a.append(nn.PixelShuffle(scale))
        self.net_a = nn.Sequential(*layers_a)
        layers_b.append(ConvAct(1, 64, 3, 1, 1, init=init, gain=gain, bias=True))
        for _ in range(numreslayers):
            layers_b.append(IncpAtt(64, (32,16,8,8), 64, 3, 1, (1,2,3,5), (1,2,3,5), 2, init=init, gain=gain, BN=BN))
        layers_b.append(ConvAct(64, 1, 3, 1, 1, act=False, init=init, gain=gain))
        self.net_b = nn.Sequential(*layers_b)

    def forward(self, x):
        x0 = self.net_a(x)
        y = self.net_b(x0)
        return (x0, x0 - y)
    
class CustomDenosingRegressionNetAtt(nn.Module):
    def __init__(self, numlayers=6, init=orthogonal_, gain=1):
        super(CustomDenosingRegressionNetAtt,self).__init__()
        layers = []
        layers.append(ConvAct(1, 64, 3, 1, 1, init=init, gain=gain, bias=True))
        layers.append(DilConv(64, (32,16,8,8), 3, 1, (1,2,3,4), (1,2,3,4), init=init, gain=gain))
        layers.append(Conv(64, 64, 3, 1, 1, init=init, gain=gain))
        for _ in range(numlayers):
            layers.append(IncpSelfAtt(64, (32,16,8,8), 64, 3, 1, (1,2,3,5), (1,2,3,5), 2, init=init, gain=gain))
        layers.append(ConvAct(64, 1, 3, 1, 1, act=False, init=init, gain=gain))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        xin = x
        y = self.net(x)
        return xin - y