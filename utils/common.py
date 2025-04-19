import math
from copy import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, uniform_, normal_, sparse_, constant_, orthogonal_
from torch.cuda import amp

# structure and few functions are borrowed from Yolo
##### basic ####
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)
    
    
class ReOrg(nn.Module):
    def __init__(self):
        super(ReOrg, self).__init__()

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Chuncat(nn.Module):
    def __init__(self, dimension=1):
        super(Chuncat, self).__init__()
        self.d = dimension

    def forward(self, x):
        x1 = []
        x2 = []
        for xi in x:
            xi1, xi2 = xi.chunk(2, self.d)
            x1.append(xi1)
            x2.append(xi2)
        return torch.cat(x1+x2, self.d)


class Shortcut(nn.Module):
    def __init__(self, dimension=0):
        super(Shortcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        return x[0]+x[1]


class Foldcut(nn.Module):
    def __init__(self, dimension=0):
        super(Foldcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        x1, x2 = x.chunk(2, self.d)
        return x1+x2

class ConvAct(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True, bias=False, init=normal_, gain=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=bias)
        self.act = nn.ReLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        # Initialize params
        self.init = init
        self.gain = gain
        self.initialize_weights()

    def forward(self, x):
        return self.act(self.conv(x))
    
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init==xavier_normal_ or self.init==xavier_uniform_ or self.init==orthogonal_:
                    self.init(m.weight, self.gain)
                elif self.init==kaiming_normal_ or self.init==kaiming_uniform_:
                    self.init(m.weight, nonlinearity='relu')
                elif self.init==normal_:
                    self.init(m.weight, mean=0, std=math.sqrt(2/(math.prod(m.weight.shape)/m.weight.shape[1])))
                if m.bias is not None:
                    constant_(m.bias, 0)              

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, bias=False, init=normal_, gain=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        # Initialize params
        self.init = init
        self.gain = gain
        self.initialize_weights()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init==xavier_normal_ or self.init==xavier_uniform_ or self.init==orthogonal_:
                    self.init(m.weight, self.gain)
                elif self.init==kaiming_normal_ or self.init==kaiming_uniform_:
                    self.init(m.weight, nonlinearity='relu')
                elif self.init==normal_:
                    self.init(m.weight, mean=0, std=math.sqrt(2/(math.prod(m.weight.shape)/m.weight.shape[1])))
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                #constant_(m.weight, 1)
                constant_(m.bias, 0)
                torch.clamp(normal_(m.weight.data, 0.0, math.sqrt(2./9./m.weight.shape[0])),-0.025,0.025)
                m.eps = 0.0001
                m.momentum = 0.95                

class DilConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c=(32,16,8,8), k=3, s=1, p=(1,2,3,4), d=(1,2,3,4), g=1, act=True, init=normal_, gain=1, BN=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(DilConv, self).__init__()
        # dilated conv
        self.conv0  = nn.Conv2d(c1, c[0], k, s, padding=p[0], dilation=d[0], groups=g, bias=False)
        self.conv1  = nn.Conv2d(c1, c[1], k, s, padding=p[1], dilation=d[1], groups=g, bias=False)
        self.conv2  = nn.Conv2d(c1, c[2], k, s, padding=p[2], dilation=d[2], groups=g, bias=False)
        self.conv3  = nn.Conv2d(c1, c[3], k, s, padding=p[3], dilation=d[3], groups=g, bias=False)
        self.bn0 = nn.BatchNorm2d(c[0]) if BN else nn.Identity()
        self.bn1 = nn.BatchNorm2d(c[1]) if BN else nn.Identity()
        self.bn2 = nn.BatchNorm2d(c[2]) if BN else nn.Identity()
        self.bn3 = nn.BatchNorm2d(c[3]) if BN else nn.Identity()
        # normal conv
        self.conv0_ = nn.Conv2d(c[0], c[0], k, s, padding=1, groups=g, bias=False)
        self.conv1_ = nn.Conv2d(c[1], c[1], k, s, padding=1, groups=g, bias=False)
        self.conv2_ = nn.Conv2d(c[2], c[2], k, s, padding=1, groups=g, bias=False)
        self.conv3_ = nn.Conv2d(c[3], c[3], k, s, padding=1, groups=g, bias=False)
        self.bn0_ = nn.BatchNorm2d(c[0]) if BN else nn.Identity()
        self.bn1_ = nn.BatchNorm2d(c[1]) if BN else nn.Identity()
        self.bn2_ = nn.BatchNorm2d(c[2]) if BN else nn.Identity()
        self.bn3_ = nn.BatchNorm2d(c[3]) if BN else nn.Identity()
        self.cat = Concat(1)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        # Initialize params
        self.init = init
        self.gain = gain
        self.initialize_weights()
        #self.act = nn.LeakyReLU(0.05) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x0 = self.act(self.bn0(self.conv0(x)))
        x01 = x0 + self.bn0_(self.conv0_(x0))
        x1 = self.act(self.bn1(self.conv1(x)))
        x11 = x1 + self.bn1_(self.conv1_(x1))
        x2 = self.act(self.bn2(self.conv2(x)))
        x21 = x2 + self.bn2_(self.conv2_(x2))
        x3 = self.act(self.bn3(self.conv3(x)))
        x31 = x3 + self.bn3_(self.conv3_(x3))
        xc = self.cat((x01,x11,x21,x31))
        return self.act(xc)
    
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init==xavier_normal_ or self.init==xavier_uniform_ or self.init==orthogonal_:
                    self.init(m.weight, self.gain)
                elif self.init==kaiming_normal_ or self.init==kaiming_uniform_:
                    self.init(m.weight, nonlinearity='relu')
                elif self.init==normal_:
                    self.init(m.weight, mean=0, std=math.sqrt(2/(math.prod(m.weight.shape)/m.weight.shape[1])))
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                #constant_(m.weight, 1)
                constant_(m.bias, 0)
                torch.clamp(normal_(m.weight.data, 0.0, math.sqrt(2./9./m.weight.shape[0])),-0.025,0.025)
                m.eps = 0.0001
                m.momentum = 0.95               

class Reshape(nn.Module):
    def __init__(self, k, reconstruct=True):  # scale, reconstruct (boolean)
        super().__init__()
        self.reconstruct = reconstruct
        self.k = k

    def forward(self, x):
        b,c,h,w = x.shape
        if self.reconstruct:
            c = int(c/(self.k*self.k))
            p = x.contiguous().view(b, c, self.k, self.k, h, w)
            p = p.contiguous().view(b, c, -1, h*w).permute(0, 1, 3, 2)
            p = p.contiguous().view(b, c*h*w, -1)
            p = nn.functional.fold(p, output_size=(h*self.k,w*self.k), kernel_size=(h,w), stride=(h,w))
        else:
            p = x.unfold(2, int(h/self.k), int(h/self.k)).unfold(3, int(w/self.k), int(w/self.k))
            p = p.contiguous().view(b, -1, int(h/self.k), int(w/self.k))
        return p
    

def DWConv(c1, c2, k=1, s=1, act=True, init=xavier_normal_, gain=1):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act, init=init)


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True, init=xavier_normal_, gain=1):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act, init=init)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act, init=init)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class Stem(nn.Module):
    # Stem
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, init=xavier_normal_, gain=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Stem, self).__init__()
        c_ = int(c2/2)  # hidden channels
        self.cv1 = Conv(c1, c_, 3, 2, init=init)
        self.cv2 = Conv(c_, c_, 1, 1, init=init)
        self.cv3 = Conv(c_, c_, 3, 2, init=init)
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.cv4 = Conv(2 * c_, c2, 1, 1, init=init)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv4(torch.cat((self.cv3(self.cv2(x)), self.pool(x)), dim=1))


class DownC(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, n=1, k=2, init=xavier_normal_, gain=1):
        super(DownC, self).__init__()
        c_ = int(c1)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, init=init)
        self.cv2 = Conv(c_, c2//2, 3, k, init=init)
        self.cv3 = Conv(c1, c2//2, 1, 1, init=init)
        self.mp = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return torch.cat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), dim=1)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), init=xavier_normal_, gain=1):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, init=init)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1, init=init)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
    

class Bottleneck(nn.Module):
    # Darknet bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, init=xavier_normal_, gain=1):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, init=init)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, init=init)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Cat(nn.Module):
    # Channel attention block
    def __init__(self, c1, c2, e=2, init=orthogonal_, gain=1):  # ch_in, ch_out, number, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.AdaptiveAvgPool2d((1,1))
        self.cv2 = ConvAct(c1, c_, 1, 1, 0)
        self.cv3 = nn.Conv2d(c_, c2, 1, 1, 0)
        self.s = nn.Sigmoid()

        # Initialize params
        self.init = init
        self.gain = gain
        self.initialize_weights()        

    def forward(self, x):
        s = self.s(self.cv3(self.cv2(self.cv1(x))))
        return x*torch.reshape(s,(s.size(0),s.size(1),1,1))
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init==xavier_normal_ or self.init==xavier_uniform_ or self.init==orthogonal_:
                    self.init(m.weight, self.gain)
                elif self.init==kaiming_normal_ or self.init==kaiming_uniform_:
                    self.init(m.weight, nonlinearity='relu')
                elif self.init==normal_:
                    self.init(m.weight, mean=0, std=math.sqrt(2/(math.prod(m.weight.shape)/m.weight.shape[1])))
                if m.bias is not None:
                    constant_(m.bias, 0)

class Pxat(nn.Module):
    # Channel attention block
    def __init__(self, c1, c2, e=2, init=orthogonal_, gain=1):  # ch_in, ch_out, number, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvAct(c1, c_, 3, 1, 1)
        self.cv2 = nn.Conv2d(c_, c2, 3, 1, 1)
        self.s = nn.Sigmoid()

        # Initialize params
        self.init = init
        self.gain = gain
        self.initialize_weights()        

    def forward(self, x):
        s = self.s(self.cv2(self.cv1(x)))
        return x*s
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init==xavier_normal_ or self.init==xavier_uniform_ or self.init==orthogonal_:
                    self.init(m.weight, self.gain)
                elif self.init==kaiming_normal_ or self.init==kaiming_uniform_:
                    self.init(m.weight, nonlinearity='relu')
                elif self.init==normal_:
                    self.init(m.weight, mean=0, std=math.sqrt(2/(math.prod(m.weight.shape)/m.weight.shape[1])))
                if m.bias is not None:
                    constant_(m.bias, 0)  

class Pat(nn.Module):
    # Patch attention block
    def __init__(self, c1, c2, e=2, k=16, init=orthogonal_, gain=1):  # ch_in, ch_out, number, groups, expansion, factor (number of patches)
        super().__init__()
        c_ = int(c2*k*k*e)  # hidden channels
        self.sh1 = Reshape(k,reconstruct=False)
        self.cv1 = nn.AdaptiveAvgPool2d((1,1))
        self.cv2 = ConvAct(c1*k*k, c_, 1, 1, 0)
        self.cv3 = nn.Conv2d(c_, c2*k*k, 1, 1, 0)
        self.s = nn.Sigmoid()
        self.sh2 = Reshape(k,reconstruct=True)

        # Initialize params
        self.init = init
        self.gain = gain
        self.initialize_weights()

    def forward(self, x):
        sh1 = self.sh1(x)
        s = self.s(self.cv3(self.cv2(self.cv1(sh1))))
        return x*self.sh2(sh1*torch.reshape(s,(s.size(0),s.size(1),1,1)))
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init==xavier_normal_ or self.init==xavier_uniform_ or self.init==orthogonal_:
                    self.init(m.weight, self.gain)
                elif self.init==kaiming_normal_ or self.init==kaiming_uniform_:
                    self.init(m.weight, nonlinearity='relu')
                elif self.init==normal_:
                    self.init(m.weight, mean=0, std=math.sqrt(2/(math.prod(m.weight.shape)/m.weight.shape[1])))
                if m.bias is not None:
                    constant_(m.bias, 0)

class IncpAtt(nn.Module):
    # Inception attention block (Variant)
    def __init__(self, c1, c, c2, k, s, p, d, e=2, act=True, init=xavier_normal_, gain=1, BN=True):  # ch_in, ch_out, number, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        c0 = sum(c)
        self.cv0 = DilConv(c1, c, k, s, p, d)
        self.cv1 = Conv(c0, c2, 3, 1, 1)
        self.cv2 = nn.AdaptiveAvgPool2d((1,1))
        self.cv3 = ConvAct(c2, c_, 1, 1, 0)
        self.cv4 = nn.Conv2d(c_, c2, 1, 1, 0)
        self.s = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(c2) if BN else nn.Identity()
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        # Initialize params
        self.init = init
        self.gain = gain
        self.initialize_weights()

    def forward(self, x):
        x1 = self.cv0(x)
        x2 = self.cv1(x1)
        s = self.s(self.cv4(self.cv3(self.cv2(x2))))
        x3 = x + x2*torch.reshape(s,(s.size(0),s.size(1),1,1))
        return x3 #self.act(self.bn(x3))
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init==xavier_normal_ or self.init==xavier_uniform_ or self.init==orthogonal_:
                    self.init(m.weight, self.gain)
                elif self.init==kaiming_normal_ or self.init==kaiming_uniform_:
                    self.init(m.weight, nonlinearity='relu')
                elif self.init==normal_:
                    self.init(m.weight, mean=0, std=math.sqrt(2/(math.prod(m.weight.shape)/m.weight.shape[1])))
                    #self.init(m.weight, mean=0, std=math.sqrt(2/(math.prod(m.weight.shape))))
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                #constant_(m.weight, 1)
                constant_(m.bias, 0)
                torch.clamp(normal_(m.weight.data, 0.0, math.sqrt(2./9./m.weight.shape[0])),-0.025,0.025)
                m.eps = 0.0001
                m.momentum = 0.95

class IncpSelfAtt(nn.Module):
    # Inception attention block (Variant)
    def __init__(self, c1, c, c2, k, s, p, d, e=2, act=True, init=xavier_normal_, gain=1):  # ch_in, ch_out, number, groups, expansion
        super().__init__()
        c0 = sum(c)
        self.cv1 = DilConv(c1, c, k, s, p, d)
        self.pxat = Pxat(c0,c2,1)
        self.cat = Cat(c2,c2,e)

        # Initialize params
        self.init = init
        self.gain = gain
        self.initialize_weights()

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.pxat(x1)
        x3 = self.cat(x2)
        return x + x3
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init==xavier_normal_ or self.init==xavier_uniform_ or self.init==orthogonal_:
                    self.init(m.weight, self.gain)
                elif self.init==kaiming_normal_ or self.init==kaiming_uniform_:
                    self.init(m.weight, nonlinearity='relu')
                elif self.init==normal_:
                    self.init(m.weight, mean=0, std=math.sqrt(2/(math.prod(m.weight.shape)/m.weight.shape[1])))
                if m.bias is not None:
                    constant_(m.bias, 0)

class RBA(nn.Module):
    def __init__(self,c1, c2, k, s ,p, e=2, act=True, init=xavier_normal_, gain=1):
        super().__init__()
        self.cv1 = ConvAct(c1, c2, k, s, p)
        self.cat = Cat(c2,c2,2)

        # Initialize params
        self.init = init
        self.gain = gain
        self.initialize_weights()

    def forward(self, x):
        x1 = self.cv1(x)
        return x + self.cat(x1)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init==xavier_normal_ or self.init==xavier_uniform_ or self.init==orthogonal_:
                    self.init(m.weight, self.gain)
                elif self.init==kaiming_normal_ or self.init==kaiming_uniform_:
                    self.init(m.weight, nonlinearity='relu')
                elif self.init==normal_:
                    self.init(m.weight, mean=0, std=math.sqrt(2/(math.prod(m.weight.shape)/m.weight.shape[1])))
                if m.bias is not None:
                    constant_(m.bias, 0)
    
##### end of basic #####

##### custom classifier #####
class CustomClassificationNet(nn.Module):
    def __init__(self, nc, init=xavier_normal_, gain=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 256, 5)
        self.conv2 = nn.Conv2d(256, 128, 3)
        self.conv3 = nn.Conv2d(128, 64, 3)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 14 * 14, 16)
        self.fc2   = nn.Linear(16, 10)
        self.fc3   = nn.Linear(10, nc)

        # Initialize params
        self.init = init
        self.gain = gain
        self.initialize_weights()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init==xavier_normal_ or self.init==xavier_uniform_ or self.init==orthogonal_:
                    self.init(m.weight, self.gain)
                elif self.init==kaiming_normal_ or self.init==kaiming_uniform_:
                    self.init(m.weight, nonlinearity='relu')
                elif self.init==normal_:
                    self.init(m.weight, mean=0, std=math.sqrt(2/(math.prod(m.weight.shape)/m.weight.shape[1])))
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                #torch.clamp(normal_(m.weight.data, 0.0, 1.0)/np.sqrt(12),-0.025,0.025)
                m.eps = 0.0001
                m.momentum = 0.95   
                constant_(m.weight, 1)             
                constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                self.init(m.weight, self.gain)
