import numpy as np
import torch.nn as nn
import torch
import math
import time
from visualDet3D.networks.backbones import resnet

from visualDet3D.networks.lib.coordconv import CoordinateConv
from visualDet3D.networks.HT import hough_transform, HT

class SelfMask(nn.Module):
    def __init__(self, c):
        super(SelfMask, self).__init__()
        self.mask1 = nn.Conv2d(c, c, kernel_size=1)
        self.max1 = nn.Softmax(dim=-1)

    def forward(self, x):

        mask1 = self.mask1(x)
        n, c, h, w = mask1.shape[0], mask1.shape[1], mask1.shape[2], mask1.shape[3]
        mask1 = mask1.view(n, c, -1)
        mask1 = self.max1(mask1)
        mask1 = mask1.view(n, c, h, w)
        x1 = x * mask1


        x1 = x1.sum(dim=-1,keepdim=True).sum(dim=-2,keepdim=True)
        #print(x1.shape)
        return x1


class YoloMono3DCore(nn.Module):
    """Some Information about YoloMono3DCore"""
    def __init__(self, backbone_arguments=dict()):
        super(YoloMono3DCore, self).__init__()
        self.backbone =resnet(**backbone_arguments)

        self.cord = nn.Sequential(
                CoordinateConv(256+512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(True))

        self.vote_index = hough_transform(72,320,3.0,3.0)
        self.vote_index = torch.tensor(self.vote_index).cuda().contiguous().float()
        self.dht = HT(self.vote_index)


        self.dht_backbone = nn.Sequential(
                CoordinateConv(16, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                SelfMask(256))

        self.bt = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 16, kernel_size=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True))

    def forward(self, x):
        #x = self.backbone(x['image'])
        #x = x[0]

        x1, x2 = self.backbone.forward1(x['image'])
        x1 = 0.1 * x1 + 0.9 * x1.detach()
        x1 = self.bt(x1)
        dht = self.dht(x1)
        dht = self.dht_backbone(dht)

        h, w = x2.shape[2], x2.shape[3]
        dht = dht.expand(-1, -1, h, w)
        
        x2 = torch.cat([x2, dht], 1)
        x2 = self.cord(x2)
        x2 = self.backbone.forward2(x2)
        return x2
