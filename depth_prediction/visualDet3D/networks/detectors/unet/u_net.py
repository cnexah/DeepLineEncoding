

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from visualDet3D.networks.backbones import resnet
from visualDet3D.networks.lib.look_ground import LookGround
from visualDet3D.networks.lib.coordconv import CoordinateConv, DisparityConv
from visualDet3D.networks.lib.ops import ModulatedDeformConvPack
from visualDet3D.networks.HT import hough_transform, HT, IHT

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

class ResConv(nn.Module):
    """Some Information about ResConv"""
    def __init__(self, *args, **kwarg):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(*args, **kwarg)

    def forward(self, x):
        x = x + self.conv(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, is_look_ground=False):
        super().__init__()
        self.is_look_ground=is_look_ground
        if not mid_channels:
            mid_channels = out_channels
        if is_look_ground:
            self.conv0 = LookGround(in_channels, baseline=0.54)
        self.conv1 =  nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            ModulatedDeformConvPack(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, P2=None, scale=1.0):
        """Forward Methods for Double Conv

        Args:
            x (torch.Tensor): [description]
            P2 ([torch.Tensor], optional): Only apply this when double conv appy disparity conv and look ground operation. Defaults to None.
            scale (float, optional): the shrink ratio of the current feature map w.r.t. the original one along with P2, e.g. 1.0/2.0/4.0. Defaults to 1.0.

        Returns:
            x: torch.Tensor
        """
        if P2 is not None:
            P = x.new_zeros([x.shape[0], 3, 4])
            P[:, :, 0:3] = P2[:, :, 0:3]
            P[:, 0:2] /= float(scale)
            x = self.conv0(dict(features=x, P2=P))
        x = self.conv1(x)

        x = self.conv2(x)

        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, is_look_ground=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels, is_look_ground)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None, **kwargs):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            if diffX > 0 or diffY > 0:
                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x, **kwargs)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNet_Core(nn.Module):
    def __init__(self, n_channels, n_classes, look_ground=True, bilinear=True, backbone_arguments=dict()):
        super(UNet_Core, self).__init__()
        self.backbone = resnet(**backbone_arguments)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.up0 = Up(512 + 256, 256, bilinear, is_look_ground=look_ground)
        self.up1 = Up(256 + 128, 128 // factor, bilinear, is_look_ground=look_ground)
        self.up2 = Up(128, 64, bilinear)
        #self.up3 = Up(64, 64, bilinear)
        #self.up4 = Up(64 + n_channels, 32, bilinear, is_look_ground=True)
        self.out_scale_8 = OutConv(64, n_classes)
        self.out_scale_4 = OutConv(64, n_classes)
        self.outc = OutConv(64, n_classes)

        self.cord = nn.Sequential(
                CoordinateConv(256+256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True))

        self.vote_index = hough_transform(44,152,3.0,3.0)
        self.vote_index = torch.tensor(self.vote_index).float().contiguous().cuda()
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
                nn.Conv2d(128, 32, kernel_size=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 16, kernel_size=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True))

    def forward(self, x, P2=None):
        #residual = x
        #x3, x4, x5, x6 = self.backbone(x)
        x3, x4 = self.backbone.forward1(x)
        x4a = 0.1 * x4 + 0.9 * x4.detach()
        x4a = self.bt(x4a)
        dht = self.dht(x4a)
        dht = self.dht_backbone(dht)

        x5 = self.backbone.forward2(x4)
        
        x5a = x5.clone()
        h, w = x5a.shape[2], x5a.shape[3]
        dht = dht.expand(-1, -1, h, w)
        x5a = torch.cat([x5a, dht], 1)
        x5a = self.cord(x5a)

        x6 = self.backbone.forward3(x5a)
        outs = {}
        #x6 = F.relu(x6 + self.fullencoders(x6))
        x = self.up0(x6, x5, P2=P2, scale=32)
        x = self.up1(x, x4, P2=P2, scale=16)
        outs['scale_8'] = self.out_scale_8(x)
        x = self.up2(x, x3)
        outs['scale_4'] = self.out_scale_4(x)
        #x = F.upsample_bilinear(x, scale_factor=4)
        x = F.interpolate(x, scale_factor=4, align_corners=True, mode='bilinear')
        #x = self.up3(x)
        #x = self.up4(x, residual)
        #x = torch.cat([x, residual], dim=1)
        outs['scale_1'] = self.outc(x)
        return outs
