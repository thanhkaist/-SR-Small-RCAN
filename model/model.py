import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math


# Super Resolution
class Net2(nn.Module):
    def __init__(self, args):
        super(Net1, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        scale = args.scale
        self.args = args

        # Define Network
        # ===========================================

        scale_factor = args.scale
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBlock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

        # ===========================================

    def forward(self, x):
        # Make a Network path
        # ===========================================
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        # block4 = self.block4(block3)
        # block5 = self.block5(block4)
        block6 = self.block6(block3)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        # ===========================================

        return F.tanh(block8)



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        # residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        # residual = self.bn2(residual)
        return x + residual



# Super Resolution
class Net1(nn.Module):
    def __init__(self, args):
        super(Net1, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        scale = args.scale
        self.args = args
        nRG = args.nRG
        # Define Network
        # ===========================================
        upsample_block_num = int(math.log(scale, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(nChannel, nFeat, kernel_size=7, padding=3),
            nn.ReLU()
        )
        block2= [RG(nFeat) for _ in range(nRG)]
        block2.append(nn.Conv2d(nFeat,nFeat,3,1,1))
        self.block2 = nn.Sequential(*block2)
        block3 = [UpsampleBlock(64, 2) for _ in range(upsample_block_num)]
        block3.append(nn.Conv2d(64, 3, kernel_size=7, padding=3))
        self.block3 = nn.Sequential(*block3)

        # ===========================================

    def forward(self, x):
        # Make a Network path
        # ===========================================
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2+block1)

        # ===========================================

        return block3

class RG(nn.Module):
    def __init__(self,in_channels = 64):
        super(RG,self).__init__()
        self.rc1 = RCAB(in_channels)
        self.rc2 = RCAB(in_channels)

    def forward(self, x):
        out = self.rc1(x)
        out = self.rc2(out)
        return out + x

class RCAB(nn.Module):

    def __init__(self,in_channels =64):
        super(RCAB,self).__init__()
        self.in_channels= in_channels
        self.filter_size = in_channels
        self.conv1 = nn.Conv2d(self.in_channels,self.filter_size,3,1,1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.filter_size,self.filter_size,3,1,1)
        self.ca1 = CA(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out =self.conv2(out)
        out = self.ca1(out )
        return x+out


class CA(nn.Module):
    def __init__(self,in_channels = 64,reduction = 8):
        super(CA,self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_bottle = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//reduction,kernel_size=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels//reduction,in_channels,kernel_size=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pooling(x)
        out = self.conv_bottle(x)
        return x*out.expand_as(x)

class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_suffle = nn.PixelShuffle(up_scale)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_suffle(x)
        x = self.relu(x)
        return x
