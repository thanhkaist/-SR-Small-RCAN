import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


# Super Resolution
class Net1(nn.Module):
    def __init__(self, args):
        super(Net1, self).__init__()
        nChannel = args.nChannel
        nFeat = args.nFeat
        scale = args.scale
        self.args = args

        # Define Network
        # ===========================================




        # ===========================================

    def forward(self, x):
        # Make a Network path
        # ===========================================



        # ===========================================

        return x