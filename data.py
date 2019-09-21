import os
import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
from PIL import Image

def RGB_np2Tensor(imgIn, imgTar, channel):

    if channel == 1:
        # rgb --> Y (gray)
        imgIn = np.sum(imgIn * np.reshape([65.481, 128.553, 24.966], [1, 1, 3]) / 255.0, axis=2, keepdims=True) + 16.0
        imgTar = np.sum(imgTar * np.reshape([65.481, 128.553, 24.966], [1, 1, 3]) / 255.0, axis=2, keepdims=True) + 16.0

    # to Tensor
    ts = (2, 0, 1)
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)
    imgTar = torch.Tensor(imgTar.transpose(ts).astype(float)).mul_(1.0)

    # normalization [-1,1]
    imgIn = (imgIn / 255.0 - 0.5) * 2
    imgTar = (imgTar / 255.0 - 0.5) * 2
    return imgIn, imgTar


def augment(imgIn, imgTar):
    if random.random() < 0.3: # horizontal flip
        imgIn = imgIn[:, ::-1, :]
        imgTar = imgTar[:, ::-1, :]

    if random.random() < 0.3: # vertical flip
        imgIn = imgIn[::-1, :, :]
        imgTar = imgTar[::-1, :, :]

    rot = random.randint(0, 3) # rotate
    imgIn = np.rot90(imgIn, rot, (0, 1))
    imgTar = np.rot90(imgTar, rot, (0, 1))

    return imgIn, imgTar

def getPatch(imgIn, imgTar, args, scale):
    (ih, iw, c) = imgIn.shape
    (th, tw) = (scale * ih, scale * iw)
    tp = args.patchSize # HR image patch size
    ip = tp // scale
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (scale * ix, scale * iy)
    imgIn = imgIn[iy:iy + ip, ix:ix + ip, :]
    imgTar = imgTar[ty:ty + tp, tx:tx + tp, :]
    return imgIn, imgTar

class DIV2K(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.scale = args.scale
        self.channel = args.nChannel
        apath = args.dataDir
        dirHR = 'DIV2K_train_HR_crop_100'
        dirLR = 'DIV2k_train_LR_bicubic_crop_100/X2'
        self.dirIn = os.path.join(apath, dirLR)
        self.dirTar = os.path.join(apath, dirHR)
        self.fileList= os.listdir(self.dirTar)
        self.nTrain = len(self.fileList)
        
    def __getitem__(self, idx):
        scale = self.scale
        args = self.args
        nameIn, nameTar = self.getFileName(idx)
        imgIn = cv2.imread(nameIn)
        imgTar = cv2.imread(nameTar)
        if self.args.need_patch:
            imgIn, imgTar = getPatch(imgIn, imgTar, self.args, scale)
        imgIn, imgTar = augment(imgIn, imgTar) 
        return RGB_np2Tensor(imgIn, imgTar, self.channel)

    def __len__(self):
        return self.nTrain   
        
    def getFileName(self, idx):
        name = self.fileList[idx]
        nameTar = os.path.join(self.dirTar, name)
        name = name[0:-4] + 'x2' + '.png'
        nameIn = os.path.join(self.dirIn, name)
        return nameIn, nameTar


class Set5(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.scale = args.scale
        self.channel = args.nChannel
        dirHR = args.HR_valDataroot
        dirLR = args.LR_valDataroot
        self.dirIn = os.path.join(dirLR)
        self.dirTar = os.path.join(dirHR)
        self.fileList = os.listdir(self.dirTar)
        self.nTrain = len(self.fileList)

    def __getitem__(self, idx):
        scale = self.scale
        nameIn, nameTar = self.getFileName(idx)
        imgIn = cv2.imread(nameIn)
        imgTar = cv2.imread(nameTar)

        return RGB_np2Tensor(imgIn, imgTar, self.channel)

    def __len__(self):
        return self.nTrain

    def getFileName(self, idx):
        name = self.fileList[idx]
        nameTar = os.path.join(self.dirTar, name)
        name = name[0:-4] + 'x2' + '.png'
        nameIn = os.path.join(self.dirIn, name)
        return nameIn, nameTar