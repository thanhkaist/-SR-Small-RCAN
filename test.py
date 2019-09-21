# ------------------------------
# EE838A
# KAIST VIC LAB
# 2019/09/17
# Sehwan Ki
# Super-Resolution
# ------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import argparse
import numpy as np
from torch.nn import init
import torch.optim as optim
import math
from math import log10

from model import model
from data import DIV2K, Set5
from utils import *
import time
from scipy import io
from PIL import Image


parser = argparse.ArgumentParser(description='Super Resolution')

# validation data
parser.add_argument('--HR_valDataroot', required=False, default='SR_data/benchmark/Set5/HR') # modifying to your SR_data folder path
parser.add_argument('--LR_valDataroot', required=False, default='SR_data/benchmark/Set5/LR_bicubic/X2') # modifying to your SR_data folder path
parser.add_argument('--valBatchSize', type=int, default=5)

parser.add_argument('--pretrained_model', default='result/Net1/model/model_lastest.pt', help='save result')


parser.add_argument('--nRG', type=int, default=0, help='number of RG block')
parser.add_argument('--nRCAB', type=int, default=0, help='number of RCAB block')
parser.add_argument('--nFeat', type=int, default=0, help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=0, help='number of color channels to use')
parser.add_argument('--patchSize', type=int, default=0, help='patch size')

parser.add_argument('--nThreads', type=int, default=8, help='number of threads for data loading')

parser.add_argument('--scale', type=float, default=2, help='scale output size /input size')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')

args = parser.parse_args()

if args.gpu == 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif args.gpu == 1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)


def get_testdataset(args):
    data_test = Set5(args)
    dataloader = torch.utils.data.DataLoader(data_test, batch_size=1,
                                             drop_last=True, shuffle=False, num_workers=int(args.nThreads), pin_memory=True)
    return dataloader


def test(args):

    # SR network
    my_model = model.Net1(args)
    my_model.apply(weights_init)
    my_model.cuda()

    my_model.load_state_dict(torch.load(args.pretrained_model))

    testdataloader = get_testdataset(args)
    my_model.eval()

    avg_psnr = 0
    count = 0
    for batch, (im_lr, im_hr) in enumerate(testdataloader):
        count = count + 1
        with torch.no_grad():
            im_lr = Variable(im_lr.cuda(), volatile=False)
            im_hr = Variable(im_hr.cuda())
            output = my_model(im_lr)

        output = output.cpu()
        output = output.data.squeeze(0)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        for t, m, s in zip(output, mean, std):
            t.mul_(s).add_(m)

        output = output.numpy()
        output *= 255.0
        output = output.clip(0, 255)

        out = Image.fromarray(np.uint8(output[0]), mode='RGB')  # output of SRCNN
        out.save('result/Net1/SR_img_%03d.png' % (count))

        # =========== Target Image ===============
        im_hr = im_hr.cpu()
        im_hr = im_hr.data.squeeze(0)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        for t, m, s in zip(im_hr, mean, std):
            t.mul_(s).add_(m)

        im_hr = im_hr.numpy()
        im_hr *= 255.0
        im_hr = im_hr.clip(0, 255)


        mse = ((im_hr[:, 8:-8,8:-8] - output[:, 8:-8,8:-8]) ** 2).mean()
        psnr = 10 * log10(255 * 255 / (mse + 10 ** (-10)))
        psnr_val = psnr
        print(str(count) + '_img PSNR: ' + str(psnr_val))
        avg_psnr += psnr

    print('AVG PSNR : ' + str(avg_psnr))


if __name__ == '__main__':
    test(args)
