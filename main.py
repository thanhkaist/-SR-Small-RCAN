# ------------------------------
# EE838A
# KAIST VIC LAB
# 2019/09/17
# Sehwan Ki
# Super-Resolution
# ------------------------------

import torch
import torch.nn as nn

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


parser = argparse.ArgumentParser(description='super-resolution')

# train data
parser.add_argument('--dataDir', default='SR_data/train', help='dataset directory') # modifying to your SR_data folder path
parser.add_argument('--saveDir', default='./result', help='datasave directory')

# validation data
parser.add_argument('--HR_valDataroot', required=False, default='SR_data/benchmark/Set5/HR') # modifying to your SR_data folder path
parser.add_argument('--LR_valDataroot', required=False, default='SR_data/benchmark/Set5/LR_bicubic/X2') # modifying to your SR_data folder path
parser.add_argument('--valBatchSize', type=int, default=5)

parser.add_argument('--load', default='Net1', help='save result')
parser.add_argument('--model_name', default='Net1', help='model to select')
parser.add_argument('--finetuning', default=False, help='finetuning the training')
parser.add_argument('--need_patch', default=True, help='get patch form image')

parser.add_argument('--nRG', type=int, default=0, help='number of RG block')
parser.add_argument('--nRCAB', type=int, default=0, help='number of RCAB block')
parser.add_argument('--nFeat', type=int, default=0, help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=0, help='number of color channels to use')
parser.add_argument('--patchSize', type=int, default=0, help='patch size')

parser.add_argument('--nThreads', type=int, default=8, help='number of threads for data loading')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--lrDecay', type=int, default=100, help='epoch of half lr')
parser.add_argument('--decayType', default='inv', help='lr decay function')
parser.add_argument('--lossType', default='L1', help='Loss type')

parser.add_argument('--period', type=int, default=10, help='period of evaluation')
parser.add_argument('--scale', type=int, default=2, help='scale output size /input size')
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

def get_dataset(args):
    data_train = DIV2K(args)
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batchSize,
                                             drop_last=True, shuffle=True, num_workers=int(args.nThreads), pin_memory=False)
    return dataloader

def get_testdataset(args):
    data_test = Set5(args)
    dataloader = torch.utils.data.DataLoader(data_test, batch_size=1,
                                             drop_last=True, shuffle=False, num_workers=int(args.nThreads), pin_memory=False)
    return dataloader

def set_loss(args):
    lossType = args.lossType
    if lossType == 'MSE':
        lossfunction = nn.MSELoss()
    elif lossType == 'L1':
        lossfunction = nn.L1Loss()
    return lossfunction


def set_lr(args, epoch, optimizer):
    lrDecay = args.lrDecay
    decayType = args.decayType
    if decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2 ** epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = args.lr * math.exp(-k * epoch)
    elif decayType == 'inv':
        k = 1 / lrDecay
        lr = args.lr / (1 + k * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test(args, model, dataloader):

    avg_psnr = 0
    psnr_val = 0
    for batch, (im_lr, im_hr) in enumerate(dataloader):
        with torch.no_grad():
            im_lr = Variable(im_lr.cuda(), volatile=False)
            im_hr = Variable(im_hr.cuda())
            output = model(im_lr)

        output = output.cpu()
        output = output.data.squeeze(0)

        # denormalization
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        for t, m, s in zip(output, mean, std):
            t.mul_(s).add_(m)

        output = output.numpy()
        output *= 255.0
        output = output.clip(0, 255)
        # output = Image.fromarray(np.uint8(output[0]), mode='RGB')

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
        # im_hr = Image.fromarray(np.uint8(im_hr[0]), mode='RGB')

        mse = ((im_hr[:, 8:-8,8:-8] - output[:, 8:-8,8:-8]) ** 2).mean()
        psnr = 10 * log10(255 * 255 / (mse + 10 ** (-10)))
        psnr_val = psnr
        avg_psnr += psnr

    return avg_psnr/args.valBatchSize


def train(args):

    # Set a Model
    my_model = model.Net1(args)
    my_model.apply(weights_init)
    my_model.cuda()

    save = saveData(args)

    Numparams = count_parameters(my_model)
    save.save_log(str(Numparams))

    last_epoch = 0
    # fine-tuning or retrain
    if args.finetuning:
        my_model, last_epoch = save.load_model(my_model)

    # load data
    dataloader = get_dataset(args)
    testdataloader = get_testdataset(args)

    start_epoch = last_epoch
    lossfunction = set_loss(args)
    lossfunction.cuda()
    total_loss = 0
    total_time = 0
    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        optimizer = optim.Adam(my_model.parameters()) # optimizer
        learning_rate = set_lr(args, epoch, optimizer)
        total_loss_ = 0
        loss_ = 0
        for batch, (im_lr, im_hr) in enumerate(dataloader):
            im_lr = Variable(im_lr.cuda())
            im_hr = Variable(im_hr.cuda())

            my_model.zero_grad()
            output = my_model(im_lr)
            loss = lossfunction(output, im_hr)
            total_loss = loss
            total_loss.backward()
            optimizer.step()

            loss_ += loss.data.cpu().numpy()
            total_loss_ += loss.data.cpu().numpy()
        loss_ = loss_ / (batch + 1)
        total_loss_ = total_loss_ / (batch + 1)

        end = time.time()
        epoch_time = (end - start)
        total_time = total_time + epoch_time

        if (epoch + 1) % args.period == 0:
            my_model.eval()
            avg_psnr = test(args, my_model, testdataloader)
            my_model.train()
            log = "[{} / {}] \tLearning_rate: {:.5f}\t Train total_loss: {:.4f}\t Train Loss: {:.4f} \t Val PSNR: {:.4f} Time: {:.4f}".format(epoch + 1,
                                                                                                                                              args.epochs, learning_rate, total_loss_, loss_, avg_psnr, total_time)
            print(log)
            save.save_log(log)
            save.save_model(my_model, epoch)
            total_time = 0


if __name__ == '__main__':
    train(args)
