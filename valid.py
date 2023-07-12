import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import numpy as np
import nibabel as nib
import imageio
import SimpleITK as sitk


def one_hot(ori, classes):

    batch, h, w, d = ori.size()
    new_gd = torch.zeros((batch, classes, h, w, d), dtype=ori.dtype).cuda()
    for j in range(classes):
        index_list = (ori == j).nonzero()

        for i in range(len(index_list)):
            batch, height, width, depth = index_list[i]
            new_gd[batch, j, height, width, depth] = 1

    return new_gd.float()


def tailor_and_concat(x, model):
    temp = []

    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    y = x.clone()

    for i in range(len(temp)):
        temp[i] = model(temp[i])

    y[..., :128, :128, :128] = temp[0]
    y[..., :128, 128:240, :128] = temp[1][..., :, 16:128, :]
    y[..., 128:240, :128, :128] = temp[2][..., 16:128, :, :]
    y[..., 128:240, 128:240, :128] = temp[3][..., 16:128, 16:128, :]
    y[..., :128, :128, 128:155] = temp[4][..., 96:123]
    y[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
    y[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
    y[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]

    return y[..., :155]


def dice_score(o, t, eps=1e-8):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den


def mIOU(o, t, eps=1e-8):
    num = (o*t).sum() + eps
    den = (o | t).sum() + eps
    return num/den


def softmax_mIOU_score(output, target):
    mIOU_score = []
    mIOU_score.append(mIOU(o=(output==1),t=(target==1)))
    mIOU_score.append(mIOU(o=(output==2),t=(target==2)))
    mIOU_score.append(mIOU(o=(output==3),t=(target==4)))
    return mIOU_score


def softmax_output_dice(output, target):
    ret = []

    # whole
    o = output > 0; t = target > 0 # ce
    ret += dice_score(o, t),
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 4)
    ret += dice_score(o, t),
    # active
    o = (output == 3);t = (target == 4)
    ret += dice_score(o, t),

    return ret


keys = 'whole', 'core', 'enhancing', 'loss'


def validate_softmax(
        valid_loader,
        model,
        names=None  # The names of the patients orderly!
        ):

    H, W, T = 240, 240, 155
    model.eval()

    dice_WT = 0.0
    dice_TC = 0.0
    dice_ET = 0.0
    for i, data in enumerate(valid_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))
        #data = [t.cuda(non_blocking=True) for t in data]
        x, target = data
        x = x.cuda()
        target = target.cuda()

        logit = tailor_and_concat(x, model)
        output = F.softmax(logit, dim=1)

        output = output[0, :, :H, :W, :T].cpu().detach().numpy()
        output = output.argmax(0)

        dice_ret = softmax_output_dice(output, target.cpu().detach().numpy().squeeze())
        dice_WT += dice_ret[0]
        dice_TC += dice_ret[1]
        dice_ET += dice_ret[2]

        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)

        print( msg, "WT:{}, TC:{}, ET:{}".format(dice_ret[0], dice_ret[1], dice_ret[2]))

    avg_WT = dice_WT / 69.0
    avg_TC = dice_TC / 69.0
    avg_ET = dice_ET / 69.0

    metric = (avg_WT + avg_TC + avg_ET) / 3
    return metric, avg_WT, avg_TC, avg_ET


