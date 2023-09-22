import torch
import torch.nn as nn
import numpy as np
import utils
import os
import torch.nn.functional as F
import random
import copy


def cutmix(input, target, prob=1.0, beta=3.0):
    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0)).cuda()
    target_b = target.clone()

    if r < prob:
        bs = input.size(0)
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]
        input_b = input[rand_index].clone()
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input_b[:, :, bbx1:bbx2, bby1:bby2]

        # adjust lambda to exactly match pixel ratio
        lam_a = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        lam_a *= torch.ones(input.size(0))

    lam_b = 1 - lam_a

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def cutout(input, target, prob=0.5):
    r = np.random.rand(1)
    lam = torch.ones(input.size(0)).cuda()
    target_b = target.clone()
    lam_a = lam
    lam_b = 1 - lam

    if r < prob:
        bs = input.size(0)
        lam = 0.75
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = 0

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def mixup(input, target, prob=0.5, beta=1.0):
    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0)).cuda()
    bs = input.size(0)
    target_a = target  # 原始的label
    target_b = target

    if r < prob:
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]  # 交换顺序的label
        lam = np.random.beta(beta, beta)
        lam_a = lam_a * lam  # 0~1范围bs大小的数组
        input = input * lam + input[rand_index] * (1 - lam)

    lam_b = 1 - lam_a

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def rand_bbox(size, lam, center=False, attcen=None):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    elif len(size) == 2:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    if attcen is None:
        # uniform
        cx = 0
        cy = 0
        if W > 0 and H > 0:
            cx = np.random.randint(W)
            cy = np.random.randint(H)
        if center:
            cx = int(W / 2)
            cy = int(H / 2)
    else:
        cx = attcen[0]
        cy = attcen[1]

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_bbox(imgsize=(224, 224), beta=1.0):
    r = np.random.rand(1)
    lam = np.random.beta(beta, beta)
    bbx1, bby1, bbx2, bby2 = rand_bbox(imgsize, lam)

    return [bbx1, bby1, bbx2, bby2]
