import random
import time
import os

import PIL.Image
import torch
import torch.nn as nn
import torchvision
import timm
import numpy as np
import matplotlib.pyplot as plt

from models.modeling import VisionTransformer


class MaskAttnNet(nn.Module):  # for embedding layer
    def __init__(self, config, args, num_classes, mask_scope=0.75):
        super(MaskAttnNet, self).__init__()
        self.device = config.device
        self.img_size = args.img_size
        self.base_model = VisionTransformer(config, self.img_size, num_classes, vis=True)
        self.base_model.load_state_dict(torch.load(args.checkpoint))
        self.num_patches = int(self.img_size / config.patches['size'][0]) ** 2
        self.k_scope = int(self.num_patches * 4 * mask_scope)
        self.norm = nn.LayerNorm(self.num_patches * 4)

    def forward(self, x):
        att_map = self.base_model(x)[-1]  # [B, 12, 785, 785] * 12
        att_map = torch.stack(att_map, dim=0).detach()  # 12, B, 12, 785, 785
        att_map = att_map.mean(dim=2).mean(dim=0)[:, 1:, 1:]  # B, 785, 785 -> B, 784, 784

        att_map = self.norm(att_map.sum(dim=1))  # B, 784
        att_map = torch.sigmoid(att_map)  # B, 784

        # 增加阈值
        threshold = att_map.kthvalue(self.k_scope, dim=1)[0]  # B
        threshold = threshold.unsqueeze(dim=-1).expand(att_map.shape)
        att_map = (att_map > threshold).type(torch.float)