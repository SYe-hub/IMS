import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, num_class):
        super(ResNet, self).__init__()
        basenet = models.resnet50(pretrained=True)
        self.conv = nn.Sequential(*list(basenet.children())[:-2])
        feadim = 2048
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(feadim, num_class)

    def forward(self, x):
        x = self.conv(x)
        fea_pool = self.avg_pool(x).view(x.size(0), -1)  # [B, 2048]
        logits = self.classifier(fea_pool)
        return logits

    def get_params(self):
        ft_layers_params = list(self.conv.parameters())
        ft_layers_ids = list(map(id, ft_layers_params))
        fresh_layer_params = filter(lambda p: id(p) not in ft_layers_ids, self.parameters())
        return ft_layers_params, fresh_layer_params







