import random
import time
import os
import torch
import torch.nn as nn
import torchvision
import timm
import numpy as np
import matplotlib.pyplot as plt
import ml_collections
import PIL.Image

from models.modeling_mask import VisionTransformer


class PatchRefineTwoStream(nn.Module):
    def __init__(self, config, args, num_classes, mask_scope=0.75):
        super(PatchRefineTwoStream, self).__init__()
        self.device = config.device
        self.img_size = args.img_size
        self.num_classes = num_classes
        self.zoom = 0.5
        self.base_model_one = VisionTransformer(config, self.img_size, num_classes, vis=True, zero_head=True)
        self.base_model_one.load_from(np.load(args.pretrained_dir))

        # self.base_model_two = VisionTransformer(config, self.img_size, num_classes, zero_head=True)
        # self.base_model_two.load_from(np.load(args.pretrained_dir))

        self.num_patches = int(self.img_size * 2 / config.patches['size'][0]) ** 2
        self.k_scope = int(self.num_patches * mask_scope)
        self.norm = nn.LayerNorm(self.num_patches)

        self.hidden_concat = config.hidden_size * 2
        self.classfier = nn.Linear(self.hidden_concat, num_classes)

    def forward(self, x, labels=None):
        if labels is not None:
            loss1, logits1, cls_feature1, att_map = self.base_model_one(x, labels)
            att_map = torch.stack(att_map, dim=0).detach()  # 12, B, 12, 785, 785
            att_map = att_map.mean(dim=2).mean(dim=0)[:, 1:, 1:]  # B, 785, 785 -> B, 784, 784
            att_map = att_map.sum(dim=1).reshape(-1, 28, 28)  # B, 28, 28
            att_map = torchvision.transforms.Resize(size=(56, 56), interpolation=PIL.Image.NEAREST)(att_map)
            att_map = att_map.reshape(-1, 56 * 56)
            # att_map = self.norm(att_map.sum(dim=1))  # B, 784 * 4
            att_map = self.norm(att_map)  # B, 784 * 4
            att_map = torch.sigmoid(att_map)  # B, 784 * 4

            min_index = torch.argsort(att_map, dim=1, descending=False)  # B  从低到高
            min_index = min_index[:, self.k_scope:] + 1
            att_map = min_index

            x = torchvision.transforms.Resize(size=(448 * 2, 448 * 2))(x)
            print('======> ', x.shape)
            # loss2, logits2, cls_feature2, _ = self.base_model_two(x, labels, mask=att_map)
            loss2, logits2, cls_feature2, _ = self.base_model_one(x, labels, mask=att_map)
            cls_concat = torch.cat((cls_feature1, cls_feature2), dim=1)
            logits_concat = self.classfier(cls_concat)
            loss_concat = nn.CrossEntropyLoss()(logits_concat.view(-1, self.num_classes), labels.view(-1))

            return loss1, loss2, loss_concat, logits1, logits2, logits_concat

        else:
            logits1, cls_feature1, att_map = self.base_model_one(x)
            att_map = torch.stack(att_map, dim=0).detach()  # 12, B, 12, 785, 785
            att_map = att_map.mean(dim=2).mean(dim=0)[:, 1:, 1:]  # B, 785, 785 -> B, 784, 784
            att_map = self.norm(att_map.sum(dim=1))  # B, 784
            att_map = torch.sigmoid(att_map)  # B, 784

            min_index = torch.argsort(att_map, dim=1, descending=False)  # B  从低到高
            min_index = min_index[:, self.k_scope:] + 1
            att_map = min_index

            # logits2, cls_feature2, attn_weights = self.base_model_two(x, mask=att_map)
            logits2, cls_feature2, attn_weights = self.base_model_one(x, mask=att_map)
            cls_concat = torch.cat((cls_feature1, cls_feature2), dim=1)
            logits_concat = self.classfier(cls_concat)
            return logits1, logits2, logits_concat

















