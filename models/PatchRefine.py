import os
import time
import random
import timm
import torch
import torch.nn as nn
import torchvision
import PIL.Image
import numpy as np
import ml_collections
import matplotlib.pyplot as plt
from skimage import measure, morphology
import cv2

from models.modeling_patch_refine import VisionTransformer

def remove_small_points(x, threshold_point):
    img_label, num = measure.label(x, neighbors=4, return_num=True)
    props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等

    resMatrix = np.zeros(img_label.shape).astype(np.int32)
    max_idx = -1
    max_area = 0
    for i in range(0, len(props)):
        if props[i].area > max_area:
            max_area = props[i].area
            max_idx = i
    if max_idx == -1 or max_area < threshold_point:
        return x
    else:
        resMatrix += (img_label == max_idx + 1).astype(np.int32)
        return resMatrix


class MaskGenerate(nn.Module):  # 生成mask的模块
    def __init__(self, k_scope=784, threshold_point=160):
        super(MaskGenerate, self).__init__()
        self.k_scope = k_scope
        self.threshold_point = threshold_point

    def forward(self, att_maps):
        att_maps = torch.stack(att_maps, dim=0).detach()  # 12, B, 12, 785, 785
        att_maps = att_maps.mean(dim=2).mean(dim=0).squeeze()  # B, 785, 785
        att_maps = att_maps[:, 0, 1:]
        B, num_patch = att_maps.shape
        grid_size = int(np.sqrt(num_patch))
        grid_size_new = grid_size * 2
        att_maps = att_maps.reshape(B, grid_size, grid_size)  # B, 28, 28
        att_maps = torchvision.transforms.Resize(size=(grid_size_new, grid_size_new), interpolation=PIL.Image.NEAREST)(att_maps)
        mask_list = []
        for att_map in att_maps:  # 1, 56, 56
            att_map = att_map.cpu().numpy()
            data_mean = att_map.mean()
            att_map = (att_map > data_mean).astype(np.int32)
            att_map = remove_small_points(att_map, threshold_point=self.threshold_point)
            mask_list.append(torch.tensor(att_map))
        masks = torch.stack(mask_list, dim=0).reshape(B, -1).type(torch.bool)

        min_index = torch.argsort(masks, dim=1, descending=True)  # B  从高到低排序
        # print('===> ', torch.sort(masks, dim=1, descending=True))
        min_index = min_index[:, :self.k_scope] + 1
        masks = min_index
        return masks


class PatchRefine(nn.Module):  # 最终模型
    def __init__(self, config, args, num_classes):
        super(PatchRefine, self).__init__()
        self.base_model = VisionTransformer(config,
                                            img_size=args.img_size,
                                            num_classes=num_classes,
                                            smoothing_value=args.smoothing_value,
                                            zero_head=True,
                                            vis=True)
        self.base_model.load_from(np.load(args.pretrained_dir))
        self.base_model.load_embedding_refine(np.load(args.pretrained_embedding_refine_dir))
        self.mask_generator = MaskGenerate()

    def forward(self, x, labels=None):
        if labels is not None:
            loss1, logits1, att_map = self.base_model(x, labels)
            with torch.no_grad():
                mask = self.mask_generator(att_map)
            loss2, logits2, _ = self.base_model(x, labels, mask=mask)
            return loss1, loss2, logits1, logits2
        else:
            logits1, att_map = self.base_model(x)
            mask = self.mask_generator(att_map)
            logits2, _ = self.base_model(x, mask=mask)
            return logits1, logits2, att_map
