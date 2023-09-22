import random
import time
import os
import torch
import torch.nn as nn
import torchvision
import timm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from skimage import measure, morphology

# from models.modeling import VisionTransformer
from models.modeling_mask import VisionTransformer
from utils.ib import calculate_MI


def remove_small_points(x):
    img_label, num = measure.label(x, neighbors=4, return_num=True)
    props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等

    resMatrix = np.zeros(img_label.shape).astype(np.int32)
    max_idx = -1
    max_area = 0
    for i in range(0, len(props)):
        if props[i].area > max_area:
            max_area = props[i].area
            max_idx = i
    resMatrix += (img_label == max_idx + 1).astype(np.int32)
    return resMatrix


def get_largest_connect_component(bw_img):
    labeled_img, num = measure.label(bw_img, background=0, return_num=True)

    max_label = 0
    max_num = 0
    for i in range(1, num + 1):  # Starting at 1 here prevents setting the background to the maximum connect component
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label).astype(np.int32)

    return lcc, max_num


def get_attention_flow_maps(att_maps, device):  # B, 12, 785, 785
    joint_maps = []
    for weights in att_maps:
        residual_att = torch.eye(weights.size(1)).to(device)  # 785×785
        aug_att_mat = weights + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
        joint_maps.append(joint_attentions[-1])
    att_maps = torch.stack(joint_maps, dim=0)
    return att_maps


def get_mask_nums(att_maps, p_min=0.35, p_max=0.85, lmd=1.0):
    batch_mean = torch.mean(att_maps, dim=1, keepdim=True) * lmd
    p_mean = torch.mean((att_maps < batch_mean).type(torch.float).sum(dim=1)) / att_maps.shape[-1]
    if p_mean < p_min:
        p_mean = p_min
    if p_mean > p_max:
        p_mean = p_max
    p_nums = int(p_mean * att_maps.shape[-1])
    return p_nums


# class MaskTwoStream(nn.Module):  # for embedding layer
#     def __init__(self, config, args, num_classes, mask_scope=0.5):
#         super(MaskTwoStream, self).__init__()
#         print('\n=====================\n=====> mask_scope={}\n======================'.format(mask_scope))
#         self.device = config.device
#         self.img_size = args.img_size
#         self.num_classes = num_classes
#         self.base_model_one = VisionTransformer(config, self.img_size, num_classes, vis=True, zero_head=True)
#         if args.pretrained_dir is not None:
#             print('===> load pretrained model : {}'.format(args.pretrained_dir))
#             self.base_model_one.load_from(np.load(args.pretrained_dir))
#         if args.checkpoint is not None:
#             print('===> load checkpoint: {}'.format(args.checkpoint))
#             self.base_model_one.load_state_dict(torch.load(args.checkpoint))
#
#         self.num_patches = int(self.img_size / config.patches['size'][0]) ** 2
#         self.k_scope = int(self.num_patches * mask_scope)
#         self.norm = nn.LayerNorm(self.num_patches)
#
#         self.hidden_concat = config.hidden_size * 2
#         self.classfier = nn.Linear(self.hidden_concat, num_classes)
#
#     def forward(self, x, labels=None):
#         print('===>ok')
#         if labels is not None:
#             loss1, logits1, cls_feature1, att_map = self.base_model_one(x, labels)
#             print('===>ok')
#             att_map = torch.stack(att_map, dim=0).detach()  # 12, B, 12, 785, 785
#             att_map = att_map.mean(dim=2).mean(dim=0)[:, 1:, 1:]  # B, 785, 785 -> B, 784, 784
#             att_map = self.norm(att_map.sum(dim=1))  # B, 784
#             att_map = torch.sigmoid(att_map)  # B, 784
#
#             sorted_index = torch.argsort(att_map, dim=1, descending=False)  # B  从低到高
#             # max_index = sorted_index[:, self.k_scope:] + 1
#             max_index = sorted_index[:, self.k_scope:]
#             print('===> ', max_index.sum())
#             #  可视化看一下效果
#             mask = torch.zeros_like(input=att_map)
#             print('===> mask=', mask.shape)
#             for i in range(max_index.shape[0]):
#                 mask[i, max_index[i]] = 1
#             print('===> mask=', mask.sum())
#             grid_size = int(np.sqrt(att_map.shape[-1]))
#             mask = mask.reshape(-1, grid_size, grid_size)  # B, 28, 28
#             mask = torch.nn.functional.interpolate(mask.unsqueeze(dim=1),
#                                                    size=[self.img_size, self.img_size],
#                                                    mode='nearest')  # B, 1, 448, 448 bilinear
#             for i in range(mask.shape[0]):
#                 img = mask[i].permute(1, 2, 0).detach().cpu().numpy()
#                 plt.imshow(img)
#                 plt.savefig('test{}.jpg'.format(i))
#                 src = torch.mul(x[i], mask[i]).detach().permute(1, 2, 0).cpu().numpy()
#                 plt.imshow(src)
#                 plt.savefig('src{}.jpg'.format(i))
#             exit(-1)
#
#             loss2, logits2, cls_feature2, _ = self.base_model_one(x, labels, mask=max_index)
#             cls_concat = torch.cat((cls_feature1, cls_feature2), dim=1)
#             logits_concat = self.classfier(cls_concat)
#             loss_concat = nn.CrossEntropyLoss()(logits_concat.view(-1, self.num_classes), labels.view(-1))
#
#             #  MI 互信息
#             # with torch.no_grad():
#             #     Z_numpy = cls_feature1.cpu().detach().numpy()
#             #     k = squareform(pdist(Z_numpy, 'euclidean'))
#             #     sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))
#             # IXZ1 = calculate_MI(x, cls_feature1, s_x=1000, s_y=sigma ** 2)
#             #
#             # with torch.no_grad():
#             #     Z_numpy = cls_feature2.cpu().detach().numpy()
#             #     k = squareform(pdist(Z_numpy, 'euclidean'))
#             #     sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))
#             # IXZ2 = calculate_MI(x, cls_feature2, s_x=1000, s_y=sigma ** 2)
#             #
#             # loss_ixz = IXZ1 + IXZ2
#
#             # return loss1, IXZ1, logits1
#             return loss1, loss2, loss_concat, logits1, logits2, logits_concat
#
#         else:
#             logits1, cls_feature1, weight_map = self.base_model_one(x)
#             att_map = torch.stack(weight_map, dim=0).detach()  # 12, B, 12, 785, 785
#             att_map = att_map.mean(dim=2).mean(dim=0)[:, 1:, 1:]  # B, 785, 785 -> B, 784, 784
#             att_map = self.norm(att_map.sum(dim=1))  # B, 784
#             att_map = torch.sigmoid(att_map)  # B, 784
#
#             sorted_index = torch.argsort(att_map, dim=1, descending=False)  # B  从低到高
#             max_index = sorted_index[:, self.k_scope:] + 1
#
#             logits2, cls_feature2, weight_map = self.base_model_one(x, mask=max_index)
#             logits2, cls_feature2, _ = self.base_model_one(x, mask=max_index)
#             cls_concat = torch.cat((cls_feature1, cls_feature2), dim=1)
#             logits_concat = self.classfier(cls_concat)
#             # return logits1, weight_map
#             return logits1, logits2, logits_concat, weight_map


class MaskTwoStream(nn.Module):  # for embedding layer
    def __init__(self, config, args, num_classes, mask_scope=0.5):
        super(MaskTwoStream, self).__init__()
        self.device = config.device
        self.img_size = args.img_size
        self.num_classes = num_classes
        self.base_model_one = VisionTransformer(config, self.img_size, num_classes, vis=True, zero_head=True)
        if args.pretrained_dir is not None:
            print('===> load pretrained model : {}'.format(args.pretrained_dir))
            self.base_model_one.load_from(np.load(args.pretrained_dir))

        self.num_patches = int(self.img_size / config.patches['size'][0]) ** 2
        self.h = int(np.sqrt(self.num_patches))
        self.k_scope = int(self.num_patches * mask_scope)
        self.norm = nn.LayerNorm(self.num_patches)

        self.hidden_concat = config.hidden_size * 2
        # self.classfier = nn.Linear(self.hidden_concat, num_classes)
        self.classifier = nn.Linear(self.hidden_concat, num_classes)

    def forward(self, x, labels=None):
        if labels is not None:
            loss1, logits1, cls_feature1, att_maps = self.base_model_one(x, labels)
            att_maps = torch.stack(att_maps, dim=0).detach()  # 12, B, 12, 785, 785

            # 直接平均
            att_maps = att_maps.mean(dim=2).mean(dim=0)[:, 1:, 1:]  # B, 785, 785 -> B, 784, 784
            att_maps = self.norm(att_maps.sum(dim=1))  # B, 784
            cls_maps = torch.sigmoid(att_maps)  # B, 784

            # Attention Flow方式
            # att_maps = get_attention_flow_maps(att_maps, device=self.device)  # B, 785, 785
            # cls_maps = att_maps.sum(dim=1)[:, 1:]  # B, 28, 28

            num_mask = get_mask_nums(cls_maps)
            # print('===> ', num_mask)
            sorted_index = torch.argsort(cls_maps, dim=1, descending=False)  # B  从低到高
            # print('===> sorted_index=', sorted_index)
            # max_index = sorted_index[:, self.k_scope:] + 1
            max_index = sorted_index[:, num_mask:] + 1

            loss2, logits2, cls_feature2, _ = self.base_model_one(x, labels, mask=max_index)
            cls_concat = torch.cat((cls_feature1, cls_feature2), dim=1)
            logits_concat = self.classifier(cls_concat)
            loss_concat = nn.CrossEntropyLoss()(logits_concat.view(-1, self.num_classes), labels.view(-1))

            #  MI 互信息
            with torch.no_grad():
                Z_numpy = cls_feature1.cpu().detach().numpy()
                k = squareform(pdist(Z_numpy, 'euclidean'))
                sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))
            IXZ1 = calculate_MI(x, cls_feature1, s_x=1000, s_y=sigma ** 2)
            # print('\n===> IXZ1=', IXZ1)
            # loss_ixz = IXZ1
            with torch.no_grad():
                Z_numpy = cls_feature2.cpu().detach().numpy()
                k = squareform(pdist(Z_numpy, 'euclidean'))
                sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))
            IXZ2 = calculate_MI(x, cls_feature2, s_x=1000, s_y=sigma ** 2)
            # print('===> IXZ2=', IXZ2)
            with torch.no_grad():
                Z_numpy = cls_concat.cpu().detach().numpy()
                k = squareform(pdist(Z_numpy, 'euclidean'))
                sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))
            IXZ3 = calculate_MI(x, cls_concat, s_x=1000, s_y=sigma ** 2)
            # print('===> IXZ3=', IXZ3)
            loss_ixz = IXZ1 + IXZ2 + IXZ3

            return loss1, loss2, loss_concat, loss_ixz, logits1, logits2, logits_concat
            # return loss1, loss2, loss_concat, logits1, logits2, logits_concat

        else:
            logits1, cls_feature1, weight_map = self.base_model_one(x)
            att_maps = torch.stack(weight_map, dim=0).detach()  # 12, B, 12, 785, 785
            # 直接平均
            att_maps = att_maps.mean(dim=2).mean(dim=0)[:, 1:, 1:]  # B, 785, 785 -> B, 784, 784
            att_maps = self.norm(att_maps.sum(dim=1))  # B, 784
            cls_maps = torch.sigmoid(att_maps)  # B, 784

            # Attention Flow方式
            # att_maps = get_attention_flow_maps(att_maps, device=self.device)  # B, 785, 785
            # cls_maps = att_maps.sum(dim=1)[:, 1:]  # B, 28, 28

            num_mask = get_mask_nums(cls_maps)
            # print('===> ', num_mask)
            sorted_index = torch.argsort(cls_maps, dim=1, descending=False)  # B  从低到高
            # max_index = sorted_index[:, self.k_scope:] + 1
            max_index = sorted_index[:, num_mask:] + 1

            logits2, cls_feature2, _ = self.base_model_one(x, mask=max_index)
            cls_concat = torch.cat((cls_feature1, cls_feature2), dim=1)
            logits_concat = self.classifier(cls_concat)

            return logits1, logits2, logits_concat, weight_map

    def get_params(self):
        # ftlayer_params = list(self.base_model_one.parameters())
        # freshlayer_params = list(self.classfier.parameters())
        freshlayer_params = list(self.classifier.parameters()) + list(self.base_model_one.head.parameters())
        ftlayer_params_ids = list(map(id, freshlayer_params))
        ftlayer_params = filter(lambda p: id(p) not in ftlayer_params_ids, self.parameters())
        return ftlayer_params, freshlayer_params


