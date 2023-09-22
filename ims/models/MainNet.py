import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from skimage import measure

from models.modeling_mainnet import VisionTransformer


def get_coordinates(attn_maps: torch.Tensor, scale=16, λ=1.3):  # B, 28, 28
    mean_v = torch.mean(attn_maps, dim=[0, 1], keepdim=True) * λ
    mask_maps = (attn_maps > mean_v).int()
    coordinates = []
    for i, m in enumerate(mask_maps):
        m = m.cpu().numpy()
        component_labels = measure.label(m)
        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        max_idx = areas.index(max(areas))

        intersection = (component_labels == (max_idx + 1)).astype(int) == 1
        prop = measure.regionprops(intersection.astype(int))
        if len(prop) == 0:
            bbox = [0, 0, 14, 14]
            print('there is one img no intersection')
        else:
            bbox = prop[0].bbox

        x_lefttop = bbox[0] * scale - 1
        y_lefttop = bbox[1] * scale - 1
        x_rightlow = bbox[2] * scale - 1
        y_rightlow = bbox[3] * scale - 1
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)
    return coordinates


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


class MainNet(nn.Module):
    def __init__(self, config, args, num_classes):
        super(MainNet, self).__init__()
        self.device = config.device
        self.img_size = args.img_size
        self.num_classes = num_classes
        self.pretrained_backbone = VisionTransformer(config, self.img_size, num_classes, vis=True, zero_head=True)
        if args.pretrained_dir is not None:
            print('===> load pretrained model : {}'.format(args.pretrained_dir))
            self.pretrained_backbone.load_from(np.load(args.pretrained_dir))
        self.scale = int(config.patches['size'][0])
        self.h = int(self.img_size / self.scale)

    def forward(self, x, labels=None):
        cls_feature1, logits1, attn_weights = self.pretrained_backbone(x)
        attn_maps = torch.stack(attn_weights, dim=0).mean(dim=2).permute(1, 0, 2, 3).detach()
        attn_weights_trans = get_attention_flow_maps(attn_maps, device=self.device)  # B, 785, 785
        batch_size = attn_weights_trans.shape[0]
        cls_attn_maps = attn_weights_trans[:, 0, 1:].reshape(batch_size, self.h, self.h)  # B, 28, 28
        coordinates = torch.tensor(get_coordinates(cls_attn_maps, self.scale))

        local_imgs = torch.zeros([batch_size, 3, 448, 448]).to(self.device)  # [B, 3, 448, 448]
        for i in range(batch_size):
            [x0, y0, x1, y1] = coordinates[i]
            local_imgs[i:i+1] = F.interpolate(x[i:i+1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(448, 448), mode='bilinear', align_corners=True)  # 3, 448, 448
        cls_feature2, logits2, _ = self.pretrained_backbone(local_imgs.detach())

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss1 = loss_fct(logits1.view(-1, self.num_classes), labels.view(-1))
            loss2 = loss_fct(logits2.view(-1, self.num_classes), labels.view(-1))
            return loss1, loss2, logits1, logits2
        else:
            return logits1, logits2, attn_weights













