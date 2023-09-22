from cProfile import label
import imp
import numpy as np
from scipy.spatial.distance import pdist, squareform
from skimage import measure, morphology
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import timm
from .ib import calculate_MI


class vib(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.encoder = timm.create_model('vit_base_patch16_224', img_size=224, pretrained=True)
        self.encoder.head = nn.Identity()
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        cls_feat = self.encoder(x)
        logit = self.head(cls_feat)
        with torch.no_grad():
            Z_numpy = cls_feat.cpu().detach().numpy()
            k = squareform(pdist(Z_numpy, 'euclidean'))
            sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))
        IXZ1 = calculate_MI(x, cls_feat, s_x=1000, s_y=sigma ** 2)
        return logit, IXZ1


class IRM(nn.Module):
    """Invariant Risk Minimization"""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def forward(self, logits, labels, unlabeled=None):
        penalty_weight = self.factor
        nll = 0.
        penalty = 0.

        '''
        for i, (x, y) in enumerate(zip(logits, labels)):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)
        '''
        nll += F.cross_entropy(logits, labels)
        penalty += self._irm_penalty(logits, labels)
        loss = nll + (penalty_weight * penalty)

        return loss


class IRM_2(nn.Module):
    """Invariant Risk Minimization"""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    @staticmethod
    def _irm_penalty(logits, y, domain_1, domain_2):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        #对一个bathsize的图片划分域
        logits_1 = logits.index_select(0, torch.IntTensor(domain_1).to("cuda"))  #生成的tensor张量需要移到cuda中！！
        logits_2 = logits.index_select(0, torch.IntTensor(domain_2).to("cuda"))
        y_1 = y.index_select(0, torch.IntTensor(domain_1).to("cuda"))
        y_2 = y.index_select(0, torch.IntTensor(domain_2).to("cuda"))

        loss_1 = F.cross_entropy(logits_1 * scale, y_1)
        loss_2 = F.cross_entropy(logits_2 * scale, y_2)
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def forward(self, logits, labels, domain_1, domain_2, unlabeled=None):
        penalty_weight = self.factor
        nll = 0.
        penalty = 0.

        '''
        for i, (x, y) in enumerate(zip(logits, labels)):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)
        '''
        nll += F.cross_entropy(logits, labels)
        penalty += self._irm_penalty(logits, labels, domain_1, domain_2)
        loss = nll + (penalty_weight * penalty)

        return loss

class IRM_3(nn.Module):
    """Invariant Risk Minimization"""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    @staticmethod
    def _irm_penalty(logits, y, domain_1, domain_2, domain_3):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        #对一个bathsize的图片划分域
        logits_1 = logits.index_select(0, torch.IntTensor(domain_1).to("cuda"))  #生成的tensor张量需要移到cuda中！！
        logits_2 = logits.index_select(0, torch.IntTensor(domain_2).to("cuda"))
        logits_3 = logits.index_select(0, torch.IntTensor(domain_3).to("cuda"))  
    

        y_1 = y.index_select(0, torch.IntTensor(domain_1).to("cuda"))
        y_2 = y.index_select(0, torch.IntTensor(domain_2).to("cuda"))
        y_3 = y.index_select(0, torch.IntTensor(domain_3).to("cuda"))
        

        loss_1 = F.cross_entropy(logits_1 * scale, y_1)
        loss_2 = F.cross_entropy(logits_2 * scale, y_2)
        loss_3 = F.cross_entropy(logits_3 * scale, y_3)
    

        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        grad_3 = autograd.grad(loss_3, [scale], create_graph=True)[0]
    

        result = 0.5* torch.sum( grad_1 * grad_2) + 0.5*torch.sum(grad_1 * grad_3 )+ +0.5* torch.sum(grad_2 * grad_3)
        return result

    def forward(self, logits, labels, domain_1, domain_2, domain_3, unlabeled=None):
        penalty_weight = self.factor
        nll = 0.
        penalty = 0.

        '''
        for i, (x, y) in enumerate(zip(logits, labels)):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)
        '''
        nll += F.cross_entropy(logits, labels)
        penalty += self._irm_penalty(logits, labels, domain_1, domain_2, domain_3)
        loss = nll + (penalty_weight * penalty)

        return loss   


class IRM_4(nn.Module):
    """Invariant Risk Minimization"""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    @staticmethod
    def _irm_penalty(logits, y, domain_1, domain_2, domain_3, domain_4):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        #对一个bathsize的图片划分域
        logits_1 = logits.index_select(0, torch.IntTensor(domain_1).to("cuda"))  #生成的tensor张量需要移到cuda中！！
        logits_2 = logits.index_select(0, torch.IntTensor(domain_2).to("cuda"))
        logits_3 = logits.index_select(0, torch.IntTensor(domain_3).to("cuda"))  
        logits_4 = logits.index_select(0, torch.IntTensor(domain_4).to("cuda"))

        y_1 = y.index_select(0, torch.IntTensor(domain_1).to("cuda"))
        y_2 = y.index_select(0, torch.IntTensor(domain_2).to("cuda"))
        y_3 = y.index_select(0, torch.IntTensor(domain_3).to("cuda"))
        y_4 = y.index_select(0, torch.IntTensor(domain_4).to("cuda"))

        loss_1 = F.cross_entropy(logits_1 * scale, y_1)
        loss_2 = F.cross_entropy(logits_2 * scale, y_2)
        loss_3 = F.cross_entropy(logits_3 * scale, y_3)
        loss_4 = F.cross_entropy(logits_4 * scale, y_4)

        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        grad_3 = autograd.grad(loss_3, [scale], create_graph=True)[0]
        grad_4 = autograd.grad(loss_4, [scale], create_graph=True)[0]

        result = 0.5* torch.sum( grad_1 * grad_2) + 0.5*torch.sum(grad_1 * grad_3 )+0.5* torch.sum(grad_1 * grad_4) +0.5* torch.sum(grad_2 * grad_3)+ 0.5*torch.sum(grad_2 * grad_4 )+ 0.5*torch.sum(grad_3 * grad_4)
        return result

    def forward(self, logits, labels, domain_1, domain_2, domain_3, domain_4, unlabeled=None):
        penalty_weight = self.factor
        nll = 0.
        penalty = 0.

        '''
        for i, (x, y) in enumerate(zip(logits, labels)):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)
        '''
        nll += F.cross_entropy(logits, labels)
        penalty += self._irm_penalty(logits, labels, domain_1, domain_2, domain_3, domain_4)
        loss = nll + (penalty_weight * penalty)

        return loss   

class IRM_5(nn.Module):
    """Invariant Risk Minimization"""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    @staticmethod
    def _irm_penalty(logits, y, domain_1, domain_2, domain_3, domain_4, domain_5):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        #对一个bathsize的图片划分域
        logits_1 = logits.index_select(0, torch.IntTensor(domain_1).to("cuda"))  #生成的tensor张量需要移到cuda中！！
        logits_2 = logits.index_select(0, torch.IntTensor(domain_2).to("cuda"))
        logits_3 = logits.index_select(0, torch.IntTensor(domain_3).to("cuda"))  
        logits_4 = logits.index_select(0, torch.IntTensor(domain_4).to("cuda"))
        logits_5 = logits.index_select(0, torch.IntTensor(domain_5).to("cuda"))

        y_1 = y.index_select(0, torch.IntTensor(domain_1).to("cuda"))
        y_2 = y.index_select(0, torch.IntTensor(domain_2).to("cuda"))
        y_3 = y.index_select(0, torch.IntTensor(domain_3).to("cuda"))
        y_4 = y.index_select(0, torch.IntTensor(domain_4).to("cuda"))
        y_5 = y.index_select(0, torch.IntTensor(domain_5).to("cuda"))

        loss_1 = F.cross_entropy(logits_1 * scale, y_1)
        loss_2 = F.cross_entropy(logits_2 * scale, y_2)
        loss_3 = F.cross_entropy(logits_3 * scale, y_3)
        loss_4 = F.cross_entropy(logits_4 * scale, y_4)
        loss_5 = F.cross_entropy(logits_5 * scale, y_5)


        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        grad_3 = autograd.grad(loss_3, [scale], create_graph=True)[0]
        grad_4 = autograd.grad(loss_4, [scale], create_graph=True)[0]
        grad_5 = autograd.grad(loss_5, [scale], create_graph=True)[0]

        result = 0.5* torch.sum( grad_1 * grad_2) + 0.5*torch.sum(grad_1 * grad_3 )+0.5* torch.sum(grad_1 * grad_4)+ 0.5* torch.sum( grad_1 * grad_5)+0.5* torch.sum(grad_2 * grad_3)+ 0.5*torch.sum(grad_2 * grad_4 )+ 0.5*torch.sum(grad_2 * grad_5)+0.5* torch.sum(grad_3 * grad_4)+ 0.5*torch.sum(grad_3 * grad_5 )+ 0.5*torch.sum(grad_4 * grad_5)
        return result

    def forward(self, logits, labels, domain_1, domain_2, domain_3, domain_4, domain_5, unlabeled=None):
        penalty_weight = self.factor
        nll = 0.
        penalty = 0.

        '''
        for i, (x, y) in enumerate(zip(logits, labels)):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)
        '''
        nll += F.cross_entropy(logits, labels)
        penalty += self._irm_penalty(logits, labels, domain_1, domain_2, domain_3, domain_4, domain_5)
        loss = nll + (penalty_weight * penalty)

        return loss   

class IRM_6(nn.Module):
    """Invariant Risk Minimization"""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    @staticmethod
    def _irm_penalty(logits, y, domain_1, domain_2, domain_3, domain_4, domain_5, domain_6):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        #对一个bathsize的图片划分域
        logits_1 = logits.index_select(0, torch.IntTensor(domain_1).to("cuda"))  #生成的tensor张量需要移到cuda中！！
        logits_2 = logits.index_select(0, torch.IntTensor(domain_2).to("cuda"))
        logits_3 = logits.index_select(0, torch.IntTensor(domain_3).to("cuda"))  
        logits_4 = logits.index_select(0, torch.IntTensor(domain_4).to("cuda"))
        logits_5 = logits.index_select(0, torch.IntTensor(domain_5).to("cuda"))
        logits_6 = logits.index_select(0, torch.IntTensor(domain_6).to("cuda"))

        y_1 = y.index_select(0, torch.IntTensor(domain_1).to("cuda"))
        y_2 = y.index_select(0, torch.IntTensor(domain_2).to("cuda"))
        y_3 = y.index_select(0, torch.IntTensor(domain_3).to("cuda"))
        y_4 = y.index_select(0, torch.IntTensor(domain_4).to("cuda"))
        y_5 = y.index_select(0, torch.IntTensor(domain_5).to("cuda"))
        y_6 = y.index_select(0, torch.IntTensor(domain_6).to("cuda"))

        loss_1 = F.cross_entropy(logits_1 * scale, y_1)
        loss_2 = F.cross_entropy(logits_2 * scale, y_2)
        loss_3 = F.cross_entropy(logits_3 * scale, y_3)
        loss_4 = F.cross_entropy(logits_4 * scale, y_4)
        loss_5 = F.cross_entropy(logits_5 * scale, y_5)
        loss_6 = F.cross_entropy(logits_6 * scale, y_6)


        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        grad_3 = autograd.grad(loss_3, [scale], create_graph=True)[0]
        grad_4 = autograd.grad(loss_4, [scale], create_graph=True)[0]
        grad_5 = autograd.grad(loss_5, [scale], create_graph=True)[0]
        grad_6 = autograd.grad(loss_6, [scale], create_graph=True)[0]


        result = 0.5* torch.sum( grad_1 * grad_2) + 0.5*torch.sum(grad_1 * grad_3 )+0.5* torch.sum(grad_1 * grad_4)+ 0.5* torch.sum( grad_1 * grad_5)+ 0.5* torch.sum( grad_1 * grad_6)+0.5* torch.sum(grad_2 * grad_3)+ 0.5*torch.sum(grad_2 * grad_4 )+ 0.5*torch.sum(grad_2 * grad_5)+ 0.5*torch.sum(grad_2 * grad_6)+0.5* torch.sum(grad_3 * grad_4)+ 0.5*torch.sum(grad_3 * grad_5 )+ + 0.5*torch.sum(grad_3 * grad_6)+ 0.5*torch.sum(grad_4 * grad_5) + 0.5*torch.sum(grad_4 * grad_6)+0.5*torch.sum(grad_5 * grad_6 )
        return result

    def forward(self, logits, labels, domain_1, domain_2, domain_3, domain_4, domain_5,domain_6,unlabeled=None):
        penalty_weight = self.factor
        nll = 0.
        penalty = 0.

        '''
        for i, (x, y) in enumerate(zip(logits, labels)):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)
        '''
        nll += F.cross_entropy(logits, labels)
        penalty += self._irm_penalty(logits, labels, domain_1, domain_2, domain_3, domain_4, domain_5, domain_6)
        loss = nll + (penalty_weight * penalty)

        return loss   
