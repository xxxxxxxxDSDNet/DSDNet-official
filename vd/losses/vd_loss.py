import math
import numpy as np

import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
import torchvision

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.basic_loss import _reduction_modes
from basicsr.losses.loss_util import weighted_loss


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def l2_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@LOSS_REGISTRY.register()
class L1_Loss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1_Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class L2_Loss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L2_Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l2_loss(pred, target, weight, reduction=self.reduction)

@LOSS_REGISTRY.register()
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, loss_weight=1.0):
        super(VGGPerceptualLoss, self).__init__()
        self.loss_weight = loss_weight
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[], mask=None, return_feature=False):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                if mask is not None:
                    _,_,H,W = x.shape
                    mask_resized = F.interpolate(mask, size=(H, W), mode='nearest')[:, 0:1, :, :]
                    x = x*mask_resized
                    y = y*mask_resized
                    loss += torch.nn.functional.l1_loss(x, y)
                else:
                    loss += torch.nn.functional.l1_loss(x, y)
                    
                if return_feature:
                    return x, y
                    
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss * self.loss_weight

def color_loss(x1, x2):
	x1_norm = torch.sqrt(torch.sum(torch.pow(x1, 2)) + 1e-8)
	x2_norm = torch.sqrt(torch.sum(torch.pow(x2, 2)) + 1e-8)
	# 内积
	x1_x2 = torch.sum(torch.mul(x1, x2))
	cosin = 1 - x1_x2 / (x1_norm * x2_norm)
	return cosin

@LOSS_REGISTRY.register()
class ColorLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0):
        super(ColorLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        if isinstance(pred,tuple):
            loss_1 = color_loss(pred[0], target)
            target_2 = F.interpolate(target, scale_factor=0.5, mode='bilinear')
            loss_2 = color_loss(pred[1], target_2)
            target_3 = F.interpolate(target_2, scale_factor=0.5, mode='bilinear')
            loss_3 = color_loss(pred[2], target_3)
            return self.loss_weight * (loss_1+loss_2+loss_3)
        else:
            return self.loss_weight * color_loss(pred, target)
        



