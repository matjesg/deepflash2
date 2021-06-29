# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_losses.ipynb (unless otherwise specified).

__all__ = ['LOSSES', 'FastaiLoss', 'WeightedLoss', 'JointLoss', 'DeepSupervisionLoss', 'get_loss']

# Cell
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import fastai
from fastai.torch_core import TensorBase
import segmentation_models_pytorch as smp
from .utils import import_package

# Cell
LOSSES = ['CrossEntropyLoss', 'DiceLoss', 'SoftCrossEntropyLoss', 'CrossEntropyDiceLoss',  'JaccardLoss', 'FocalLoss', 'LovaszLoss', 'TverskyLoss']

# Cell
class FastaiLoss(_Loss):
    'Wrapper class around loss function for handling different tensor types.'
    def __init__(self, loss, axis=1):
        super().__init__()
        self.loss = loss
        self.axis=axis

    #def _contiguous(self, x): return TensorBase(x.contiguous())
    def _contiguous(self,x):
        return TensorBase(x.contiguous()) if isinstance(x,torch.Tensor) else x

    def forward(self, *input):
        #input = map(self._contiguous, input)
        input = [self._contiguous(x) for x in input]
        return self.loss(*input) #

# Cell
# from https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/joint_loss.py
class WeightedLoss(_Loss):
    '''
    Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    '''
    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight

class JointLoss(_Loss):
    'Wrap two loss functions into one. This class computes a weighted sum of two losses.'

    def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)

# Cell
import math
class DeepSupervisionLoss(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, inputs, targets):
        loss = 0
        div = 1
        for i, input_head in enumerate(inputs):
            #target = interpolate(targets, size=input_head.shape, mode='nearest')
            targets_scaled = torch.nn.functional.adaptive_max_pool2d(targets.float(), input_head.shape[-2:]).long()
            loss += self.criterion(input_head, targets_scaled)/div
            div *= 2
            if i>2: break
        return loss

# Cell
def get_loss(loss_name, mode='multiclass', deep_supervision=False, classes=[1], smooth_factor=0., alpha=0.5, beta=0.5, gamma=2.0, reduction='mean', **kwargs):
    'Load losses from based on loss_name'

    assert loss_name in LOSSES, f'Select one of {LOSSES}'

    if loss_name=="CrossEntropyLoss":
        loss = fastai.losses.CrossEntropyLossFlat(axis=1)

    if loss_name=="SoftCrossEntropyLoss":
        loss = smp.losses.SoftCrossEntropyLoss(smooth_factor=smooth_factor, **kwargs)

    elif loss_name=="DiceLoss":
        loss = smp.losses.DiceLoss(mode=mode, classes=classes, **kwargs)

    elif loss_name=="JaccardLoss":
        loss = smp.losses.JaccardLoss(mode=mode, classes=classes, **kwargs)

    elif loss_name=="FocalLoss":
        loss = smp.losses.FocalLoss(mode=mode, alpha=alpha, gamma=gamma, reduction=reduction, **kwargs)

    elif loss_name=="LovaszLoss":
        loss = smp.losses.LovaszLoss(mode=mode, **kwargs)

    elif loss_name=="TverskyLoss":
        kornia = import_package('kornia')
        loss = kornia.losses.TverskyLoss(alpha=alpha, beta=beta, **kwargs)

    elif loss_name=="CrossEntropyDiceLoss":
        dc = smp.losses.DiceLoss(mode=mode, classes=classes, **kwargs)
        ce = fastai.losses.CrossEntropyLossFlat(axis=1)
        loss = JointLoss(ce, dc, 1, 1)

    if deep_supervision:
        return DeepSupervisionLoss(loss)

    else:
        return loss