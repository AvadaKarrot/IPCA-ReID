# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .l2loss import L2LOSS


def make_loss_hdr_pretrain(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER

    l2loss = L2LOSS()


    def loss_func(img_reconstruct=None,target_image = None):
        L2LOSS = l2loss(img_reconstruct, target_image)
        return cfg.MODEL.L2_LOSS_WEIGHT * L2LOSS

    return loss_func


