import torch
import torch.nn as nn


class CaptionLoss(nn.Module):
    def __init__(self,reduction=None):
        super(CaptionLoss, self).__init__()

    def forward(self, logits):
        caption_loss = nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
        return caption_loss