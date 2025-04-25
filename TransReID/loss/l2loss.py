import torch.nn as nn
import torch
class L2LOSS(nn.Module):

    def forward(self, x,y):
        return torch.mean((x-y)**2)