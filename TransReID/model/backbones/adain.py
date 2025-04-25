import torch
import torch.nn as nn
# 2017 Oral  Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization 
##############  AdaIN-pytorch from https://github.com/aadhithya/AdaIN-pytorch/blob/master/model.py
def compute_mean_std(
    feats: torch.Tensor, eps=1e-8, infer=False
) -> torch.Tensor:
    assert (
        len(feats.shape) == 4 or len(feats.shape) == 3
    ), "feature map should be 4-dimensional of the form N,C,H,W! or 3-dimensional of the form N,Seq,D!"
    #  * Doing this to support ONNX.js inference.
    # if infer:
    #     n = 1
    #     c = 512  # * fixed for vgg19
    # else:
    #     n, c, _, _ = feats.shape
    n, seq, c =feats.shape # b* 129 *768

    mean = torch.mean(feats[:, 1:, :], dim=-2).view(n, 1, c) # NO class token
    std = torch.std(feats[:, 1:, :], dim=-2).view(n, 1, c) + eps

    return mean, std
class AdaIN(nn.Module): # zwq add nn.Module in the ()
    """
    Adaptive Instance Normalization as proposed in
    'Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization' by Xun Huang, Serge Belongie.
    """
    def __init__(self):
        super().__init__()
        # self.adain_xishu = adain_xishu

    def _compute_mean_std(
        self, feats: torch.Tensor, eps=1e-8, infer=False
    ) -> torch.Tensor:
        return compute_mean_std(feats, eps, infer)

    def __call__(
        self,
        content_feats: torch.Tensor,
        infer: bool = False,
    ) -> torch.Tensor:
        """
        __call__ Adaptive Instance Normalization as proposaed in
        'Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization' by Xun Huang, Serge Belongie.

        Args:
            content_feats (torch.Tensor): Content features
            style_feats (torch.Tensor): 估计

        Returns:
            torch.Tensor: [description]
        """
        # # 计算内容特征的均值和方差
        c_mean, c_std = self._compute_mean_std(content_feats, infer=infer) # 129, 1, 768
        # alpha_range = (0.8,1.2)
        # beta_range = (0.8,1.2)
        # alpha_range = (0.4,1.6)
        # beta_range = (0.4,1.6)     
        alpha_range = (0.4,1.6)
        beta_range = (0.4,1.6)        

        # 随机生成alpha和beta
        alpha = torch.FloatTensor(1).uniform_(*alpha_range).item()
        beta = torch.FloatTensor(1).uniform_(*beta_range).item()     

        # 计算风格特征的均值和方差 估计
        s_mean = alpha * c_mean  
        s_std = beta * c_std

        normalized = (s_std * (content_feats - c_mean) / c_std) + s_mean

        return normalized