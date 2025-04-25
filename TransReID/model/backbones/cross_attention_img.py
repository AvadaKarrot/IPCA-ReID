import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp, Block

import math

class Cross_Attention_img(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        ############### zwq 
        self.qq = nn.Linear(dim, dim, bias=qkv_bias) # deal with text feature
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  # Only compute key and value (img feature)

        ##############################
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        ##### debug 
        self.norm = nn.LayerNorm(dim)


    def forward(self, x, text_features):
        # img_feat: b * 129 * d
        # text_feat: L * d
        B, N, C = x.shape

        text_features = text_features.unsqueeze(0).repeat(B, 1, 1) # 文本特征作为query
        q = text_features #     text 特征作为query
        
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # Unpack key and value
        # Use provided text as query
        q = self.qq(q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale # L*129
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C) # L*768
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
# zwq 
class Cross_Attention_img_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, text_L = 14):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Cross_Attention_img(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)            

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.linear_map = nn.Linear(text_L,129)

    def forward(self, x, text_feat):

        x = x + self.drop_path((self.linear_map((self.attn(x, text_feat)).transpose(-2,-1))).transpose(-2,-1))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x   