import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp, Block

import math



class edge_CrossAttention(nn.Module):
    def __init__(self, dim, dim_edge, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        # self.wq = nn.Conv1d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.wk = nn.Linear(dim_edge, dim, bias=qkv_bias) # convert edge dim to general dims
        # self.wk = nn.Conv1d(dim_edge, dim, 1, 1, 0, bias=qkv_bias)
        self.wv = nn.Linear(dim_edge, dim, bias=qkv_bias)
        # self.wv = nn.Conv1d(dim_edge, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    @staticmethod
    def _separate_heads(x, num_heads):
        """Separate the input tensor into the specified number of attention heads."""
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    @staticmethod
    def _recombine_heads(x):
        """Recombine the separated attention heads into a single tensor."""
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q, kv):
        q_s = q.shape
        k_s = kv.shape


        # k = k.reshape(k_s[0], k_s[1], -1)
        # v = v.reshape(v_s[0], v_s[1], -1) # b* 96 * 256

        # B, N, C = x.shape
        q = self.wq(q)# B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(kv)# BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(kv)# BNC -> BNH(C/H) -> BHN(C/H)

        q= self._separate_heads(q, self.num_heads)
        k= self._separate_heads(k, self.num_heads)
        v= self._separate_heads(v, self.num_heads)

        _, _, _, c_per_head = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        # print('qv after: {}'.format(attn))
        atn = attn / math.sqrt(c_per_head)
        
        # print('attn is nan : {}'.format(torch.isnan(attn).any()))
        # attn = attn.softmax(dim=-1)
        attn = torch.softmax(atn, dim=-1)
        # print('after softmax: {}'.format(attn))
        # print('attn is nan : {}'.format(torch.isnan(attn).any()))
        attn = self.attn_drop(attn)
        # print('attn is nan : {}'.format(torch.isnan(attn).any()))
        

        x = attn @ v # b, head, seq, edge_seq
    # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self._recombine_heads(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print('q is nan: {}'.format(torch.isnan(q).any()))
        # print('k is nan: {}'.format(torch.isnan(k).any()))
        # print('v is nan: {}'.format(torch.isnan(v).any()))
        # print('x is nan: {}'.format(torch.isnan(x).any()))
        # print('attn is nan: {}'.format(torch.isnan(attn).any()))
        # assert not torch.isnan(x).any()
        # assert not torch.isnan(attn).any()
        return x

    # def forward(self, q, k, v): # Tensor -> Tensor

    #     B, N, C = x.shape
    #     q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
    #     k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
    #     v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

    #     attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)

    #     x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x

################# using CONV
class edge_CrossAttention(nn.Module):
    def __init__(self, dim, dim_edge, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wq = nn.Conv1d(dim, dim, 1, 1, 0, bias=qkv_bias)
        # self.wk = nn.Linear(dim_edge, dim, bias=qkv_bias) # convert edge dim to general dims
        self.wk = nn.Conv1d(dim_edge, dim, 1, 1, 0, bias=qkv_bias)
        # self.wv = nn.Linear(dim_edge, dim, bias=qkv_bias)
        self.wv = nn.Conv1d(dim_edge, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    @staticmethod
    def _separate_heads(x, num_heads):
        """Separate the input tensor into the specified number of attention heads."""
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    @staticmethod
    def _recombine_heads(x):
        """Recombine the separated attention heads into a single tensor."""
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q, kv):
        q_s = q.shape
        k_s = kv.shape


        # k = k.reshape(k_s[0], k_s[1], -1)
        # v = v.reshape(v_s[0], v_s[1], -1) # b* 96 * 256

        # B, N, C = x.shape
        q = self.wq(q.permute(0,2,1))# B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(kv.permute(0,2,1))# BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(kv.permute(0,2,1))# BNC -> BNH(C/H) -> BHN(C/H)


        q= self._separate_heads(q.permute(0,2,1), self.num_heads)
        k= self._separate_heads(k.permute(0,2,1), self.num_heads)
        v= self._separate_heads(v.permute(0,2,1), self.num_heads)

        _, _, _, c_per_head = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        # print('qv after: {}'.format(attn))
        atn = attn / math.sqrt(c_per_head)
        
        # print('attn is nan : {}'.format(torch.isnan(attn).any()))
        # attn = attn.softmax(dim=-1)
        attn = torch.softmax(atn, dim=-1)
        # print('after softmax: {}'.format(attn))
        # print('attn is nan : {}'.format(torch.isnan(attn).any()))
        attn = self.attn_drop(attn)
        # print('attn is nan : {}'.format(torch.isnan(attn).any()))
        

        x = attn @ v # b, head, seq, edge_seq
    # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self._recombine_heads(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print('q is nan: {}'.format(torch.isnan(q).any()))
        # print('k is nan: {}'.format(torch.isnan(k).any()))
        # print('v is nan: {}'.format(torch.isnan(v).any()))
        # print('x is nan: {}'.format(torch.isnan(x).any()))
        # print('attn is nan: {}'.format(torch.isnan(attn).any()))
        # assert not torch.isnan(x).any()
        # assert not torch.isnan(attn).any()
        return x

    # def forward(self, q, k, v): # Tensor -> Tensor

    #     B, N, C = x.shape
    #     q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
    #     k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
    #     v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

    #     attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)

    #     x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x

class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, edge_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = edge_CrossAttention(
            dim, edge_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x