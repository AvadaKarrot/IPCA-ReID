import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_base_patch16_224_TransReID_Prompt, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

from model.backbones.hdr.hdrnet_ice import HDRPointwiseNN
from model.backbones.cross_attention_block import edge_CrossAttention
from model.backbones.deformable_attention_2d import DeformableAttention2D
from model.backbones.cross_attention_text import Cross_Attention_text, Cross_Attention_text_Block
from model.backbones.cross_attention_img import Cross_Attention_img, Cross_Attention_img_Block
from model.csa_block.cov_setting import CovMatrix_AIAW
from model.backbones.vit_pytorch import Block
from model.maple.build_transformer_maple import build_transformer_maple
from model.maple.build_transformer_capmaple import build_transformer_caption
import numpy as np

import torch.distributions  as tdist

from decimal import Decimal, localcontext
# from model.maple import clip
from model.maple import clip
from model.prompt.promptlearner import *
from model.prompt.textencoder import *

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x
####################### zwq CLIP
def weights_init_cross_attention(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('LayerNorm') != -1:
        nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(m.weight, 1.0)
#####################################

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


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

# 2017 Oral  Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization 
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
        alpha_range = (0.8,1.2)
        beta_range = (0.8,1.2)

        # 随机生成alpha和beta
        alpha = torch.FloatTensor(1).uniform_(*alpha_range).item()
        beta = torch.FloatTensor(1).uniform_(*beta_range).item()     

        # 计算风格特征的均值和方差 估计
        s_mean = alpha * c_mean  
        s_std = beta * c_std

        normalized = (s_std * (content_feats - c_mean) / c_std) + s_mean

        return normalized

####################################################################

################################## zwq改进的AdaIN：用迪利克雷函数生成变换统计变量的矩阵 idea from Instance-Aware Domain Generalization for Face Anti-Spoofing
class AdaIN_Catogory(AdaIN):
    def __init__(self, base_style_num, dim=768):
        super(AdaIN_Catogory, self).__init__()
        self.dim = dim
        self.base_style_num = base_style_num
        self.concentration_coeff = float(1.0/self.base_style_num)
        self.concentration = torch.tensor([self.concentration_coeff] * self.base_style_num, device='cuda')
        self._dirichlet = tdist.dirichlet.Dirichlet(concentration=self.concentration)

        self.register_buffer("proto_mean", torch.zeros((self.base_style_num, self.dim), requires_grad=False))
        self.register_buffer("proto_std", torch.zeros((self.base_style_num, self.dim), requires_grad=False))

    # def precise_computation(self, base_style_num):
    #     with localcontext() as ctx:
    #         ctx.prec = 9
    #         base_style_num = Decimal(base_style_num)
    #         concentration_coeff = Decimal('1') / base_style_num    
    #         return concentration_coeff

    def __call__( self,
        content_feats: torch.Tensor,
        infer: bool = False,
    ) -> torch.Tensor:
        B, seq, dim = content_feats.size()
        c_mean, c_std = self._compute_mean_std(content_feats, infer=infer) # 129, 1, 768

        combine_weights = self._dirichlet.sample((B,)) # B, L

        # 新风格统计参数                                
        s_mean = combine_weights@(self.proto_mean.data) ###### B, 768
        s_std = combine_weights@(self.proto_std.data) #       B, 768

        normalized = (s_std.unsqueeze(dim=1) * (content_feats - c_mean) / c_std) + s_mean.unsqueeze(dim=1)

        return normalized

# ################################## zwq改进的AdaIN：用迪利克雷函数生成变换统计变量的矩阵 idea from Instance-Aware Domain Generalization for Face Anti-Spoofing
# class AdaIN_dirichlet(AdaIN):
#     def __init__(self, base_style_num, dim=768):
#         super(AdaIN_dirichlet, self).__init__()
#         self.dim = dim
#         self.base_style_num = base_style_num
#         self.concentration_coeff = float(1.0/self.base_style_num)
#         self.concentration = torch.tensor([self.concentration_coeff] * self.base_style_num, device='cuda')
#         self._dirichlet = tdist.dirichlet.Dirichlet(concentration=self.concentration)

#     # def precise_computation(self, base_style_num):
#     #     with localcontext() as ctx:
#     #         ctx.prec = 9
#     #         base_style_num = Decimal(base_style_num)
#     #         concentration_coeff = Decimal('1') / base_style_num    
#     #         return concentration_coeff

#     def __call__( self,
#         content_feats: torch.Tensor,
#         infer: bool = False,
#     ) -> torch.Tensor:
#         B, seq, dim = content_feats.size()
#         c_mean, c_std = self._compute_mean_std(content_feats, infer=infer) # 129, 1, 768

#         combine_weights = self._dirichlet.sample((B,)) # B, L

#         # 新风格统计参数                                
#         s_mean = combine_weights@c_mean.squeeze(dim=1) ###### B, 768
#         s_std = combine_weights@c_std.squeeze(dim=1) #       B, 768

#         normalized = (s_std.unsqueeze(dim=1) * (content_feats - c_mean) / c_std) + s_mean.unsqueeze(dim=1)

#         return normalized


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        hdr_path = cfg.HDRNET.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.text = cfg.CLIP.TEXT_FEAT # True use CLIP
        self.text_feat = torch.load(cfg.CLIP.TEXT_FEAT_PATH) if self.text else None
        self.text_feat_query = torch.load(cfg.CLIP.TEXT_FEAT_QUERY_PATH) if self.text else None
        if self.text and torch.cuda.is_available():
            self.text_feat = self.text_feat.to('cuda')  # Move to default CUDA device immediately
            self.text_feat_query = self.text_feat_query.to('cuda')  # Move to default CUDA device immediately
            L, _ = self.text_feat_query.shape
                        ############## normalize test
            # self.text_feat = self.text_feat / self.text_feat.norm(dim=0, keepdim= True)    # b*129*768
            # self.cross_attn_text = Cross_Attention_text(dim=self.in_planes)
            # self.cross_attn_text.apply(weights_init_cross_attention) ####### zwq 
            self.cross_attn_text = Cross_Attention_text_Block(
                dim = self.in_planes, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.
            )
            self.cross_attn_text.apply(weights_init_cross_attention)
            self.cross_attn_img =   Cross_Attention_img_Block(
                dim = self.in_planes, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,text_L=L
            )
            self.cross_attn_img.apply(weights_init_cross_attention)
        ###############################zwq
        self.hdr = cfg.HDRNET.HDR_NET

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA: # 增加相机信息
            camera_num = camera_num
        else:
            camera_num = 0

        ############################zwq
        if cfg.HDRNET.HDR_NET:
            self.hdrnet = HDRPointwiseNN(cfg)
            # self.feature_fusion = edge_CrossAttention(dim=self.in_planes, dim_edge=(int(cfg.HDRNET.NET_INPUT_SIZE[0]*cfg.HDRNET.NET_INPUT_SIZE[1]/(cfg.HDRNET.SPATIAL_BIN)**2)))
            self.feature_fusion = edge_CrossAttention(dim=self.in_planes, dim_edge=(int(cfg.HDRNET.NET_INPUT_SIZE[0]*cfg.HDRNET.NET_INPUT_SIZE[1]/(2**(int(np.log2(cfg.HDRNET.NET_INPUT_SIZE[0]/cfg.HDRNET.SPATIAL_BIN))))**2)))
            # self.feature_fusion = DeformableAttention2D(dim=self.in_planes, group_queries=False)
        ############################zwq



            self.hdrnet.load_param(hdr_path)
            print('Loading pretrained HDRNet model......from {}'.format(hdr_path))
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, hdr=cfg.HDRNET.HDR_NET, text_feat = cfg.CLIP.TEXT_FEAT, adain_norm=cfg.MODEL.ADAIN, csa=cfg.MODEL.CSA)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        ####### use adain
        self.use_adain_norm = cfg.MODEL.ADAIN
        if cfg.MODEL.ADAIN:
            self.ada_in = AdaIN()            # 定义1x1卷积层将通道数从6减少到3
            # self.conv1x1 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1)

            
        ##### 使用CSA，每个epoch更新数据统计参数，使用Instance Adain校准新统计参数(Dirichlet)\
        self.csa = cfg.MODEL.CSA
        if self.csa:
            self.csada_in= AdaIN_Catogory(base_style_num=cfg.MODEL.CSA_STYLE_NUM, dim=self.in_planes)
        self.white = cfg.MODEL.WHITE
        if self.white:
            self.cov_matrix_layer = CovMatrix_AIAW(dim=self.in_planes, relax_denom=0)
            # self.final_conv = nn.Sequential(
            #     nn.Conv1d(768,768,kernel_size=3, stride = 1, padding=1, bias=False),
            #     nn.BatchNorm1d(768),
            # )
            self.final_block = nn.Sequential(
                Block(dim=768, num_heads=12, mlp_ratio=4, qkv_bias=False,norm_layer=nn.LayerNorm),
                nn.LayerNorm(768),
            )
            # self.final_conv.apply(weights_init_kaiming)
            self.final_block.apply(weights_init_kaiming)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)


    def forward(self, x, label=None, cam_label= None, low = None, full=None, cal_covstat=False):

        global_feat = self.base(x, cam_label=cam_label)
        # feat = global_feat 
        if cal_covstat:
            # adain+white                
            feat_org = global_feat
            # feat_aug = self.ada_in(feat) + feat
            feat_aug = self.ada_in(feat_org) 
            # csa+white
            # feat_org = self.final_block(feat)
            # feat_aug = self.final_block((self.csada_in(feat) + feat))
            feat_org_aug = torch.cat((feat_org, feat_aug), dim=0) # 2B, 129, 768
            B, seq, dim = feat_org_aug.shape
            ################### 计算白化
            eye, reverse_eye = self.cov_matrix_layer.get_eye_matrix()
            f_covariance = torch.bmm(feat_org_aug.transpose(1,2), feat_org_aug).div(seq-1) + (eye * 1e-7)
            off_diag_elements = f_covariance * reverse_eye # 512, 768,768
            self.cov_matrix_layer.set_pair_covariance(torch.var(off_diag_elements, dim=0))
            return 0

            ##################### debug before adain
        # if self.text:
        #     # normalize features
        #     # feat = feat / feat.norm(dim=1, keepdim= True)    # b*129*768
        #     cls_feat = global_feat[:,0,:].unsqueeze(1)
        #     fuse_feat = cls_feat +  self.cross_attn_text(cls_feat, self.text_feat)# B*129*768
        #     global_feat[:, 0, :] = fuse_feat.squeeze(1)
        if self.text:
            # normalize features
            # feat = feat / feat.norm(dim=1, keepdim= True)    # b*129*768

            # feat = feat + self.cros_attn_text( feat, self.text_feat)+\
            #         self.cross_attn_img(feat, self.text_feat_query)# B*129*768s
            # self.text_feat_query = self.text_feat_query.to(global_feat.device)
            global_feat = global_feat +  self.cross_attn_text(global_feat, self.text_feat_query)# B*129*768
            
        ######## use class token as class feature
        ###########################zwq 
        if self.use_adain_norm and self.training:
            feat_aug = self.ada_in(global_feat) # b*129*768
            if self.white and self.training:  
                csa_output = {}          
                eye, mask_matrix, margin, num_remove_cov = self.cov_matrix_layer.get_mask_matrix()
                csa_output['eye'] = eye
                csa_output['mask_matrix'] = mask_matrix
                csa_output['margin'] = margin
                csa_output['num_remove_cov'] = num_remove_cov
                csa_output["feat_aug"] = feat_aug
                csa_output["feat_org"] = global_feat
            global_feat = feat_aug
        ############### CLIP test + img        
        if self.csa and self.training:
            csa_output = {}
            feat_org = self.final_block(feat_aug)
            # feat = feat + self.csada_in(feat) # use 是否残差
            feat_aug = self.final_block(self.csada_in(feat_aug) + feat_aug)
            # feat = feat_aug + feat_org
            if self.white and self.training:      # 在使用sca的情况下，用白噪声      
                eye, mask_matrix, margin, num_remove_cov = self.cov_matrix_layer.get_mask_matrix()
                csa_output['eye'] = eye
                csa_output['mask_matrix'] = mask_matrix
                csa_output['margin'] = margin
                csa_output['num_remove_cov'] = num_remove_cov
                csa_output["feat_aug"] = feat_aug
                csa_output["feat_org"] = feat_org
        # ############### CLIP test + img 
        # if self.text:
        #     # normalize features
        #     feat = feat / feat.norm(dim=1, keepdim= True)    # b*129*768

        #     # feat = feat + self.cross_attn_text( feat, self.text_feat)+\
        #     #         self.cross_attn_img(feat, self.text_feat_query)# B*129*768
        #     feat = feat +  self.cross_attn_text(feat, self.text_feat_query)# B*129*768
        ######## use class token as class feature
        global_feat = global_feat[:, 0]
        ####################  # ############### CLIP test + img 466放在class token和风格做cross attention
        # if self.text:
        #     # normalize features
        #     # feat = feat / feat.norm(dim=1, keepdim= True)    # b*129*768

        #     # feat = feat + self.cros_attn_text( feat, self.text_feat)+\
        #     #         self.cross_attn_img(feat, self.text_feat_query)# B*129*768s
        #     global_feat = global_feat.unsqueeze(1) +  self.cross_attn_text(global_feat.unsqueeze(1), self.text_feat_query)# B*129*768
        #     global_feat = global_feat[:,0]
        ######## use class token as class feature
        ###### HDRNet is edge feature 
        if self.hdr:
            img_reconstruct, edge_feat = self.hdrnet(low, full)
            fusion_feat = global_feat + self.feature_fusion(global_feat, edge_feat)
            global_feat = fusion_feat[:, 0]    
        ###########################
        feat = self.bottleneck(global_feat) # b*768


        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            if self.hdr:
                return cls_score, global_feat, img_reconstruct
                
#########################################################################################            
            if self.white:
                return cls_score, global_feat, csa_output
#########################################################################################           
            return cls_score, global_feat # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        ######## debug #################
        param_dict = param_dict['model']
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))         

############################# origin load param is under
    # def load_param(self, trained_path):
    #     param_dict = torch.load(trained_path)
    #     for i in param_dict:
    #         self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
    #     print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

# class build_transformer_prompt(build_transformer):

#     def __init__(self, num_classes, camera_num, view_num, cfg, factory):
#         super(build_transformer_prompt, self).__init__(self, num_classes, camera_num, view_num, cfg, factory)
#         ################## new parameter
#         self.aug_prob = cfg.MODEL.AUG_PROB
#         self.aug_layer = cfg.MODEL.AUG_LAYER

#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](aug_prob=self.aug_prob, aug_layer=self.aug_layer, img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
#                                                         stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
#                                                         drop_rate= cfg.MODEL.DROP_OUT,
#                                                         attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, hdr=cfg.HDRNET.HDR_NET, text_feat = cfg.CLIP.TEXT_FEAT, adain_norm=cfg.MODEL.ADAIN)        

#         print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

#         domain_text = {'source': "A photo of a human taken during the day"}
#         with open('prompts.txt','r') as f:
#             for ind,l in enumerate(f):
#                 domain_text.update({str(ind):l.strip()})
#         self.offsets = nn.Parameter(torch.zeros(len(domain_text)-1,129,768)) #skip day
#         import clip
#         self.text_model, _=  clip.load("/home/test/LIVA/ZWQ/pretrained/ViT-L-14.pt.1", device='cuda')
#         self.domain_tk = dict([(k,clip.tokenize(t)) for k,t in domain_text.items()])  # 字典，存文字向量，总长77
    
#     def forward(self, x, label=None, low=None, full=None):

#         B = x.shape[0]
        
#         features = self.base(x, before=True, after=False) # before augment

#         if np.random.rand(1)>self.aug_prob:
#             oids = np.random.choice(np.arange(len(self.offsets)),B)
#             change = torch.cat([self.offsets[oid:oid+1].cuda().mean(dim=(2,3),keepdims=True) for oid in oids ],0)
#             features = features + change
        
#         global_feat = self.base(features, before=False, after=True)
#             ###########################zwq
#         if self.use_adain_norm and self.training:
#             feat = self.ada_in(global_feat) # b*129*768
#         ############### CLIP test + img 
#         if self.text:
#             # normalize features
#             # feat = feat / feat.norm(dim=1, keepdim= True)    # b*129*768
#             feat = feat + self.cross_attn_text( feat, self.text_feat) # B*129*768

#         ######## use class token as class feature
#         global_feat = feat[:, 0]    
#         ###### HDRNet is edge feature 
#         if self.hdr:
#             img_reconstruct, edge_feat = self.hdrnet(low, full)
#             fusion_feat = global_feat + self.feature_fusion(global_feat, edge_feat)
#             global_feat = fusion_feat[:, 0]    
#         ###########################
#         feat = self.bottleneck(global_feat) # b*768

#         if self.training:
#             if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
#                 cls_score = self.classifier(feat, label)
#             else:
#                 cls_score = self.classifier(feat)
#             if self.hdr:
#                 return cls_score, global_feat, img_reconstruct
#             return cls_score, global_feat # global feature for triplet loss
#         else:
#             if self.neck_feat == 'after':
#                 # print("Test with feature after BN")
#                 return feat
#             else:
#                 # print("Test with feature before BN")
#                 return global_feat
#     def opt_offsets(self, x):
#         with torch.no_grad():
#             features = self.base(x, before=True, after=False)
#         for i, val in enumerate(self.domain_tk.items()):
#             name, dtk = val
#             if name == 'day':
#                 continue
#             with torch.no_grad(): 

#                 wo_aug_im_embed = features #features是需要aug层之后返回的浅层特征
#                 wo_aug_im_embed  = wo_aug_im_embed/wo_aug_im_embed.norm(dim=-1,keepdim=True)

#                 source_text_embed = self.text_model.encode_text(self.domain_tk['source'].cuda()) # source 
#                 source_text_embed = source_text_embed/source_text_embed.norm(dim=-1, keepdim=True)

#                 target_text_embed = self.text_model.encode_text(dtk.cuda())
#                 target_text_embed = target_text_embed/target_text_embed.norm(dim=-1, keepdim=True)

#                 text_off = target_text_embed - source_text_embed
#                 text_off = text_off/text_off.norm(dim=-1,keepdim=True)

#                 wo_aug_im_tsl = wo_aug_im_embed + text_off
#                 wo_aug_im_tsl = wo_aug_im_tsl/wo_aug_im_tsl.norm(dim=-1, keepdim=True)
#                 wo_aug_im_tsl = wo_aug_im_tsl.unsqueeze(1).permute(0,2,1)
#             aug_feat = features.detach() + self.offsets[i-1:i]
#             x = self.base(aug_feat, before=False, after=True)

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
'''
_tokenizer = _Tokenizer()
class PromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super(PromptLearner, self).__init__()
        # classnames = ['the person\'s arms', 'the person\'s head', 'the person\'s hands',  'the person\'s feet', 'the person\'s legs', 'the person\'s clothes', 'the person\'s body']
        classnames = [
            'a human during the day', 
            'a human on the cloudy morning',
            'a human on the sunny morning',
            'a human on the sunny afternoon',
            'a human  indoor',
            'a human outdoor',
            'a human standing still',
            'a human walking',
            'a human cycling',
            'a human carrying a bag',
            'a human with a backpack',
            'a human occluded',
            'human blurred',
            'a human clear',
        ]
        n_cls = len(classnames)
        n_ctx = cfg.COOP.N_CTX
        ctx_init = cfg.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:

            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")        

        self.ctx = nn.Parameter(ctx_vectors) # prompt作为可学习参数

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) ## 类别token化
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS特征
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS特征
        self.n_cls = n_cls
        self.n_ctx = n_ctx        
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor ## 类别token化
        self.name_lens = name_lens     
        self.class_token_position = cfg.COOP.CLASS_TOKEN_POSITION 

    def forward(self):
        ctx = self.ctx ### 可学习参数 "a photo of" # 3 * 768
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # 14 * 3 * 768

        prefix = self.token_prefix # 14 1 768
        suffix = self.token_suffix #  14, 73,768
        if self.class_token_position == 'end':
            prompts = torch.cat(
                [
                    prefix, # n_cls, 1, 768
                    ctx, # n_cls, n_ctx, 768
                    suffix, # n_cls, *, 768 # class token + EOS
                ],
                dim=1
            )
        elif self.class_token_position == 'middle':
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)            
        else:
            raise ValueError
        return prompts # 14 77 768

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        # classnames = ['the person\'s arms', 'the person\'s head', 'the person\'s hands',  'the person\'s feet', 'the person\'s legs', 'the person\'s clothes', 'the person\'s body']
        classnames = [
            'a human during the day', 
            'a human on the cloudy morning',
            'a human on the sunny morning',
            'a human on the sunny afternoon',
            'a human  indoor',
            'a human outdoor',
            'a human standing still',
            'a human walking',
            'a human cycling',
            'a human carrying a bag',
            'a human with a backpack',
            'a human occluded',
            'a human blurred',
            'a human clear',
        ]
        n_cls = len(classnames)
        n_ctx = cfg.MAPLE.N_CTX
        ctx_init = cfg.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super(TextEncoder, self).__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # to shape [seq_len, batch, dim]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # to shape [batch, seq_len, dim]
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class TextEncoder_MaPLE(nn.Module):
    def __init__(self, clip_model):
        super(TextEncoder, self).__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
    def forward(self, prompts, tokenized_prompts,compound_prompts_deeper_text): 
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # to shape [seq_len, batch, dim]
        ## ccompound_prompts_deeper_text
        combined = [x, compound_prompts_deeper_text, 0]
        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute(1, 0, 2)  # to shape [batch, seq_len, dim]
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
'''

def load_clip_cpu(): ############  本地加载CLIP模型 包含文本+图像特征提取
    model_path = "/home/test/LIVA/ZWQ/pretrained/ViT-L-14.pt.1"
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model 

class build_transformer_prompt(nn.Module):
    def __init__(self, num_classes, camera_num, cfg, factory):
        super(build_transformer_prompt, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        hdr_path = cfg.HDRNET.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        self.text = cfg.CLIP.TEXT_FEAT # True use CLIP
        if self.text and torch.cuda.is_available():
            self.text_feat = torch.load(cfg.CLIP.TEXT_FEAT_PATH) if self.text else None
            self.text_feat_query = torch.load(cfg.CLIP.TEXT_FEAT_QUERY_PATH) if self.text else None
            self.text_feat = self.text_feat.to('cuda')  # Move to default CUDA device immediately
            self.text_feat_query = self.text_feat_query.to('cuda')  # Move to default CUDA device immediately
            L, _ = self.text_feat_query.shape
                        ############## normalize test
            # self.text_feat = self.text_feat / self.text_feat.norm(dim=0, keepdim= True)    # b*129*768
            # self.cross_attn_text = Cross_Attention_text(dim=self.in_planes)
            # self.cross_attn_text.apply(weights_init_cross_attention) ####### zwq 
            self.cross_attn_text = Cross_Attention_text_Block(
                dim = self.in_planes, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.
            )
            self.cross_attn_text.apply(weights_init_cross_attention)
            # self.cross_attn_img =   Cross_Attention_img_Block(
            #     dim = self.in_planes, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,text_L=14
            # )
            # self.cross_attn_img.apply(weights_init_cross_attention)
        ###############################zwq

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, hdr=cfg.HDRNET.HDR_NET, text_feat = cfg.CLIP.TEXT_FEAT, adain_norm=cfg.MODEL.ADAIN, csa=cfg.MODEL.CSA)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        ####### use adain
        self.use_adain_norm = cfg.MODEL.ADAIN
        if cfg.MODEL.ADAIN:
            self.ada_in = AdaIN()            # 定义1x1卷积层将通道数从6减少到3
            # self.conv1x1 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1)

        ##### 使用CSA，每个epoch更新数据统计参数，使用Instance Adain校准新统计参数(Dirichlet)\
        self.csa = cfg.MODEL.CSA
        if self.csa:
            self.csada_in= AdaIN_Catogory(base_style_num=cfg.MODEL.CSA_STYLE_NUM, dim=self.in_planes)
        self.white = cfg.MODEL.WHITE
        if self.white:
            self.cov_matrix_layer = CovMatrix_AIAW(dim=self.in_planes, relax_denom=0)
            # self.final_conv = nn.Sequential(
            #     nn.Conv1d(768,768,kernel_size=3, stride = 1, padding=1, bias=False),
            #     nn.BatchNorm1d(768),
            # )
            self.final_block = nn.Sequential(
                Block(dim=768, num_heads=12, mlp_ratio=4, qkv_bias=False,norm_layer=nn.LayerNorm),
                nn.LayerNorm(768),
            )
            # self.final_conv.apply(weights_init_kaiming)
            self.final_block.apply(weights_init_kaiming)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        ########### prompt learning
        if cfg.MODEL.PROMPT:
            print(f"Loading CLIP (backbone: {'ViT-L-14'})")
            clip_model = load_clip_cpu()
            clip_model.float()
            print('Building Prompt Learning CLIP')
            self.prompt_learner = PromptLearner(cfg, clip_model) # 假设给了一些固定的图像描述(例如 人的头，手...)
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts  ## 类别token化 # 14*77
            self.text_encoder = TextEncoder(clip_model) # 不进行文本特征提取学习 冻结
            # 冻结 self.text_encoder 的所有参数
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.dtype = clip_model.dtype

    def forward(self, x, label=None, cam_label= None, low = None, full=None, cal_covstat=False):

        global_feat = self.base(x, cam_label=cam_label)
    
        #### 文本特征(可学习prompt+类别token化)
        prompts = self.prompt_learner() #### 类别token化 
        tokenized_prompts = self.tokenized_prompts 
        text_features = self.text_encoder(prompts, tokenized_prompts)
        #  要不要norm呢?
        # text_features = text_features / text_features.norm(dim=1, keepdim= True)    # b*129*768

        if self.text:
        # if self.text and self.training:
            global_feat = global_feat +  self.cross_attn_text(global_feat, text_features)# B*129*768 
            # global_feat = global_feat[:, 0] ############ Warning
        ######## use class token as class feature
        ###########################zwq 
        if self.use_adain_norm and self.training:
            feat = self.ada_in(global_feat) # b*129*768
            if self.white and self.training:  
                csa_output = {}           
                eye, mask_matrix, margin, num_remove_cov = self.cov_matrix_layer.get_mask_matrix()
                csa_output['eye'] = eye
                csa_output['mask_matrix'] = mask_matrix
                csa_output['margin'] = margin
                csa_output['num_remove_cov'] = num_remove_cov
                csa_output["feat_aug"] = feat 
                csa_output["feat_org"] = global_feat

        ######## use class token as class feature
            global_feat = feat[:, 0]
        else:
            global_feat = global_feat[:, 0] 
        ####################  # ############### CLIP test + img 466放在class token和风格做cross attention
        # if self.text:
        #     # normalize features
        #     # feat = feat / feat.norm(dim=1, keepdim= True)    # b*129*768

        #     # feat = feat + self.cros_attn_text( feat, self.text_feat)+\
        #     #         self.cross_attn_img(feat, self.text_feat_query)# B*129*768s
        #     global_feat = global_feat.unsqueeze(1) +  self.cross_attn_text(global_feat.unsqueeze(1), self.text_feat_query)# B*129*768
        #     global_feat = global_feat[:,0]
        ######## use class token as class feature
        feat = self.bottleneck(global_feat) # b*768

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
      
#########################################################################################            
            if self.white:
                return cls_score, global_feat, csa_output
#########################################################################################           
            return cls_score, global_feat # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        ######## debug #################
        param_dict = param_dict['model']
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))         

############################# origin load param is under
    # def load_param(self, trained_path):
    #     param_dict = torch.load(trained_path)
    #     for i in param_dict:
    #         self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
    #     print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
'''
class build_transformer_caption(nn.Module):
    def __init__(self, num_classes, camera_num, cfg, factory):
        super(build_transformer_caption, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        hdr_path = cfg.HDRNET.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, hdr=cfg.HDRNET.HDR_NET, text_feat = cfg.CLIP.TEXT_FEAT, adain_norm=cfg.MODEL.ADAIN, csa=cfg.MODEL.CSA)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        ####### use adain
        self.use_adain_norm = cfg.MODEL.ADAIN
        if cfg.MODEL.ADAIN:
            self.ada_in = AdaIN()            # 定义1x1卷积层将通道数从6减少到3
            # self.conv1x1 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1)

        ##### 使用CSA，每个epoch更新数据统计参数，使用Instance Adain校准新统计参数(Dirichlet)\
        self.csa = cfg.MODEL.CSA
        if self.csa:
            self.csada_in= AdaIN_Catogory(base_style_num=cfg.MODEL.CSA_STYLE_NUM, dim=self.in_planes)
        self.white = cfg.MODEL.WHITE
        if self.white:
            self.cov_matrix_layer = CovMatrix_AIAW(dim=self.in_planes, relax_denom=0)
            # self.final_conv = nn.Sequential(
            #     nn.Conv1d(768,768,kernel_size=3, stride = 1, padding=1, bias=False),
            #     nn.BatchNorm1d(768),
            # )
            self.final_block = nn.Sequential(
                Block(dim=768, num_heads=12, mlp_ratio=4, qkv_bias=False,norm_layer=nn.LayerNorm),
                nn.LayerNorm(768),
            )
            # self.final_conv.apply(weights_init_kaiming)
            self.final_block.apply(weights_init_kaiming)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        ########### caption learning
        self.text = cfg.CLIP.TEXT_FEAT # Cross Attention 
        if self.text:
            # self.text_feat = torch.load(cfg.CLIP.TEXT_FEAT_PATH) if self.text else None
            # self.text_feat_query = torch.load(cfg.CLIP.TEXT_FEAT_QUERY_PATH) if self.text else None
            # self.text_feat = self.text_feat.to('cuda')  # Move to default CUDA device immediately
            # self.text_feat_query = self.text_feat_query.to('cuda')  # Move to default CUDA device immediately
            # self.text_feat = self.text_feat / self.text_feat.norm(dim=0, keepdim= True)    # b*129*768
            self.cross_attn_text = Cross_Attention_text_Block(
                dim = self.in_planes, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.
            )
            self.cross_attn_text.apply(weights_init_cross_attention)
        ###############################zwq
        self.coop = cfg.COOP.COOP_PROMPT
        self.maple = cfg.MAPLE.MAPLE_PROMPT
        if cfg.COOP.COOP_PROMPT: # 可学习Prompt：ctx
            print(f"Loading CLIP (backbone: {'ViT-L-14'})")
            clip_model = load_clip_cpu()
            clip_model.float()
            print('Building Prompt Learning CLIP')
            self.prompt_learner = CapPromptLearner(cfg, clip_model) # 假设给了一些固定的图像描述(例如 人的头，手...)
            # self.tokenized_prompts = self.prompt_learner.tokenized_prompts  ## 类别token化 # 14*77
            self.text_encoder = TextEncoder(clip_model) # 不进行文本特征提取学习 冻结
            # 冻结 self.text_encoder 的所有参数
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.dtype = clip_model.dtype
        elif cfg.MAPLE.MAPLE_PROMPT: #多模态可学习Prompt: 文本-图像
            print(f"Loading CLIP (backbone: {'ViT-L-14'})")
            clip_model = load_clip_cpu()
            clip_model.float()
            print('Building Prompt Learning CLIP')
            self.prompt_learner = CapMultiModalPromptLearner(cfg, clip_model) # 假设给了一些固定的图像描述(例如 人的头，手...)
            # self.tokenized_prompts = self.prompt_learner.tokenized_prompts  ## 类别token化 # 14*77
            self.text_encoder = TextEncoder_MaPLE(clip_model) # 不进行文本特征提取学习 冻结
            # 冻结 self.text_encoder 的所有参数
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.dtype = clip_model.dtype            
        else:  # 固定的Prompt
            clip_model = load_clip_cpu()
            clip_model.float()
            print('Building Caption-drve CLIP')
            self.text_encoder = TextEncoder_Fix(clip_model) # 不进行文本特征提取学习 冻结
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.dtype = clip_model.dtype                    



    def forward(self, x, captions=None,label=None, cam_label= None, low = None, full=None, cal_covstat=False):

        #  captions: batchbatch中每个图像特有的captions

        if self.coop:
            global_feat = self.base(x, cam_label=cam_label)
        
            #### 文本特征(可学习prompt+类别token化)
            prompts, tokenized_prompts = self.prompt_learner(captions) #### 类别token化   ## 类别token化 # 14*77
            text_features = self.text_encoder(prompts, tokenized_prompts) 
        elif self.maple:
            #### 文本特征(可学习prompt+类别token化)
            prompts, tokenized_prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(captions) #### 类别token化 
            text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
            
            global_feat = self.base(x, cam_label=cam_label, 
                                    shared_ctx=shared_ctx, compound_deeper_prompts=deep_compound_prompts_vision)
        else:
            captions_tokens = clip.tokenize('a photo of '+ caption for caption in captions).cuda()
            text_features = self.text_encoder(captions_tokens)
            global_feat = self.base(x, cam_label=cam_label)
        #  要不要norm呢?
        # text_features = text_features / text_features.norm(dim=1, keepdim= True)    # b*129*768
        if True:
            img_features = global_feat.norm(dim=-1, keepdim=True)
            text_features = text_features.norm(dim=-1, keepdim=True)
            similarity = text_features.cpu().numpy() @ img_features.cpu().numpy().T
            
        if self.text: # self.text图像文本特征融合
        # if self.text and self.training:
            global_feat = global_feat +  self.cross_attn_text(global_feat, text_features)# B*129*768 
            # global_feat = global_feat[:, 0] ############ Warning
        ######## use class token as class feature
        ###########################zwq 
        if self.use_adain_norm and self.training:
            feat = self.ada_in(global_feat) # b*129*768
            if self.white and self.training:  
                csa_output = {}           
                eye, mask_matrix, margin, num_remove_cov = self.cov_matrix_layer.get_mask_matrix()
                csa_output['eye'] = eye
                csa_output['mask_matrix'] = mask_matrix
                csa_output['margin'] = margin
                csa_output['num_remove_cov'] = num_remove_cov
                csa_output["feat_aug"] = feat 
                csa_output["feat_org"] = global_feat

        ######## use class token as class feature
            global_feat = feat[:, 0]
        else:
            global_feat = global_feat[:, 0] 
        ####################  # ############### CLIP test + img 466放在class token和风格做cross attention
        # if self.text:
        #     # normalize features
        #     # feat = feat / feat.norm(dim=1, keepdim= True)    # b*129*768

        #     # feat = feat + self.cros_attn_text( feat, self.text_feat)+\
        #     #         self.cross_attn_img(feat, self.text_feat_query)# B*129*768s
        #     global_feat = global_feat.unsqueeze(1) +  self.cross_attn_text(global_feat.unsqueeze(1), self.text_feat_query)# B*129*768
        #     global_feat = global_feat[:,0]
        ######## use class token as class feature
        feat = self.bottleneck(global_feat) # b*768

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
      
#########################################################################################            
            if self.white:
                return cls_score, global_feat, csa_output
#########################################################################################           
            return cls_score, global_feat # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        ######## debug #################
        param_dict = param_dict['model']
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))         

############################# origin load param is under
    # def load_param(self, trained_path):
    #     param_dict = torch.load(trained_path)
    #     for i in param_dict:
    #         self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
    #     print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
'''

class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0


        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.text = cfg.CLIP.TEXT_FEAT
        if self.text and torch.cuda.is_available():
            self.text_feat = torch.load(cfg.CLIP.TEXT_FEAT_PATH) if self.text else None
            self.text_feat_query = torch.load(cfg.CLIP.TEXT_FEAT_QUERY_PATH) if self.text else None
            self.text_feat = self.text_feat.to('cuda')  # Move to default CUDA device immediately
            self.text_feat_query = self.text_feat_query.to('cuda')  # Move to default CUDA device immediately
            L, _ = self.text_feat_query.shape
                        ############## normalize test
            # self.text_feat = self.text_feat / self.text_feat.norm(dim=0, keepdim= True)    # b*129*768
            # self.cross_attn_text = Cross_Attention_text(dim=self.in_planes)
            # self.cross_attn_text.apply(weights_init_cross_attention) ####### zwq 
            self.cross_attn_text = Cross_Attention_text_Block(
                dim = self.in_planes, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.
            )
            self.cross_attn_text.apply(weights_init_cross_attention)
            self.cross_attn_img =   Cross_Attention_img_Block(
                dim = self.in_planes, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,text_L=14
            )
            self.cross_attn_img.apply(weights_init_cross_attention)
        ###############################zwq
        self.use_adain_norm = cfg.MODEL.ADAIN
        if cfg.MODEL.ADAIN:
            self.ada_in = AdaIN()
        ########### prompt learning
        if cfg.MODEL.PROMPT:
            print(f"Loading CLIP (backbone: {'ViT-L-14'})")
            clip_model = load_clip_cpu()
            clip_model.float()
            print('Building Prompt Learning CLIP')
            self.prompt_learner = PromptLearner(cfg, clip_model) # 假设给了一些固定的图像描述(例如 人的头，手...)
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts  ## 类别token化 # 14*77
            self.text_encoder = TextEncoder(clip_model) # 不进行文本特征提取学习 冻结
            # 冻结 self.text_encoder 的所有参数
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.dtype = clip_model.dtype               
        ###### class
        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label= None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        # Adain
        if self.use_adain_norm and self.training:
            ada_feat = self.ada_in(b1_feat)
            global_feat = ada_feat[:, 0]
        else: # orig
            global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_base_patch16_224_TransReID_Prompt': vit_base_patch16_224_TransReID_Prompt,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model_debug(cfg, num_class, camera_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        elif cfg.MODEL.CAPTION:
            model = build_transformer_caption(num_class, camera_num, cfg, __factory_T_type)
            print('===========building transformer with caption ===========')
            
        elif cfg.MODEL.PROMPT:
            if cfg.MAPLE.MAPLE_PROMPT:
                model = build_transformer_maple(num_class, camera_num, cfg, __factory_T_type)  
            else: 
                model = build_transformer_prompt(num_class, camera_num, cfg, __factory_T_type) # 普通CoOP      
        else:
            model = build_transformer(num_class, camera_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
