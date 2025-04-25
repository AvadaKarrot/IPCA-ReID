import torch
import torch.nn as nn
import numpy as np
from model.maple.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

_tokenizer = _Tokenizer()
class TextEncoder_Fix(nn.Module): # 固定的Prompt
    def __init__(self, clip_model):
        super(TextEncoder_Fix, self).__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
    def forward(self, text): 
        x = self.token_embedding(text).type(self.dtype) # [bs, n_ctx, dim] ## 文本编码 
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # to shape [seq_len, batch, dim]
        x = self.transformer(x)
        # x = outputs[0]
        x = x.permute(1, 0, 2)  # to shape [batch, seq_len, dim] bs*77*512
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

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

######################  图像文本特征融合######################
from timm.models.vision_transformer import _cfg, Mlp, Block
class Cross_Attention_text_proj(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        ############### zwq 
        self.qq = nn.Linear(dim, dim, bias=qkv_bias) # deal with text feature
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  # Only compute key and value (image feature)

        ##############################
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        ##### debug 
        self.norm = nn.LayerNorm(dim)

        # self.textproj = nn.Linear(768, 512, bias=qkv_bias)


    def forward(self, x, text_features):

        # B, C = text_features.shape
        B, C = x.shape[0], x.shape[1]
        q = x #     img 特征作为query
        # text_features = self.textproj(text_features)
        text_features = text_features.unsqueeze(0).repeat(B,1,1) # 文本特征作为key和value
        kv = self.kv(text_features).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # Unpack key and value
        # Use provided img features as query
        q = q.unsqueeze(1)
        q = self.qq(q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C) # b, seq, 512
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Cross_Attention_text(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        ############### zwq 
        self.qq = nn.Linear(dim, dim, bias=qkv_bias) # deal with text feature
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  # Only compute key and value (image feature)

        ##############################
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        ##### debug 
        self.norm = nn.LayerNorm(dim)
        self.test_proj = nn.Linear(512, 768, bias=qkv_bias)

    def forward(self, x, text_features):

        
        B = x.shape[0]
        q = x #     img 特征作为query
        text_features = text_features.unsqueeze(0).repeat(B,1,1) # 文本特征作为key和valueW
        text_features = self.test_proj(text_features)
        B, _, C = text_features.shape
        kv = self.kv(text_features).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # Unpack key and value
        # Use provided img features as query
        q = self.qq(q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C) # b, seq, 512
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Cross_Attention_text_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Cross_Attention_text(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)            

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, text_feat):
        if len(x.shape) == 2:
            x = x + self.drop_path(self.attn(x, text_feat)).squeeze(1) # 将 (256, 1, 512) 变为 (256, 512)
        else:
            x = x + self.drop_path(self.attn(x, text_feat))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x   
    
class Cross_Attention_text_proj_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Cross_Attention_text_proj(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)            

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, text_feat):
        if len(x.shape) == 2:
            x = x + self.drop_path(self.attn(x, text_feat)).squeeze(1) # 将 (256, 1, 512) 变为 (256, 512)
        else:
            x = x + self.drop_path(self.attn(x, text_feat))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x   

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE
        
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(cfg, self.h_resolution, self.w_resolution,self.vision_stride_size)
        # clip_model.to("cuda")

        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        self.fixcaption = cfg.CLIP.TEXT_FEAT # 是否使用fix文本特征
        # self.text_feat = torch.load(cfg.CLIP.TEXT_FEAT_PATH) if self.fixcaption else None # 文本特征路径
        print(f'Using fixcaption {self.fixcaption}')
        if self.fixcaption:
            ################### 文本图像特征融合——cross attention ###################
            self.cross_attn_text_proj = Cross_Attention_text_proj_Block(
                dim = self.in_planes_proj, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.
            )
            self.cross_attn_text_proj.apply(weights_init_cross_attention)
            ################### 文本图像特征融合——cross attention ###################
            ################### 文本图像特征融合——cross attention ###################
            self.cross_attn_text = Cross_Attention_text_Block(
                dim = self.in_planes, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.
            )
            self.cross_attn_text.apply(weights_init_cross_attention)
            text_encoder = TextEncoder_Fix(clip_model).cuda() # 不进行文本特征提取学习 冻结
            # 冻结 self.text_encoder 的所有参数
            for param in text_encoder.parameters():
                param.requires_grad = False
            fix_captions_tokens = clip.tokenize([f'A photo of {caption}.' for caption in cfg.CLIP.PROMPT]).cuda() # 9*77
            print(f'The fix captions are {cfg.CLIP.PROMPT}')
            with torch.no_grad():
                self.text_feat = text_encoder(fix_captions_tokens)
        print(f'The shape of the text_feat is {self.text_feat.shape}')
        self.gradcam = cfg.TEST.GRADCAM # 与分类任务不同，测试不输出分类器结果，所以GRADCAM时候需要专门输出分类结果


    def forward(self, x, label=None, cam_label= None, view_label=None):
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) #B,512  B,128,512
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) #B,512  B,128,512
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        if self.fixcaption:
            img_feature = image_features + self.cross_attn_text(image_features, self.text_feat) # B*129*768 and 12,768
            img_feature = img_feature[:,0]
            
            img_feature_proj = img_feature_proj + self.cross_attn_text_proj(img_feature_proj, self.text_feat) # B*768            

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj]

        else:
            if self.gradcam:
                # cls_score = self.classifier(feat)
                cls_score_proj = self.classifier_proj(feat_proj)
                return cls_score_proj
            else:
                if self.neck_feat == 'after':
                    # print("Test with feature after BN")
                    return torch.cat([feat, feat_proj], dim=1)
                else:
                    return torch.cat([img_feature, img_feature_proj], dim=1)




    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        param_dict = param_dict['model']
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from model.maple.clip import clip
def load_clip_to_cpu(cfg, h_resolution, w_resolution, vision_stride_size):
    backbone_name = cfg.MODEL.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.MAPLE.N_CTX,
                      "person_reid": cfg.INPUT.PERSON_REID,
                      "person_img_size": cfg.INPUT.SIZE_TRAIN,
                      "vision_stride_size":cfg.MODEL.STRIDE_SIZE}
    model = clip.build_model(state_dict or model.state_dict(), design_details, h_resolution, w_resolution, vision_stride_size)
    print('Loading pretrained Clip model......from {}'.format(model_path))
    return model