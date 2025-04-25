import torch
import os
import sys
import copy
import torch.nn as nn
import numpy as np
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from model.maple.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from model.prompt.promptlearner import *
from model.prompt.textencoder import *
from model.backbones.adain import *
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将自定义的clip文件夹路径添加到sys.path的前面
sys.path.insert(0, os.path.join(current_dir, 'clip'))

_tokenizer = _Tokenizer()

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


    def forward(self, x, text_features):

        B, C = text_features.shape
        B = x.shape[0]
        q = x #     img 特征作为query
        text_features = text_features.unsqueeze(0).permute(1,0,2) # 文本特征作为key和value
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
        text_features = text_features.unsqueeze(0).permute(1,0,2) # 文本特征作为key和value
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

####################################################################
from model.maple.clip import clip
from model.maple.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def load_clip_to_cpu_maple(cfg, h_resolution, w_resolution, vision_stride_size):
    # model_path = "/home/test/LIVA/ZWQ/pretrained/ViT-L-14.pt.1"
    backbone_name = cfg.MODEL.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.MAPLE.N_CTX,
                      "person_reid": cfg.INPUT.PERSON_REID,
                      "person_img_size": cfg.INPUT.SIZE_TRAIN,
                      "vision_stride_size":cfg.MODEL.STRIDE_SIZE}
    model = clip.build_model(state_dict or model.state_dict(), design_details,h_resolution, w_resolution, vision_stride_size)
    print('Loading pretrained Clip model......from {}'.format(model_path))
    return model

# 其他都可以用load_clip_to_cpu: capion; caption-coop; caption-cocoop
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

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=512,
        dropout=0.4
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

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
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE_TRAIN[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            ######### zwq
            len_ctx = len(ctx_init.split(" "))
            ############################# 
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype) # 1, 77, 768
            # ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            ##  zwq
            ctx_vectors = embedding[0, 1: 1 + len_ctx, :]
            ############33333333333333333333333333333333333333333333333333333
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
        # self.proj = nn.Identity()
        # self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn) # 14, 77
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # torch.Size([14, 77, 768])

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
        ctx = self.ctx # 3,768

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # 14,3,768

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix) # 14,77,768 # text whole embedding

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index])) # 2,768 -> 2,768 # 图像特征的深层提示 2，768
        # Now the other way around
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        # return prompts, self.ctx, self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required
## prompts:14,77,768
## self.proj(self.ctx):2,768
## self.compound_prompts_text 8 list 2,768
## visual_deep_prompts 8 list 2,768
class TextEncoder_MaPLE(nn.Module):
    def __init__(self, clip_model):
        super(TextEncoder_MaPLE, self).__init__()
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

class build_transformer_caption(nn.Module):
    def __init__(self, num_classes, camera_num, cfg):
        super(build_transformer_caption, self).__init__()
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
        self.sie_coe = cfg.MODEL.SIE_COE

        print('using Transformer_type: {} as a backbone'.format(self.model_name))
        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes)) # 6,768(market),15,769(msmt17)
            trunc_normal_(self.cv_embed, std=.02)
            print('train camera number is : {}'.format(camera_num))
        else:
            camera_num = 0

        ########## use adain 使用风格迁移增强local feature ##########
        self.use_adain_norm = cfg.MODEL.ADAIN
        if cfg.MODEL.ADAIN:
            self.ada_in = AdaIN()         

        self.gap = nn.AdaptiveAvgPool2d(1)

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
            self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
            self.classifier_proj.apply(weights_init_classifier)
            
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        ###################### caption learning ######################
        # h_resolution, w_resolution, vision_stride_size为了修改原模型参数的positional embedding
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        self.coop = cfg.COOP.COOP_PROMPT #可学习token
        self.cocoop = cfg.COCOOP.COCOOP_PROMPT #泛化的可学习token
        self.maple = cfg.MAPLE.MAPLE_PROMPT #耦合可学习token
        if self.maple: # maple需要更改detailed_design
            clip_model = load_clip_to_cpu_maple(cfg, self.h_resolution, self.w_resolution, self.vision_stride_size)
        else:
            clip_model = load_clip_to_cpu(cfg, self.h_resolution, self.w_resolution, self.vision_stride_size)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.caption = cfg.MODEL.CAPTION # 有标签进行文本特征学习
        if self.caption:
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
            ################### 文本图像特征融合——cross attention ###################            
        if cfg.COOP.COOP_PROMPT: # 可学习Prompt：ctx
            print('Building Cap-coop-drive CLIP')
            print('Building Prompt Learning CLIP')
            self.prompt_learner = CapPromptLearner(cfg, clip_model) # 假设给了一些固定的图像描述(例如 人的头，手...)
            # self.tokenized_prompts = self.prompt_learner.tokenized_prompts  ## 类别token化 # 14*77
            self.text_encoder = TextEncoder(clip_model) # 不进行文本特征提取学习 冻结
            # 冻结 self.text_encoder 的所有参数
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.dtype = clip_model.dtype
        elif cfg.COCOOP.COCOOP_PROMPT:  # 泛化的可学习Prompt
            print('Building Cap-cocoop-drive CLIP')
            print('Builing Prompt Learning CLIP')
            self.prompt_learner = CapConditionalPromptLearner(cfg, clip_model)
            self.text_encoder = TextEncoder(clip_model) # 不进行文本特征学习 冻结
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.dtype = clip_model.dtype            
        
        elif cfg.MAPLE.MAPLE_PROMPT: #多模态可学习Prompt: 文本-图像
            print(f"Loading Cap-MaPLe-drive CLIP")
            print('Building Prompt Learning CLIP')
            self.prompt_learner = CapMultiModalPromptLearner(cfg, clip_model) 
            # self.tokenized_prompts = self.prompt_learner.tokenized_prompts  ## 类别token化 # 14*77
            self.text_encoder = TextEncoder_MaPLE(clip_model) # 不进行文本特征提取学习 冻结
            # 冻结 self.text_encoder 的所有参数
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.dtype = clip_model.dtype            
        else:  ###################### 固定的Prompt ######################
            print('Building Caption-drive CLIP')
            self.text_encoder = TextEncoder_Fix(clip_model) # 不进行文本特征提取学习 冻结
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.dtype = clip_model.dtype                    
        # self.image_projection = ProjectionHead(embedding_dim=512)
        self.text_proj = cfg.MODEL.TEXT_PROJ # 
        if cfg.MODEL.TEXT_PROJ:
            self.text_projectoin = ProjectionHead(embedding_dim=512)

        self.gradcam = cfg.TEST.GRADCAM # 与分类任务不同，测试不输出分类器结果，所以GRADCAM时候需要专门输出分类结果

    def caption_features(self, captions, img_features=None):
        if self.coop:
            ###################### prompts: ctx更新的captions特征;captions_token######################
            prompts, captions_tokens = self.prompt_learner(captions) #### 类别token化   ## 类别token化 # 14*77
            text_features = self.text_encoder(prompts, captions_tokens) # 128*77 -> 128*768
            return text_features
        elif self.cocoop:
            ###################### prompts: ctx更新的captions特征;captions_token######################
            prompts, captions_tokens = self.prompt_learner(captions, img_features) #### 类别token化   ## 类别token化 # 14*77
            text_features = self.text_encoder(prompts, captions_tokens) # 128*77 -> 128*768
            return text_features
        elif self.maple:
            prompts, tokenized_prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(captions) #### 类别token化 
            text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
            return text_features, shared_ctx, deep_compound_prompts_vision
        else:
            captions_tokens = clip.tokenize([f'A photo of {caption}.' for caption in captions]).cuda()
            text_features = self.text_encoder(captions_tokens) # 128*77 -> 128*768
            text_features = self.text_projectoin(text_features)

        return text_features

    def caption_fusion(self, img, text): # img: b*129*768, text:b*768
        fused_features = []
        for i in range(len(img)):
            img_feature = img[i:i+1] # 1*129*768
            text_features = text[i:i+1]     # 1*768
            # img_feature = img_feature +  self.cross_attn_text(img_feature, text_features)# B*129*768
            fuse_features = self.cross_attn_text(img_feature, text_features)
            # fused_features.append(img_feature)
            fused_features.append(fuse_features)
        fused_features = torch.cat(fused_features, dim=0)
        return fused_features
    
    def forward(self, x, label=None, captions=None,cam_label= None):
        '''
        captions: batchbatch中每个图像特有的captions
        '''
        if cam_label != None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label]
        else:
            cv_embed = None
        if self.coop:
            ###################### 文本特征(可学习prompt+类别token化) ######################
            text_features = self.caption_features(captions)    
            if self.text_proj:       
                text_features = self.text_projectoin(text_features) # ADD A LEARNABLE TEXT PROJECTION TO TRAIN TEXT
            ###################### 图像特征 ######################
            image_features_last, image_features, image_features_proj = self.image_encoder(
                x, cv_embed
            )        
            if not self.use_adain_norm: # if not adain 
                img_feature_last = image_features_last[:,0]
            # img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]
        elif self.cocoop:
            ###################### 图像特征 ######################
            image_features_last, image_features, image_features_proj = self.image_encoder(
                x, cv_embed
            )        
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]           
            ###################### 文本特征(可学习prompt+类别token化) ######################
            text_features = self.caption_features(captions, img_feature_proj) 
            if self.text_proj:
                text_features = self.text_projectoin(text_features) # ADD A LEARNABLE TEXT PROJECTION TO TRAIN TEXT                        
        elif self.maple:
            #### 文本特征(可学习prompt+类别token化)
            text_features,shared_ctx, deep_compound_prompts_vision = self.caption_features(captions)
            if self.text_proj:
                text_features = self.text_projectoin(text_features) # ADD A LEARNABLE TEXT PROJECTION TO TRAIN TEXT
            ###################### 图像特征 ######################
            image_features_last, image_features, image_features_proj = self.image_encoder(
                x, shared_ctx, deep_compound_prompts_vision
            )  
            if not self.use_adain_norm: # if not adain 
                img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]     
        else:
            ###################### 可学习文本 ######################
            captions_tokens = clip.tokenize('A photo of '+ caption + '.' for caption in captions).cuda()
            text_features = self.text_encoder(captions_tokens)
            if self.text_proj:
                text_features = self.text_projectoin(text_features) # ADD A LEARNABLE TEXT PROJECTION TO TRAIN TEXT
            ###################### 图像特征 ######################
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) # clip output 512
            if not self.use_adain_norm:
                img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0] # test time 
            img_feature_proj = image_features_proj[:,0]

        if self.training:
        # if True:
            # text_features = self.caption_features(captions)
            # global_feat = global_feat + self.caption_fusion(global_feat, text_features) # B*129*768 
            ###################### img_features_proj: b,129,512, 其他是b,129,768 ######################
            # img_feature_last = image_features_last + self.cross_attn_text(img_feature_last, text_features) # B*129*768
            # img_feature = image_features_last + self.cross_attn_text(img_feature, text_features) # B*129*768
            if self.use_adain_norm:
                img_feature_last = self.ada_in(image_features_last) # b*129*768
                img_feature_last = img_feature_last[:,0]

            img_feature = image_features + self.cross_attn_text(image_features, text_features) # B*129*768
            img_feature = img_feature[:,0]
            
            img_feature_proj = img_feature_proj + self.cross_attn_text_proj(img_feature_proj, text_features) # B*768
            # contrastive loss
            # text_features = text_features / text_features.norm(dim=1, keepdim=True)
            # image_features = image_features / image_features.norm(dim=1, keepdim=True)
            # logit_scale = self.logit_scale.exp()
            # logits = logit_scale * image_features @ text_features.t()
            # image_features = global_feat / global_feat.norm(dim=2, keepdim= True)
            # global_feat = global_feat / global_feat.norm(dim=2, keepdim= True)
            # global_feat = global_feat + self.caption_fusion(global_feat, text_features) # B*129*768

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)
        # feat = global_featenen

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_proj = self.classifier_proj(feat_proj)
            # if self.caption:
            #     pass
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
                    # print("Test with feature before BN")
                    return torch.cat([img_feature, img_feature_proj], dim=1)

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



def make_model(cfg, num_class, camera_num):
    model = build_transformer_caption(num_class, camera_num, cfg)
    return model

