import torch
from model.backbones.resnet import ResNet, Bottleneck
import copy
import torch.nn as nn
from model.backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_base_patch16_224_TransReID_Prompt, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from model.backbones.cross_attention_text import Cross_Attention_text, Cross_Attention_text_Block
from model.backbones.cross_attention_img import Cross_Attention_img, Cross_Attention_img_Block
from model.csa_block.cov_setting import CovMatrix_AIAW
from model.backbones.vit_pytorch import Block
from model.maple import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from model.prompt.promptlearner import *
from model.prompt.textencoder import *
import os, sys
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
from model.maple import clip
from model.maple.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def load_clip_to_cpu_maple(cfg):
    model_path = "/home/test/LIVA/ZWQ/pretrained/ViT-L-14.pt.1"

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
                      "maple_length": cfg.MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model
def load_clip_to_cpu(cfg):
    model_path = "/home/test/LIVA/ZWQ/pretrained/ViT-B-16.pt" # 文本图像特征提取器对齐         

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
                      "person_img_size": cfg.INPUT.SIZE_TRAIN}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
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
    def __init__(self, num_classes, camera_num, cfg, factory):
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

        ####### use adain
        self.use_adain_norm = cfg.MODEL.ADAIN
        if cfg.MODEL.ADAIN:
            self.ada_in = AdaIN()            # 定义1x1卷积层将通道数从6减少到3
            # self.conv1x1 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1)

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
        self.maple = cfg.MAPLE.MAPLE_PROMPT #耦合可学习token
        clip_model = load_clip_to_cpu(cfg, self.h_resolution, self.w_resolution,self.vision_stride_size)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.caption = cfg.MODEL.CAPTION # 有标签进行文本特征学习
        if self.caption:
            ################### 文本图像特征融合——cross attention ###################
            self.cross_attn_text = Cross_Attention_text_Block(
                dim = self.in_planes, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.
            )
            self.cross_attn_text.apply(weights_init_cross_attention)
            ################### 文本图像特征融合——cross attention ###################

        if cfg.COOP.COOP_PROMPT: # 可学习Prompt：ctx
            print(f"Loading CLIP (backbone: {'ViT-L-14'})")
            clip_model = load_clip_to_cpu()
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
            clip_model = load_clip_to_cpu()
            clip_model.float()
            print('Building Prompt Learning CLIP')
            self.prompt_learner = CapMultiModalPromptLearner(cfg, clip_model) # 假设给了一些固定的图像描述(例如 人的头，手...)
            # self.tokenized_prompts = self.prompt_learner.tokenized_prompts  ## 类别token化 # 14*77
            self.text_encoder = TextEncoder_MaPLE(clip_model) # 不进行文本特征提取学习 冻结
            # 冻结 self.text_encoder 的所有参数
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.dtype = clip_model.dtype            
        else:  ###################### 固定的Prompt ######################
            print('Building Caption-drve CLIP')
            self.text_encoder = TextEncoder_Fix(clip_model) # 不进行文本特征提取学习 冻结
            for param in self.text_encoder.parameters():
                param.requires_grad = True
            self.dtype = clip_model.dtype                    
        

        # self.image_projection = ProjectionHead(embedding_dim=512)
        # self.text_projectoin = ProjectionHead(embedding_dim=512)

    def caption_features(self, captions):

        if self.coop:
            prompts, captions_tokens = self.prompt_learner(captions) #### 类别token化   ## 类别token化 # 14*77
            text_features = self.text_encoder(prompts, captions_tokens) # 128*77 -> 128*768
            return text_features
        elif self.maple:
            prompts, tokenized_prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(captions) #### 类别token化 
            text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
            return text_features, shared_ctx, deep_compound_prompts_vision
        else:
            captions_tokens = clip.tokenize([f'a photo of {caption}.' for caption in captions]).cuda()
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
            global_feat = self.base(x, cam_label=cam_label)
            
            #### 文本特征(可学习prompt+类别token化)
            text_features = self.caption_features(captions) 
        
        elif self.maple:
            #### 文本特征(可学习prompt+类别token化)
            text_features,shared_ctx, deep_compound_prompts_vision = self.caption_features(captions)
            
            global_feat = self.base(x, cam_label=cam_label, 
                                    shared_ctx=shared_ctx, compound_deeper_prompts=deep_compound_prompts_vision)

        else:
            
            # captions_tokens = clip.tokenize('a photo of '+ caption for caption in captions).cuda()
            # text_features = self.text_encoder(captions_tokens)
            # global_feat = self.base(x, cam_label=cam_label)
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) # clip output 512
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        if self.training:
            pass
            # text_features = self.caption_features(captions)
            # global_feat = global_feat + self.caption_fusion(global_feat, text_features) # B*129*768
            # global_feat = global_feat + self.cross_attn_text(global_feat, text_features) # B*129*768
            # contrastive loss
            # text_features = text_features / text_features.norm(dim=1, keepdim=True)
            # image_features = image_features / image_features.norm(dim=1, keepdim=True)
            # logit_scale = self.logit_scale.exp()
            # logits = logit_scale * image_features @ text_features.t()
            # image_features = global_feat / global_feat.norm(dim=2, keepdim= True)
            # global_feat = global_feat / global_feat.norm(dim=2, keepdim= True)
            # global_feat = global_feat + self.caption_fusion(global_feat, text_features) # B*129*768

        #  要不要norm呢?   # b*129*768
        if self.use_adain_norm and self.training:
            feat = self.ada_in(global_feat) # b*129*768
            global_feat = feat[:, 0]
        else:
            # global_feat = global_feat[:, 0] 
            pass
        ####################  # ############### CLIP test + img 466放在class token和风格做cross attention

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
