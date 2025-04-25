from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch
import torch.nn as nn
import copy
from collections import OrderedDict
_tokenizer = _Tokenizer()
from model.maple import clip
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

class CapPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super(CapPromptLearner, self).__init__()

        n_ctx = cfg.COOP.N_CTX ## 提示词个数 a photo of (3)
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
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :] # 3*612
            prompt_prefix = ctx_init

        else:

            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")        

        if cfg.MODEL.PROMPT:
            self.ctx = nn.Parameter(ctx_vectors)  # Prompt 作为可学习参数
        else:
            self.ctx = ctx_vectors  # Prompt 作为固定参数
        # self.ctx = nn.Parameter(ctx_vectors) # prompt作为可学习参数
        self.prompt_prefix = prompt_prefix # str: "A photo of"
        self.n_ctx = n_ctx        
        self.class_token_position = cfg.COOP.CLASS_TOKEN_POSITION 
        self.clip_model = clip_model
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
    
    def forward(self, captions):
        n_cls = len(captions)
        ctx = self.ctx.cuda()
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1) # batch * 3 * 768(512)
        captions = [cap.replace("_", " ") for cap in captions] # str: "A person."
        # captions = [cap.lower() for cap in captions] # str: "a person."
        cap_lens = [len(_tokenizer.encode(cap)) for cap in captions] # [int:, ]
        prompts = [self.prompt_prefix + " " + cap + "." for cap in captions] # str: "a photo of a person."

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda() ## 类别token化 (n_cls, n_tkn) cap:256,77
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype) # bs*77*512
        prefix = embedding[:, :1, :]  # SOS
        suffix = embedding[:, 1 + self.n_ctx :, :]                  
        prompts = self.construct_prompts(ctx, prefix, suffix)        # bs*77*512                             

        return prompts, tokenized_prompts # 14 77 768

class CapConditionalPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super(CapConditionalPromptLearner, self).__init__()

        n_ctx = cfg.COCOOP.N_CTX ## 提示词个数 a photo of (3)
        ctx_init = cfg.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim # 图像特征提取器的输出特征维度 512
        
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

        if cfg.MODEL.PROMPT: # 是否用可学习的Prompt 必须在Default中加入MODEL.PROMPT: True
            self.ctx = nn.Parameter(ctx_vectors)  # Prompt 作为可学习参数
        else:
            self.ctx = ctx_vectors  # Prompt 作为固定参数
        # self.ctx = nn.Parameter(ctx_vectors) # prompt作为可学习参数
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        self.prompt_prefix = prompt_prefix # str: "a photo of"
        self.n_ctx = n_ctx        
        self.class_token_position = cfg.COCOOP.CLASS_TOKEN_POSITION 
        self.clip_model = clip_model
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
    
    def forward(self, captions, im_features):
        n_cls = len(captions)
        ctx = self.ctx.cuda() # n_ctx, ctx_dim
        bias = self.meta_net(im_features) # bs, ctx_dim
        bias = bias.unsqueeze(1) # bs,1,ctx_dim
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1) # batch * 3(n_ctx) * 768(ctx_dim)
        ctx_shifted = ctx + bias # bs, n_ctx, ctx_dim
        captions = [cap.replace("_", " ") for cap in captions] # str: "a photo of a person."
        captions = [cap.lower() for cap in captions] # 描述→小写(服务器已改)
        cap_lens = [len(_tokenizer.encode(cap)) for cap in captions] # [int:, ]
        prompts = [self.prompt_prefix + " " + cap + "." for cap in captions] # str: "a photo of a person."

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda() ## 类别token化 (n_cls, n_tkn) cap:256,77
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype) # bs*77*512
        prefix = embedding[:, :1, :]  # SOS
        suffix = embedding[:, 1 + self.n_ctx :, :]                  
        prompts = self.construct_prompts(ctx_shifted, prefix, suffix)                                    

        return prompts, tokenized_prompts # 14 77 768

class CapMultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super(CapMultiModalPromptLearner, self).__init__()

        n_ctx = cfg.MAPLE.N_CTX ## 提示词个数 a photo of (3)
        ctx_init = cfg.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        ####### modal
        assert cfg.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        
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

        print('caption based MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")        

        self.proj = nn.Linear(ctx_dim, 768)
        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt (visual)
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        self.ctx = nn.Parameter(ctx_vectors) # prompt作为可学习参数
        self.prompt_prefix = prompt_prefix # str: "a photo of"
        self.n_ctx = n_ctx        
        self.class_token_position = cfg.COOP.CLASS_TOKEN_POSITION 
        self.clip_model = clip_model
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
    
    def forward(self, captions):
        n_cls = len(captions) #n_cls = bs
        ctx = self.ctx.cuda()
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1) # batch * 3 * 768
        captions = [cap.replace("_", " ") for cap in captions] # str: "a photo of a person."
        cap_lens = [len(_tokenizer.encode(cap)) for cap in captions] # [int:, ]
        prompts = [self.prompt_prefix + " " + cap + "." for cap in captions] # str: "a photo of a person."

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda() ## 类别token化 (n_cls, n_tkn) 
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype)
        prefix = embedding[:, :1, :]  # SOS
        suffix = embedding[:, 1 + self.n_ctx :, :]                  
        prompts = self.construct_prompts(ctx, prefix, suffix)                                    
        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))

        return prompts, tokenized_prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts # 14 77 768