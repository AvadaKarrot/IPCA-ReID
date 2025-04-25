from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch
import torch.nn as nn
import copy

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


class TextEncoder(nn.Module):
    def __init__(self, clip_model): #dfdddd
        super(TextEncoder, self).__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) # [bs, 77, dim] + [77, dim]
        x = x.permute(1, 0, 2)  # to shape [seq_len, batch, dim]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # to shape [batch, seq_len, dim]
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection # 14,768

        return x


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