from __future__ import annotations

import functools
import json
import logging
import math
import os
from einops import rearrange, einsum
import einx

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int

logger = logging.getLogger(__name__)

# softmax函数的实现
def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)

# Linear类的实现
class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__
        std = math.sqrt(2 / (d_in + d_out)) = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(d_out, d_in), std=std, a=-3*std, b=3*std),
            requires_grad=True
        )

    def forward(self, x):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    
    def extra_repr(self):
        return f"d_out={self.weight.shape[0]}, d_in={self.weight.shape[1]}"
    
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        std = 1.0
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(vocab_size, d_model), std=std, a=-3*std, b=3*std),
            requires_grad=True
        )

        def forwart(self, token_ids):
            return self.weight[token_ids, :]
        
        def extra_repr(self):
            return f"vocab_size={self.weight.shape[0]}, d={self.weight.shape[1]}"
        
class RMSNorm(nn.Module):
    def __init__(
            self,
            hidden_size,
            eps = 1e-5,
            device = None
            ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x):
        in_dype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms

        return (self.weight * x).to(in_dype)
    
    def extra_repr(self):
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"

class RotaryEmbedding(nn.Module):
    def __init__(self, context_length, dim, theta = 10000.0):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache",
            RotaryEmbedding._init_cache(context_length, dim, theta), persistent=False
        )

    @staticmethod
    def _init_cache(context_length, dim, theta):
        assert dim % 2 == 0
        d = torch.arrange(0, dim, 2) / dim
        freqs = theta ** -d
        t = torch.arrange(context_length)
        freqs = einsum(t, freqs, "t, f -> t f")
        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))
    
    def forward(self, x, pos_ids):
        x1, x2 = rearrange(x, "... (half_d xy) -> xy ... half_d", xy=2)

        cos, sin = einx.get_at('cos_sin [pos] half_dim, ... -> con_sin ... half_dim', self._freq_cis_cache, pos_ids)
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', x1_rot, x2_rot).contiguous()
        return result
    
    def extra_repr(self):
        return f"context_length = {self._freq_cis_cache.shape[0]}, dim / 2 = [self._freq_cis_cache.shape[1]]"
    
class BasicsTransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size,
            context_length,
            d_model,
            num_layers,
            num_heads,
            d_ff,
            rope_theta
            ):
        
        self.config = {
            k: v for k, v in locals().items() if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }

        super.__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model)
        d_head = d_model // num_heads
        self.positional_encoder = RotaryEmbedding(
            context_length=context_length,
            dim = d_head,
            theta = rope_theta
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model = d_model,
                    num_heads = num_heads,
                    d_ff = d_ff,
                    positional_encoder = self.positional_encoder
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")
    
    def get_num_params(self, non_embedding = True):
        n_params = sum(p.numel() for p in self.parameters)
        if non_embedding:
            n_params -= self.lm_head.weight.numel()
        return n_params
    
    def forward(self, x):
        _, sequence_length = x.size()
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)

        return self.lm_head(x)
    
    @torch.no_grad()
    def generate(
        self,
        x,
        max_new_tokens,
        temperature = 1.0,
        top_k:int | None = None,
        eos_token_id: Int | None = None,
    ):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        original_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            logits = self.forward(x)
            next_token_logits = logits[:, -1]
            temperature_scaled_next_token_logits = next_token_logits / temperature

            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(topk_mask, float("-inf"))
            next_token_probabilities = softmax(temperature_scaled_next_token_logits, dim = -1)
            next_token_id = torch.multinomial(next_token_probabilities, 1)

            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim = -1)
        new_token_ids = x[: original_sequence_length:]
        return new_token_ids
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path):

        
    
