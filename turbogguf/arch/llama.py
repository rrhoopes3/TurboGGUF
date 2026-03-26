"""LLaMA architecture handler.

Covers: LLaMA 2, LLaMA 3, LLaMA 3.1, CodeLlama, and derivatives.
"""

from typing import Dict
import torch.nn as nn
from turbogguf.arch.base import ArchHandler


class LlamaHandler(ArchHandler):
    """Handler for LLaMA-family models."""

    def get_embedding(self, model: nn.Module) -> nn.Embedding:
        return model.model.embed_tokens

    def get_lm_head(self, model: nn.Module) -> nn.Linear:
        return model.lm_head

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        return model.model.layers

    def get_attn_projs(self, layer: nn.Module) -> Dict[str, nn.Linear]:
        attn = layer.self_attn
        return {
            "q_proj": attn.q_proj,
            "k_proj": attn.k_proj,
            "v_proj": attn.v_proj,
            "o_proj": attn.o_proj,
        }

    def get_mlp_projs(self, layer: nn.Module) -> Dict[str, nn.Linear]:
        mlp = layer.mlp
        return {
            "gate_proj": mlp.gate_proj,
            "up_proj": mlp.up_proj,
            "down_proj": mlp.down_proj,
        }

    def get_pre_attn_norm(self, layer: nn.Module) -> nn.Module:
        return layer.input_layernorm

    def get_post_attn_norm(self, layer: nn.Module) -> nn.Module:
        return layer.post_attention_layernorm

    def get_final_norm(self, model: nn.Module) -> nn.Module:
        return model.model.norm

    def get_head_dim(self, model: nn.Module) -> int:
        config = model.config
        if hasattr(config, "head_dim"):
            return config.head_dim
        return config.hidden_size // config.num_attention_heads

    def get_num_heads(self, model: nn.Module) -> int:
        return model.config.num_attention_heads

    def get_num_kv_heads(self, model: nn.Module) -> int:
        config = model.config
        if hasattr(config, "num_key_value_heads"):
            return config.num_key_value_heads
        return config.num_attention_heads  # MHA fallback

    def get_hidden_size(self, model: nn.Module) -> int:
        return model.config.hidden_size
