"""Qwen2 architecture handler.

Qwen2 uses LLaMA-like structure but with bias in QKV projections.
"""

from typing import Dict
import torch.nn as nn
from turbogguf.arch.llama import LlamaHandler


class Qwen2Handler(LlamaHandler):
    """Handler for Qwen2 models."""

    def has_bias(self) -> bool:
        return True  # Qwen2 has bias in q/k/v projections

    def get_attn_projs(self, layer: nn.Module) -> Dict[str, nn.Linear]:
        attn = layer.self_attn
        return {
            "q_proj": attn.q_proj,
            "k_proj": attn.k_proj,
            "v_proj": attn.v_proj,
            "o_proj": attn.o_proj,
        }
