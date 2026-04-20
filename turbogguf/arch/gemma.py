"""Gemma architecture handler.

Covers: Gemma 1 (GemmaForCausalLM), Gemma 2 (Gemma2ForCausalLM),
and Gemma 4 (Gemma4ForCausalLM).

Key differences from LLaMA:
  - RMSNorm uses (1 + weight) scaling instead of weight scaling
  - Gemma2/4 have 4 norms per layer (pre/post-attn + pre/post-MLP)
  - Embedding and lm_head weights are typically tied
  - Embedding output is scaled by sqrt(hidden_size) at runtime
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from turbogguf.arch.llama import LlamaHandler


class GemmaHandler(LlamaHandler):
    """Handler for Gemma 1 models.

    Same LLaMA-like structure but with (1+weight) RMSNorm and tied embeddings.
    """

    def has_tied_embeddings(self, model: nn.Module) -> bool:
        config = model.config
        return getattr(config, "tie_word_embeddings", True)

    def extract_norm_gamma(self, norm: nn.Module) -> torch.Tensor:
        # Gemma RMSNorm: output = (1 + weight) * x / rms(x)
        return (1.0 + norm.weight.data.clone()).float()

    def reset_norm_to_identity(self, norm: nn.Module) -> None:
        # (1 + 0) = 1 → identity scaling
        norm.weight.data.fill_(0.0)


class Gemma2Handler(GemmaHandler):
    """Handler for Gemma 2 models.

    Gemma2 has 4 norms per decoder layer:
      - input_layernorm (pre-attention)
      - post_attention_layernorm (post-attention output, before residual add)
      - pre_feedforward_layernorm (pre-MLP)
      - post_feedforward_layernorm (post-MLP output, before residual add)

    The post-attention and post-MLP output norms are fused approximately
    into o_proj and down_proj respectively.
    """

    def get_post_attn_norm(self, layer: nn.Module) -> nn.Module:
        # The norm that feeds into MLP (gate/up projections)
        return layer.pre_feedforward_layernorm

    def get_post_attn_output_norm(self, layer: nn.Module) -> Optional[nn.Module]:
        # Post-attention output norm (applied to attn output before residual add)
        return layer.post_attention_layernorm

    def get_post_mlp_output_norm(self, layer: nn.Module) -> Optional[nn.Module]:
        # Post-MLP output norm (applied to MLP output before residual add)
        return layer.post_feedforward_layernorm

    def get_attn_projs(self, layer: nn.Module) -> Dict[str, nn.Linear]:
        attn = layer.self_attn
        return {
            "q_proj": attn.q_proj,
            "k_proj": attn.k_proj,
            "v_proj": attn.v_proj,
            "o_proj": attn.o_proj,
        }

    def has_bias(self) -> bool:
        # Gemma2 attention_bias is configurable, default False
        return False


class Gemma4Handler(Gemma2Handler):
    """Handler for Gemma 4 models.

    Gemma 4 text decoder uses the same structure as Gemma 2.
    MoE layers (if present) use the same gate/up/down projection names.
    """
    pass
