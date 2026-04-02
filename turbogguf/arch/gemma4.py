"""Gemma 4 architecture handler.

Covers: Gemma 4 31B (dense), Gemma 4 multimodal (Gemma4ForConditionalGeneration).

Gemma 4 is a multimodal model whose text decoder follows the LLaMA pattern.
The outer Gemma4ForConditionalGeneration wraps vision/audio encoders plus
a language_model (the actual text decoder). This handler navigates through
the wrapper to access the text decoder weights for rotation.

Key differences from LLaMA:
  - Multimodal wrapper: weights live under model.language_model.model.*
  - Sandwich norms: optional pre/post feedforward norms per layer
  - Tied lm_head: embedding weights may be shared with output projection
  - Hybrid attention: sliding window + full attention (transparent to rotation)
"""

from typing import Dict, Optional
import torch.nn as nn
from turbogguf.arch.llama import LlamaHandler


class Gemma4Handler(LlamaHandler):
    """Handler for Gemma 4 models (including multimodal wrapper)."""

    def _get_text_model(self, model: nn.Module) -> nn.Module:
        """Navigate to the inner text decoder.

        Handles both:
          - Gemma4ForConditionalGeneration (has .language_model)
          - Gemma4ForCausalLM (is the text model directly)
        """
        if hasattr(model, "language_model"):
            return model.language_model
        return model

    def _get_config(self, model: nn.Module):
        """Get the text model config, handling nested configs."""
        text = self._get_text_model(model)
        config = text.config
        # Some multimodal wrappers nest text config under .text_config
        if hasattr(config, "text_config"):
            return config.text_config
        return config

    def get_embedding(self, model: nn.Module) -> nn.Embedding:
        return self._get_text_model(model).model.embed_tokens

    def get_lm_head(self, model: nn.Module) -> Optional[nn.Linear]:
        text = self._get_text_model(model)
        head = getattr(text, "lm_head", None)
        if head is None:
            return None
        # Check for tied weights (shared with embedding)
        emb = text.model.embed_tokens
        if head.weight.data_ptr() == emb.weight.data_ptr():
            return None
        return head

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        return self._get_text_model(model).model.layers

    def get_final_norm(self, model: nn.Module) -> nn.Module:
        return self._get_text_model(model).model.norm

    def get_pre_ffn_norm(self, layer: nn.Module):
        """Return pre-feedforward norm (Gemma sandwich norm pattern)."""
        return getattr(layer, "pre_feedforward_layernorm", None)

    def get_post_ffn_norm(self, layer: nn.Module):
        """Return post-feedforward norm (Gemma sandwich norm pattern)."""
        return getattr(layer, "post_feedforward_layernorm", None)

    def get_head_dim(self, model: nn.Module) -> int:
        config = self._get_config(model)
        if hasattr(config, "head_dim"):
            return config.head_dim
        return config.hidden_size // config.num_attention_heads

    def get_num_heads(self, model: nn.Module) -> int:
        return self._get_config(model).num_attention_heads

    def get_num_kv_heads(self, model: nn.Module) -> int:
        config = self._get_config(model)
        if hasattr(config, "num_key_value_heads"):
            return config.num_key_value_heads
        return config.num_attention_heads

    def get_hidden_size(self, model: nn.Module) -> int:
        return self._get_config(model).hidden_size

    def has_tied_lm_head(self, model: nn.Module) -> bool:
        text = self._get_text_model(model)
        head = getattr(text, "lm_head", None)
        if head is None:
            return True
        emb = text.model.embed_tokens
        return head.weight.data_ptr() == emb.weight.data_ptr()
