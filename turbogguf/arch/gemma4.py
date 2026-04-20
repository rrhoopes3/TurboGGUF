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

Note: Gemma4RMSNorm.forward does `output * weight` (plain weight, init=1.0),
NOT `output * (1 + weight)`. That 1+weight pattern is Gemma 3 only. The base
class extract_norm_gamma / reset_norm_to_identity are correct as-is.
"""

from typing import Dict, Optional
import torch.nn as nn
from turbogguf.arch.llama import LlamaHandler


class Gemma4Handler(LlamaHandler):
    """Handler for Gemma 4 models (including multimodal wrapper)."""

    def _get_text_model(self, model: nn.Module) -> nn.Module:
        """Navigate to the inner Gemma4TextModel.

        Structure varies by loading method:
          - Gemma4ForConditionalGeneration: model.model.language_model
          - Gemma4ForCausalLM: model.model
          - Direct Gemma4TextModel: model itself
        """
        # ConditionalGeneration: model -> Gemma4Model -> language_model
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model
        # CausalLM or direct: model -> Gemma4TextModel (has .layers)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model
        # Already the text model
        if hasattr(model, "layers"):
            return model
        raise AttributeError(
            f"Cannot find Gemma4TextModel in {type(model).__name__}. "
            f"Children: {[n for n, _ in model.named_children()]}"
        )

    def _get_config(self, model: nn.Module):
        """Get the text model config, handling nested configs."""
        text = self._get_text_model(model)
        config = text.config
        # Some multimodal wrappers nest text config under .text_config
        if hasattr(config, "text_config"):
            return config.text_config
        return config

    def get_embedding(self, model: nn.Module) -> nn.Embedding:
        return self._get_text_model(model).embed_tokens

    def get_lm_head(self, model: nn.Module) -> Optional[nn.Linear]:
        head = self.get_tied_lm_head_module(model)
        if head is None:
            return None
        # Check for tied weights (shared with embedding)
        emb = self._get_text_model(model).embed_tokens
        if head.weight.data_ptr() == emb.weight.data_ptr():
            return None
        return head

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        return self._get_text_model(model).layers

    def get_final_norm(self, model: nn.Module) -> nn.Module:
        return self._get_text_model(model).norm

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
        head = self.get_tied_lm_head_module(model)
        if head is None:
            return True
        emb = self._get_text_model(model).embed_tokens
        return head.weight.data_ptr() == emb.weight.data_ptr()

    def get_tied_lm_head_module(self, model: nn.Module):
        # For Gemma4ForConditionalGeneration, lm_head is not a top-level
        # attribute — it lives inside the text model (model.model.language_model).
        # Fall back to the outer model if the text model doesn't have one.
        text = self._get_text_model(model)
        head = getattr(text, "lm_head", None)
        if head is None:
            head = getattr(model, "lm_head", None)
        return head

    def uses_tied_lm_head_for_gguf(self) -> bool:
        # Gemma4 requires a SEPARATE output.weight for correct logit computation.
        # The output_norm has large learned gamma values (up to ~510x) that must
        # be fused into lm_head BEFORE rotation; otherwise gamma doesn't commute
        # with the Hadamard rotation and output logits are completely wrong.
        #
        # The pre-built llama.cpp binary (≤b8638) looks for tensor named "output"
        # (not "output.weight") for Gemma4 due to a missing LLM_TENSOR_OUTPUT entry
        # in the Gemma4 arch tensor list.  The pipeline post-processes the GGUF to
        # rename "output.weight" → "output" for compatibility.  See cli.py pipeline.
        return False
