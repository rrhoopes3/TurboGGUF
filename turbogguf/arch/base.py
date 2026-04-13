"""Base class for architecture-specific model handlers.

Each handler maps the abstract rotation interface to concrete model attribute
names (e.g., model.model.layers vs model.transformer.h).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch
import torch.nn as nn


class ArchHandler(ABC):
    """Abstract architecture handler for weight access."""

    @abstractmethod
    def get_embedding(self, model: nn.Module) -> nn.Embedding:
        """Return the token embedding layer."""

    @abstractmethod
    def get_lm_head(self, model: nn.Module) -> nn.Linear:
        """Return the language model head (output projection)."""

    @abstractmethod
    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        """Return the list of transformer layers."""

    @abstractmethod
    def get_attn_projs(self, layer: nn.Module) -> Dict[str, nn.Linear]:
        """Return attention projections: {q, k, v, o}_proj."""

    @abstractmethod
    def get_mlp_projs(self, layer: nn.Module) -> Dict[str, nn.Linear]:
        """Return MLP projections: {gate, up, down}_proj."""

    @abstractmethod
    def get_pre_attn_norm(self, layer: nn.Module) -> nn.Module:
        """Return the pre-attention layer norm (RMSNorm/LayerNorm)."""

    @abstractmethod
    def get_post_attn_norm(self, layer: nn.Module) -> nn.Module:
        """Return the norm between attention and MLP (pre-MLP norm).

        For LLaMA this is post_attention_layernorm. For Gemma2 this is
        pre_feedforward_layernorm. In both cases, this is the norm whose
        output feeds into the MLP's gate/up projections.
        """

    @abstractmethod
    def get_final_norm(self, model: nn.Module) -> nn.Module:
        """Return the final output norm before lm_head."""

    @abstractmethod
    def get_head_dim(self, model: nn.Module) -> int:
        """Return the attention head dimension."""

    @abstractmethod
    def get_num_heads(self, model: nn.Module) -> int:
        """Return number of query attention heads."""

    @abstractmethod
    def get_num_kv_heads(self, model: nn.Module) -> int:
        """Return number of key/value attention heads (for GQA)."""

    @abstractmethod
    def get_hidden_size(self, model: nn.Module) -> int:
        """Return the hidden dimension size."""

    def get_post_attn_output_norm(self, layer: nn.Module) -> Optional[nn.Module]:
        """Return the post-attention output norm (applied to attn output before residual add).

        Only Gemma2/4 have this. Returns None for architectures without it.
        """
        return None

    def get_post_mlp_output_norm(self, layer: nn.Module) -> Optional[nn.Module]:
        """Return the post-MLP output norm (applied to MLP output before residual add).

        Only Gemma2/4 have this. Returns None for architectures without it.
        """
        return None

    def uses_rms_norm(self) -> bool:
        """Whether this architecture uses RMSNorm (True) or LayerNorm (False)."""
        return True

    def has_bias(self) -> bool:
        """Whether linear layers have bias terms."""
        return False

    def has_tied_embeddings(self, model: nn.Module) -> bool:
        """Whether embedding and lm_head share the same weight tensor."""
        return False

    def extract_norm_gamma(self, norm: nn.Module) -> torch.Tensor:
        """Extract the effective scaling factor (gamma) from a norm module.

        Standard RMSNorm: gamma = weight
        Gemma RMSNorm:    gamma = 1 + weight
        """
        return norm.weight.data.clone().float()

    def reset_norm_to_identity(self, norm: nn.Module) -> None:
        """Reset norm to identity scaling (gamma=1).

        Standard RMSNorm: set weight = 1.0
        Gemma RMSNorm:    set weight = 0.0 (since gamma = 1 + weight)
        """
        norm.weight.data.fill_(1.0)
