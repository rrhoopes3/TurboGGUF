"""Base class for architecture-specific model handlers.

Each handler maps the abstract rotation interface to concrete model attribute
names (e.g., model.model.layers vs model.transformer.h).
"""

from abc import ABC, abstractmethod
from typing import Dict, List
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
        """Return the post-attention layer norm."""

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

    def get_pre_ffn_norm(self, layer: nn.Module):
        """Return pre-feedforward norm if this architecture has one (e.g., Gemma sandwich norms)."""
        return None

    def get_post_ffn_norm(self, layer: nn.Module):
        """Return post-feedforward norm if this architecture has one (e.g., Gemma sandwich norms)."""
        return None

    def uses_rms_norm(self) -> bool:
        """Whether this architecture uses RMSNorm (True) or LayerNorm (False)."""
        return True

    def has_bias(self) -> bool:
        """Whether linear layers have bias terms."""
        return False

    def has_tied_lm_head(self, model: nn.Module) -> bool:
        """Whether lm_head weights are tied to embedding weights."""
        return False
