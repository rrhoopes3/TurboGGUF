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

    def get_pre_ffn_norm(self, layer: nn.Module):
        """Return pre-feedforward norm if this architecture has one (e.g., Gemma sandwich norms)."""
        return None

    def get_post_ffn_norm(self, layer: nn.Module):
        """Return post-feedforward norm if this architecture has one (e.g., Gemma sandwich norms)."""
        return None

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

    def is_linear_attention_layer(self, layer: nn.Module) -> bool:
        """Whether this layer uses a linear-attention / state-space token mixer
        (e.g. Gated DeltaNet in Qwen 3.5/3.6 MoE) instead of standard softmax
        attention.

        When True, callers must use get_linear_attn_projs() for rotation/fusion
        instead of get_attn_projs() — the layer has no q/k/v/o_proj.
        """
        return False

    def get_linear_attn_projs(self, layer: nn.Module):
        """Linear-attention block info, or None for standard-attention layers.

        Shape:
            {
                "in_projs": [nn.Linear, ...]   # all projections whose input is
                                               # the residual hidden dim; will
                                               # all absorb pre-norm gamma and
                                               # right-rotate by Q under R1.
                "out_proj": nn.Linear,         # single projection back into the
                                               # residual; left-rotated by Q^T
                                               # under R1.
            }
        Internal recurrent / conv / per-head weights are assumed not to touch
        the residual-basis hidden dim and are left untouched by R1.
        """
        return None

    def is_moe_layer(self, layer: nn.Module) -> bool:
        """Whether this decoder layer uses a Mixture-of-Experts FFN block.

        When True, callers must use get_moe() instead of get_mlp_projs() for
        FFN-side rotation/fusion, because the projections live in 3D Parameter
        tensors (fused across experts) rather than per-layer nn.Linears.
        """
        return False

    def get_moe(self, layer: nn.Module):
        """Return MoE block info for a layer, or None for dense layers.

        Shape:
            {
                "experts": module holding 3D Parameters `gate_up_proj`
                    (E, 2*I, H) and `down_proj` (E, H, I),
                "router": module whose `.weight` is the routing matrix
                    (num_experts, H),
                "shared_expert": None | {"gate_proj", "up_proj", "down_proj"}
                    nn.Linears for an always-on shared expert (Qwen 3.5 MoE),
                "shared_expert_gate": None | nn.Linear producing a scalar
                    gate for the shared expert.
            }
        """
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

    def has_tied_embeddings(self, model: nn.Module) -> bool:
        """Alias for has_tied_lm_head — whether embedding and lm_head share the same weight tensor."""
        return self.has_tied_lm_head(model)

    def get_tied_lm_head_module(self, model: nn.Module):
        """Return the lm_head nn.Module regardless of tied-weight status.

        Unlike get_lm_head() (which returns None for tied heads), this returns
        the raw module so fuse_all_norms can un-tie it and fuse gamma into it.
        For most architectures lm_head is a direct model attribute; override
        in handlers where it is nested (e.g. Gemma4 multimodal wrapper).
        """
        return getattr(model, "lm_head", None)

    def uses_tied_lm_head_for_gguf(self) -> bool:
        """Whether the GGUF format always uses tied token_embd for output logits.

        When True, llama.cpp's architecture definition has no LLM_TENSOR_OUTPUT
        entry — it always reuses token_embd.weight for both input embeddings and
        output logits.  In this case the rotation pipeline must:
          - NOT break the lm_head tie in fuse_all_norms
          - NOT fuse output_norm gamma into lm_head (keep gamma in output_norm.weight)
          - NOT save output.weight in the GGUF

        llama.cpp still applies output_norm at runtime, so the gamma is correctly
        applied: logits = output_norm(x_rotated) @ token_embd_rotated^T.
        """
        return False

    def extract_norm_gamma(self, norm: nn.Module) -> torch.Tensor:
        """Extract the effective scaling factor (gamma) from a norm module.

        Standard RMSNorm: gamma = weight  (weight initialized to 1.0)
        Gemma RMSNorm:    gamma = 1 + weight  (weight initialized to 0.0)

        Override in architecture handlers with non-standard parameterization.
        """
        return norm.weight.data.clone().float()

    def reset_norm_to_identity(self, norm: nn.Module) -> None:
        """Reset norm to identity scaling (gamma = 1).

        Standard RMSNorm: set weight = 1.0
        Gemma RMSNorm:    set weight = 0.0 (since gamma = 1 + weight)

        Uses nn.Parameter reassignment (not .data.fill_) so the change
        persists through accelerate's device_map dispatch.  See note in
        rotation.fuse_rms_norm_into_linear.
        """
        norm.weight = nn.Parameter(torch.ones_like(norm.weight.data))
