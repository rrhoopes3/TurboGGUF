"""Qwen3 MoE architecture handler (Qwen3MoE / Qwen3.5 MoE / Qwen3.6 MoE).

Covers the HuggingFace `Qwen3MoeForCausalLM` and `Qwen3_5MoeForCausalLM`
families.  These share the same layer skeleton (LLaMA-style pre-norm +
GQA attention with per-head QK-norm), but the FFN block is a sparse MoE:

    layer.mlp.experts.gate_up_proj  (num_experts, 2*I, H)   # fused gate+up
    layer.mlp.experts.down_proj     (num_experts, H, I)
    layer.mlp.gate.weight           (num_experts, H)        # router

Qwen3.5 MoE additionally has an always-on shared expert:

    layer.mlp.shared_expert.{gate,up,down}_proj  # standard nn.Linear
    layer.mlp.shared_expert_gate                 # nn.Linear(H, 1)

Some layers (config.mlp_only_layers) use a dense Qwen3MoeMLP instead.
Those are handled by the inherited Qwen2Handler dense path.

Attention has QK-norm (q_norm, k_norm on head_dim applied after
q_proj/k_proj).  These do not interact with R1 (residual stream) or R2
(per-head v/o), so they are left untouched.
"""

from typing import Dict
import torch
import torch.nn as nn
from turbogguf.arch.qwen2 import Qwen2Handler


def _norm_uses_1_plus_weight(norm: nn.Module) -> bool:
    """Whether an RMSNorm module parameterizes gamma as (1 + weight).

    Qwen 3.5 MoE (and beyond) follow Gemma's convention: weight is initialized
    to zero and the effective scale is (1 + weight).  Plain Qwen3 MoE uses the
    standard T5/LLaMA RMSNorm with weight initialized to 1.  We detect this at
    rotation time by class name — every model in this family advertises it
    clearly (Qwen3_5MoeRMSNorm, Qwen3_6MoeRMSNorm, GemmaRMSNorm, ...).
    """
    name = type(norm).__name__
    return (
        name.startswith("Qwen3_5")
        or name.startswith("Qwen3_6")
        or name.startswith("Qwen3_7")
        or "Gemma" in name
    )


class Qwen3MoeHandler(Qwen2Handler):
    """Handler for Qwen3 / Qwen3.5 / Qwen3.6 MoE models.

    Also handles the multimodal wrapper `Qwen3_5MoeForConditionalGeneration`:
    unwraps through `model.model.language_model` to reach the text decoder.
    The vision encoder (model.model.visual) is left untouched — rotation is
    text-only, which is fine because llama.cpp's Qwen3.5 MoE inference is
    text-only anyway.

    Also handles the hybrid linear/full-attention layout (Qwen 3.5/3.6): when
    a layer uses GatedDeltaNet (`layer.linear_attn`) instead of standard
    attention (`layer.self_attn`), R1 rotation is applied to the DeltaNet
    input/output projections rather than to q/k/v/o_proj.
    """

    def _text(self, model: nn.Module) -> nn.Module:
        """Navigate to the text decoder, handling multimodal wrappers."""
        inner = getattr(model, "model", model)
        # Multimodal wrapper: model.model.language_model is the TextModel
        if hasattr(inner, "language_model"):
            return inner.language_model
        # CausalLM: model.model IS the TextModel
        if hasattr(inner, "layers"):
            return inner
        # Already the text model
        if hasattr(model, "layers"):
            return model
        raise AttributeError(
            f"Cannot find Qwen3 MoE text decoder in {type(model).__name__}"
        )

    def _text_config(self, model: nn.Module):
        # Prefer the text model's own config — for multimodal wrappers it's
        # already the text_config, for CausalLM it's the top-level config
        # (which holds the actually-populated model shape).  Falling back to
        # cfg.text_config on a CausalLM would read stale defaults since the
        # top-level __init__ doesn't propagate overrides there.
        text = self._text(model)
        return getattr(text, "config", model.config)

    def has_bias(self) -> bool:
        return False

    def get_embedding(self, model: nn.Module) -> nn.Embedding:
        return self._text(model).embed_tokens

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        return self._text(model).layers

    def get_final_norm(self, model: nn.Module) -> nn.Module:
        return self._text(model).norm

    def get_lm_head(self, model: nn.Module):
        head = getattr(model, "lm_head", None)
        if head is None:
            return None
        emb = self._text(model).embed_tokens
        if head.weight.data_ptr() == emb.weight.data_ptr():
            return None  # tied — caller will un-tie via fuse_all_norms
        return head

    def get_tied_lm_head_module(self, model: nn.Module):
        return getattr(model, "lm_head", None)

    def get_hidden_size(self, model: nn.Module) -> int:
        return self._text_config(model).hidden_size

    def get_head_dim(self, model: nn.Module) -> int:
        cfg = self._text_config(model)
        if hasattr(cfg, "head_dim"):
            return cfg.head_dim
        return cfg.hidden_size // cfg.num_attention_heads

    def get_num_heads(self, model: nn.Module) -> int:
        return self._text_config(model).num_attention_heads

    def get_num_kv_heads(self, model: nn.Module) -> int:
        cfg = self._text_config(model)
        return getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)

    def is_linear_attention_layer(self, layer: nn.Module) -> bool:
        return hasattr(layer, "linear_attn")

    def get_linear_attn_projs(self, layer: nn.Module):
        if not self.is_linear_attention_layer(layer):
            return None
        la = layer.linear_attn
        # Qwen3.5/3.6 GatedDeltaNet has 4 parallel input projections whose
        # input is the residual hidden dim, and a single out_proj back to it.
        # Internal state (conv1d, dt_bias, A_log, norm) does not touch the
        # rotated basis and is left untouched by R1.
        in_projs = []
        for name in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]:
            p = getattr(la, name, None)
            if p is not None:
                in_projs.append(p)
        return {"in_projs": in_projs, "out_proj": la.out_proj}

    def get_attn_projs(self, layer: nn.Module) -> Dict[str, nn.Linear]:
        # Linear-attention layers have no self_attn — return all-None so the
        # standard attention path is skipped.  Callers must dispatch on
        # is_linear_attention_layer() first.
        if self.is_linear_attention_layer(layer):
            return {"q_proj": None, "k_proj": None, "v_proj": None, "o_proj": None}
        return super().get_attn_projs(layer)

    def is_moe_layer(self, layer: nn.Module) -> bool:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return False
        experts = getattr(mlp, "experts", None)
        return experts is not None and hasattr(experts, "gate_up_proj")

    def get_moe(self, layer: nn.Module):
        if not self.is_moe_layer(layer):
            return None
        mlp = layer.mlp
        result = {
            "experts": mlp.experts,
            "router": mlp.gate,
            "shared_expert": None,
            "shared_expert_gate": None,
        }
        shared = getattr(mlp, "shared_expert", None)
        if shared is not None:
            result["shared_expert"] = {
                "gate_proj": shared.gate_proj,
                "up_proj": shared.up_proj,
                "down_proj": shared.down_proj,
            }
        seg = getattr(mlp, "shared_expert_gate", None)
        if seg is not None:
            result["shared_expert_gate"] = seg
        return result

    def get_mlp_projs(self, layer: nn.Module) -> Dict[str, nn.Linear]:
        if self.is_moe_layer(layer):
            return None
        return super().get_mlp_projs(layer)

    def has_tied_lm_head(self, model: nn.Module) -> bool:
        cfg = model.config
        if getattr(cfg, "tie_word_embeddings", False):
            return True
        tc = getattr(cfg, "text_config", None)
        if tc is not None and getattr(tc, "tie_word_embeddings", False):
            return True
        head = getattr(model, "lm_head", None)
        if head is None:
            return True
        emb = self._text(model).embed_tokens
        return head.weight.data_ptr() == emb.weight.data_ptr()

    def extract_norm_gamma(self, norm: nn.Module) -> torch.Tensor:
        if _norm_uses_1_plus_weight(norm):
            return (1.0 + norm.weight.data.float()).clone()
        return norm.weight.data.clone().float()

    def reset_norm_to_identity(self, norm: nn.Module) -> None:
        if _norm_uses_1_plus_weight(norm):
            norm.weight = nn.Parameter(torch.zeros_like(norm.weight.data))
        else:
            norm.weight = nn.Parameter(torch.ones_like(norm.weight.data))
