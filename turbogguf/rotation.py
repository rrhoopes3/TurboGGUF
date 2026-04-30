"""Core rotation engine: fuse RMSNorm + apply R1/R2 Hadamard rotations.

The rotation is absorbed into model weights offline, producing a mathematically
equivalent model whose weight distributions are more uniform and thus more
amenable to low-bit quantization.

Theory:
  - RMSNorm is rotation-invariant: ||Rx|| = ||x|| for orthogonal R
  - Fuse gamma into adjacent weights, then insert rotation R between layers
  - R gets absorbed: embed @ R, W @ R (inputs), R^T @ W (outputs)
  - At each layer boundary R^T @ R = I, so the chain is identity
  - Result: bit-identical FP16 outputs, but weights are now "spread out"

Precision: the pipeline upcasts every parameter to fp32 before fusing/rotating
and only casts back to the original storage dtype after R1+R2 are complete.
Per-tensor round-trips (dtype -> fp32 -> matmul -> dtype, repeated across norm
fusion + R1 + R2 + bias rotation) accumulated bf16 ulps and were the source of
the LLaMA-family drift that prompted this refactor.

References:
  - QuaRot: https://arxiv.org/abs/2404.00456
  - TurboQuant: https://arxiv.org/abs/2504.19874
"""

from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from turbogguf.arch.base import ArchHandler
from turbogguf.hadamard import random_hadamard_matrix


def _collect_param_dtypes(model: nn.Module) -> dict[str, torch.dtype]:
    """Snapshot every parameter's storage dtype keyed by qualified name."""
    return {name: p.dtype for name, p in model.named_parameters()}


def _cast_all_params(model: nn.Module, target_dtype: torch.dtype) -> int:
    """Cast every parameter (and registered buffer that's float-like) to target_dtype.

    Uses nn.Parameter rebinding so the change survives accelerate dispatch
    hook removal and downstream save_pretrained, matching the pattern used
    elsewhere in this file. Returns the number of parameters cast.
    """
    cast_count = 0
    for module in model.modules():
        for attr_name, param in list(module._parameters.items()):
            if param is None or param.dtype == target_dtype:
                continue
            if not param.dtype.is_floating_point:
                continue
            module._parameters[attr_name] = nn.Parameter(
                param.data.to(target_dtype),
                requires_grad=param.requires_grad,
            )
            cast_count += 1
    return cast_count


def _restore_param_dtypes(
    model: nn.Module,
    original_dtypes: dict[str, torch.dtype],
) -> int:
    """Cast each parameter back to the dtype recorded by _collect_param_dtypes."""
    restored = 0
    for name, param in list(model.named_parameters()):
        target = original_dtypes.get(name)
        if target is None or param.dtype == target:
            continue
        parent_path, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        parent._parameters[attr] = nn.Parameter(
            param.data.to(target),
            requires_grad=param.requires_grad,
        )
        restored += 1
    return restored


@torch.no_grad()
def fuse_rms_norm_into_linear(
    norm: nn.Module,
    linears: list[nn.Linear],
    handler: Optional[ArchHandler] = None,
) -> None:
    """Fuse RMSNorm/LayerNorm learnable scale (gamma) into downstream linears."""
    if not hasattr(norm, "weight"):
        return

    if handler is not None:
        gamma = handler.extract_norm_gamma(norm)
    else:
        gamma = norm.weight.data.float()

    for linear in linears:
        dtype = linear.weight.dtype
        W = linear.weight.data.float()
        g = gamma.to(W.device)
        linear.weight = nn.Parameter((W * g[None, :]).to(dtype))

    if handler is not None:
        handler.reset_norm_to_identity(norm)
    else:
        norm.weight = nn.Parameter(torch.ones_like(norm.weight.data))


@torch.no_grad()
def fuse_norm_into_linear_output(
    norm: nn.Module,
    linears: list[nn.Linear],
    handler: Optional[ArchHandler] = None,
) -> None:
    """Fuse RMSNorm/LayerNorm gamma into upstream linears (output-side scaling)."""
    if not hasattr(norm, "weight"):
        return

    if handler is not None:
        gamma = handler.extract_norm_gamma(norm)
    else:
        gamma = norm.weight.data.float()

    for linear in linears:
        dtype = linear.weight.dtype
        W = linear.weight.data.float()
        g = gamma.to(W.device)
        linear.weight = nn.Parameter((W * g[:, None]).to(dtype))

        if linear.bias is not None:
            bias_dtype = linear.bias.dtype
            linear.bias = nn.Parameter((g * linear.bias.data.float()).to(bias_dtype))

    if handler is not None:
        handler.reset_norm_to_identity(norm)
    else:
        norm.weight = nn.Parameter(torch.ones_like(norm.weight.data))


@torch.no_grad()
def rotate_weight_right(
    linear: nn.Linear,
    Q: torch.Tensor,
    transpose: bool = False,
) -> None:
    """Apply W' = W @ Q (or W @ Q^T) in-place."""
    dtype = linear.weight.dtype
    W = linear.weight.data.float()
    R = (Q.T if transpose else Q).to(W.device)
    linear.weight = nn.Parameter((W @ R).to(dtype))


@torch.no_grad()
def rotate_weight_left(
    linear: nn.Linear,
    Q: torch.Tensor,
    transpose: bool = False,
) -> None:
    """Apply W' = Q @ W (or Q^T @ W) in-place."""
    dtype = linear.weight.dtype
    W = linear.weight.data.float()
    R = (Q.T if transpose else Q).to(W.device)
    linear.weight = nn.Parameter((R @ W).to(dtype))


@torch.no_grad()
def rotate_embedding(
    embedding: nn.Embedding,
    Q: torch.Tensor,
) -> None:
    """Apply E' = E @ Q in-place."""
    dtype = embedding.weight.dtype
    E = embedding.weight.data.float()
    embedding.weight = nn.Parameter((E @ Q.to(E.device)).to(dtype))


@torch.no_grad()
def _scale_weight_input_dim(module, gamma: torch.Tensor, attr: str = "weight") -> None:
    """Multiply a 2D weight's input (last) dim by gamma."""
    w = getattr(module, attr)
    dtype = w.dtype
    W = w.data.float()
    g = gamma.to(W.device)
    setattr(module, attr, nn.Parameter((W * g[None, :]).to(dtype)))


@torch.no_grad()
def _scale_expert_weight_input_dim(owner, attr: str, gamma: torch.Tensor) -> None:
    """Multiply a 3D expert weight's last (input) dim by gamma."""
    w = getattr(owner, attr)
    dtype = w.dtype
    W = w.data.float()
    g = gamma.to(W.device)
    setattr(owner, attr, nn.Parameter((W * g[None, None, :]).to(dtype)))


@torch.no_grad()
def fuse_pre_ffn_norm_into_moe(norm: nn.Module, moe: dict, handler: ArchHandler) -> None:
    """Fuse pre-FFN RMSNorm gamma into every input-side weight of an MoE block."""
    if not hasattr(norm, "weight"):
        return
    gamma = handler.extract_norm_gamma(norm)

    experts = moe["experts"]
    _scale_expert_weight_input_dim(experts, "gate_up_proj", gamma)
    _scale_weight_input_dim(moe["router"], gamma)

    se = moe.get("shared_expert")
    if se is not None:
        _scale_weight_input_dim(se["gate_proj"], gamma)
        _scale_weight_input_dim(se["up_proj"], gamma)

    seg = moe.get("shared_expert_gate")
    if seg is not None:
        _scale_weight_input_dim(seg, gamma)

    handler.reset_norm_to_identity(norm)


@torch.no_grad()
def rotate_moe_R1(moe: dict, Q: torch.Tensor) -> None:
    """Apply the residual-stream rotation Q to every weight in an MoE block."""
    experts = moe["experts"]

    gup = experts.gate_up_proj
    dtype = gup.dtype
    W = gup.data.float()
    experts.gate_up_proj = nn.Parameter(torch.matmul(W, Q.to(W.device)).to(dtype))

    dp = experts.down_proj
    dtype = dp.dtype
    W = dp.data.float()
    experts.down_proj = nn.Parameter(torch.matmul(Q.T.to(W.device), W).to(dtype))

    rotate_weight_right(moe["router"], Q)

    se = moe.get("shared_expert")
    if se is not None:
        rotate_weight_right(se["gate_proj"], Q)
        rotate_weight_right(se["up_proj"], Q)
        rotate_weight_left(se["down_proj"], Q, transpose=True)

    seg = moe.get("shared_expert_gate")
    if seg is not None:
        rotate_weight_right(seg, Q)


@torch.no_grad()
def rotate_head_weights(
    v_proj: nn.Linear,
    o_proj: nn.Linear,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    seed: int = 42,
) -> None:
    """Apply R2 per-head Hadamard rotation to v_proj and o_proj."""
    H = random_hadamard_matrix(head_dim, seed=seed)
    dtype_v = v_proj.weight.dtype
    dtype_o = o_proj.weight.dtype

    V = v_proj.weight.data.float()
    Hv = H.to(V.device)
    for h in range(num_kv_heads):
        start = h * head_dim
        end = start + head_dim
        V[start:end, :] = Hv @ V[start:end, :]
    v_proj.weight = nn.Parameter(V.to(dtype_v))

    # Qwen-family attention uses a learned v_proj bias. Since R2 rotates the
    # value head output basis, the bias lives in that basis too and must be
    # rotated head-by-head with the same H.
    if v_proj.bias is not None:
        dtype_b = v_proj.bias.dtype
        B = v_proj.bias.data.float()
        Hb = H.to(B.device)
        for h in range(num_kv_heads):
            start = h * head_dim
            end = start + head_dim
            B[start:end] = Hb @ B[start:end]
        v_proj.bias = nn.Parameter(B.to(dtype_b))

    O = o_proj.weight.data.float()
    Ho = H.to(O.device)
    for h in range(num_heads):
        start = h * head_dim
        end = start + head_dim
        O[:, start:end] = O[:, start:end] @ Ho.T
    o_proj.weight = nn.Parameter(O.to(dtype_o))


@torch.no_grad()
def fuse_all_norms(model: nn.Module, handler: ArchHandler) -> None:
    """Fuse all RMSNorm/LayerNorm weights into adjacent linear layers."""
    layers = handler.get_layers(model)

    for layer in tqdm(layers, desc="Fusing norms"):
        pre_norm = handler.get_pre_attn_norm(layer)
        if handler.is_linear_attention_layer(layer):
            la = handler.get_linear_attn_projs(layer)
            fuse_rms_norm_into_linear(pre_norm, la["in_projs"], handler=handler)
        else:
            attn = handler.get_attn_projs(layer)
            attn_linears = [p for p in [attn["q_proj"], attn["k_proj"], attn["v_proj"]] if p is not None]
            fuse_rms_norm_into_linear(pre_norm, attn_linears, handler=handler)

        post_norm = handler.get_post_attn_norm(layer)

        if handler.is_moe_layer(layer):
            moe = handler.get_moe(layer)
            fuse_pre_ffn_norm_into_moe(post_norm, moe, handler)
            continue

        mlp = handler.get_mlp_projs(layer)
        pre_ffn = handler.get_pre_ffn_norm(layer)
        if pre_ffn is not None:
            fuse_rms_norm_into_linear(
                pre_ffn,
                [mlp["gate_proj"], mlp["up_proj"]],
                handler=handler,
            )
            fuse_norm_into_linear_output(post_norm, [attn["o_proj"]], handler=handler)

            post_ffn = handler.get_post_ffn_norm(layer)
            if post_ffn is not None:
                fuse_norm_into_linear_output(post_ffn, [mlp["down_proj"]], handler=handler)
        else:
            fuse_rms_norm_into_linear(
                post_norm,
                [mlp["gate_proj"], mlp["up_proj"]],
                handler=handler,
            )

    final_norm = handler.get_final_norm(model)
    lm_head = handler.get_lm_head(model)
    if lm_head is not None:
        fuse_rms_norm_into_linear(final_norm, [lm_head], handler=handler)
    elif handler.uses_tied_lm_head_for_gguf():
        pass
    else:
        outer_head = handler.get_tied_lm_head_module(model)
        if outer_head is not None:
            emb = handler.get_embedding(model)
            outer_head.weight = nn.Parameter(emb.weight.data.clone())
            fuse_rms_norm_into_linear(final_norm, [outer_head], handler=handler)

            if hasattr(model, "config"):
                model.config.tie_word_embeddings = False
                if hasattr(model.config, "text_config"):
                    model.config.text_config.tie_word_embeddings = False


@torch.no_grad()
def apply_R1(
    model: nn.Module,
    handler: ArchHandler,
    seed: int = 42,
) -> None:
    """Apply R1: residual stream Hadamard rotation."""
    hidden_size = handler.get_hidden_size(model)
    Q = random_hadamard_matrix(hidden_size, seed=seed)

    rotate_embedding(handler.get_embedding(model), Q)

    layers = handler.get_layers(model)
    for layer in tqdm(layers, desc="Applying R1"):
        if handler.is_linear_attention_layer(layer):
            la = handler.get_linear_attn_projs(layer)
            for proj in la["in_projs"]:
                rotate_weight_right(proj, Q)
            rotate_weight_left(la["out_proj"], Q, transpose=True)
        else:
            attn = handler.get_attn_projs(layer)
            for name in ["q_proj", "k_proj", "v_proj"]:
                if attn[name] is not None:
                    rotate_weight_right(attn[name], Q)
            rotate_weight_left(attn["o_proj"], Q, transpose=True)

        if handler.is_moe_layer(layer):
            rotate_moe_R1(handler.get_moe(layer), Q)
        else:
            mlp = handler.get_mlp_projs(layer)
            rotate_weight_right(mlp["gate_proj"], Q)
            rotate_weight_right(mlp["up_proj"], Q)
            rotate_weight_left(mlp["down_proj"], Q, transpose=True)

    if not handler.uses_tied_lm_head_for_gguf():
        lm_head = handler.get_tied_lm_head_module(model)
        if lm_head is not None:
            rotate_weight_right(lm_head, Q)


@torch.no_grad()
def apply_R2(
    model: nn.Module,
    handler: ArchHandler,
    seed: int = 43,
) -> None:
    """Apply R2: per-head Hadamard rotation within attention."""
    head_dim = handler.get_head_dim(model)
    num_heads = handler.get_num_heads(model)
    num_kv_heads = handler.get_num_kv_heads(model)

    layers = handler.get_layers(model)
    skipped_gated = 0
    skipped_linear = 0
    for layer in tqdm(layers, desc="Applying R2"):
        if handler.is_linear_attention_layer(layer):
            skipped_linear += 1
            continue

        attn = handler.get_attn_projs(layer)
        if attn["v_proj"] is None:
            continue

        expected_out = num_heads * head_dim
        if attn["q_proj"] is not None and attn["q_proj"].out_features > expected_out:
            skipped_gated += 1
            continue

        rotate_head_weights(
            v_proj=attn["v_proj"],
            o_proj=attn["o_proj"],
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seed=seed,
        )
    if skipped_gated:
        print(f"  (R2 skipped on {skipped_gated} gated-attention layer(s))")
    if skipped_linear:
        print(f"  (R2 skipped on {skipped_linear} linear-attention layer(s))")


@torch.no_grad()
def rotate_model(
    model: nn.Module,
    handler: Optional[ArchHandler] = None,
    seed: int = 42,
    apply_r2: bool = True,
    verbose: bool = True,
    rotation_precision: str = "fp32",
) -> dict:
    """Full rotation pipeline: fuse norms -> R1 -> R2.

    Args:
        rotation_precision: "fp32" (default) upcasts every weight to fp32 for
            the duration of fusion + R1 + R2 and casts back at the end. This
            eliminates the per-tensor round-trip drift that previously caused
            bf16 LLaMA models to deviate from the original logits. Set to
            "original" to keep the legacy per-helper round-trip behavior, e.g.
            when memory is tight on huge models — at the cost of measurable
            drift over many layers.
    """
    if handler is None:
        from turbogguf.arch import get_handler

        handler = get_handler(model)

    if rotation_precision not in ("fp32", "original"):
        raise ValueError(
            f"rotation_precision must be 'fp32' or 'original', got {rotation_precision!r}"
        )

    layers = handler.get_layers(model)
    moe_layers = [i for i, l in enumerate(layers) if handler.is_moe_layer(l)]
    linear_attn_layers = [
        i for i, l in enumerate(layers) if handler.is_linear_attention_layer(l)
    ]

    if verbose:
        print(f"Model: {type(model).__name__}")
        print(f"Hidden size: {handler.get_hidden_size(model)}")
        print(f"Num layers: {len(layers)}")
        print(f"Num heads: {handler.get_num_heads(model)}")
        print(f"Num KV heads: {handler.get_num_kv_heads(model)}")
        print(f"Head dim: {handler.get_head_dim(model)}")
        if moe_layers:
            first = handler.get_moe(layers[moe_layers[0]])
            num_experts = first["experts"].gate_up_proj.shape[0]
            has_shared = first["shared_expert"] is not None
            print(
                f"MoE: {len(moe_layers)}/{len(layers)} layers sparse, "
                f"{num_experts} experts/layer"
                f"{' + shared expert' if has_shared else ''}"
            )
            print(
                "  Warning: rotation benefit on MoE FFNs is smaller than on dense FFNs; "
                "expert weights share the residual rotation but each expert's outliers "
                "are not individually flattened."
            )
        if linear_attn_layers:
            print(
                f"Linear attention: {len(linear_attn_layers)}/{len(layers)} layers "
                f"(e.g. GatedDeltaNet/SSM). R2 will be skipped on these; R1 is applied."
            )
        print(f"Seed: {seed}")
        print(f"Rotation precision: {rotation_precision}")
        print()

    sample_param = next(model.parameters())
    storage_dtype = sample_param.dtype
    storage_dtype_str = str(storage_dtype).removeprefix("torch.")

    original_dtypes: dict[str, torch.dtype] = {}
    if rotation_precision == "fp32" and storage_dtype != torch.float32:
        if verbose:
            print(
                f"Upcasting weights {storage_dtype_str} -> float32 for rotation "
                "(prevents per-tensor round-trip drift)..."
            )
        original_dtypes = _collect_param_dtypes(model)
        cast_count = _cast_all_params(model, torch.float32)
        if verbose:
            print(f"  Upcast {cast_count} parameter tensor(s).")
            print()

    if verbose:
        if handler.has_tied_lm_head(model):
            if handler.uses_tied_lm_head_for_gguf():
                print("Note: lm_head tied to embedding - keeping tied (llama.cpp uses token_embd for output)")
            else:
                print("Note: lm_head is tied to embedding - will un-tie for clean fusion")
        print("Step 1/3: Fusing RMSNorm weights into linear layers...")
    fuse_all_norms(model, handler)

    if verbose:
        print("Step 2/3: Applying R1 (residual stream rotation)...")
    apply_R1(model, handler, seed=seed)

    if apply_r2:
        if verbose:
            print("Step 3/3: Applying R2 (per-head rotation)...")
        apply_R2(model, handler, seed=seed + 1)
    elif verbose:
        print("Step 3/3: Skipping R2 (per-head rotation disabled)")

    if original_dtypes:
        if verbose:
            print(
                f"Downcasting weights float32 -> {storage_dtype_str} for storage..."
            )
        restored = _restore_param_dtypes(model, original_dtypes)
        if verbose:
            print(f"  Downcast {restored} parameter tensor(s).")

    metadata = {
        "turbogguf_version": "0.1.0",
        "rotation_seed": seed,
        "r1_applied": True,
        "r2_applied": apply_r2,
        "rotation_precision": rotation_precision,
        "storage_dtype": storage_dtype_str,
        "hidden_size": handler.get_hidden_size(model),
        "head_dim": handler.get_head_dim(model),
        "num_heads": handler.get_num_heads(model),
        "num_kv_heads": handler.get_num_kv_heads(model),
        "num_layers": len(layers),
        "num_moe_layers": len(moe_layers),
        "num_linear_attn_layers": len(linear_attn_layers),
        "architecture": type(model).__name__,
    }

    if verbose:
        print("\nRotation complete. Model weights are now quantization-friendly.")
        print("Save with model.save_pretrained() then quantize with llama-quantize.")

    return metadata
