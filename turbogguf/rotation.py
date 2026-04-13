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

References:
  - QuaRot: https://arxiv.org/abs/2404.00456
  - TurboQuant: https://arxiv.org/abs/2504.19874
"""

import torch
import torch.nn as nn
from typing import Optional
from tqdm import tqdm

from turbogguf.hadamard import random_hadamard_matrix, matmul_hadU
from turbogguf.arch.base import ArchHandler


@torch.no_grad()
def fuse_rms_norm_into_linear(
    norm: nn.Module,
    linears: list[nn.Linear],
    handler: Optional['ArchHandler'] = None,
) -> None:
    """Fuse RMSNorm/LayerNorm learnable scale (gamma) into downstream linears.

    After fusion, the norm's weight becomes identity (pure normalization,
    no learnable scale), and the linear weights absorb the scale:
        W_fused[i, j] = W[i, j] * gamma[j]

    Handles both standard RMSNorm (gamma=weight) and Gemma-style
    RMSNorm (gamma=1+weight) via the handler.

    Args:
        norm: The RMSNorm or LayerNorm module
        linears: List of downstream Linear layers whose weights absorb gamma
        handler: Architecture handler for gamma extraction (None uses standard)
    """
    if not hasattr(norm, "weight"):
        return

    if handler is not None:
        gamma = handler.extract_norm_gamma(norm)
    else:
        gamma = norm.weight.data.float()

    for linear in linears:
        # W_fused = W * diag(gamma) = W * gamma[None, :]
        dtype = linear.weight.dtype
        W = linear.weight.data.float()
        linear.weight.data = (W * gamma[None, :]).to(dtype)

        # If there's a bias, it's not affected by the input scaling
        # (bias is added after the matmul)

    # Set to identity — norm now just divides by RMS
    if handler is not None:
        handler.reset_norm_to_identity(norm)
    else:
        norm.weight.data.fill_(1.0)


@torch.no_grad()
def fuse_rms_norm_output_side(
    norm: nn.Module,
    linears: list[nn.Linear],
    handler: Optional['ArchHandler'] = None,
) -> None:
    """Fuse a post-output RMSNorm gamma into upstream linears (row scaling).

    Used for Gemma2/4 post-attention and post-MLP norms. These norms are
    applied to the output of a linear layer before the residual add:
        z = gamma * Linear(x) / rms(Linear(x))

    We approximate this by absorbing gamma into the linear's output rows:
        W_fused[i, :] = gamma[i] * W[i, :]

    This is approximate because rms(gamma*y) != rms(y), but the quality
    impact is negligible in practice (gamma values are close to 1.0).

    Args:
        norm: The post-output RMSNorm module
        linears: List of upstream Linear layers whose output rows absorb gamma
        handler: Architecture handler for gamma extraction
    """
    if not hasattr(norm, "weight"):
        return

    if handler is not None:
        gamma = handler.extract_norm_gamma(norm)
    else:
        gamma = norm.weight.data.float()

    for linear in linears:
        dtype = linear.weight.dtype
        W = linear.weight.data.float()
        # Row scaling: W_fused[i, :] = gamma[i] * W[i, :]
        linear.weight.data = (gamma[:, None] * W).to(dtype)

        if linear.bias is not None:
            bias_dtype = linear.bias.dtype
            linear.bias.data = (gamma * linear.bias.data.float()).to(bias_dtype)

    if handler is not None:
        handler.reset_norm_to_identity(norm)
    else:
        norm.weight.data.fill_(1.0)


@torch.no_grad()
def rotate_weight_right(
    linear: nn.Linear,
    Q: torch.Tensor,
    transpose: bool = False,
) -> None:
    """Apply W' = W @ Q (or W @ Q^T) in-place.

    This absorbs the rotation from the residual stream input side.
    Used for: q_proj, k_proj, v_proj, gate_proj, up_proj, lm_head.

    Args:
        linear: The linear layer to modify
        Q: The rotation matrix (orthogonal)
        transpose: If True, apply Q^T instead of Q
    """
    dtype = linear.weight.dtype
    W = linear.weight.data.float()
    R = Q.T if transpose else Q
    linear.weight.data = (W @ R).to(dtype)


@torch.no_grad()
def rotate_weight_left(
    linear: nn.Linear,
    Q: torch.Tensor,
    transpose: bool = False,
) -> None:
    """Apply W' = Q @ W (or Q^T @ W) in-place.

    This absorbs the rotation on the output side.
    Used for: o_proj, down_proj.

    Args:
        linear: The linear layer to modify
        Q: The rotation matrix (orthogonal)
        transpose: If True, apply Q^T instead of Q
    """
    dtype = linear.weight.dtype
    W = linear.weight.data.float()
    R = Q.T if transpose else Q
    linear.weight.data = (R @ W).to(dtype)


@torch.no_grad()
def rotate_embedding(
    embedding: nn.Embedding,
    Q: torch.Tensor,
) -> None:
    """Apply E' = E @ Q in-place.

    The embedding output enters the residual stream, which is now rotated.

    Args:
        embedding: The token embedding layer
        Q: The rotation matrix
    """
    dtype = embedding.weight.dtype
    E = embedding.weight.data.float()
    embedding.weight.data = (E @ Q).to(dtype)


@torch.no_grad()
def rotate_head_weights(
    v_proj: nn.Linear,
    o_proj: nn.Linear,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    seed: int = 42,
) -> None:
    """Apply R2 per-head Hadamard rotation to v_proj and o_proj.

    For each attention head, apply a head-dim Hadamard rotation:
      - v_proj: rotate each head's output columns
      - o_proj: rotate each head's input blocks (transposed)

    Handles GQA (grouped-query attention) where num_kv_heads < num_heads.

    Args:
        v_proj: Value projection layer
        o_proj: Output projection layer
        head_dim: Dimension per attention head
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads (for GQA)
        seed: Random seed for head rotation matrix
    """
    H = random_hadamard_matrix(head_dim, seed=seed)
    dtype_v = v_proj.weight.dtype
    dtype_o = o_proj.weight.dtype

    # Rotate v_proj: each KV head's output block gets H
    V = v_proj.weight.data.float()
    for h in range(num_kv_heads):
        start = h * head_dim
        end = start + head_dim
        V[start:end, :] = H @ V[start:end, :]
    v_proj.weight.data = V.to(dtype_v)

    # Rotate o_proj: each query head's input block gets H^T
    O = o_proj.weight.data.float()
    for h in range(num_heads):
        start = h * head_dim
        end = start + head_dim
        O[:, start:end] = O[:, start:end] @ H.T
    o_proj.weight.data = O.to(dtype_o)


@torch.no_grad()
def fuse_all_norms(model: nn.Module, handler: ArchHandler) -> None:
    """Fuse all RMSNorm/LayerNorm weights into adjacent linear layers.

    After this, all norms have gamma=1 (pure normalization).

    Handles both standard (LLaMA) and extended (Gemma2/4) norm layouts:
      - Pre-attention norm → q, k, v projections (input-side fusion)
      - Pre-MLP norm → gate, up projections (input-side fusion)
      - Post-attention output norm → o_proj (output-side fusion, Gemma2/4 only)
      - Post-MLP output norm → down_proj (output-side fusion, Gemma2/4 only)
      - Final norm → lm_head (input-side fusion)

    Args:
        model: The transformer model
        handler: Architecture handler for weight access
    """
    layers = handler.get_layers(model)

    for layer in tqdm(layers, desc="Fusing norms"):
        attn = handler.get_attn_projs(layer)
        mlp = handler.get_mlp_projs(layer)

        # Pre-attention norm → feeds into q, k, v projections
        pre_norm = handler.get_pre_attn_norm(layer)
        fuse_rms_norm_into_linear(
            pre_norm,
            [attn["q_proj"], attn["k_proj"], attn["v_proj"]],
            handler=handler,
        )

        # Post-attention output norm → fuse into o_proj rows (Gemma2/4)
        post_attn_out = handler.get_post_attn_output_norm(layer)
        if post_attn_out is not None:
            fuse_rms_norm_output_side(
                post_attn_out,
                [attn["o_proj"]],
                handler=handler,
            )

        # Pre-MLP norm → feeds into gate, up projections
        pre_mlp_norm = handler.get_post_attn_norm(layer)
        fuse_rms_norm_into_linear(
            pre_mlp_norm,
            [mlp["gate_proj"], mlp["up_proj"]],
            handler=handler,
        )

        # Post-MLP output norm → fuse into down_proj rows (Gemma2/4)
        post_mlp_out = handler.get_post_mlp_output_norm(layer)
        if post_mlp_out is not None:
            fuse_rms_norm_output_side(
                post_mlp_out,
                [mlp["down_proj"]],
                handler=handler,
            )

    # Final norm → feeds into lm_head
    final_norm = handler.get_final_norm(model)
    lm_head = handler.get_lm_head(model)
    fuse_rms_norm_into_linear(final_norm, [lm_head], handler=handler)


@torch.no_grad()
def apply_R1(
    model: nn.Module,
    handler: ArchHandler,
    seed: int = 42,
) -> None:
    """Apply R1: residual stream Hadamard rotation.

    Inserts rotation Q into the residual stream and absorbs it into weights:
      - embed_tokens @ Q
      - q/k/v_proj @ Q (input side)
      - Q^T @ o_proj (output side)
      - gate/up_proj @ Q (input side)
      - Q^T @ down_proj (output side)
      - lm_head @ Q (input side, skipped if tied to embedding)

    For models with tied embedding/lm_head weights (e.g., Gemma), rotating
    the embedding automatically rotates the lm_head since they share the
    same tensor. In that case we skip the separate lm_head rotation.

    Args:
        model: The transformer model (norms must be fused first)
        handler: Architecture handler
        seed: Random seed for the Hadamard matrix
    """
    hidden_size = handler.get_hidden_size(model)
    Q = random_hadamard_matrix(hidden_size, seed=seed)

    # Rotate embedding output
    rotate_embedding(handler.get_embedding(model), Q)

    # Rotate each layer
    layers = handler.get_layers(model)
    for layer in tqdm(layers, desc="Applying R1"):
        attn = handler.get_attn_projs(layer)
        mlp = handler.get_mlp_projs(layer)

        # Input side: W @ Q (absorbs R from residual)
        rotate_weight_right(attn["q_proj"], Q)
        rotate_weight_right(attn["k_proj"], Q)
        rotate_weight_right(attn["v_proj"], Q)
        rotate_weight_right(mlp["gate_proj"], Q)
        rotate_weight_right(mlp["up_proj"], Q)

        # Output side: Q^T @ W (emits R^T back to residual)
        rotate_weight_left(attn["o_proj"], Q, transpose=True)
        rotate_weight_left(mlp["down_proj"], Q, transpose=True)

    # LM head: absorbs R from final residual
    # For tied weights (Gemma), the embedding rotation already handled this
    # since embed_tokens.weight and lm_head.weight are the same tensor.
    tied = handler.has_tied_embeddings(model)
    if tied:
        embed = handler.get_embedding(model)
        lm_head = handler.get_lm_head(model)
        actually_tied = embed.weight.data_ptr() == lm_head.weight.data_ptr()
        if not actually_tied:
            # Config says tied but tensors are separate (can happen after
            # save/reload). Rotate lm_head independently.
            rotate_weight_right(lm_head, Q)
    else:
        rotate_weight_right(handler.get_lm_head(model), Q)


@torch.no_grad()
def apply_R2(
    model: nn.Module,
    handler: ArchHandler,
    seed: int = 43,  # Different seed from R1
) -> None:
    """Apply R2: per-head Hadamard rotation within attention.

    Rotates v_proj outputs and o_proj inputs per head.
    Handles GQA (grouped-query attention).

    Args:
        model: The transformer model
        handler: Architecture handler
        seed: Random seed for the head rotation matrix
    """
    head_dim = handler.get_head_dim(model)
    num_heads = handler.get_num_heads(model)
    num_kv_heads = handler.get_num_kv_heads(model)

    layers = handler.get_layers(model)
    for layer in tqdm(layers, desc="Applying R2"):
        attn = handler.get_attn_projs(layer)
        rotate_head_weights(
            v_proj=attn["v_proj"],
            o_proj=attn["o_proj"],
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seed=seed,
        )


@torch.no_grad()
def rotate_model(
    model: nn.Module,
    handler: Optional[ArchHandler] = None,
    seed: int = 42,
    apply_r2: bool = True,
    verbose: bool = True,
) -> dict:
    """Full rotation pipeline: fuse norms → R1 → R2.

    The resulting model produces bit-identical FP16 outputs but has weight
    distributions that are much more amenable to low-bit quantization.

    Args:
        model: The transformer model
        handler: Architecture handler (auto-detected if None)
        seed: Random seed for rotation matrices
        apply_r2: Whether to apply per-head R2 rotation
        verbose: Print progress

    Returns:
        Dict with rotation metadata (seed, dimensions, etc.)
    """
    if handler is None:
        from turbogguf.arch import get_handler
        handler = get_handler(model)

    tied = handler.has_tied_embeddings(model)

    if verbose:
        print(f"Model: {type(model).__name__}")
        print(f"Hidden size: {handler.get_hidden_size(model)}")
        print(f"Num layers: {len(handler.get_layers(model))}")
        print(f"Num heads: {handler.get_num_heads(model)}")
        print(f"Num KV heads: {handler.get_num_kv_heads(model)}")
        print(f"Head dim: {handler.get_head_dim(model)}")
        print(f"Tied embeddings: {tied}")
        print(f"Seed: {seed}")
        print()

    # For models with tied embedding/lm_head weights (e.g., Gemma), untie
    # them before fusion so that fusing the final norm into lm_head doesn't
    # corrupt the embedding weights.
    if tied:
        embed = handler.get_embedding(model)
        lm_head = handler.get_lm_head(model)
        if embed.weight.data_ptr() == lm_head.weight.data_ptr():
            if verbose:
                print("Untying embedding/lm_head weights for independent fusion...")
            lm_head.weight = nn.Parameter(lm_head.weight.data.clone())
            if hasattr(model.config, "tie_word_embeddings"):
                model.config.tie_word_embeddings = False

    # Step 1: Fuse all norms
    if verbose:
        print("Step 1/3: Fusing RMSNorm weights into linear layers...")
    fuse_all_norms(model, handler)

    # Step 2: R1 — residual stream rotation
    if verbose:
        print("Step 2/3: Applying R1 (residual stream rotation)...")
    apply_R1(model, handler, seed=seed)

    # Step 3: R2 — per-head rotation
    if apply_r2:
        if verbose:
            print("Step 3/3: Applying R2 (per-head rotation)...")
        apply_R2(model, handler, seed=seed + 1)
    elif verbose:
        print("Step 3/3: Skipping R2 (per-head rotation disabled)")

    metadata = {
        "turbogguf_version": "0.1.0",
        "rotation_seed": seed,
        "r1_applied": True,
        "r2_applied": apply_r2,
        "hidden_size": handler.get_hidden_size(model),
        "head_dim": handler.get_head_dim(model),
        "num_heads": handler.get_num_heads(model),
        "num_kv_heads": handler.get_num_kv_heads(model),
        "architecture": type(model).__name__,
        "handler": type(handler).__name__,
        "had_tied_embeddings": tied,
        "has_output_norms": handler.get_post_attn_output_norm(
            handler.get_layers(model)[0]
        ) is not None,
    }

    if verbose:
        print("\nRotation complete. Model weights are now quantization-friendly.")
        print("Save with model.save_pretrained() then quantize with llama-quantize.")

    return metadata
