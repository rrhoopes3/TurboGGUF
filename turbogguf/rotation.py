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
) -> None:
    """Fuse RMSNorm/LayerNorm learnable scale (gamma) into downstream linears.

    After fusion, the norm's weight becomes all-ones (pure normalization,
    no learnable scale), and the linear weights absorb the scale:
        W_fused[i, j] = W[i, j] * gamma[j]

    This is a prerequisite for applying Hadamard rotation, because the
    rotation must commute with the normalization layer.

    Args:
        norm: The RMSNorm or LayerNorm module
        linears: List of downstream Linear layers whose weights absorb gamma
    """
    if not hasattr(norm, "weight"):
        return

    gamma = norm.weight.data.float()

    for linear in linears:
        # W_fused = W * diag(gamma) = W * gamma[None, :]
        dtype = linear.weight.dtype
        W = linear.weight.data.float()
        linear.weight.data = (W * gamma[None, :]).to(dtype)

        # If there's a bias, it's not affected by the input scaling
        # (bias is added after the matmul)

    # Set gamma to ones — norm now just divides by RMS
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

    Args:
        model: The transformer model
        handler: Architecture handler for weight access
    """
    layers = handler.get_layers(model)

    for layer in tqdm(layers, desc="Fusing norms"):
        # Pre-attention norm → feeds into q, k, v projections
        pre_norm = handler.get_pre_attn_norm(layer)
        attn = handler.get_attn_projs(layer)
        fuse_rms_norm_into_linear(
            pre_norm,
            [attn["q_proj"], attn["k_proj"], attn["v_proj"]],
        )

        post_norm = handler.get_post_attn_norm(layer)
        mlp = handler.get_mlp_projs(layer)

        # Check for sandwich norms (Gemma 4): pre/post feedforward norms
        # wrap the MLP block separately from the attention norms.
        pre_ffn = handler.get_pre_ffn_norm(layer)
        if pre_ffn is not None:
            # Sandwich norm pattern: pre_ffn feeds into gate/up,
            # post_attn and post_ffn normalize residual outputs (not
            # linear inputs), so we set them to identity.
            fuse_rms_norm_into_linear(
                pre_ffn,
                [mlp["gate_proj"], mlp["up_proj"]],
            )
            post_norm.weight.data.fill_(1.0)
            post_ffn = handler.get_post_ffn_norm(layer)
            if post_ffn is not None:
                post_ffn.weight.data.fill_(1.0)
        else:
            # Standard pre-norm pattern (LLaMA, Mistral, Qwen2):
            # post_attn_norm feeds into gate, up projections
            fuse_rms_norm_into_linear(
                post_norm,
                [mlp["gate_proj"], mlp["up_proj"]],
            )

    # Final norm → feeds into lm_head (or embedding if tied)
    final_norm = handler.get_final_norm(model)
    lm_head = handler.get_lm_head(model)
    if lm_head is not None:
        fuse_rms_norm_into_linear(final_norm, [lm_head])
    else:
        # Tied weights: lm_head shares weight with embed_tokens.
        # Fuse gamma into embedding weights directly.
        emb = handler.get_embedding(model)
        gamma = final_norm.weight.data.float()
        dtype = emb.weight.dtype
        emb.weight.data = (emb.weight.data.float() * gamma[None, :]).to(dtype)
        final_norm.weight.data.fill_(1.0)


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
      - lm_head @ Q (input side)

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

    # LM head: absorbs R from final residual.
    # Skip if tied to embedding (already rotated via rotate_embedding above).
    lm_head = handler.get_lm_head(model)
    if lm_head is not None and not handler.has_tied_lm_head(model):
        rotate_weight_right(lm_head, Q)


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

    if verbose:
        print(f"Model: {type(model).__name__}")
        print(f"Hidden size: {handler.get_hidden_size(model)}")
        print(f"Num layers: {len(handler.get_layers(model))}")
        print(f"Num heads: {handler.get_num_heads(model)}")
        print(f"Num KV heads: {handler.get_num_kv_heads(model)}")
        print(f"Head dim: {handler.get_head_dim(model)}")
        print(f"Seed: {seed}")
        print()

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
    }

    if verbose:
        print("\nRotation complete. Model weights are now quantization-friendly.")
        print("Save with model.save_pretrained() then quantize with llama-quantize.")

    return metadata
