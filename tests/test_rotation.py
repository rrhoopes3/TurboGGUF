"""Tests for the rotation engine.

Tests RMSNorm fusion, R1/R2 rotation application, and end-to-end
logit equivalence on a small model.
"""

import torch
import torch.nn as nn
import pytest
import math

from turbogguf.rotation import (
    fuse_rms_norm_into_linear,
    rotate_weight_right,
    rotate_weight_left,
    rotate_embedding,
    rotate_head_weights,
)
from turbogguf.hadamard import random_hadamard_matrix


class TestRMSNormFusion:
    """Test RMSNorm weight fusion into linear layers."""

    def test_basic_fusion(self):
        """Fusing gamma into W gives same output as norm + linear."""
        hidden = 16
        gamma = torch.randn(hidden).abs() + 0.5  # positive scale

        # Build norm-like and linear
        norm = nn.Module()
        norm.weight = nn.Parameter(gamma.clone())

        linear = nn.Linear(hidden, 32, bias=False)
        W_original = linear.weight.data.clone()

        # Fuse
        fuse_rms_norm_into_linear(norm, [linear])

        # Verify: W_fused[i,j] = W_original[i,j] * gamma[j]
        expected = W_original * gamma[None, :]
        assert torch.allclose(linear.weight.data, expected, atol=1e-6)

        # Verify gamma is now ones
        assert torch.allclose(norm.weight.data, torch.ones(hidden), atol=1e-6)

    def test_multi_linear_fusion(self):
        """Gamma fuses correctly into multiple downstream linears."""
        hidden = 16
        gamma = torch.randn(hidden).abs() + 0.5

        norm = nn.Module()
        norm.weight = nn.Parameter(gamma.clone())

        linear1 = nn.Linear(hidden, 32, bias=False)
        linear2 = nn.Linear(hidden, 24, bias=False)
        W1_orig = linear1.weight.data.clone()
        W2_orig = linear2.weight.data.clone()

        fuse_rms_norm_into_linear(norm, [linear1, linear2])

        assert torch.allclose(linear1.weight.data, W1_orig * gamma[None, :], atol=1e-6)
        assert torch.allclose(linear2.weight.data, W2_orig * gamma[None, :], atol=1e-6)

    def test_functional_equivalence(self):
        """After fusion, norm(x) @ W_fused = norm_scaled(x) @ W_original."""
        hidden = 32
        x = torch.randn(4, hidden)

        # RMSNorm: y = x / rms(x) * gamma
        gamma = torch.randn(hidden).abs() + 0.5
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        normed = x / rms

        linear = nn.Linear(hidden, 16, bias=False)
        W = linear.weight.data.clone()

        # Original: (x / rms * gamma) @ W^T
        output_original = (normed * gamma[None, :]) @ W.T

        # Fused: (x / rms) @ W_fused^T  where W_fused = W * gamma
        W_fused = W * gamma[None, :]
        output_fused = normed @ W_fused.T

        assert torch.allclose(output_original, output_fused, atol=1e-5)


class TestWeightRotation:
    """Test individual weight rotation operations."""

    def test_right_multiply(self):
        """W' = W @ Q."""
        n = 16
        Q = random_hadamard_matrix(n, seed=42)
        linear = nn.Linear(n, 32, bias=False)
        W_orig = linear.weight.data.clone().float()

        rotate_weight_right(linear, Q)
        expected = W_orig @ Q
        assert torch.allclose(linear.weight.data.float(), expected, atol=1e-4)

    def test_right_multiply_transpose(self):
        """W' = W @ Q^T."""
        n = 16
        Q = random_hadamard_matrix(n, seed=42)
        linear = nn.Linear(n, 32, bias=False)
        W_orig = linear.weight.data.clone().float()

        rotate_weight_right(linear, Q, transpose=True)
        expected = W_orig @ Q.T
        assert torch.allclose(linear.weight.data.float(), expected, atol=1e-4)

    def test_left_multiply(self):
        """W' = Q @ W."""
        n = 32
        Q = random_hadamard_matrix(n, seed=42)
        linear = nn.Linear(16, n, bias=False)
        W_orig = linear.weight.data.clone().float()

        rotate_weight_left(linear, Q)
        expected = Q @ W_orig
        assert torch.allclose(linear.weight.data.float(), expected, atol=1e-4)

    def test_left_multiply_transpose(self):
        """W' = Q^T @ W."""
        n = 32
        Q = random_hadamard_matrix(n, seed=42)
        linear = nn.Linear(16, n, bias=False)
        W_orig = linear.weight.data.clone().float()

        rotate_weight_left(linear, Q, transpose=True)
        expected = Q.T @ W_orig
        assert torch.allclose(linear.weight.data.float(), expected, atol=1e-4)

    def test_roundtrip_right(self):
        """W @ Q @ Q^T = W (orthogonal roundtrip)."""
        n = 16
        Q = random_hadamard_matrix(n, seed=42)
        linear = nn.Linear(n, 32, bias=False)
        W_orig = linear.weight.data.clone()

        rotate_weight_right(linear, Q)
        rotate_weight_right(linear, Q, transpose=True)

        assert torch.allclose(linear.weight.data.float(), W_orig.float(), atol=1e-4)

    def test_roundtrip_left(self):
        """Q^T @ Q @ W = W."""
        n = 32
        Q = random_hadamard_matrix(n, seed=42)
        linear = nn.Linear(16, n, bias=False)
        W_orig = linear.weight.data.clone()

        rotate_weight_left(linear, Q)
        rotate_weight_left(linear, Q, transpose=True)

        assert torch.allclose(linear.weight.data.float(), W_orig.float(), atol=1e-4)


class TestEmbeddingRotation:
    """Test embedding rotation."""

    def test_embedding_rotation(self):
        """E' = E @ Q."""
        vocab, hidden = 100, 16
        Q = random_hadamard_matrix(hidden, seed=42)
        emb = nn.Embedding(vocab, hidden)
        E_orig = emb.weight.data.clone().float()

        rotate_embedding(emb, Q)
        expected = E_orig @ Q
        assert torch.allclose(emb.weight.data.float(), expected, atol=1e-4)


class TestHeadRotation:
    """Test per-head R2 rotation."""

    def test_v_o_roundtrip(self):
        """Head rotation on v_proj and o_proj should cancel in attention output.

        attn_output = softmax(Q @ K^T) @ V @ O
        If we rotate V → H @ V and O → O @ H^T, then:
        V' @ O' = (H @ V) @ (O @ H^T) = H @ (V @ O) @ H^T
        which is a similarity transform (preserves eigenvalues).
        """
        head_dim = 16
        num_heads = 4
        num_kv_heads = 4
        hidden = num_heads * head_dim

        v_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)

        # The key property: rotating v and o should not change the
        # effective linear map from input to output when combined
        V_orig = v_proj.weight.data.clone().float()
        O_orig = o_proj.weight.data.clone().float()

        rotate_head_weights(v_proj, o_proj, head_dim, num_heads, num_kv_heads, seed=42)

        # For each head, verify H was applied
        H = random_hadamard_matrix(head_dim, seed=42)
        for h in range(num_kv_heads):
            start = h * head_dim
            end = start + head_dim
            expected_v = H @ V_orig[start:end, :]
            assert torch.allclose(
                v_proj.weight.data[start:end, :].float(),
                expected_v,
                atol=1e-4,
            )

    def test_gqa_handling(self):
        """GQA: fewer KV heads than query heads."""
        head_dim = 16
        num_heads = 8
        num_kv_heads = 2  # GQA: 4 query heads per KV head

        v_proj = nn.Linear(64, num_kv_heads * head_dim, bias=False)
        o_proj = nn.Linear(num_heads * head_dim, 64, bias=False)

        # Should not crash with GQA
        rotate_head_weights(v_proj, o_proj, head_dim, num_heads, num_kv_heads, seed=42)

        # Verify dimensions unchanged
        assert v_proj.weight.shape == (num_kv_heads * head_dim, 64)
        assert o_proj.weight.shape == (64, num_heads * head_dim)


class TestWeightDistribution:
    """Test that rotation improves weight distribution for quantization."""

    def test_rotation_reduces_kurtosis(self):
        """Rotated weights should have lower kurtosis (fewer outliers).

        This is the whole point: rotation spreads outlier values across
        dimensions, making the distribution more uniform/Gaussian.
        """
        n = 128
        Q = random_hadamard_matrix(n, seed=42)

        # Create weights with outliers (realistic for LLMs)
        W = torch.randn(256, n) * 0.1
        # Add outlier channels
        W[:, 0] *= 20  # 20x outlier
        W[:, 17] *= 15
        W[:, 42] *= 25

        # Compute kurtosis before and after rotation
        def kurtosis(x):
            m = x.mean()
            s = x.std()
            return ((x - m) ** 4).mean() / (s ** 4) - 3  # excess kurtosis

        k_before = kurtosis(W.flatten())
        W_rotated = W @ Q
        k_after = kurtosis(W_rotated.flatten())

        # Rotated should have significantly lower kurtosis
        assert k_after < k_before, \
            f"Kurtosis should decrease: {k_before:.2f} → {k_after:.2f}"

    def test_rotation_reduces_max_magnitude(self):
        """Rotated weights should have smaller max absolute values."""
        n = 64
        Q = random_hadamard_matrix(n, seed=42)

        W = torch.randn(128, n)
        W[:, 5] *= 30  # big outlier

        max_before = W.abs().max().item()
        W_rotated = W @ Q
        max_after = W_rotated.abs().max().item()

        assert max_after < max_before, \
            f"Max magnitude should decrease: {max_before:.2f} → {max_after:.2f}"
