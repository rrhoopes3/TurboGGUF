"""Tests for Gemma architecture support.

Tests Gemma-specific features:
  1. (1+weight) RMSNorm gamma extraction and identity reset
  2. Post-attention/MLP output norm fusion (output-side, approximate)
  3. Tied embedding/lm_head weight handling
  4. End-to-end rotation on a mini Gemma2-like model with 4 norms per layer
"""

import torch
import torch.nn as nn
import pytest
import math

from turbogguf.rotation import (
    fuse_rms_norm_into_linear,
    fuse_rms_norm_output_side,
    rotate_weight_right,
    rotate_weight_left,
    rotate_embedding,
)
from turbogguf.hadamard import random_hadamard_matrix
from turbogguf.arch.gemma import GemmaHandler, Gemma2Handler


# ---------------------------------------------------------------------------
# Gemma-style RMSNorm: output = (1 + weight) * x / rms(x)
# ---------------------------------------------------------------------------

class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm where gamma = 1 + weight."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))  # gamma = 1 + 0 = 1
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (1.0 + self.weight) * x / rms


class StandardRMSNorm(nn.Module):
    """Standard RMSNorm for comparison."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


# ---------------------------------------------------------------------------
# Mini Gemma2 model with 4 norms per layer + tied weights
# ---------------------------------------------------------------------------

class MiniAttention(nn.Module):
    def __init__(self, hidden: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class MiniMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniGemma2Block(nn.Module):
    """Gemma2-style block with 4 norms: pre/post-attn + pre/post-MLP."""
    def __init__(self, hidden: int, num_heads: int, intermediate: int):
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(hidden)
        self.self_attn = MiniAttention(hidden, num_heads)
        self.post_attention_layernorm = GemmaRMSNorm(hidden)
        self.pre_feedforward_layernorm = GemmaRMSNorm(hidden)
        self.mlp = MiniMLP(hidden, intermediate)
        self.post_feedforward_layernorm = GemmaRMSNorm(hidden)

    def forward(self, x):
        # Attention with pre/post norms
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = self.post_attention_layernorm(x)
        x = residual + x

        # MLP with pre/post norms
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x
        return x


class MiniGemma2Model(nn.Module):
    """Mini Gemma2-like model for testing rotation."""
    def __init__(self, vocab: int = 100, hidden: int = 64, num_heads: int = 4,
                 num_layers: int = 2, intermediate: int = 128, tie_weights: bool = True):
        super().__init__()

        class _Config:
            tie_word_embeddings = tie_weights
            hidden_size = hidden
            num_attention_heads = num_heads
            num_key_value_heads = num_heads
            head_dim = hidden // num_heads
            num_hidden_layers = num_layers

        self.config = _Config()

        # Wrap inner model to match HF structure: model.model.layers
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([
            MiniGemma2Block(hidden, num_heads, intermediate)
            for _ in range(num_layers)
        ])
        self.model.norm = GemmaRMSNorm(hidden)

        self.lm_head = nn.Linear(hidden, vocab, bias=False)

        if tie_weights:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGemmaNormGamma:
    """Test Gemma (1+weight) norm gamma extraction and reset."""

    def test_extract_gamma_gemma_norm(self):
        """Gemma handler extracts gamma = 1 + weight."""
        handler = GemmaHandler()
        norm = GemmaRMSNorm(16)
        norm.weight.data = torch.full((16,), 0.5)  # gamma should be 1.5

        gamma = handler.extract_norm_gamma(norm)
        assert torch.allclose(gamma, torch.full((16,), 1.5))

    def test_extract_gamma_zero_init(self):
        """Default Gemma norm (weight=0) gives gamma=1."""
        handler = GemmaHandler()
        norm = GemmaRMSNorm(16)  # weight = 0

        gamma = handler.extract_norm_gamma(norm)
        assert torch.allclose(gamma, torch.ones(16))

    def test_reset_to_identity(self):
        """Reset sets weight=0 so gamma=1+0=1."""
        handler = GemmaHandler()
        norm = GemmaRMSNorm(16)
        norm.weight.data = torch.randn(16)

        handler.reset_norm_to_identity(norm)
        assert torch.allclose(norm.weight.data, torch.zeros(16))

    def test_fusion_with_gemma_norm(self):
        """Fusing Gemma norm into linear uses (1+weight) as gamma."""
        handler = GemmaHandler()
        hidden = 16
        norm = GemmaRMSNorm(hidden)
        norm.weight.data = torch.full((hidden,), 0.3)  # gamma = 1.3

        linear = nn.Linear(hidden, 32, bias=False)
        W_orig = linear.weight.data.clone()

        fuse_rms_norm_into_linear(norm, [linear], handler=handler)

        expected_gamma = torch.full((hidden,), 1.3)
        expected_W = W_orig * expected_gamma[None, :]
        assert torch.allclose(linear.weight.data, expected_W, atol=1e-6)
        assert torch.allclose(norm.weight.data, torch.zeros(hidden), atol=1e-6)

    def test_functional_equivalence_gemma_norm(self):
        """After fusion, Gemma norm(x) @ W_fused matches original computation."""
        hidden = 32
        handler = GemmaHandler()
        x = torch.randn(4, hidden)

        norm = GemmaRMSNorm(hidden)
        norm.weight.data = torch.randn(hidden) * 0.1  # small deviations from 1
        linear = nn.Linear(hidden, 16, bias=False)

        # Original output
        with torch.no_grad():
            orig = linear(norm(x)).clone()

        # Fuse and compute
        fuse_rms_norm_into_linear(norm, [linear], handler=handler)
        with torch.no_grad():
            fused = linear(norm(x))

        assert torch.allclose(orig, fused, atol=1e-5), \
            f"Max diff: {(orig - fused).abs().max():.6f}"


class TestOutputNormFusion:
    """Test output-side (approximate) norm fusion for Gemma2 post-norms."""

    def test_output_side_row_scaling(self):
        """Output-side fusion scales linear weight rows by gamma."""
        hidden = 16
        norm = GemmaRMSNorm(hidden)
        norm.weight.data = torch.full((hidden,), 0.2)  # gamma = 1.2

        linear = nn.Linear(32, hidden, bias=False)
        W_orig = linear.weight.data.clone()

        handler = GemmaHandler()
        fuse_rms_norm_output_side(norm, [linear], handler=handler)

        expected_gamma = torch.full((hidden,), 1.2)
        expected_W = expected_gamma[:, None] * W_orig
        assert torch.allclose(linear.weight.data, expected_W, atol=1e-6)
        assert torch.allclose(norm.weight.data, torch.zeros(hidden), atol=1e-6)

    def test_output_side_with_bias(self):
        """Output-side fusion also scales bias by gamma."""
        hidden = 16
        norm = GemmaRMSNorm(hidden)
        norm.weight.data = torch.full((hidden,), 0.5)  # gamma = 1.5

        linear = nn.Linear(32, hidden, bias=True)
        W_orig = linear.weight.data.clone()
        b_orig = linear.bias.data.clone()

        handler = GemmaHandler()
        fuse_rms_norm_output_side(norm, [linear], handler=handler)

        gamma = torch.full((hidden,), 1.5)
        assert torch.allclose(linear.weight.data, gamma[:, None] * W_orig, atol=1e-6)
        assert torch.allclose(linear.bias.data, gamma * b_orig, atol=1e-6)

    def test_output_fusion_approximate_equivalence(self):
        """Output-side fusion is approximately correct (not exact due to RMS denominator)."""
        hidden = 16
        x = torch.randn(4, 32)

        norm = GemmaRMSNorm(hidden)
        norm.weight.data = torch.randn(hidden) * 0.05  # small deviations
        linear = nn.Linear(32, hidden, bias=False)

        # Original: norm(linear(x))
        with torch.no_grad():
            orig = norm(linear(x)).clone()

        # Fused: norm_identity(linear_fused(x))
        handler = GemmaHandler()
        fuse_rms_norm_output_side(norm, [linear], handler=handler)
        with torch.no_grad():
            fused = norm(linear(x))

        # Should be close but not exact (approximate fusion)
        max_diff = (orig - fused).abs().max().item()
        assert max_diff < 0.1, f"Output-side fusion too inaccurate: max_diff={max_diff:.6f}"


class TestTiedEmbeddings:
    """Test handling of tied embedding/lm_head weights."""

    def test_detect_tied_weights(self):
        model = MiniGemma2Model(tie_weights=True)
        handler = Gemma2Handler()
        assert handler.has_tied_embeddings(model)
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    def test_detect_untied_weights(self):
        model = MiniGemma2Model(tie_weights=False)
        handler = Gemma2Handler()
        assert not handler.has_tied_embeddings(model)
        assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()

    def test_rotation_with_tied_weights(self):
        """Full rotation on a model with tied weights should not crash or corrupt."""
        torch.manual_seed(42)
        model = MiniGemma2Model(vocab=50, hidden=64, num_heads=4,
                                num_layers=2, intermediate=128, tie_weights=True)
        model.eval()
        handler = Gemma2Handler()
        input_ids = torch.randint(0, 50, (2, 8))

        # Get original output
        with torch.no_grad():
            logits_orig = model(input_ids).clone()

        # Import and run rotation
        from turbogguf.rotation import rotate_model
        rotate_model(model, handler=handler, seed=42, apply_r2=True, verbose=False)

        # Weights should now be untied
        assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()

        # Output should be close (not exact due to approximate post-norm fusion)
        with torch.no_grad():
            logits_rotated = model(input_ids)

        max_diff = (logits_orig - logits_rotated).abs().max().item()
        # Allow larger tolerance due to post-norm approximation
        assert max_diff < 1.0, f"Logits differ by {max_diff:.4f} (should be < 1.0)"


class TestGemma2Handler:
    """Test Gemma2Handler returns correct norm layers."""

    def test_pre_attn_norm(self):
        model = MiniGemma2Model()
        handler = Gemma2Handler()
        layer = handler.get_layers(model)[0]
        norm = handler.get_pre_attn_norm(layer)
        assert norm is layer.input_layernorm

    def test_post_attn_norm_returns_pre_feedforward(self):
        """get_post_attn_norm should return the pre-MLP norm (pre_feedforward_layernorm)."""
        model = MiniGemma2Model()
        handler = Gemma2Handler()
        layer = handler.get_layers(model)[0]
        norm = handler.get_post_attn_norm(layer)
        assert norm is layer.pre_feedforward_layernorm

    def test_post_attn_output_norm(self):
        model = MiniGemma2Model()
        handler = Gemma2Handler()
        layer = handler.get_layers(model)[0]
        norm = handler.get_post_attn_output_norm(layer)
        assert norm is layer.post_attention_layernorm

    def test_post_mlp_output_norm(self):
        model = MiniGemma2Model()
        handler = Gemma2Handler()
        layer = handler.get_layers(model)[0]
        norm = handler.get_post_mlp_output_norm(layer)
        assert norm is layer.post_feedforward_layernorm

    def test_llama_handler_no_output_norms(self):
        """LLaMA handler should return None for output norms."""
        from turbogguf.arch.llama import LlamaHandler
        handler = LlamaHandler()
        # Use a duck-typed layer object
        layer = type('Layer', (), {
            'input_layernorm': StandardRMSNorm(16),
            'post_attention_layernorm': StandardRMSNorm(16),
        })()
        assert handler.get_post_attn_output_norm(layer) is None
        assert handler.get_post_mlp_output_norm(layer) is None


class TestGemma2EndToEnd:
    """End-to-end rotation tests on mini Gemma2 model."""

    @pytest.fixture
    def model_and_input(self):
        torch.manual_seed(42)
        model = MiniGemma2Model(
            vocab=50, hidden=64, num_heads=4,
            num_layers=2, intermediate=128, tie_weights=True,
        )
        model.eval()
        input_ids = torch.randint(0, 50, (2, 8))
        return model, input_ids

    def test_norms_reset_after_fusion(self, model_and_input):
        """All Gemma RMSNorm weights should be 0 after fusion (gamma=1+0=1)."""
        model, _ = model_and_input
        from turbogguf.rotation import rotate_model
        handler = Gemma2Handler()
        rotate_model(model, handler=handler, seed=42, verbose=False)

        for layer in model.model.layers:
            for norm_name in ['input_layernorm', 'post_attention_layernorm',
                              'pre_feedforward_layernorm', 'post_feedforward_layernorm']:
                norm = getattr(layer, norm_name)
                assert torch.allclose(norm.weight.data, torch.zeros_like(norm.weight.data), atol=1e-6), \
                    f"{norm_name} weight should be 0 after fusion, got {norm.weight.data}"
        assert torch.allclose(
            model.model.norm.weight.data,
            torch.zeros_like(model.model.norm.weight.data),
            atol=1e-6,
        )

    def test_rotation_completes_without_error(self, model_and_input):
        """Full rotation pipeline on Gemma2 model completes without error.

        Note: For randomly-initialized weights, rotation may not improve
        distributions (no outlier channels to spread). The real benefit is
        on trained models. Here we just verify correctness.
        """
        model, input_ids = model_and_input

        from turbogguf.rotation import rotate_model
        handler = Gemma2Handler()
        metadata = rotate_model(model, handler=handler, seed=42, verbose=False)

        assert metadata["r1_applied"] is True
        assert metadata["r2_applied"] is True
        assert metadata["handler"] == "Gemma2Handler"
        assert metadata["has_output_norms"] is True
        assert metadata["had_tied_embeddings"] is True

        # Model should still produce output without NaN/Inf
        with torch.no_grad():
            logits = model(input_ids)
        assert not torch.isnan(logits).any(), "NaN in output after rotation"
        assert not torch.isinf(logits).any(), "Inf in output after rotation"


class TestArchRegistry:
    """Test that Gemma models are registered."""

    def test_gemma_registered(self):
        from turbogguf.arch import ARCH_REGISTRY
        assert "GemmaForCausalLM" in ARCH_REGISTRY
        assert "Gemma2ForCausalLM" in ARCH_REGISTRY
        assert "Gemma4ForCausalLM" in ARCH_REGISTRY

    def test_gemma_handler_types(self):
        from turbogguf.arch import ARCH_REGISTRY
        assert ARCH_REGISTRY["GemmaForCausalLM"] is GemmaHandler
        assert ARCH_REGISTRY["Gemma2ForCausalLM"] is Gemma2Handler
        from turbogguf.arch.gemma import Gemma4Handler
        assert ARCH_REGISTRY["Gemma4ForCausalLM"] is Gemma4Handler
