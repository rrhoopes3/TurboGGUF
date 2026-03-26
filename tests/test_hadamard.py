"""Tests for Hadamard matrix generation and fast butterfly multiply."""

import torch
import pytest
import math

from turbogguf.hadamard import (
    hadamard_matrix,
    random_hadamard_matrix,
    matmul_hadU,
    matmul_hadU_right,
    matmul_hadU_left,
    _sylvester,
    _is_power_of_2,
    _factorize_for_hadamard,
)


class TestSylvester:
    """Test Sylvester Hadamard construction."""

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128])
    def test_correct_size(self, n):
        H = _sylvester(n)
        assert H.shape == (n, n)

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32])
    def test_entries_are_pm1(self, n):
        H = _sylvester(n)
        assert torch.all((H == 1) | (H == -1))

    @pytest.mark.parametrize("n", [2, 4, 8, 16, 32, 64])
    def test_orthogonality(self, n):
        """H @ H^T = n * I for Hadamard matrices."""
        H = _sylvester(n)
        product = H @ H.T
        expected = n * torch.eye(n)
        assert torch.allclose(product, expected, atol=1e-6)

    def test_non_power_of_2_raises(self):
        with pytest.raises(AssertionError):
            _sylvester(3)


class TestHadamardMatrix:
    """Test the general hadamard_matrix() function."""

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128, 256])
    def test_power_of_2(self, n):
        H = hadamard_matrix(n)
        assert H.shape == (n, n)
        product = H @ H.T
        expected = n * torch.eye(n)
        assert torch.allclose(product, expected, atol=1e-5)

    def test_size_12(self):
        """Test precomputed H_12."""
        H = hadamard_matrix(12)
        assert H.shape == (12, 12)
        product = H @ H.T
        expected = 12 * torch.eye(12)
        assert torch.allclose(product, expected, atol=1e-5)

    def test_size_24_via_kronecker(self):
        """Test H_24 = kron(H_12, H_2)."""
        H = hadamard_matrix(24)
        assert H.shape == (24, 24)
        product = H @ H.T
        expected = 24 * torch.eye(24)
        assert torch.allclose(product, expected, atol=1e-4)

    def test_kronecker_construction(self):
        """Test Kronecker product for composite sizes like 24 = 12 * 2."""
        H = hadamard_matrix(24)
        assert H.shape == (24, 24)
        product = H @ H.T
        expected = 24 * torch.eye(24)
        assert torch.allclose(product, expected, atol=1e-4)

    def test_caching(self):
        """Repeated calls return same matrix."""
        H1 = hadamard_matrix(16)
        H2 = hadamard_matrix(16)
        assert torch.equal(H1, H2)


class TestRandomHadamard:
    """Test randomized orthogonal Hadamard matrices."""

    @pytest.mark.parametrize("n", [4, 8, 16, 32, 64, 128])
    def test_orthogonality(self, n):
        """Q @ Q^T = I for random Hadamard."""
        Q = random_hadamard_matrix(n, seed=42)
        product = Q @ Q.T
        assert torch.allclose(product, torch.eye(n), atol=1e-5), \
            f"Not orthogonal for n={n}: max deviation {(product - torch.eye(n)).abs().max():.6f}"

    @pytest.mark.parametrize("n", [4, 8, 16, 32, 64])
    def test_transpose_is_inverse(self, n):
        """Q^T @ Q = I (transpose equals inverse for orthogonal matrices)."""
        Q = random_hadamard_matrix(n, seed=42)
        product = Q.T @ Q
        assert torch.allclose(product, torch.eye(n), atol=1e-5)

    def test_deterministic_from_seed(self):
        """Same seed → same matrix."""
        Q1 = random_hadamard_matrix(32, seed=123)
        Q2 = random_hadamard_matrix(32, seed=123)
        assert torch.equal(Q1, Q2)

    def test_different_seeds_differ(self):
        """Different seeds → different matrices."""
        Q1 = random_hadamard_matrix(32, seed=1)
        Q2 = random_hadamard_matrix(32, seed=2)
        assert not torch.equal(Q1, Q2)

    def test_preserves_norms(self):
        """Orthogonal matrices preserve vector norms."""
        Q = random_hadamard_matrix(64, seed=42)
        x = torch.randn(64)
        assert torch.allclose(
            torch.norm(Q @ x),
            torch.norm(x),
            atol=1e-5,
        )

    def test_preserves_dot_products(self):
        """Orthogonal matrices preserve inner products."""
        Q = random_hadamard_matrix(32, seed=42)
        x = torch.randn(32)
        y = torch.randn(32)
        dot_original = x @ y
        dot_rotated = (Q @ x) @ (Q @ y)
        assert torch.allclose(dot_original, dot_rotated, atol=1e-4)


class TestButterflyMultiply:
    """Test fast O(n log n) Hadamard multiplication."""

    @pytest.mark.parametrize("n", [4, 8, 16, 32, 64, 128])
    def test_matches_dense(self, n):
        """Butterfly multiply matches dense Q @ x."""
        Q = random_hadamard_matrix(n, seed=42)
        x = torch.randn(n)

        # Dense
        result_dense = x @ Q

        # Butterfly
        result_butterfly = matmul_hadU(x.unsqueeze(0), seed=42).squeeze(0)

        assert torch.allclose(result_dense, result_butterfly, atol=1e-4), \
            f"Max diff: {(result_dense - result_butterfly).abs().max():.6f}"

    @pytest.mark.parametrize("n", [4, 8, 16, 32, 64])
    def test_transpose_matches_dense(self, n):
        """Butterfly transpose matches dense Q^T @ x."""
        Q = random_hadamard_matrix(n, seed=42)
        x = torch.randn(n)

        result_dense = x @ Q.T
        result_butterfly = matmul_hadU(x.unsqueeze(0), seed=42, transpose=True).squeeze(0)

        assert torch.allclose(result_dense, result_butterfly, atol=1e-4)

    def test_batched(self):
        """Butterfly works on batched inputs."""
        n = 32
        batch = 8
        X = torch.randn(batch, n)
        result = matmul_hadU(X, seed=42)
        assert result.shape == (batch, n)

        # Verify each row matches individual transform
        Q = random_hadamard_matrix(n, seed=42)
        for i in range(batch):
            expected = X[i] @ Q
            assert torch.allclose(result[i], expected, atol=1e-4)

    def test_matrix_right_multiply(self):
        """matmul_hadU_right: W @ Q."""
        n = 16
        W = torch.randn(32, n)
        Q = random_hadamard_matrix(n, seed=42)

        result = matmul_hadU_right(W, seed=42)
        expected = W @ Q
        assert torch.allclose(result, expected, atol=1e-4)

    def test_matrix_left_multiply(self):
        """matmul_hadU_left: Q @ W."""
        n = 16
        W = torch.randn(n, 32)
        Q = random_hadamard_matrix(n, seed=42)

        result = matmul_hadU_left(W, seed=42)
        expected = Q @ W
        assert torch.allclose(result, expected, atol=5e-3), \
            f"Max diff: {(result - expected).abs().max():.6f}"

    def test_roundtrip(self):
        """Apply Q then Q^T recovers original."""
        n = 64
        x = torch.randn(1, n)
        rotated = matmul_hadU(x, seed=42)
        recovered = matmul_hadU(rotated, seed=42, transpose=True)
        assert torch.allclose(x, recovered, atol=1e-4)


class TestFactorization:
    """Test dimension factorization for Hadamard construction."""

    def test_power_of_2(self):
        k, p = _factorize_for_hadamard(64)
        assert k == 1
        assert p == 64

    def test_composite_24(self):
        k, p = _factorize_for_hadamard(24)
        assert k * p == 24
        assert _is_power_of_2(p)

    def test_composite_48(self):
        k, p = _factorize_for_hadamard(48)
        assert k * p == 48
        assert _is_power_of_2(p)

    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            _factorize_for_hadamard(7)  # No Hadamard of size 7
