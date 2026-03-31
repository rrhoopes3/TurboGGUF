"""Tests for TurboQuant+ KV cache compression integration."""

import numpy as np
import pytest


class TestPolarQuant:
    """Test PolarQuant scalar quantization."""

    def test_roundtrip_preserves_shape(self):
        from turbogguf.turboquant_plus.polar_quant import PolarQuant
        pq = PolarQuant(d=64, bit_width=2, seed=42)
        x = np.random.randn(64)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)
        assert x_hat.shape == x.shape

    def test_batch_roundtrip(self):
        from turbogguf.turboquant_plus.polar_quant import PolarQuant
        pq = PolarQuant(d=32, bit_width=2, seed=42)
        X = np.random.randn(5, 32)
        indices, norms = pq.quantize(X)
        X_hat = pq.dequantize(indices, norms)
        assert X_hat.shape == X.shape

    def test_mse_decreases_with_bits(self):
        from turbogguf.turboquant_plus.polar_quant import PolarQuant
        x = np.random.randn(64)
        mses = []
        for bits in [1, 2, 3]:
            pq = PolarQuant(d=64, bit_width=bits, seed=42)
            indices, norms = pq.quantize(x)
            x_hat = pq.dequantize(indices, norms)
            mse = np.mean((x - x_hat) ** 2)
            mses.append(mse)
        # More bits should give lower MSE
        assert mses[0] > mses[1] > mses[2]

    def test_zero_vector(self):
        from turbogguf.turboquant_plus.polar_quant import PolarQuant
        pq = PolarQuant(d=32, bit_width=2, seed=42)
        x = np.zeros(32)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)
        assert np.allclose(x_hat, 0.0, atol=1e-10)


class TestQJL:
    """Test QJL 1-bit quantizer."""

    def test_roundtrip_shape(self):
        from turbogguf.turboquant_plus.qjl import QJL
        qjl = QJL(d=64, seed=42)
        r = np.random.randn(64) * 0.1
        signs, norms = qjl.quantize(r)
        r_hat = qjl.dequantize(signs, norms)
        assert r_hat.shape == r.shape
        assert signs.dtype == np.int8

    def test_signs_are_pm1(self):
        from turbogguf.turboquant_plus.qjl import QJL
        qjl = QJL(d=64, seed=42)
        r = np.random.randn(64)
        signs, _ = qjl.quantize(r)
        assert set(np.unique(signs)).issubset({-1, 1})


class TestTurboQuant:
    """Test full TurboQuant (PolarQuant + QJL)."""

    def test_roundtrip(self):
        from turbogguf.turboquant_plus.turboquant import TurboQuant
        tq = TurboQuant(d=64, bit_width=3, seed=42)
        x = np.random.randn(64)
        compressed = tq.quantize(x)
        x_hat = tq.dequantize(compressed)
        assert x_hat.shape == x.shape
        # Should reconstruct reasonably well
        rel_error = np.linalg.norm(x - x_hat) / np.linalg.norm(x)
        assert rel_error < 1.0  # At least better than random

    def test_bit_width_validation(self):
        from turbogguf.turboquant_plus.turboquant import TurboQuant
        with pytest.raises(ValueError, match="bit_width >= 2"):
            TurboQuant(d=64, bit_width=1)

    def test_compression_ratio(self):
        from turbogguf.turboquant_plus.turboquant import TurboQuant
        tq = TurboQuant(d=128, bit_width=3, seed=42)
        ratio = tq.compression_ratio(original_bits_per_value=16)
        assert ratio > 1.0  # Should compress
        assert ratio < 16.0  # Should be reasonable

    def test_inner_product_preservation(self):
        from turbogguf.turboquant_plus.turboquant import TurboQuant
        tq = TurboQuant(d=128, bit_width=3, seed=42)
        x = np.random.randn(128)
        y = np.random.randn(128)
        ip_original = np.dot(x, y)
        x_hat = tq.dequantize(tq.quantize(x))
        y_hat = tq.dequantize(tq.quantize(y))
        ip_approx = np.dot(x_hat, y_hat)
        # Inner product should be roughly preserved
        rel_error = abs(ip_original - ip_approx) / (abs(ip_original) + 1e-10)
        assert rel_error < 2.0  # Loose bound for 3-bit


class TestTurboQuantMSE:
    """Test MSE-only variant."""

    def test_roundtrip(self):
        from turbogguf.turboquant_plus.turboquant import TurboQuantMSE
        tq = TurboQuantMSE(d=64, bit_width=3, seed=42)
        x = np.random.randn(64)
        indices, norms = tq.quantize(x)
        x_hat = tq.dequantize(indices, norms)
        assert x_hat.shape == x.shape


class TestKVCacheCompressor:
    """Test KV cache compression."""

    def test_compress_decompress(self):
        from turbogguf.turboquant_plus.kv_cache import KVCacheCompressor
        compressor = KVCacheCompressor(head_dim=32, k_bits=3, v_bits=3)
        k_cache = np.random.randn(2, 4, 8, 32)  # 2 layers, 4 heads, 8 seq, 32 dim
        v_cache = np.random.randn(2, 4, 8, 32)
        compressed = compressor.compress(k_cache, v_cache)
        k_hat, v_hat = compressor.decompress(compressed)
        assert k_hat.shape == k_cache.shape
        assert v_hat.shape == v_cache.shape

    def test_memory_stats(self):
        from turbogguf.turboquant_plus.kv_cache import KVCacheCompressor
        compressor = KVCacheCompressor(head_dim=128, k_bits=3, v_bits=3)
        stats = compressor.memory_stats(seq_len=4096, num_layers=32, num_heads=32)
        assert stats["compression_ratio"] > 1.0
        assert stats["compressed_mb"] < stats["original_mb"]


class TestCodebook:
    """Test codebook construction."""

    def test_1bit_centroids(self):
        from turbogguf.turboquant_plus.codebook import optimal_centroids
        c = optimal_centroids(1, 128)
        assert len(c) == 2
        assert c[0] < 0 < c[1]
        assert np.isclose(c[0], -c[1])  # Symmetric

    def test_2bit_centroids(self):
        from turbogguf.turboquant_plus.codebook import optimal_centroids
        c = optimal_centroids(2, 128)
        assert len(c) == 4
        assert np.all(np.diff(c) > 0)  # Sorted

    def test_3bit_centroids(self):
        from turbogguf.turboquant_plus.codebook import optimal_centroids
        c = optimal_centroids(3, 128)
        assert len(c) == 8
        assert np.all(np.diff(c) > 0)


class TestRotation:
    """Test rotation matrix generation."""

    def test_dense_rotation_orthogonal(self):
        from turbogguf.turboquant_plus.rotation import random_rotation_dense
        rng = np.random.default_rng(42)
        Q = random_rotation_dense(32, rng)
        # Q @ Q^T should be identity
        assert np.allclose(Q @ Q.T, np.eye(32), atol=1e-10)

    def test_dense_rotation_det_positive(self):
        from turbogguf.turboquant_plus.rotation import random_rotation_dense
        rng = np.random.default_rng(42)
        Q = random_rotation_dense(32, rng)
        sign, _ = np.linalg.slogdet(Q)
        assert sign > 0

    def test_hadamard_matrix(self):
        from turbogguf.turboquant_plus.rotation import hadamard_matrix
        H = hadamard_matrix(8)
        assert H.shape == (8, 8)
        # H @ H^T = n * I
        assert np.allclose(H @ H.T, 8 * np.eye(8))

    def test_fast_walsh_hadamard(self):
        from turbogguf.turboquant_plus.rotation import fast_walsh_hadamard_transform, hadamard_matrix
        x = np.random.randn(8)
        # FWHT should match dense: H @ x / sqrt(n)
        H = hadamard_matrix(8)
        expected = H @ x / np.sqrt(8)
        result = fast_walsh_hadamard_transform(x)
        assert np.allclose(result, expected, atol=1e-10)


class TestUtils:
    """Test bit packing utilities."""

    def test_pack_unpack_roundtrip(self):
        from turbogguf.turboquant_plus.utils import pack_bits, unpack_bits
        signs = np.array([1, -1, 1, 1, -1, -1, 1, -1, 1, 1], dtype=np.int8)
        packed = pack_bits(signs)
        unpacked = unpack_bits(packed, d=10)
        assert np.array_equal(signs, unpacked)

    def test_pack_unpack_batch(self):
        from turbogguf.turboquant_plus.utils import pack_bits, unpack_bits
        rng = np.random.default_rng(42)
        signs = rng.choice([-1, 1], size=(3, 16)).astype(np.int8)
        packed = pack_bits(signs)
        unpacked = unpack_bits(packed, d=16)
        assert np.array_equal(signs, unpacked)

    def test_memory_footprint(self):
        from turbogguf.turboquant_plus.utils import memory_footprint_bytes
        stats = memory_footprint_bytes(n_vectors=1000, d=128, bit_width=3)
        assert stats["compression_ratio"] > 1.0
        assert stats["total_bytes"] < stats["original_fp16_bytes"]


class TestOutlierTurboQuant:
    """Test outlier channel strategy."""

    def test_fractional_bits(self):
        from turbogguf.turboquant_plus.outlier import OutlierTurboQuant
        oq = OutlierTurboQuant(d=128, target_bits=2.5, seed=42)
        assert 2.4 <= oq.effective_bits <= 2.6

    def test_roundtrip(self):
        from turbogguf.turboquant_plus.outlier import OutlierTurboQuant
        oq = OutlierTurboQuant(d=64, target_bits=2.5, seed=42)
        x = np.random.randn(64)
        compressed = oq.quantize(x)
        x_hat = oq.dequantize(compressed)
        assert x_hat.shape == x.shape

    def test_compression_ratio(self):
        from turbogguf.turboquant_plus.outlier import OutlierTurboQuant
        oq = OutlierTurboQuant(d=128, target_bits=2.5, seed=42)
        ratio = oq.compression_ratio()
        assert ratio > 1.0
