"""Outlier channel strategy for non-integer bit precision.

Paper Section on Non-Integer Bit Precision:
Split channels into outlier (higher bits) and non-outlier (lower bits).

Examples:
- 2.5-bit: 32/128 outlier at 3b + 96/128 normal at 2b = 2.5 avg
- 3.5-bit: 64/128 outlier at 4b + 64/128 normal at 3b = 3.5 avg

Source: https://github.com/TheTom/turboquant_plus
"""

import numpy as np
from dataclasses import dataclass

from turbogguf.turboquant_plus.polar_quant import PolarQuant
from turbogguf.turboquant_plus.qjl import QJL


@dataclass
class OutlierCompressedVector:
    """Container for outlier-strategy compressed vector."""
    outlier_indices: np.ndarray
    outlier_norms: np.ndarray
    normal_indices: np.ndarray
    normal_norms: np.ndarray
    qjl_signs: np.ndarray
    residual_norms: np.ndarray
    effective_bits: float


def _compute_channel_split(d: int, target_bits: float) -> tuple[int, int, int, int]:
    """Compute how many channels get higher vs lower bit-width.

    Returns:
        (n_outlier, outlier_bits, n_normal, normal_bits)
    """
    low_bits = int(np.floor(target_bits))
    high_bits = low_bits + 1
    frac = target_bits - low_bits

    n_outlier = int(round(d * frac))
    n_normal = d - n_outlier

    return n_outlier, high_bits, n_normal, low_bits


class OutlierTurboQuant:
    """TurboQuant with outlier channel strategy for non-integer bit rates.

    Splits channels into outlier (higher bit-width) and normal (lower bit-width)
    to achieve fractional average bit rates like 2.5 or 3.5 bits per channel.

    Usage:
        oq = OutlierTurboQuant(d=128, target_bits=2.5, seed=42)
        compressed = oq.quantize(x)
        x_hat = oq.dequantize(compressed)
    """

    def __init__(self, d: int, target_bits: float, seed: int = 42):
        self.d = d
        self.target_bits = target_bits

        n_outlier, high_bits, n_normal, low_bits = _compute_channel_split(d, target_bits)
        self.n_outlier = n_outlier
        self.n_normal = n_normal
        self.high_bits = high_bits
        self.low_bits = low_bits

        self.effective_bits = (n_outlier * high_bits + n_normal * low_bits) / d

        self.outlier_idx = np.arange(n_outlier)
        self.normal_idx = np.arange(n_outlier, d)

        self.pq_outlier = PolarQuant(n_outlier, bit_width=high_bits - 1, seed=seed) if n_outlier > 0 else None
        self.pq_normal = PolarQuant(n_normal, bit_width=low_bits - 1, seed=seed + 500) if n_normal > 0 else None
        self.qjl = QJL(d, seed=seed + 1000)

    def quantize(self, x: np.ndarray) -> OutlierCompressedVector:
        """Quantize with outlier channel split."""
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]

        x_outlier = x[:, self.outlier_idx]
        x_normal = x[:, self.normal_idx]

        if self.pq_outlier is not None:
            out_idx, out_norms, out_residual = self.pq_outlier.quantize_and_residual(
                x_outlier if x.shape[0] > 1 else x_outlier[0]
            )
        else:
            out_idx = np.array([])
            out_norms = np.array([])
            out_residual = np.zeros_like(x_outlier)

        if self.pq_normal is not None:
            norm_idx, norm_norms, norm_residual = self.pq_normal.quantize_and_residual(
                x_normal if x.shape[0] > 1 else x_normal[0]
            )
        else:
            norm_idx = np.array([])
            norm_norms = np.array([])
            norm_residual = np.zeros_like(x_normal)

        if single:
            full_residual = np.zeros(self.d)
            full_residual[self.outlier_idx] = out_residual if out_residual.ndim == 1 else out_residual[0]
            full_residual[self.normal_idx] = norm_residual if norm_residual.ndim == 1 else norm_residual[0]
        else:
            batch = x.shape[0]
            full_residual = np.zeros((batch, self.d))
            full_residual[:, self.outlier_idx] = out_residual
            full_residual[:, self.normal_idx] = norm_residual

        qjl_signs, residual_norms = self.qjl.quantize(full_residual)

        return OutlierCompressedVector(
            outlier_indices=out_idx,
            outlier_norms=out_norms,
            normal_indices=norm_idx,
            normal_norms=norm_norms,
            qjl_signs=qjl_signs,
            residual_norms=residual_norms,
            effective_bits=self.effective_bits,
        )

    def dequantize(self, compressed: OutlierCompressedVector) -> np.ndarray:
        """Dequantize outlier-strategy compressed vector."""
        single = compressed.qjl_signs.ndim == 1

        if self.pq_outlier is not None:
            x_outlier = self.pq_outlier.dequantize(compressed.outlier_indices, compressed.outlier_norms)
        else:
            x_outlier = np.zeros(0)

        if self.pq_normal is not None:
            x_normal = self.pq_normal.dequantize(compressed.normal_indices, compressed.normal_norms)
        else:
            x_normal = np.zeros(0)

        x_qjl = self.qjl.dequantize(compressed.qjl_signs, compressed.residual_norms)

        if single:
            x_hat = np.zeros(self.d)
            if self.n_outlier > 0:
                x_hat[self.outlier_idx] = x_outlier
            if self.n_normal > 0:
                x_hat[self.normal_idx] = x_normal
            x_hat += x_qjl
        else:
            batch = compressed.qjl_signs.shape[0]
            x_hat = np.zeros((batch, self.d))
            if self.n_outlier > 0:
                x_hat[:, self.outlier_idx] = x_outlier
            if self.n_normal > 0:
                x_hat[:, self.normal_idx] = x_normal
            x_hat += x_qjl

        return x_hat

    def compression_ratio(self, original_bits: int = 16) -> float:
        """Compression ratio vs original precision."""
        per_vector_bits = self.d * self.effective_bits + 32 + 64
        original = self.d * original_bits
        return original / per_vector_bits
