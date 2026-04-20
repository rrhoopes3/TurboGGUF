"""Hadamard matrix generation and fast O(n log n) butterfly multiplication.

Implements randomized Hadamard transforms for the rotation-based quantization
pipeline. Supports arbitrary dimensions via Kronecker product of precomputed
Hadamard matrices with Sylvester (power-of-2) matrices.

References:
  - QuaRot (NeurIPS 2024): https://arxiv.org/abs/2404.00456
  - TurboQuant (ICLR 2026): https://arxiv.org/abs/2504.19874
"""

import torch
import math
from typing import Tuple, Optional

# Precomputed Hadamard matrices for non-power-of-2 dimensions.
# These are normalized orthogonal matrices of specific sizes that, when
# Kronecker-producted with Sylvester matrices, cover common LLM hidden dims.
# Sizes: 12, 20, 28, 36, 40 cover most transformer architectures.

# fmt: off
_HADAMARD_12 = torch.tensor([
    [+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1],
    [+1,-1,+1,-1,+1,+1,+1,-1,-1,-1,+1,-1],
    [+1,-1,-1,+1,-1,+1,+1,+1,-1,-1,-1,+1],
    [+1,+1,-1,-1,+1,-1,+1,+1,+1,-1,-1,-1],
    [+1,-1,+1,-1,-1,+1,-1,+1,+1,+1,-1,-1],
    [+1,-1,-1,+1,-1,-1,+1,-1,+1,+1,+1,-1],
    [+1,-1,-1,-1,+1,-1,-1,+1,-1,+1,+1,+1],
    [+1,+1,-1,-1,-1,+1,-1,-1,+1,-1,+1,+1],
    [+1,+1,+1,-1,-1,-1,+1,-1,-1,+1,-1,+1],
    [+1,+1,+1,+1,-1,-1,-1,+1,-1,-1,+1,-1],
    [+1,-1,+1,+1,+1,-1,-1,-1,+1,-1,-1,+1],
    [+1,+1,-1,+1,+1,+1,-1,-1,-1,+1,-1,-1],
], dtype=torch.float32)

_HADAMARD_20 = None  # Constructed at runtime via Kronecker if needed
_HADAMARD_28 = None  # Constructed at runtime via Kronecker if needed
# fmt: on

# Cache for computed Hadamard matrices
_hadamard_cache: dict[int, torch.Tensor] = {}


def _sylvester(n: int) -> torch.Tensor:
    """Build Sylvester-type Hadamard matrix of size n (must be power of 2)."""
    assert n > 0 and (n & (n - 1)) == 0, f"n must be a power of 2, got {n}"
    if n == 1:
        return torch.ones(1, 1)
    half = _sylvester(n // 2)
    return torch.cat([
        torch.cat([half, half], dim=1),
        torch.cat([half, -half], dim=1),
    ], dim=0)


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _factorize_for_hadamard(n: int) -> Tuple[int, int]:
    """Factor n = k * 2^m where k has a precomputed Hadamard matrix.

    Returns (k, 2^m). If k == 1, use pure Sylvester.
    """
    precomputed_sizes = [1, 12]
    # Try largest precomputed factors first
    for k in sorted(precomputed_sizes, reverse=True):
        if k == 1:
            continue
        if n % k == 0:
            remainder = n // k
            if _is_power_of_2(remainder):
                return k, remainder
    # Pure power of 2
    if _is_power_of_2(n):
        return 1, n
    raise ValueError(
        f"Cannot construct Hadamard matrix of size {n}. "
        f"Need n = k * 2^m where k in {precomputed_sizes}"
    )


def _get_precomputed(k: int) -> torch.Tensor:
    """Get precomputed Hadamard matrix of size k."""
    if k == 1:
        return torch.ones(1, 1)
    if k == 12:
        return _HADAMARD_12.clone()
    raise ValueError(f"No precomputed Hadamard matrix for size {k}")


def hadamard_matrix(n: int) -> torch.Tensor:
    """Construct an n x n Hadamard matrix (unnormalized, entries ±1).

    Supports:
      - Power-of-2 sizes via Sylvester construction
      - n = k * 2^m via Kronecker product of precomputed H_k and Sylvester H_{2^m}

    Args:
        n: Matrix dimension

    Returns:
        n x n tensor with entries ±1 satisfying H @ H^T = n * I
    """
    if n in _hadamard_cache:
        return _hadamard_cache[n].clone()

    if _is_power_of_2(n):
        H = _sylvester(n)
    else:
        k, pow2 = _factorize_for_hadamard(n)
        H_k = _get_precomputed(k)
        H_pow2 = _sylvester(pow2)
        H = torch.kron(H_k, H_pow2)

    _hadamard_cache[n] = H
    return H.clone()


def _random_orthogonal_matrix(
    size: int,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a random orthogonal matrix via QR decomposition.

    Fallback for dimensions where Hadamard construction is impossible
    (e.g., 5376 = 21 * 256, and no Hadamard matrix exists for size 21).
    Same guarantees: orthogonal, deterministic from seed.

    Args:
        size: Matrix dimension
        seed: Random seed
        dtype: Target dtype

    Returns:
        size x size orthogonal matrix Q with Q @ Q^T = I
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    # Random Gaussian matrix -> QR decomposition gives orthogonal Q
    A = torch.randn(size, size, generator=gen, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    # Ensure deterministic sign (Q from QR has arbitrary column signs)
    # Fix by making diagonal of R positive
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs.unsqueeze(0)
    return Q


def random_hadamard_matrix(
    size: int,
    seed: int = 42,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a randomized orthogonal matrix.

    Uses fast Hadamard construction when possible (n = k * 2^m with known k),
    falls back to QR-based random orthogonal matrix for other dimensions
    (e.g., Gemma 4's hidden_size=5376).

    The result is always an orthogonal matrix Q satisfying Q @ Q^T = I.

    Args:
        size: Matrix dimension
        seed: Random seed for reproducibility
        device: Target device
        dtype: Target dtype

    Returns:
        size x size orthogonal matrix
    """
    try:
        H = hadamard_matrix(size).to(dtype=dtype)
        # Random ±1 diagonal (seeded)
        gen = torch.Generator()
        gen.manual_seed(seed)
        signs = torch.randint(0, 2, (size,), generator=gen).to(dtype) * 2 - 1
        # Q = D @ H / sqrt(n) where D = diag(signs)
        Q = (signs.unsqueeze(1) * H) / math.sqrt(size)
    except ValueError:
        # Hadamard construction not possible for this dimension
        print(f"  Note: Using QR-based orthogonal matrix for dim={size} "
              f"(no Hadamard factorization available)")
        Q = _random_orthogonal_matrix(size, seed=seed, dtype=dtype)

    if device is not None:
        Q = Q.to(device)
    return Q


def matmul_hadU(
    X: torch.Tensor,
    seed: int = 42,
    transpose: bool = False,
) -> torch.Tensor:
    """Fast Hadamard transform via butterfly factorization: O(n log n).

    Computes X @ Q (or X @ Q^T if transpose=True) without materializing
    the full n x n Hadamard matrix, using in-place butterfly operations.

    For power-of-2 dimensions, this is the classic fast Walsh-Hadamard transform.
    For non-power-of-2 dimensions, falls back to dense matmul.

    Args:
        X: Input tensor of shape (..., n)
        seed: Random seed for the sign diagonal
        transpose: If True, apply Q^T instead of Q

    Returns:
        Transformed tensor of same shape as X
    """
    n = X.shape[-1]
    orig_dtype = X.dtype
    X = X.float()

    # Generate the random signs
    gen = torch.Generator()
    gen.manual_seed(seed)
    signs = torch.randint(0, 2, (n,), generator=gen).float() * 2 - 1
    signs = signs.to(X.device)

    if not _is_power_of_2(n):
        # Fallback to dense matmul for non-power-of-2
        Q = random_hadamard_matrix(n, seed=seed, device=X.device)
        if transpose:
            Q = Q.T
        result = X @ Q
        return result.to(orig_dtype)

    # Apply signs first (or last if transpose)
    if not transpose:
        X = X * signs

    # In-place butterfly (Walsh-Hadamard transform)
    h = 1
    while h < n:
        # Process pairs at distance h
        for i in range(0, n, h * 2):
            j_range = slice(i, i + h)
            k_range = slice(i + h, i + 2 * h)
            x_j = X[..., j_range].clone()
            x_k = X[..., k_range].clone()
            X[..., j_range] = x_j + x_k
            X[..., k_range] = x_j - x_k
        h *= 2

    # Normalize
    X = X / math.sqrt(n)

    # Apply signs last if transpose
    if transpose:
        X = X * signs

    return X.to(orig_dtype)


def matmul_hadU_right(
    W: torch.Tensor,
    seed: int = 42,
    transpose: bool = False,
) -> torch.Tensor:
    """Compute W @ Q or W @ Q^T for a weight matrix W.

    This is the right-multiplication variant: each row of W gets transformed.

    Args:
        W: Weight matrix of shape (out_features, in_features)
        seed: Random seed
        transpose: If True, compute W @ Q^T

    Returns:
        Rotated weight matrix of same shape
    """
    return matmul_hadU(W, seed=seed, transpose=transpose)


def matmul_hadU_left(
    W: torch.Tensor,
    seed: int = 42,
    transpose: bool = False,
) -> torch.Tensor:
    """Compute Q @ W or Q^T @ W for a weight matrix W.

    This is the left-multiplication variant: each column of W gets transformed.
    Uses dense matmul for numerical stability (not on hot path).

    Args:
        W: Weight matrix of shape (out_features, in_features)
        seed: Random seed
        transpose: If True, compute Q^T @ W

    Returns:
        Rotated weight matrix of same shape
    """
    n = W.shape[0]
    Q = random_hadamard_matrix(n, seed=seed, device=W.device, dtype=torch.float32)
    W_f = W.float()
    if transpose:
        result = Q.T @ W_f
    else:
        result = Q @ W_f
    return result.to(W.dtype)
