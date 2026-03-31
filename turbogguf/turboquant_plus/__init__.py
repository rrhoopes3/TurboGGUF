"""TurboQuant+: KV cache compression via PolarQuant + QJL.

Integrates the TurboQuant+ library (https://github.com/TheTom/turboquant_plus)
for runtime KV cache compression. Complementary to TurboGGUF's weight rotation:
- TurboGGUF: improves weight quantization quality (offline, before GGUF)
- TurboQuant+: compresses KV cache at inference time (runtime, in llama.cpp)

Reference: TurboQuant (ICLR 2026) https://arxiv.org/abs/2504.19874
"""

from turbogguf.turboquant_plus.polar_quant import PolarQuant
from turbogguf.turboquant_plus.qjl import QJL
from turbogguf.turboquant_plus.turboquant import TurboQuant, TurboQuantMSE, CompressedVector
from turbogguf.turboquant_plus.kv_cache import KVCacheCompressor

__all__ = ["PolarQuant", "QJL", "TurboQuant", "TurboQuantMSE", "CompressedVector", "KVCacheCompressor"]
