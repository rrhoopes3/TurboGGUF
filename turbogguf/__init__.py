"""TurboGGUF: Rotation-aware GGUF quantizer.

Applies TurboQuant/QuaRot-style Hadamard rotation to LLM weights before
GGUF quantization. The rotation eliminates outlier features, making low-bit
quantization (Q2_K, Q3_K) dramatically more effective.

No llama.cpp patching required — rotations are absorbed into weights offline.
"""

__version__ = "0.1.0"
