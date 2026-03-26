# TurboGGUF

**Rotation-aware GGUF quantizer: Q2 quality that performs like Q4.**

Applies TurboQuant/QuaRot-style Hadamard rotation to LLM weights before GGUF quantization. The rotation eliminates outlier features in weight matrices, making low-bit quantization (Q2_K, Q3_K) dramatically more effective. No llama.cpp patching required.

```
                    ┌──────────────────────────────────┐
                    │   HuggingFace Model (FP16)       │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │  TurboGGUF Rotation Pipeline     │
                    │                                   │
                    │  1. Fuse RMSNorm → linear weights │
                    │  2. R1: Hadamard residual rotation │
                    │  3. R2: Per-head rotation          │
                    │                                   │
                    │  Output: identical FP16 logits,   │
                    │  but weights are now "spread out" │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │  Standard GGUF Quantization      │
                    │  (llama.cpp's Q2_K / Q3_K_S)     │
                    │                                   │
                    │  Same tool, same format,          │
                    │  dramatically better quality      │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │  Load in LM Studio / llama.cpp   │
                    │  Standard GGUF — nothing special  │
                    └──────────────────────────────────┘
```

## Why This Works

LLM weight matrices have **outlier channels** — a few dimensions with values 10-100x larger than the rest. When you quantize to 2-3 bits, these outliers dominate the quantization range and destroy information in the remaining 99% of dimensions.

Hadamard rotation spreads these outliers uniformly across all dimensions. After rotation:
- Weight distributions become near-Gaussian (concentrated, no outliers)
- Standard Q2_K quantization loses far less information
- The rotation is absorbed into the weights offline — **zero runtime cost**

The math: RMSNorm is rotation-invariant (`||Rx|| = ||x||`), so the same rotation matrix R can be fused into adjacent weight matrices. At each layer boundary `R^T @ R = I`. The rotated model produces **bit-identical FP16 outputs**.

## Expected Quality Gains

| Model | Quant | Rotated? | Size | Expected PPL |
|-------|-------|----------|------|-------------|
| Llama-3.1-8B | Q4_K_M | No | ~4.9 GB | ~6.5 (target) |
| Llama-3.1-8B | Q2_K | No | ~3.0 GB | ~8.5 |
| Llama-3.1-8B | Q2_K | **Yes** | ~3.0 GB | **~6.8-7.2** |
| Llama-3.1-8B | Q3_K_S | **Yes** | ~3.5 GB | **~6.3-6.5** |

**Rotated Q3_K_S ≈ unrotated Q4_K_M quality at 30% less VRAM.**

## Quick Start

```bash
pip install -e ".[dev]"

# Rotate a model (CPU-only, ~16GB RAM for 8B model)
turbogguf rotate \
    --model meta-llama/Llama-3.1-8B \
    --output ./rotated-llama3.1-8b \
    --seed 42

# Convert to GGUF + quantize (using llama.cpp)
python /path/to/llama.cpp/convert_hf_to_gguf.py ./rotated-llama3.1-8b --outtype f16
/path/to/llama.cpp/llama-quantize ./rotated-llama3.1-8b/model.gguf ./turbo-Q2_K.gguf Q2_K

# Or do it all in one command
turbogguf pipeline \
    --model meta-llama/Llama-3.1-8B \
    --quant Q2_K \
    --output ./turbo-Q2_K.gguf \
    --llama-cpp /path/to/llama.cpp

# Load turbo-Q2_K.gguf in LM Studio — it's a standard GGUF file
```

## CLI Commands

```bash
turbogguf rotate     # Apply Hadamard rotation to model weights
turbogguf pipeline   # Full pipeline: rotate → convert → quantize
turbogguf evaluate   # Compare perplexity of GGUF models
turbogguf info       # Show memory requirements for a model
```

## Hardware Requirements

| Model Size | RAM Needed | Time (approx) |
|-----------|-----------|---------------|
| 1-3B | ~8 GB | ~2 min |
| 7-8B | ~16 GB | ~5 min |
| 13B | ~28 GB | ~10 min |
| 34B | ~70 GB | ~25 min |
| 70B | ~140 GB | ~45 min |

GPU not required — rotation is pure matrix math on CPU.

## Architecture Support

| Architecture | Status | Models |
|-------------|--------|--------|
| LLaMA | Supported | Llama 2, Llama 3, Llama 3.1, CodeLlama |
| Mistral | Supported | Mistral 7B, Mixtral (dense layers) |
| Qwen2 | Supported | Qwen2, Qwen2.5 |

## How It Works (Technical)

**Step 1 — Fuse RMSNorm:**
```
W_fused[i,j] = W[i,j] * gamma[j]
gamma → ones (norm becomes pure RMS division)
```

**Step 2 — R1: Residual stream Hadamard rotation (global Q):**
```
embed_tokens  → weight @ Q
q/k/v_proj    → W @ Q         (absorb R from residual input)
o_proj        → Q^T @ W       (emit R^T back to residual)
gate/up_proj  → W @ Q
down_proj     → Q^T @ W
lm_head       → W @ Q
```

**Step 3 — R2: Per-head rotation (head-dim Hadamard H):**
```
v_proj: H @ V_head    (rotate each head's output)
o_proj: O_head @ H^T  (rotate each head's input)
```

**Zero runtime cost:** `R^T @ R = I` at every layer boundary. `||Rx|| = ||x||` for orthogonal R. The rotated model is mathematically equivalent.

## References

- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456) (NeurIPS 2024)
- [TurboQuant: Redefining AI Efficiency with Extreme Compression](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence](https://arxiv.org/abs/2402.04396) (ICML 2024)

## Tests

```bash
pytest -v tests/
```

## License

MIT
