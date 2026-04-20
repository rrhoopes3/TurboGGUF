# TurboGGUF

> **Status: shelved.** The underlying math (QuaRot-style Hadamard rotation) is sound and is published research. This repo now has a working fix for Qwen-family `v_proj.bias` rotation and a bf16-equivalence test on Qwen2.5-3B. Remaining bugs (bf16 precision drift on pure-LLaMA architectures, no forward-equivalence gate in the pipeline) prevent claiming any universal Q2/Q3 win. See [Status](#status) below.

Applies TurboQuant/QuaRot-style Hadamard rotation to LLM weights before GGUF quantization. The rotation eliminates outlier features in weight matrices, making low-bit quantization (Q2_K, Q3_K) more effective on architectures where it works. No llama.cpp patching required.

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

## Measured Results

### Qwen2.5-3B (pure transformer, after `v_proj.bias` fix)

wiki.test.raw, `n_ctx=512`, llama.cpp b8638, RTX 3090 24 GB.

| Build | PPL | ± | Size |
|-------|-----|---|------|
| Stock bf16 (baseline) | 8.4230 | 0.0566 | 5.8 GB |
| TurboGGUF bf16 | 8.4217 | 0.0566 | 6.4 GB |
| Stock Q6_K | 8.4807 | 0.0572 | 2.4 GB |
| TurboGGUF Q6_K | 8.4652 | 0.0569 | 2.7 GB |
| Stock Q4_K_M | 8.8059 | 0.0596 | 1.8 GB |
| TurboGGUF Q4_K_M | 8.8420 | 0.0594 | 2.0 GB |
| Stock Q3_K_M | 14.3488 | 0.1033 | 1.5 GB |
| **TurboGGUF Q3_K_M** | **10.0950** | 0.0690 | 1.7 GB |
| Stock Q2_K | 21,636 | 207 | 1.2 GB |
| **TurboGGUF Q2_K** | **26.70** | 0.207 | 1.3 GB |

**Findings:**

- **bf16 equivalence confirmed** (turbo 8.4217 vs stock 8.4230, within ±0.06 noise): the rotation pipeline is mathematically correct on Qwen2 after the `v_proj.bias` fix.
- **Q6 and Q4 are neutral:** deltas fall inside the error bars, which is expected — at 4+ bits the quantizer has enough levels that outliers aren't destructive.
- **Q3 is a real win:** rotated 10.10 vs stock 14.35 PPL (−30%). But stock Q4 (8.81) still beats rotated Q3 at similar size, so "rotated Q3 ≈ stock Q4" does not hold on this model.
- **Q2 rescue is dramatic but not practically useful:** stock Q2 is catastrophic (21,636 PPL), rotated Q2 is 26.70 — usable in the sense of not-completely-broken, but still worse than stock Q3 (14.35) at similar file size.

### Qwen3.6-35B-A3B (hybrid Mamba + attention MoE) — pre-fix data, indicative only

These numbers were collected before the `v_proj.bias` fix and without a forward-equivalence gate in the pipeline. Treat as suggestive, not validated.

| Build | PPL | ± | Size |
|-------|-----|---|------|
| Stock Q6_K | 6.7305 | 0.0438 | 28.5 GB |
| Stock Q4_K_M | 6.8216 | 0.0445 | 21.2 GB |
| TurboGGUF Q6_K | 6.7553 | 0.0440 | 28.5 GB |
| TurboGGUF Q3_K_M | 7.2943 | 0.0485 | 16.7 GB |

`qwen35moe` has `full_attention_interval=4`: only 10 of 40 layers are classical attention. The remaining 30 are SSM layers that the standard Q/K/V/O + gate/up/down rotation recipe doesn't touch. Benefit ceiling on this architecture is inherently limited regardless of other fixes.

## Status

Shelved pending the following fixes:

1. **bf16/fp16 precision drift on pure-LLaMA/Yi architectures.** Yi-1.5-9B bf16 turbo PPL drifts to 8.42 vs stock 5.76 even with R1 alone. Not a rotation-math bug — looks like bf16→fp32→bf16 round-trip accumulation in the rotation path. Needs fp32 accumulation throughout, or at minimum a gate that refuses to save when forward logits diverge past a threshold.
2. **No forward-equivalence gate in the pipeline.** When rotation produces mathematically wrong weights, the pipeline currently saves them silently. A gate comparing original-model logits vs rotated-model logits on a fixed probe batch would catch catastrophic bugs (like the pre-fix `v_proj.bias` bug) at rotation time instead of via downstream perplexity runs.
3. **Headline claim not independently validated.** "Q2 quality that performs like Q4" comes from the QuaRot paper on Llama-3.1-8B (INT4). On Qwen2.5-3B (this repo's first architecture fully validated at bf16), rotated Q3 beats stock Q3 but does not match stock Q4. Claim should be either re-measured on Llama-3.1-8B directly with this implementation, or dropped.

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
turbogguf rotate       # Apply Hadamard rotation to model weights
turbogguf pipeline     # Full pipeline: rotate → convert → quantize
turbogguf evaluate     # Compare perplexity of GGUF models
turbogguf info         # Show memory requirements for a model
turbogguf kv-compress  # Show KV cache compression stats (TurboQuant+)
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

| Architecture | Status | Notes |
|---|---|---|
| Qwen2 / Qwen2.5 | Validated at bf16 (Qwen2.5-3B) | `v_proj.bias` rotation fixed; Q3 improves over stock Q3, Q2 still not viable |
| LLaMA / Yi | Pipeline runs, correctness not confirmed | bf16 forward drift observed on Yi-1.5-9B — see Status |
| Mistral | Pipeline runs, not re-validated post-fix | |
| Gemma 4 | Pipeline runs end-to-end, correctness not verified | |
| Qwen3 hybrid (SSM+attention MoE) | Partial by design | Only ~25% of layers are attention; rotation cannot reach SSM layers |

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

## TurboQuant+ KV Cache Compression

TurboGGUF includes [TurboQuant+](https://github.com/TheTom/turboquant_plus), a KV cache compression library that complements weight rotation. While TurboGGUF improves **weight quantization** (offline, before GGUF), TurboQuant+ compresses the **KV cache at inference time** using PolarQuant + QJL.

**Combined approach:** Use TurboGGUF rotation for better weight quantization, **plus** TurboQuant+ KV cache compression for reduced memory during inference.

### KV Cache Compression Formats

| Format | Compression | PPL vs q8_0 | Notes |
|--------|-------------|-------------|-------|
| turbo4 | 3.8x | +0.23% | Best quality after q8_0 |
| turbo3 | 4.6-5.1x | +1.06% | Maximum compression balance |
| turbo2 | 6.4x | +6.48% | Extreme compression |

### Usage

```bash
# Check KV cache savings for your model
turbogguf kv-compress --model meta-llama/Llama-3.1-8B --seq-len 8192

# Use with llama.cpp at inference time
llama-server -m turbo-Q2_K.gguf --cache-type-k turbo3 --cache-type-v turbo3
```

### Python API

```python
from turbogguf.turboquant_plus import KVCacheCompressor

compressor = KVCacheCompressor(head_dim=128, k_bits=3, v_bits=3)
compressed = compressor.compress(k_cache, v_cache)
k_hat, v_hat = compressor.decompress(compressed)
print(compressor.memory_stats(seq_len=4096, num_layers=32, num_heads=32))
```

## References

- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456) (NeurIPS 2024)
- [TurboQuant: Redefining AI Efficiency with Extreme Compression](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence](https://arxiv.org/abs/2402.04396) (ICML 2024)
- [TurboQuant+ (KV Cache Compression)](https://github.com/TheTom/turboquant_plus)

## Tests

```bash
pytest -v tests/
```

## License

MIT
