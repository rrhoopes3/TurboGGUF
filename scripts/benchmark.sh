#!/bin/bash
# Benchmark: compare rotated vs unrotated quantization quality
#
# Usage: ./scripts/benchmark.sh <model_id> <quant_type> [llama_cpp_dir]
#
# Produces two GGUF files and compares perplexity:
#   1. Standard quantization (no rotation)
#   2. TurboGGUF quantization (with rotation)

set -euo pipefail

MODEL_ID="${1:?Usage: benchmark.sh <model_id> <quant_type> [llama_cpp_dir]}"
QUANT_TYPE="${2:?Specify quant type (Q2_K, Q3_K_S, etc.)}"
LLAMA_CPP="${3:-$HOME/llama.cpp}"

CONVERTER="${LLAMA_CPP}/convert_hf_to_gguf.py"
QUANTIZER="${LLAMA_CPP}/build/bin/llama-quantize"
PERPLEXITY="${LLAMA_CPP}/build/bin/llama-perplexity"

if [ ! -f "$QUANTIZER" ]; then QUANTIZER="${LLAMA_CPP}/llama-quantize"; fi
if [ ! -f "$PERPLEXITY" ]; then PERPLEXITY="${LLAMA_CPP}/llama-perplexity"; fi

MODEL_NAME=$(echo "$MODEL_ID" | tr '/' '-')
WORKDIR="./benchmark_${MODEL_NAME}_${QUANT_TYPE}"
mkdir -p "$WORKDIR"

echo "============================================"
echo "TurboGGUF Benchmark"
echo "Model: $MODEL_ID"
echo "Quant: $QUANT_TYPE"
echo "============================================"
echo

# 1. Standard quantization (no rotation)
echo "[1/4] Converting original model to GGUF..."
python "$CONVERTER" "$MODEL_ID" --outtype f16 --outfile "${WORKDIR}/original-f16.gguf"
echo

echo "[2/4] Standard ${QUANT_TYPE} (no rotation)..."
"$QUANTIZER" "${WORKDIR}/original-f16.gguf" "${WORKDIR}/standard-${QUANT_TYPE}.gguf" "$QUANT_TYPE"
echo

# 2. TurboGGUF quantization (with rotation)
echo "[3/4] TurboGGUF ${QUANT_TYPE} (with rotation)..."
turbogguf pipeline \
    --model "$MODEL_ID" \
    --quant "$QUANT_TYPE" \
    --output "${WORKDIR}/turbo-${QUANT_TYPE}.gguf" \
    --llama-cpp "$LLAMA_CPP"
echo

# 3. Compare perplexity
echo "[4/4] Evaluating perplexity..."
turbogguf evaluate \
    --gguf "${WORKDIR}/standard-${QUANT_TYPE}.gguf" \
    --gguf "${WORKDIR}/turbo-${QUANT_TYPE}.gguf" \
    --llama-perplexity "$PERPLEXITY" \
    --output "${WORKDIR}/results.json"

echo
echo "Results saved to ${WORKDIR}/results.json"
echo "GGUF files in ${WORKDIR}/"
