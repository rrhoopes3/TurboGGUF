#!/bin/bash
# TurboGGUF: rotate → convert → quantize pipeline
#
# Usage: ./scripts/quantize.sh <model_id> <quant_type> <output.gguf> [llama_cpp_dir]
#
# Example:
#   ./scripts/quantize.sh meta-llama/Llama-3.1-8B Q2_K ./llama3.1-8b-turbo-Q2_K.gguf /path/to/llama.cpp

set -euo pipefail

MODEL_ID="${1:?Usage: quantize.sh <model_id> <quant_type> <output.gguf> [llama_cpp_dir]}"
QUANT_TYPE="${2:?Specify quant type (Q2_K, Q3_K_S, Q4_K_M, etc.)}"
OUTPUT="${3:?Specify output GGUF path}"
LLAMA_CPP="${4:-$HOME/llama.cpp}"

ROTATED_DIR="$(mktemp -d)/turbogguf_rotated"
CONVERTER="${LLAMA_CPP}/convert_hf_to_gguf.py"
QUANTIZER="${LLAMA_CPP}/build/bin/llama-quantize"

# Try alternative quantizer path
if [ ! -f "$QUANTIZER" ]; then
    QUANTIZER="${LLAMA_CPP}/llama-quantize"
fi

echo "============================================"
echo "TurboGGUF Pipeline"
echo "Model:  $MODEL_ID"
echo "Quant:  $QUANT_TYPE"
echo "Output: $OUTPUT"
echo "============================================"
echo

echo "[1/3] Rotating model weights..."
turbogguf rotate --model "$MODEL_ID" --output "$ROTATED_DIR" --seed 42
echo

echo "[2/3] Converting to GGUF (F16)..."
python "$CONVERTER" "$ROTATED_DIR" --outtype f16 --outfile "${ROTATED_DIR}/model.gguf"
echo

echo "[3/3] Quantizing to ${QUANT_TYPE}..."
"$QUANTIZER" "${ROTATED_DIR}/model.gguf" "$OUTPUT" "$QUANT_TYPE"
echo

# Cleanup
rm -rf "$ROTATED_DIR"

echo "============================================"
echo "Done! Output: $OUTPUT"
echo "Size: $(du -h "$OUTPUT" | cut -f1)"
echo "Load in LM Studio — standard GGUF file."
echo "============================================"
