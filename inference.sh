#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

# libgomp requires a positive integer. Some managed shells export an empty or
# otherwise invalid value, which makes every PyTorch worker print a warning.
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=1
fi

MODEL_PATH=${MODEL_PATH:-pretrained_models/CAMO}
INPUT_DIR=${INPUT_DIR:-samples/FLIR-IVSR/lq}
OUTPUT_DIR=${OUTPUT_DIR:-results/FLIR-IVSR}
GPU_ID=${GPU_ID:-0}

test -d "$MODEL_PATH" || { echo "Missing CAMO checkpoint: $MODEL_PATH" >&2; exit 2; }
test -d "$INPUT_DIR" || { echo "Missing input directory: $INPUT_DIR" >&2; exit 2; }

CUDA_VISIBLE_DEVICES="$GPU_ID" python inference.py \
    --input_dir "$INPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --output_path "$OUTPUT_DIR" \
    --is_vae_st
