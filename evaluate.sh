#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

PRED_DIR=${PRED_DIR:-results/FLIR-IVSR}
GT_DIR=${GT_DIR:-samples/FLIR-IVSR/gt}
OUTPUT_DIR=${OUTPUT_DIR:-results/FLIR-IVSR/metrics}
METRICS=${METRICS:-psnr,ssim,lpips,dists,musiq,brisque,dover,fastervqa}
DEVICE=${DEVICE:-cuda}

test -d "$PRED_DIR" || { echo "Missing prediction directory: $PRED_DIR" >&2; exit 2; }
test -d "$GT_DIR" || { echo "Missing ground-truth directory: $GT_DIR" >&2; exit 2; }

python evaluate.py \
    --pred "$PRED_DIR" \
    --gt "$GT_DIR" \
    --out "$OUTPUT_DIR" \
    --metrics "$METRICS" \
    --device "$DEVICE"
