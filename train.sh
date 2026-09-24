#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

export TOKENIZERS_PARALLELISM=false

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=1
fi

MODEL_PATH=${MODEL_PATH:-pretrained_models/CogVideoX1.5-5B}
DATA_ROOT=${DATA_ROOT:-datasets/train/FLIR-IVSR-video}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/CAMO}
NUM_PROCESSES=${NUM_PROCESSES:-4}
REPORT_TO=${REPORT_TO:-none}

test -d "$MODEL_PATH" || { echo "Missing base model: $MODEL_PATH" >&2; exit 2; }
test -d "$DATA_ROOT" || { echo "Missing training dataset: $DATA_ROOT" >&2; exit 2; }
test -d "$DATA_ROOT/gt" || { echo "Missing ground-truth video directory: $DATA_ROOT/gt" >&2; exit 2; }
test -d "$DATA_ROOT/turb_lr" || { echo "Missing turbulent LR video directory: $DATA_ROOT/turb_lr" >&2; exit 2; }

accelerate launch \
    --config_file finetune/accelerate_config.yaml \
    --num_processes "$NUM_PROCESSES" \
    finetune/train.py \
    --model_path "$MODEL_PATH" \
    --model_name camo \
    --model_type real-sr \
    --training_type sft \
    --output_dir "$OUTPUT_DIR" \
    --report_to "$REPORT_TO" \
    --tracker_name CAMO \
    --data_root "$DATA_ROOT" \
    --video_column "$DATA_ROOT/gt" \
    --lq_video_column "$DATA_ROOT/turb_lr" \
    --train_resolution 25x320x640 \
    --train_epochs 1000 \
    --train_steps 10000 \
    --seed 42 \
    --batch_size 2 \
    --gradient_accumulation_steps 1 \
    --mixed_precision bf16 \
    --learning_rate 2e-5 \
    --gradient_checkpointing true \
    --max_grad_norm 0.1 \
    --lr_scheduler constant_with_warmup \
    --num_workers 8 \
    --pin_memory true \
    --nccl_timeout 1800 \
    --stastic_frequency 500 \
    --checkpointing_steps 1000 \
    --checkpointing_limit 2 \
    --do_validation false \
    --is_latent false \
    --is_cache true \
    --empty_prompt true \
    --prompt_cache prompt_embeddings \
    --sr_noise_step 399 \
    --noise_step 0 \
    --degradation_config finetune/configs/degradation.yaml \
    --enable_random_frame_drop true \
    --max_frame_drop_count 5
