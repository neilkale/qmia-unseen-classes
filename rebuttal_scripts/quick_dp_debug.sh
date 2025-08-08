#!/usr/bin/env bash
set -euo pipefail

# Single-GPU
export CUDA_VISIBLE_DEVICES=0

MODEL_DIR=./models/
DATA_DIR=./data/

# Fixed DP parameters
DP_ENABLE="--enable_dp"
DP_MAX_GRAD_NORM="--dp_max_grad_norm 1.0"
# Use noise multiplier mode for debugging epsilon updates
DP_NOISE_MULTIPLIER="--dp_noise_multiplier 1.0"
DP_TARGET_EPSILON=""  # leave empty to disable target-epsilon mode
DP_TARGET_DELTA="--dp_target_delta 8e-5"
DP_SECURE_RNG=""  # set to "--dp_secure_rng" to enable secure RNG

# Base setup
BASE_ARCHITECTURE=cifar-resnet-50-dp
BASE_DATASET=cifar20/0_16

# Train base model (short run)
echo "[quick_dp_debug] Training base model (noise=1.0) for 3 epochs on single GPU (no scheduler)" | tee -a quick_dp_debug.log
python train_base.py --dataset=$BASE_DATASET --architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
  --batch_size=64 --scheduler="" --lr=0.05 --weight_decay=5e-4 --epochs=1 \
  --rerun \
  $DP_ENABLE $DP_MAX_GRAD_NORM $DP_NOISE_MULTIPLIER $DP_TARGET_DELTA $DP_SECURE_RNG 