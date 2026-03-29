#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=/path/to/expanded_diffusion_config.yaml
AUTOENCODER_CHECKPOINT=/path/to/autoencoder_checkpoint.ckpt
DIFFUSION_CHECKPOINT=/path/to/diffusion_checkpoint
ORIGINAL_DIR=/home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache
EDITED_DIR=/home16T/home8T_1/leitingting/sledge_workspace/exp/caches/tiered_crossing_raw_cache
OUTPUT_DIR=/home16T/home8T_1/leitingting/sledge_workspace/exp/exp/half_denoise_from_tiered_cache

export CUDA_VISIBLE_DEVICES=0

python $SLEDGE_DEVKIT_ROOT/sledge/script/run_half_denoise_from_tiered_cache.py \
  --original-dir "$ORIGINAL_DIR" \
  --edited-dir "$EDITED_DIR" \
  --output "$OUTPUT_DIR" \
  --config "$CONFIG_PATH" \
  --autoencoder-checkpoint "$AUTOENCODER_CHECKPOINT" \
  --diffusion-checkpoint "$DIFFUSION_CHECKPOINT" \
  --num-inference-timesteps 24 \
  --guidance-scale 3.0 \
  --low-noise-start-step-seq 6,8,10 \
  --repair-attempts 6 \
  --alignment-threshold 0.70 \
  --min-preservation-ratio 0.95 \
  --diff-threshold 1e-4 \
  --diff-mask-dilation 2 \
  --roi-mask-dilation 1 \
  --pedestrian-roi-strength 1.0 \
  --roadside-anchor-strength 1.0 \
  --lane-anchor-strength 0.95 \
  --crossing-corridor-strength 0.80 \
  --generic-roi-strength 0.90 \
  --seed 0 \
  --save-visuals \
  --save-latents
