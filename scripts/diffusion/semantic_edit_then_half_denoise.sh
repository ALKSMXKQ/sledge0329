#!/usr/bin/env bash
set -euo pipefail

JOB_NAME=semantic_edit_then_half_denoise
CONFIG_PATH=/path/to/expanded_diffusion_config.yaml
AUTOENCODER_CHECKPOINT=/path/to/autoencoder_checkpoint.ckpt
DIFFUSION_CHECKPOINT=/path/to/diffusion_checkpoint
INPUT_DIR=/path/to/source_autoencoder_cache
OUTPUT_DIR=/path/to/output_semantic_half_denoise
PROMPT="创建突发行人横穿马路场景，即自车前方有一名行人从路边突然横穿进入车道"

export CUDA_VISIBLE_DEVICES=0

python $SLEDGE_DEVKIT_ROOT/sledge/script/run_semantic_edit_then_half_denoise.py \
  --input-dir "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --prompt "$PROMPT" \
  --config "$CONFIG_PATH" \
  --autoencoder-checkpoint "$AUTOENCODER_CHECKPOINT" \
  --diffusion-checkpoint "$DIFFUSION_CHECKPOINT" \
  --num-inference-timesteps 24 \
  --guidance-scale 3.0 \
  --low-noise-start-step-seq 6,8,10 \
  --repair-attempts 6 \
  --variants-per-scene 3 \
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
