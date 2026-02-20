#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-0}"

torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" flow_train_ARC.py \
  --ddp \
  --data-root "raw_data/ARC-AGI" \
  --train-split "training" \
  --eval-split "evaluation" \
  --num-demos 3 \
  --image-size 30 \
  --num-colors 12 \
  --embed-dim 512 \
  --depth 10 \
  --num-heads 8 \
  --mlp-ratio 4.0 \
  --dropout 0.0 \
  --no-framewise-causal-attention \
  --attention-backend "auto" \
  --batch-size 4 \
  --eval-batch-size 8 \
  --log-every-steps 5 \
  --eval-every-steps "${EVAL_EVERY_STEPS}" \
  --epochs 20 \
  --learning-rate 2e-4 \
  --weight-decay 0 \
  --include-rearc \
  --bf16-autocast \
  --sample-steps 40 \
  --save-path "saves/flow_context_vit/checkpoint_last.pt" \
  --best-save-path "saves/flow_context_vit/checkpoint_best.pt" \
  --use-wandb \
  --wandb-project "VisionARC" \
  --wandb-run-name "flow-context-vit-ddp"
