#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-500}"

torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" flow_train_discrete_ARC.py \
  --ddp \
  --data-root "raw_data/ARC-AGI" \
  --train-split "training" \
  --eval-split "evaluation" \
  --include-rearc \
  --num-demos 3 \
  --image-size 32 \
  --num-colors 12 \
  --embed-dim 512 \
  --depth 10 \
  --num-heads 8 \
  --mlp-ratio 4.0 \
  --dropout 0.0 \
  --attention-backend "auto" \
  --batch-size 4 \
  --eval-batch-size 8 \
  --log-every-steps 5 \
  --eval-every-steps "${EVAL_EVERY_STEPS}" \
  --epochs 100 \
  --learning-rate 3e-4 \
  --lr-scheduler "cosine" \
  --weight-decay 0 \
  --bf16-autocast \
  --discrete-rate 5.0 \
  --reverse-sampler "sample" \
  --sample-steps 40 \
  --save-path "saves/flow_context_vit_discrete/checkpoint_last.pt" \
  --best-save-path "saves/flow_context_vit_discrete/checkpoint_best.pt" \
  --use-wandb \
  --wandb-project "VisionARC" \
  --wandb-num-vis-samples 8 \
  --wandb-vis-scale 8 \
  --wandb-run-name "flow-context-vit-discrete-ddp"
