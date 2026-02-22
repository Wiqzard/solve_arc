#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=8

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-500}"
NESTED_DROPOUT="${NESTED_DROPOUT:-0}"

nested_dropout_args=()
if [[ "${NESTED_DROPOUT}" == "1" ]]; then
  nested_dropout_args+=(--nested-dropout)
fi

dataset_args=(
  --data-root "raw_data/ARC-AGI"
  --train-split "training"
  --eval-split "evaluation"
  --include-rearc
  --rearc-path "raw_data/re_arc"
  #--rearc-limit -1
  --include-barc
  --barc-path "raw_data/BARC"
  #--barc-limit -1
  --max-demos 3
  --image-size 32
  --num-colors 12
  --flow-train-resolution-aug
  --flow-train-translation-aug
  --num-workers 8
)

model_args=(
  --embed-dim 576
  --depth 16
  --num-heads 18
  --mlp-ratio 4.0
  --dropout 0.0
  --attention-backend "auto"
  --framewise-causal-attention
  --rope-3d
  --rope-base 256.0
)

training_args=(
  --seed 42
  --batch-size 4
  --eval-batch-size 4
  --log-every-steps 5
  --eval-every-steps "${EVAL_EVERY_STEPS}"
  --epochs 20 #20 #100
  --learning-rate 3e-4
  --lr-scheduler "cosine"
  --min-learning-rate 1e-6
  --weight-decay 0
  --bf16-autocast
  --compile
  --discrete-scheduler "cosine"
  --sample-steps 40
  --save-path "saves/flow_context_vit_discrete/checkpoint_last.pt"
  --best-save-path "saves/flow_context_vit_discrete/checkpoint_best.pt"
  # --reverse-sampler "sample"
)
#--wandb-project "VisionARC"

log_args=(
  --use-wandb
  --wandb-project "solve_arc-VARC"
  --wandb-num-vis-samples 8
  --wandb-vis-scale 8
  --wandb-train-vis-every-steps "${EVAL_EVERY_STEPS}"
  --wandb-train-vis-samples 2
  --wandb-run-name "flow-context-vit-discrete-ddp"
)

torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" flow_train_discrete_ARC.py \
  --ddp \
  "${dataset_args[@]}" \
  "${nested_dropout_args[@]}" \
  "${model_args[@]}" \
  "${training_args[@]}" \
  "${log_args[@]}"
