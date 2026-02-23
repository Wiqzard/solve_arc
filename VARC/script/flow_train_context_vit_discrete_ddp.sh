#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=8
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# -----------------------------------------------------------------------------
# Distributed
# -----------------------------------------------------------------------------
distributed_args=(
  --ddp
  --dist-backend "nccl"
  --dist-url "env://"
)

# -----------------------------------------------------------------------------
# Dataset / Loader
# -----------------------------------------------------------------------------
dataset_args=(
  --data-root "raw_data/ARC-AGI"
  --train-split "training"
  --eval-split "evaluation"
  --max-demos 10
  --image-size 32
  --num-colors 12
  --num-workers 8
  --include-rearc
  --include-barc
  --flow-train-translation-aug
  --flow-train-resolution-aug
  --rearc-path "raw_data/re_arc"
  --rearc-limit -1
  --barc-path "raw_data/BARC"
  --barc-limit -1
)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model_args=(
  --model-arch "flow_vit"
  --n-loops 1
  --patch-size 2
  --embed-dim 512
  --depth 10
  --num-heads 8
  --mlp-ratio 4.0
  --dropout 0.0
  --framewise-causal-attention true #false
  --attention-backend "auto"
  --mask-pad-attention true #false
  --mask-intra-frame-pad-attention false
  --rope-3d true #false
  --rope-base 256.0
)

# -----------------------------------------------------------------------------
# Optimization / Training
# -----------------------------------------------------------------------------
training_args=(
  --bf16-autocast
  --compile
  --batch-size 8
  --eval-batch-size 8
  --epochs 20
  --learning-rate 3e-4
  --lr-scheduler "cosine"
  --min-learning-rate 1e-6
  --weight-decay 0.0
  --max-grad-norm 1.0
  --seed 42
  --log-every-steps 5
  --compile-mode "default"
  --nested-dropout
  # --loss-on-target-only
)

# -----------------------------------------------------------------------------
# Discrete Flow
# -----------------------------------------------------------------------------
discrete_args=(
  --loss-function "cross_entropy"
  --discrete-rate 5.0
  --discrete-scheduler "cosine"
  --discrete-poly-n 2.0
  --discrete-vp-beta-min 0.1
  --discrete-vp-beta-max 20.0
  --reverse-sampler "sample"
  --sample-steps 40
  --train-time-discretization-steps 1000
)

# -----------------------------------------------------------------------------
# Eval / Checkpoint
# -----------------------------------------------------------------------------
eval_args=(
  --eval-every 1
  --eval-every-steps 500
  --save-path "saves/flow_context_vit_discrete/checkpoint_last.pt"
  --best-save-path "saves/flow_context_vit_discrete/checkpoint_best.pt"
)

# -----------------------------------------------------------------------------
# W&B settings (values kept at defaults)
# -----------------------------------------------------------------------------
wandb_args=(
  --use-wandb
  --wandb-project "VisionARC"
  --wandb-run-name "flow-context-vit-discrete"
  --wandb-num-vis-samples 8
  --wandb-vis-scale 8
  --wandb-train-vis-every-steps 500
  --wandb-train-vis-samples 2
)

# -----------------------------------------------------------------------------
# Optional boolean flags (default is OFF). Uncomment to enable.
# -----------------------------------------------------------------------------
optional_flag_args=(
  # --verbose
)

torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" flow_train_discrete_ARC.py \
  "${distributed_args[@]}" \
  "${dataset_args[@]}" \
  "${model_args[@]}" \
  "${training_args[@]}" \
  "${discrete_args[@]}" \
  "${eval_args[@]}" \
  "${wandb_args[@]}" \
  "${optional_flag_args[@]}"
