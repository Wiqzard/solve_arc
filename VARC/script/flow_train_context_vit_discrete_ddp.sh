#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# -----------------------------------------------------------------------------
# Launcher / distributed defaults
# -----------------------------------------------------------------------------
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"   # choices: nccl, gloo
DIST_URL="${DIST_URL:-env://}"

# -----------------------------------------------------------------------------
# Dataset + loader defaults
# -----------------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-raw_data/ARC-AGI}"
TRAIN_SPLIT="${TRAIN_SPLIT:-training}"
EVAL_SPLIT="${EVAL_SPLIT:-evaluation}"
MAX_DEMOS="${MAX_DEMOS:-3}"
IMAGE_SIZE="${IMAGE_SIZE:-30}"
PATCH_SIZE="${PATCH_SIZE:-1}"
NUM_COLORS="${NUM_COLORS:-12}"
NUM_WORKERS="${NUM_WORKERS:-0}"

INCLUDE_REARC="${INCLUDE_REARC:-0}"    # 0/1
REARC_PATH="${REARC_PATH:-raw_data/re_arc}"
REARC_LIMIT="${REARC_LIMIT:--1}"

INCLUDE_BARC="${INCLUDE_BARC:-0}"      # 0/1
BARC_PATH="${BARC_PATH:-raw_data/BARC}"
BARC_LIMIT="${BARC_LIMIT:--1}"

FLOW_TRAIN_TRANSLATION_AUG="${FLOW_TRAIN_TRANSLATION_AUG:-0}"  # 0/1
FLOW_TRAIN_RESOLUTION_AUG="${FLOW_TRAIN_RESOLUTION_AUG:-0}"    # 0/1
NESTED_DROPOUT="${NESTED_DROPOUT:-0}"                          # 0/1

# -----------------------------------------------------------------------------
# Model defaults
# -----------------------------------------------------------------------------
MODEL_ARCH="${MODEL_ARCH:-flow_vit}"   # choices: flow_vit, flow_vit_looped
N_LOOPS="${N_LOOPS:-2}"
EMBED_DIM="${EMBED_DIM:-512}"
DEPTH="${DEPTH:-10}"
NUM_HEADS="${NUM_HEADS:-8}"
MLP_RATIO="${MLP_RATIO:-4.0}"
DROPOUT="${DROPOUT:-0.1}"
FRAMEWISE_CAUSAL_ATTENTION="${FRAMEWISE_CAUSAL_ATTENTION:-false}"          # true/false
ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"                              # auto/flex/sdpa
MASK_PAD_ATTENTION="${MASK_PAD_ATTENTION:-false}"                           # true/false
MASK_INTRA_FRAME_PAD_ATTENTION="${MASK_INTRA_FRAME_PAD_ATTENTION:-false}"  # true/false
ROPE_3D="${ROPE_3D:-false}"                                                 # true/false
ROPE_BASE="${ROPE_BASE:-256.0}"

# -----------------------------------------------------------------------------
# Optimization + training defaults
# -----------------------------------------------------------------------------
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-20}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"  # choices: cosine, none
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
VERBOSE="${VERBOSE:-0}"                 # 0/1

BF16_AUTOCAST="${BF16_AUTOCAST:-0}"     # 0/1
COMPILE="${COMPILE:-0}"                 # 0/1
COMPILE_MODE="${COMPILE_MODE:-default}" # default/reduce-overhead/max-autotune

# -----------------------------------------------------------------------------
# Discrete flow defaults
# -----------------------------------------------------------------------------
LOSS_ON_TARGET_ONLY="${LOSS_ON_TARGET_ONLY:-0}"         # 0/1
LOSS_FUNCTION="${LOSS_FUNCTION:-generalized_kl}"        # cross_entropy/generalized_kl
DISCRETE_RATE="${DISCRETE_RATE:-5.0}"
DISCRETE_SCHEDULER="${DISCRETE_SCHEDULER:-exponential}" # exponential/condot/polynomial/vp/linear_vp/cosine
DISCRETE_POLY_N="${DISCRETE_POLY_N:-2.0}"
DISCRETE_VP_BETA_MIN="${DISCRETE_VP_BETA_MIN:-0.1}"
DISCRETE_VP_BETA_MAX="${DISCRETE_VP_BETA_MAX:-20.0}"
REVERSE_SAMPLER="${REVERSE_SAMPLER:-sample}"            # sample/argmax
SAMPLE_STEPS="${SAMPLE_STEPS:-40}"
TRAIN_TIME_DISCRETIZATION_STEPS="${TRAIN_TIME_DISCRETIZATION_STEPS:-1000}"

# -----------------------------------------------------------------------------
# Evaluation + checkpoint defaults
# -----------------------------------------------------------------------------
EVAL_EVERY="${EVAL_EVERY:-1}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-0}"
SAVE_PATH="${SAVE_PATH:-saves/flow_context_vit_discrete/checkpoint_last.pt}"
BEST_SAVE_PATH="${BEST_SAVE_PATH:-saves/flow_context_vit_discrete/checkpoint_best.pt}"

# -----------------------------------------------------------------------------
# W&B defaults
# -----------------------------------------------------------------------------
USE_WANDB="${USE_WANDB:-0}"  # 0/1
WANDB_PROJECT="${WANDB_PROJECT:-VisionARC}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-flow-context-vit-discrete}"
WANDB_NUM_VIS_SAMPLES="${WANDB_NUM_VIS_SAMPLES:-8}"
WANDB_VIS_SCALE="${WANDB_VIS_SCALE:-8}"
WANDB_TRAIN_VIS_EVERY_STEPS="${WANDB_TRAIN_VIS_EVERY_STEPS:-0}"
WANDB_TRAIN_VIS_SAMPLES="${WANDB_TRAIN_VIS_SAMPLES:-2}"

dist_args=(
  --ddp
  --dist-backend "${DIST_BACKEND}"
  --dist-url "${DIST_URL}"
)

dataset_args=(
  --data-root "${DATA_ROOT}"
  --train-split "${TRAIN_SPLIT}"
  --eval-split "${EVAL_SPLIT}"
  --max-demos "${MAX_DEMOS}"
  --image-size "${IMAGE_SIZE}"
  --patch-size "${PATCH_SIZE}"
  --num-colors "${NUM_COLORS}"
  --num-workers "${NUM_WORKERS}"
  --rearc-path "${REARC_PATH}"
  --rearc-limit "${REARC_LIMIT}"
  --barc-path "${BARC_PATH}"
  --barc-limit "${BARC_LIMIT}"
)

model_args=(
  --model-arch "${MODEL_ARCH}"
  --n-loops "${N_LOOPS}"
  --embed-dim "${EMBED_DIM}"
  --depth "${DEPTH}"
  --num-heads "${NUM_HEADS}"
  --mlp-ratio "${MLP_RATIO}"
  --dropout "${DROPOUT}"
  --framewise-causal-attention "${FRAMEWISE_CAUSAL_ATTENTION}"
  --attention-backend "${ATTENTION_BACKEND}"
  --mask-pad-attention "${MASK_PAD_ATTENTION}"
  --mask-intra-frame-pad-attention "${MASK_INTRA_FRAME_PAD_ATTENTION}"
  --rope-3d "${ROPE_3D}"
  --rope-base "${ROPE_BASE}"
)

train_args=(
  --seed "${SEED}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --lr-scheduler "${LR_SCHEDULER}"
  --min-learning-rate "${MIN_LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --log-every-steps "${LOG_EVERY_STEPS}"
  --compile-mode "${COMPILE_MODE}"
)

discrete_args=(
  --loss-function "${LOSS_FUNCTION}"
  --discrete-rate "${DISCRETE_RATE}"
  --discrete-scheduler "${DISCRETE_SCHEDULER}"
  --discrete-poly-n "${DISCRETE_POLY_N}"
  --discrete-vp-beta-min "${DISCRETE_VP_BETA_MIN}"
  --discrete-vp-beta-max "${DISCRETE_VP_BETA_MAX}"
  --reverse-sampler "${REVERSE_SAMPLER}"
  --sample-steps "${SAMPLE_STEPS}"
  --train-time-discretization-steps "${TRAIN_TIME_DISCRETIZATION_STEPS}"
)

eval_ckpt_args=(
  --eval-every "${EVAL_EVERY}"
  --eval-every-steps "${EVAL_EVERY_STEPS}"
  --save-path "${SAVE_PATH}"
  --best-save-path "${BEST_SAVE_PATH}"
)

wandb_args=(
  --wandb-project "${WANDB_PROJECT}"
  --wandb-run-name "${WANDB_RUN_NAME}"
  --wandb-num-vis-samples "${WANDB_NUM_VIS_SAMPLES}"
  --wandb-vis-scale "${WANDB_VIS_SCALE}"
  --wandb-train-vis-every-steps "${WANDB_TRAIN_VIS_EVERY_STEPS}"
  --wandb-train-vis-samples "${WANDB_TRAIN_VIS_SAMPLES}"
)

flag_args=()
if [[ "${INCLUDE_REARC}" == "1" ]]; then
  flag_args+=(--include-rearc)
fi
if [[ "${INCLUDE_BARC}" == "1" ]]; then
  flag_args+=(--include-barc)
fi
if [[ "${FLOW_TRAIN_TRANSLATION_AUG}" == "1" ]]; then
  flag_args+=(--flow-train-translation-aug)
fi
if [[ "${FLOW_TRAIN_RESOLUTION_AUG}" == "1" ]]; then
  flag_args+=(--flow-train-resolution-aug)
fi
if [[ "${NESTED_DROPOUT}" == "1" ]]; then
  flag_args+=(--nested-dropout)
fi
if [[ "${LOSS_ON_TARGET_ONLY}" == "1" ]]; then
  flag_args+=(--loss-on-target-only)
fi
if [[ "${BF16_AUTOCAST}" == "1" ]]; then
  flag_args+=(--bf16-autocast)
fi
if [[ "${COMPILE}" == "1" ]]; then
  flag_args+=(--compile)
fi
if [[ "${VERBOSE}" == "1" ]]; then
  flag_args+=(--verbose)
fi
if [[ "${USE_WANDB}" == "1" ]]; then
  flag_args+=(--use-wandb)
fi

torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" flow_train_discrete_ARC.py \
  "${dist_args[@]}" \
  "${dataset_args[@]}" \
  "${model_args[@]}" \
  "${train_args[@]}" \
  "${discrete_args[@]}" \
  "${eval_ckpt_args[@]}" \
  "${wandb_args[@]}" \
  "${flag_args[@]}"
