#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# 2D Cellular Automata dataset generation (paper-aligned recipe):
# - long rollouts
# - random spatiotemporal windows
# - mixed prediction horizons (1 and 5)
# -----------------------------------------------------------------------------
python script/generate_ca_2d_dataset.py \
  --output-root "raw_data/CA-2D-DFM" \
  --train-split "training" \
  --eval-split "evaluation" \
  --num-train-tasks 2048 \
  --num-eval-tasks 256 \
  --train-examples-per-task 12 \
  --test-examples-per-task 4 \
  --sim-size 96 \
  --crop-size 31 \
  --rollout-steps 1000 \
  --warmup-steps 60 \
  --densities "0.15,0.25,0.35,0.45,0.55" \
  --horizons "1,5" \
  --rule-profile "paper_like" \
  --seed 42 \
  --overwrite

# -----------------------------------------------------------------------------
# Discrete flow-matching pretraining on 2D CA data.
# -----------------------------------------------------------------------------
python flow_train_discrete_ARC.py \
  --data-root "raw_data/CA-2D-DFM" \
  --train-split "training" \
  --eval-split "evaluation" \
  --max-demos 10 \
  --image-size 32 \
  --num-colors 12 \
  --model-arch "flow_vit" \
  --n-loops 1 \
  --patch-size 2 \
  --embed-dim 512 \
  --depth 10 \
  --num-heads 8 \
  --mlp-ratio 4.0 \
  --dropout 0.0 \
  --framewise-causal-attention true \
  --attention-backend "auto" \
  --mask-pad-attention true \
  --mask-intra-frame-pad-attention false \
  --rope-3d true \
  --rope-base 256.0 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --epochs 20 \
  --learning-rate 3e-4 \
  --lr-scheduler "cosine" \
  --min-learning-rate 1e-6 \
  --weight-decay 0.0 \
  --max-grad-norm 1.0 \
  --num-workers 8 \
  --seed 42 \
  --compile true \
  --compile-mode "default" \
  --bf16-autocast true \
  --flow-train-translation-aug false \
  --flow-train-resolution-aug false \
  --nested-dropout false \
  --loss-function "cross_entropy" \
  --discrete-scheduler "cosine" \
  --discrete-rate 5.0 \
  --discrete-poly-n 2.0 \
  --discrete-vp-beta-min 0.1 \
  --discrete-vp-beta-max 20.0 \
  --reverse-sampler "sample" \
  --sample-steps 40 \
  --train-time-discretization-steps 1000 \
  --eval-every 1 \
  --eval-every-steps 500 \
  --save-path "saves/flow_context_vit_discrete_ca2d/checkpoint_last.pt" \
  --best-save-path "saves/flow_context_vit_discrete_ca2d/checkpoint_best.pt" \
  --use-wandb false \
  --wandb-project "solve_arc-VARC" \
  --wandb-run-name "flow-context-vit-discrete-ca2d"
