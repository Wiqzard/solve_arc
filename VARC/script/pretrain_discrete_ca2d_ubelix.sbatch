#!/usr/bin/env bash
#SBATCH --job-name=ca2d_dfm_pt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=h200:8
#SBATCH --time=08:00:00
#SBATCH --output=outputs/ubelix_ca2d/slurm-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p outputs/ubelix_ca2d

module load Anaconda3 >/dev/null 2>&1
eval "$(conda shell.bash hook)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mira}"
conda activate "${CONDA_ENV_NAME}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

DATA_ROOT="${DATA_ROOT:-raw_data/CA-2D-DFM}"
TRAIN_TASKS="${TRAIN_TASKS:-2048}"
EVAL_TASKS="${EVAL_TASKS:-256}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"

python script/generate_ca_2d_dataset.py \
  --output-root "${DATA_ROOT}" \
  --train-split "training" \
  --eval-split "evaluation" \
  --num-train-tasks "${TRAIN_TASKS}" \
  --num-eval-tasks "${EVAL_TASKS}" \
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

torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" flow_train_discrete_ARC.py \
  --ddp \
  --dist-backend "nccl" \
  --dist-url "env://" \
  --data-root "${DATA_ROOT}" \
  --train-split "training" \
  --eval-split "evaluation" \
  --max-demos 10 \
  --image-size 32 \
  --num-colors 12 \
  --num-workers 8 \
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
  --epochs "${TRAIN_EPOCHS}" \
  --learning-rate 3e-4 \
  --lr-scheduler "cosine" \
  --min-learning-rate 1e-6 \
  --weight-decay 0.0 \
  --max-grad-norm 1.0 \
  --seed 42 \
  --compile true \
  --compile-mode "default" \
  --bf16-autocast true \
  --flow-train-translation-aug false \
  --flow-train-resolution-aug false \
  --nested-dropout false \
  --loss-function "cross_entropy" \
  --discrete-rate 5.0 \
  --discrete-scheduler "cosine" \
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
  --wandb-run-name "flow-context-vit-discrete-ca2d-${SLURM_JOB_ID}"
