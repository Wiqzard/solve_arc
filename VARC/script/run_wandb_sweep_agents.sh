#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <entity/project/sweep_id>"
  exit 1
fi

SWEEP_ID="$1"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Suppress noisy pydantic field-attribute warnings emitted by wandb/pydantic internals.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:.*attribute with value .* was provided to the .*Field\\(\\).* function, which has no effect.*:UserWarning}"

# Each sweep run uses torch.distributed.run with NPROC_PER_NODE GPUs.
# Run a single agent per node to avoid GPU oversubscription.
wandb agent "${SWEEP_ID}"

# bash script/run_wandb_sweep_agents.sh   wiqzard/solve_arc-VARC/kp30fgij 8
