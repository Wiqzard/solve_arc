#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <entity/project/sweep_id> [num_agents]"
  exit 1
fi

SWEEP_ID="$1"
NUM_AGENTS="${2:-${NUM_AGENTS:-1}}"

for ((i=0; i<NUM_AGENTS; i++)); do
  CUDA_VISIBLE_DEVICES="${i}" wandb agent "${SWEEP_ID}" &
done

wait
