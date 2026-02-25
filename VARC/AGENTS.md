# Agent Workflow Rules

1. After implementing requested code or doc changes, always run a minimal validation check.
2. Always commit the resulting changes.
3. Always push the commit to the active remote branch unless the user explicitly says not to push.

## UBELIX / SLURM Execution Rules

1. Prefer running experiments on UBELIX via `ssh ubelix` (default access method).
2. If the alias is unavailable, fallback to `ssh ss24i671@submit03.unibe.ch`.
3. When requesting GPUs on UBELIX, always request the maximum number of GPUs that can be allocated concurrently for the selected GPU type/partition.
4. Prefer high-end GPU types when available (for example RTX 4090), while still requesting the maximum allocatable count.
5. For interactive runs, use `srun` with explicit GPU type/count and request the maximum allowed count.
6. For batch runs, use `sbatch` with explicit `#SBATCH --partition=gpu` and `#SBATCH --gpus-per-node=<type>:<max_count>`.

### UBELIX GPU capacity notes (queried on 2026-02-24 via `sinfo`)

- `gpu:h200:8` (max `8` per node)
- `gpu:h100:8` (max `8` per node)
- `gpu:rtx4090:8` (max `8` per node)
- `gpu:rtx3090:8` (max `8` per node)
- `gpu:a100:6` (max `6` per node)

Preferred default for new jobs in this repo:
- partition-level high-end + max-count request: `#SBATCH --partition=gpu` and `#SBATCH --gpus-per-node=h200:8`
- if queue pressure is high, fallback to `h100:8`, then `rtx4090:8`

### QoS-specific limits observed for user `ss24i671` (`job_gratis`)

- per-job GPU limits: `h200=0`, `h100=1`, `rtx4090=2`, total `gpu<=3`
- per `h100:1` request, scheduler enforces approximately `CPU<=16` and `RAM<=92160MB`
- practical high-end default for this user: `#SBATCH --partition=gpu` and `#SBATCH --gpus-per-node=h100:1`
