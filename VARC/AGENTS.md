# Agent Workflow Rules

1. After implementing requested code or doc changes, always run a minimal validation check.
2. Always commit the resulting changes.
3. Always push the commit to the active remote branch unless the user explicitly says not to push.

## UBELIX / SLURM Execution Rules

1. Prefer running experiments on UBELIX via `ssh ss24i671@submit03.unibe.ch`.
2. When requesting GPUs on UBELIX, always request the maximum number of GPUs that can be allocated concurrently for the selected GPU type/partition.
3. Prefer high-end GPU types when available (for example RTX 4090), while still requesting the maximum allocatable count.
4. For interactive runs, use `srun` with explicit GPU type/count and request the maximum allowed count.
5. For batch runs, use `sbatch` with explicit `#SBATCH --partition=gpu` and `#SBATCH --gpus-per-node=<type>:<max_count>`.
