#!/usr/bin/env bash
# One closed-loop experiment as a cluster job:
#   sbatch -J <name> -o <log> slurm/slurm_run.sh <config.yaml>
# One array task per shard:
#   sbatch -a 0-3 slurm/slurm_run.sh <config.yaml> 4
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=16:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
shard=()
if [ "$#" -ge 2 ]; then
  shard=(--shard "${SLURM_ARRAY_TASK_ID}/$2")
fi
exec "${DECSIM_PYTHON:-.venv/bin/python}" -m decsim collect "$1" "${shard[@]}"
