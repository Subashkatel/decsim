#!/bin/bash
# One offline shard per array task, one shards.tsv line each:
#   python -m experiments.offline_run plan configs/<name>.yaml [seeds_per_shard]
#   sbatch --array=1-<N> experiments/slurm_offline.sh <run_dir>
#   python -m experiments.offline_run merge <run_dir>
#SBATCH --job-name=decsim-offline
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=4:00:00

cd "$SLURM_SUBMIT_DIR"
# one shard, one core; parallelism comes from the array, not from BLAS
export OMP_NUM_THREADS=1
exec "${DECSIM_PYTHON:-.venv/bin/python}" -m experiments.offline_run \
    shard "$1" "$SLURM_ARRAY_TASK_ID"
