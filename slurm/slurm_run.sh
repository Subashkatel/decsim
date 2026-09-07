#!/usr/bin/env bash
# One experiment as a cluster job. Every argument after the config
# reaches `decsim collect` as written; an array task adds its own shard
# and its own folder under $RUN, so the whole sweep is one line:
#   RUN=results/weak_ler sbatch -a 0-199 slurm/slurm_run.sh \
#     configs/weak_ler.yaml --shots-per-unit 50000
# then `decsim combine $RUN/*` folds the tasks' folders into one report.
# Without an array this runs the whole sweep in one job, and --out and
# --processes pass through like anything else.
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=16:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
config=$1
shift
task=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  run_dir=${RUN:?an array task needs RUN, the folder its shards write in}
  task=(
    --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
    --out "${run_dir}/${SLURM_ARRAY_TASK_ID}"
  )
fi
exec "${DECSIM_PYTHON:-.venv/bin/python}" -m decsim collect \
  "$config" "${task[@]}" "$@"
