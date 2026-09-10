#!/usr/bin/env bash
# One array task of a campaign: the shard OFFSET + task id of SHARDS,
# written under $RUN/<shard>. Every argument after the config reaches
# `decsim collect` as written (PLAN.md has the submit lines).
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
config=$1
shift
run_dir=${RUN:?the folder the shards write in}
shards=${SHARDS:?the shard count of the whole campaign distance}
offset=${OFFSET:-0}
shard=$((offset + SLURM_ARRAY_TASK_ID))
exec "${DECSIM_PYTHON:-.venv/bin/python}" -m decsim collect \
  "$config" --shard "${shard}/${shards}" --out "${run_dir}/${shard}" "$@"
