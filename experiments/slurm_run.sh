#!/usr/bin/env bash
# One closed-loop experiment as a cluster job:
#   sbatch -J <name> -o <log> experiments/slurm_run.sh <config.yaml>
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=16:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec "${DECSIM_PYTHON:-.venv/bin/python}" -m experiments.run "$1"
