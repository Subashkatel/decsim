#!/usr/bin/env bash
set -euo pipefail

configurations_file=$1
output_root=$2
shift 2
mapfile -t configurations < "$configurations_file"
configuration=${configurations[$SLURM_ARRAY_TASK_ID]}
exec "${DECSIM_PYTHON:-python}" -m experiments.run_surface \
  "$@" --config "$configuration" --output "$output_root"
