#!/usr/bin/env bash
set -euo pipefail

configurations_file=$1
shift
mapfile -t configurations < "$configurations_file"
configuration=${configurations[$SLURM_ARRAY_TASK_ID]}
exec "$@" --config "$configuration"
