#!/bin/bash
# Relay-BP decode times on one whole A100 SXM4 80GB, one decode at a
# time, for each region file make_regions.py wrote that is named on the
# command line. Slurm runs a copy of this file, so TOOLS names this
# folder. The SXM boards take only short jobs here, so a job times one
# region or a few. Usage:
#   OUT=dir PY=<venv>/bin/python TOOLS=<this folder> \
#     sbatch -o <log> time_a100.sbatch region.npz ...
#SBATCH -J relay_a100_times
#SBATCH -c 10
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=sxm
set -u
module load cudatoolkit/12.9
mkdir -p "$OUT"
nvidia-smi -L
for region in "$@"; do
  name=$(basename "$region" .npz)
  distance=${name#region_d}
  distance=${distance%%_*}
  # the largest regions take the longest decodes, so they time fewer
  requests=2000
  if [ "$distance" -ge 11 ]; then requests=500; fi
  "$PY" -W ignore "$TOOLS/time_decodes.py" --region "$region" \
    --device a100 --partition whole --requests "$requests" \
    --out "$OUT/a100_whole_${name#region_}.csv"
done
