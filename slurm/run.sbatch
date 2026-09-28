#!/bin/bash
#SBATCH -c 16 --mem 32G -t 24:00:00
# submit: sbatch --array 0-<last id from run.py --list> slurm/run.sbatch <checkout>/experiments/NAME/run.py results/DATE_NAME
# The task imports the checkout run.py sits in, two folders above it.
checkout="$(cd "$(dirname "$1")/../.." && pwd)"
[ -f "$checkout/decsim/experiment_runner.py" ] || { echo "run.sbatch: $checkout is not a decsim checkout; give the path of experiments/<name>/run.py inside one" >&2; exit 1; }
export PYTHONPATH="$checkout${PYTHONPATH:+:$PYTHONPATH}"
python "$1" "$SLURM_ARRAY_TASK_ID" --workers "$SLURM_CPUS_PER_TASK" --out "$2"
