#!/bin/bash
#SBATCH -c 16 --mem 32G -t 24:00:00
# submit from the checkout: sbatch --array 0-$(( $(PYTHONPATH=. python experiments/NAME/run.py --list | wc -l) - 1 )) slurm/run.sbatch experiments/NAME/run.py results/DATE_NAME
export PYTHONPATH="$SLURM_SUBMIT_DIR"
python "$1" "$SLURM_ARRAY_TASK_ID" --workers "$SLURM_CPUS_PER_TASK" --out "$2"
