#!/bin/bash
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 16G --time 12:00:00
# Runs an experiment round after round to its end from one command. Each
# step plans the next round with `decsim plan`, submits it with
# slurm/round.sh, and queues the next step behind that round's arrays,
# until the plan says every point has stopped. The next step waits with
# afterany, so a task that died or ran out of time is planned again.
#
#   slurm/run_all_rounds.sh <experiment folder> <decsim plan options and yamls>
#   slurm/run_all_rounds.sh results/my_run/run --tasks 1000 --cores 1 \
#     --hours 24 configs/experiments/my_study/my_study.yaml
#
# Start it from a checkout pinned at a commit; every step runs there, and
# slurm/round.sh refuses a tree with uncommitted changes. DECSIM_PYTHON
# names the interpreter, as in slurm/round.sh.
set -euo pipefail
if [ -n "${SLURM_JOB_ID:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
fi
if [ $# -lt 2 ]; then
  echo "usage: slurm/run_all_rounds.sh <experiment folder>" \
    "<decsim plan options and yamls>" >&2
  exit 1
fi
python=${DECSIM_PYTHON:-python}
experiment=$(realpath -m "$1")
shift

planned=$("$python" -m decsim plan "$@" --out "$experiment")
round=$(echo "$planned" | awk '/^submit it:/ {print $NF}')
if [ -z "$round" ]; then
  echo "every point has stopped; decsim status $experiment"
  exit 0
fi

submitted=$(slurm/round.sh "$experiment" "$round")
echo "$submitted"
job_ids=$(echo "$submitted" | awk '/^Submitted batch job/ {print $4}' | paste -sd:)
if [ -z "$job_ids" ]; then
  echo "round $round submitted no job; stopping" >&2
  exit 1
fi
sbatch --dependency "afterany:$job_ids" \
  --job-name "decsim-after-round$round" \
  --output "$experiment/round$round/next_step.log" \
  slurm/run_all_rounds.sh "$experiment" "$@"
