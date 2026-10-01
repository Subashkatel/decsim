#!/bin/bash
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 32 --mem 160G --time 6-00:00:00
# One experiment yaml run to its end on one node: `decsim collect` runs
# every point until its collection rule stops it, one piece per core,
# and saves each piece the moment it ends. A job stopped by its time
# limit loses only the pieces still running, and the same line again
# carries on from the saved ones.
#
#   sbatch --output <folder>/collect_%j.log slurm/collect.sbatch <yaml> <folder>
#
# Submit from a checkout pinned at a commit: the job imports decsim from
# the folder it was submitted from, so a tree with uncommitted changes
# is refused. DECSIM_PYTHON names the interpreter, as in slurm/round.sh.
set -euo pipefail
if [ $# -ne 2 ]; then
  echo "usage: sbatch slurm/collect.sbatch <experiment yaml> <experiment folder>" >&2
  exit 1
fi
cd "$SLURM_SUBMIT_DIR"
python=${DECSIM_PYTHON:-python}
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing to start: $PWD has uncommitted changes; submit from a" \
    "checkout pinned at a commit" >&2
  exit 1
fi
echo "decsim tree: $PWD"
echo "decsim commit: $(git rev-parse HEAD)"
exec "$python" -m decsim collect "$1" --out "$2" \
  --processes "$SLURM_CPUS_PER_TASK"
