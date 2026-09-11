#!/usr/bin/env bash
# One experiment as a cluster job. Every argument after the config
# reaches `decsim collect` as written; an array task adds its own shard
# and its own folder under $RUN, so the whole sweep is one line:
#   RUN=results/weak_ler sbatch -a 0-199 slurm/slurm_run.sh \
#     configs/weak_ler.yaml --shots-per-unit 50000
# then `decsim combine $RUN/*` folds the tasks' folders into one report.
# Without an array this runs the whole sweep in one job, and --out and
# --processes pass through like anything else.
#
# $SHARDS is how many shards the sweep is cut into. The array's own task
# count is the default, which is right only when the array is the whole
# sweep; give $SHARDS whenever it is not, so a partial re-run of an
# array that ran 500 shards is one line:
#   RUN=results/weak_ler SHARDS=500 sbatch -a 447-499 slurm/slurm_run.sh \
#     configs/weak_ler.yaml --shots-per-unit 50000
#
# $ALLOW_DIRTY lets a job start from a tree with uncommitted changes.
# Without it a dirty tree is refused, because every task imports the
# tree as it stands when that task starts: an edit or a commit landing
# while the array is still launching gives different tasks different
# code, and the folders would name a commit none of them ran. Submit
# from a worktree pinned at a commit instead
# (docs/how-to/run_a_sweep_on_slurm.md).
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
python=${DECSIM_PYTHON:-.venv/bin/python}

# The tree the interpreter imports decsim from, which is the code this
# task runs, and which is not always the folder the job was submitted
# from. gem5 prints its version, build date, host and command line at
# every start for the same reason (tmp/resources/gem5,
# src/python/m5/main.py:524-537).
checkout=$("$python" -c \
  'import pathlib, decsim; print(pathlib.Path(decsim.__file__).resolve().parent.parent)')
if commit=$(git -C "$checkout" rev-parse HEAD 2>/dev/null); then
  :
else
  commit=unknown
fi
if porcelain=$(git -C "$checkout" status --porcelain 2>/dev/null); then
  if [ -n "$porcelain" ]; then
    dirty=1
  else
    dirty=0
  fi
else
  dirty=unknown
fi
echo "decsim tree: $checkout"
echo "decsim commit: $commit, dirty: $dirty"
if [ "$dirty" != unknown ]; then
  # the interpreter may have no git of its own, so hand it what was seen
  export DECSIM_TREE_DIRTY="$dirty"
fi
if [ "$dirty" = 1 ] && [ -z "${ALLOW_DIRTY:-}" ]; then
  echo "refusing to start: $checkout has uncommitted changes, so this" \
    "task would import whatever the tree holds when it starts; commit" \
    "them, submit from a worktree pinned at a commit, or set ALLOW_DIRTY=1" >&2
  exit 1
fi

task=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  run_dir=${RUN:?an array task needs RUN, the folder its shards write in}
  shards=${SHARDS:-${SLURM_ARRAY_TASK_COUNT}}
  echo "shard: ${SLURM_ARRAY_TASK_ID} of ${shards}"
  task=(
    --shard "${SLURM_ARRAY_TASK_ID}/${shards}"
    --out "${run_dir}/${SLURM_ARRAY_TASK_ID}"
  )
fi
exec "$python" -m decsim collect \
  "$config" "${task[@]}" "$@"
