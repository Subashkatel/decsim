#!/usr/bin/env bash
# One round of an experiment on Slurm. `decsim plan` wrote the round's
# pieces in <experiment>/round<k>/plan.csv and each task's cores,
# memory and hours in tasks.csv; this submits one job array per shape
# of job tasks.csv names, and each array task runs its share:
#   slurm/round.sh results/weak_ler 1
# DRY_RUN=1 prints the sbatch lines and submits nothing.
#
# No account, partition or QOS is written here: sbatch reads them from
# SBATCH_ACCOUNT, SBATCH_PARTITION and SBATCH_QOS in the environment
# (sbatch(1), "INPUT ENVIRONMENT VARIABLES"), so a user sets them once:
#   SBATCH_QOS=short slurm/round.sh results/weak_ler 1
# The QOS enforces its own limits on running jobs and cores; tasks past
# them wait in the queue. Its limit on submitted jobs is different: it
# counts each array task as a job and rejects a submission past it, so
# a round whose tasks and the user's queued jobs pass $SUBMIT_LIMIT
# (1000 unless set, Della's short QOS) is refused before any array.
#
# Inside the array, where SLURM_ARRAY_TASK_ID is set, this script is the
# job: it runs `decsim collect --plan <plan.csv> --task <id>` with one
# process per core it was given.
#
# $ALLOW_DIRTY lets a round start from a tree whose state git does not
# vouch for. Without it a dirty tree is refused, at submission and in
# every task, because every task imports the tree as it stands when that
# task starts: an edit or a commit landing while the array is still
# launching gives different tasks different code, and the pieces would
# name a commit none of them ran. A tree git cannot read at all is
# refused for the same reason: the job must name the code it ran, and
# there nothing can. Submit from a worktree pinned at a commit instead
# (docs/how-to/run_a_sweep_on_slurm.md).
set -euo pipefail
python=${DECSIM_PYTHON:-python}

# Refuses a tree git does not vouch for, and hands the interpreter what
# git saw, since the interpreter may have no git of its own. The tree is
# the one the interpreter imports decsim from, which is the code a task
# runs and not always the folder it was submitted from; gem5 prints its
# version, build date, host and command line at every start for the
# same reason (gem5 src/python/m5/main.py:524-537).
refuse_an_unnamed_tree() {
  local checkout commit porcelain dirty
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
    export DECSIM_TREE_DIRTY="$dirty"
  fi
  if [ "$dirty" = 1 ] && [ -z "${ALLOW_DIRTY:-}" ]; then
    echo "refusing to start: $checkout has uncommitted changes, so a" \
      "task would import whatever the tree holds when it starts; commit" \
      "them, submit from a worktree pinned at a commit, or set ALLOW_DIRTY=1" >&2
    exit 1
  fi
  if [ "$dirty" = unknown ] && [ -z "${ALLOW_DIRTY:-}" ]; then
    echo "refusing to start: git says nothing about $checkout, so a" \
      "task cannot name the code it ran; submit from a git checkout" \
      "pinned at a commit, or set ALLOW_DIRTY=1" >&2
    exit 1
  fi
}

# Refuses a round that would pass the submit limit partway through: a
# half-submitted round leaves tasks no array holds.
refuse_a_round_past_the_submit_limit() {
  local task_count queued limit
  task_count=$(awk 'NR > 1' "$tasks_file" | wc -l)
  queued=0
  if command -v squeue > /dev/null; then
    queued=$(squeue -h -r -u "$(id -un)" | wc -l)
  fi
  limit=${SUBMIT_LIMIT:-1000}
  if ! [[ "$limit" =~ ^[1-9][0-9]*$ ]]; then
    echo "refusing to submit: SUBMIT_LIMIT must be a whole number of jobs" \
      "of at least 1, got '$limit'" >&2
    exit 1
  fi
  if [ $((task_count + queued)) -gt "$limit" ]; then
    echo "refusing to submit: round $round has $task_count tasks and" \
      "$queued of your jobs are queued, past the submit limit of $limit" \
      "(SUBMIT_LIMIT); plan the round again with fewer --tasks, or wait" \
      "for queued jobs to end" >&2
    exit 1
  fi
}

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
  refuse_an_unnamed_tree
  exec "$python" -m decsim collect \
    --plan "$1/round$2/plan.csv" \
    --task "$SLURM_ARRAY_TASK_ID" \
    --processes "${SLURM_CPUS_PER_TASK:-1}"
fi

if [ $# -ne 2 ]; then
  echo "usage: slurm/round.sh <experiment folder> <round>" >&2
  exit 1
fi
experiment=$(realpath "$1")
round=$2
round_dir=$experiment/round$round
tasks_file=$round_dir/tasks.csv
if [ ! -f "$tasks_file" ]; then
  echo "$tasks_file is not there; decsim plan writes it" >&2
  exit 1
fi
refuse_an_unnamed_tree
refuse_a_round_past_the_submit_limit
script=$(realpath "$0")

# Each shape of job, cores:memory:hours, and the tasks that share it.
declare -A tasks_by_shape
while IFS=, read -r task cores memory_mb hours _estimate; do
  if [ "$task" = task ]; then
    continue
  fi
  shape="$cores:$memory_mb:$hours"
  tasks_by_shape[$shape]+="${tasks_by_shape[$shape]:+,}$task"
done < "$tasks_file"

mapfile -t shapes < <(printf '%s\n' "${!tasks_by_shape[@]}" | sort)
for shape in "${shapes[@]}"; do
  IFS=: read -r cores memory_mb hours <<< "$shape"
  tasks=${tasks_by_shape[$shape]}
  line=(
    sbatch
    --job-name "decsim-round$round"
    --array "$tasks"
    --nodes 1
    --ntasks 1
    --cpus-per-task "$cores"
    --mem "${memory_mb}M"
    --time "$hours:00:00"
    --output "$round_dir/%a/log.txt"
    "$script" "$experiment" "$round"
  )
  if [ -n "${DRY_RUN:-}" ]; then
    echo "${line[*]}"
    continue
  fi
  # Slurm opens each task's log before the task runs, so its folder
  # must exist already; the task's manifest goes in the same folder.
  IFS=, read -r -a task_list <<< "$tasks"
  for task in "${task_list[@]}"; do
    mkdir -p "$round_dir/$task"
  done
  "${line[@]}"
done
