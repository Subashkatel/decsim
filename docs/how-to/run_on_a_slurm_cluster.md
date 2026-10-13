[decsim docs](../README.md) › [How-to guides](README.md)

# How to run a sweep on a Slurm cluster

decsim writes no job script. A run split over many jobs is three
commands, and how you split the seeds, size the jobs and submit them is
yours: a Slurm array, a loop of `sbatch`, or any other scheduler.

1. `decsim run my_study.py --record --out DIR` records every task into
   the folder the jobs share, once, before any job starts.
2. `decsim run my_study.py --share INDEX/COUNT --out DIR` in each job
   runs job INDEX's share of every task's seeds into that folder, or
   `--seeds FIRST:END` runs seeds FIRST to END - 1.
3. `decsim run --fold --out DIR` folds what the jobs saved.

The folded files are the ones one local run writes, row for row but the
wall clock: a shot's seed fixes the shot, whichever job runs it.

## 1. Commit the run file

Every job imports the decsim tree and the run file as they stand when
the job starts. So recording the folder, and every job, refuse a
checkout with uncommitted changes, or one git cannot read. Commit
first, or run from a worktree pinned at a commit; `ALLOW_DIRTY=1` lets a
run go on anyway, and its pieces record that the tree was dirty.

## 2. Record the folder

```bash
decsim run my_study.py --record --out results/my_study
```

This builds every task, so a machine that cannot be built stops here,
before anything is queued, and writes `run.json`, a copy of the run
file and one record per task. A job checks the folder against them: a
job of another tree, of a run file edited since the record, or of a
task built with other settings (another `--set` or DECSIM_SET) is
refused. Give the jobs the same `--set` flags as the record, or export
DECSIM_SET in the shell that submits them, which sbatch passes on.

Recording a folder again is allowed when only the stop rules changed,
for instance a pilot's `max_shots` raised for the full run; the jobs
then go on from the pieces already saved.

## 3. Write the job script

A job array of twenty, each job running its twentieth of every task:

```bash
#!/bin/bash
#SBATCH --array=0-19
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=results/my_study/logs/%A_%a.log
python -m decsim run my_study.py --out results/my_study --share $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT --processes 4
```

`--share INDEX/COUNT` cuts each task's `max_shots` seeds into COUNT
runs of consecutive seeds and runs the one numbered INDEX, counted from
0 as the array's ids are. A task of 2,000,000 shots and one of 20,000 are both
cut in twenty, so the script needs no shot counts. To name the seeds
yourself, `--seeds FIRST:END` runs seeds FIRST to END - 1 of every
task, stopped at its `max_shots`; a task that stops on failures or time
has no `max_shots` to share, so it needs `--seeds`. What a job does with
its seeds:

- It skips seeds a saved piece holds, so a job that ran out of time is
  submitted again as it was and runs only what it did not save.
- It cuts the rest into pieces of the task's `piece_rounds`, each saved
  the moment it ends, so a killed job loses one piece a process.
- `--processes N` runs N pieces at once; ask Slurm for N cores.
- `--only NAME` runs one task, so tasks of very different costs can go
  to arrays of their own: a share holds equal shots, not equal time.
- It folds nothing.

Give no two jobs the same seeds at once: the fold refuses a shot two
pieces hold, so one of them would have to be deleted. Size the walltime
and memory from a pilot: `decsim run my_study.py --shots 20 --out
pilot` on one
compute node writes `sim_wall_seconds_per_shot` per task in
`pilot/sweep.csv`, and each piece's `piece.json` holds the
`peak_memory_mb` of the process that ran it.

A task whose threshold learns online runs its pieces one after another,
each from the state the one before saved, so `--share` and `--seeds`
refuse it; run
it in one job with `decsim run my_study.py --only NAME --out DIR`.

## 4. Submit, then fold

```bash
mkdir -p results/my_study/logs
array=$(sbatch --parsable my_study.sbatch)
sbatch --dependency=afterany:$array --wrap "python -m decsim run --fold --out results/my_study"
```

`afterany` starts the fold once every job of the array has ended,
however it ended. Run the fold yourself at any time too, even while
jobs run, to read the shots saved so far:

```bash
decsim run --fold --out results/my_study
```

## 5. Set the job for your cluster

decsim reads nothing of the cluster. The account, partition, QOS,
cores, memory and walltime are the job script's `#SBATCH` lines, or
sbatch's own SBATCH_ACCOUNT, SBATCH_PARTITION and SBATCH_QOS variables
in the submitting shell. A cluster that routes short jobs to a test
QOS, or asks jobs to use a share of the time they request, is served by
the walltime and the seeds a job gets, both of which the script sets.
