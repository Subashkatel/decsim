[decsim docs](../README.md) › [How-to guides](README.md)

# How to run a sweep on Slurm

A real sweep is millions of shots. On a cluster each point runs as one
task of a Slurm array, and one fold job adds up what the tasks saved.
One command starts both:

```bash
SBATCH_QOS=short decsim run configs/experiments/switching/redo_window_switching.yaml \
  --slurm --out $PWD/results/redo_window_switching
```

## 1. Size the pieces and say when a point stops

A piece is one point's block of shots that one process runs from start
to finish and saves whole. The yaml's `collection` section sets its
size in QEC rounds with `piece_rounds`, 20,000 unless it says otherwise
(`configs/reference.yaml`). A shot's rounds add up every patch's
rounds. So a piece takes about as long at any history length.

The same section says when a point stops:

- `max_failures` is the target.
- `max_shots` and `max_core_seconds` are the caps.
- `min_shots` is the minimum.

A task checks the stop after every piece it saves, so a point stops
within a piece of its target.

## 2. Pin the code the tasks will run

A task imports the tree as it stands when it starts. A commit landing
between submitting and starting gives the task code nobody chose. So
submit from a worktree pinned at a commit, and leave the tree you work
in free:

```bash
git worktree add ../decsim-pinned <commit>
cd ../decsim-pinned
```

`decsim run --slurm` refuses a tree with uncommitted changes, or one
git cannot read. It checks once when you submit, and again in every
task. `ALLOW_DIRTY=1` overrides the refusals.

Every piece's `piece.json` records the commit and the dirty flag, so a
result names its code. The interpreter decides which tree is imported,
not the directory you submit from. An environment installed with
`pip install -e` from another checkout imports that checkout, so a
worktree needs `PYTHONPATH` naming it.

## 3. Submit

```bash
SBATCH_QOS=short decsim run configs/experiments/switching/redo_window_switching.yaml \
  --slurm --out $PWD/results/redo_window_switching
```

Without `--out` the results go to a new `results/<date>_<name>/`. The
command does this, in order:

1. It refuses a tree git does not vouch for (section 2).
2. It records every point in the results folder, which builds each
   point's machine, so a broken config is refused before anything is
   queued. It copies the run file into the folder.
3. It writes `run.sbatch`, one array with a task per point: task `i`
   runs point `i` of the folder's copy of the run file until it stops.
   A task asks for `--cores` cores (default 4) and runs that many
   pieces at once, `--hours` hours (default 24) and `--memory-mb` MB
   (default 16384). Task `i` logs to `logs/i.log`.
4. It writes `fold.sbatch`, one job that folds every saved piece into
   the csv files at the top of the folder, as `decsim run --fold --out
   <folder>` does.
5. It submits `run.sbatch`, then `fold.sbatch` with `--dependency
   afterany` on the array, so the fold runs however the tasks ended.

`--dry-run` records the points and writes both files, and submits
nothing.

decsim writes no account, partition or QOS. `sbatch` reads them from
the environment variables SBATCH_ACCOUNT, SBATCH_PARTITION and
SBATCH_QOS, and every job inherits the environment it was submitted
from.

## 4. If a task dies or runs out of time

Nothing is lost but the piece it was running. A piece folder appears
only once the piece is whole, so nothing is counted twice. Submit the
same command again: each task starts from the pieces its point saved,
and a point that stopped runs nothing.

## 5. Read the numbers

```bash
decsim run --fold --out results/redo_window_switching
```

This folds every saved piece into the csv files at the top of the
folder, at any time, even while tasks run. It reads what the folder
recorded of each point, not the run file, so a point the run file no
longer sweeps is still counted. `sweep.csv` holds one row per point:
its values, its `state` (`running`, `target`, `minimum`, `cap` or
`time cap`), its counts, and every estimate and exact interval of its
contiguous prefix. A point's rounds are the sum of `shots.csv`'s
`executed_rounds` over its rows, and its core seconds the sum of
`sim_wall_seconds`. Each point's `points/<name>/machine.json` holds
every setting it ran with ([How to compare two runs](compare_two_runs.md)).

## Read next

- [Your first sweep](../tutorials/first_sweep.md): a small sweep end to end, with the
  error bars explained.
- [The run folder](../reference/run_folder.md): what a piece and a run folder hold.
- [The commands](../reference/cli.md): every flag of `run`.
