[decsim docs](../README.md) › [How-to guides](README.md)

# How to run a sweep on Slurm

A real sweep is millions of shots. On a cluster it runs in batches.
Each batch is one written plan of pieces, and Slurm arrays run it. One
command starts the loop:

```bash
SBATCH_QOS=short decsim run configs/experiments/switching/redo_window_switching.yaml \
  --slurm --tasks 300 --out $PWD/results/redo_window_switching
```

Each step of the loop plans the next batch from what the saved pieces
say, submits it, and queues the next step behind it, until every point
has stopped. `decsim status` folds every saved piece into the numbers
at any time.

## 1. Size the pieces and say when a point stops

A piece is one point's block of shots that one process runs from start
to finish. The yaml's `collection` section sets its size in QEC rounds
with `piece_rounds`, 20,000 unless it says otherwise
(`configs/reference.yaml`). A shot's rounds add up every patch's
rounds. So a piece takes about as long at any history length.

The same section says when a point stops:

- `max_failures` is the target.
- `max_shots` and `max_core_seconds` are the caps.
- `min_shots` is the minimum.

A piece must finish inside one task's walltime, so size `piece_rounds`
from a point's seconds a shot.

## 2. Pin the code the batches will run

A task imports the tree as it stands when it starts. A commit landing
between submitting and starting gives the task code nobody chose. So
submit from a worktree pinned at a commit, and leave the tree you work
in free:

```bash
git worktree add ../decsim-pinned <commit>
cd ../decsim-pinned
```

`decsim run --slurm` refuses a tree with uncommitted changes, or one
git cannot read. It checks once when you submit, at every step of the
loop, and again in every task. `ALLOW_DIRTY=1` overrides the refusals.

Every piece's `piece.json` records the commit and the dirty flag, so a
result names its code. The interpreter decides which tree is imported,
not the directory you submit from. An environment installed with
`pip install -e` from another checkout imports that checkout, so a
worktree needs `PYTHONPATH` naming it.

## 3. Start the loop

```bash
SBATCH_QOS=short decsim run configs/experiments/switching/redo_window_switching.yaml \
  --slurm --tasks 300 --out $PWD/results/redo_window_switching
```

Without `--out` the results go to a new `results/<date>_<name>/`, and
every later step is handed that folder. One step does this, in order:

1. It refuses a tree git does not vouch for (section 2).
2. Before the first batch, it runs one shot of each machine shape:
   seed 0 of the shape's point with the fewest rounds, as `decsim run
   --only NAME --seed 0`, in the interpreter the jobs will use
   (`DECSIM_PYTHON`, else the one you ran). Points that differ only in
   their numbers are one shape. The shot is thrown away. A broken
   import, build or config is caught before anything is queued.
3. It plans the next batch into
   `results/redo_window_switching/batches/<k>/plan.csv`, with its
   pieces dealt to at most `--tasks` tasks, and `tasks.csv`, with each
   task's cores, memory and hours. When every point has stopped it
   says so and the loop ends.
4. It refuses the batch if its tasks and the jobs you have queued pass
   `SUBMIT_LIMIT` (1,000 unless you set it), since Slurm counts each
   array task as a submitted job and rejects a submission past the
   limit. A `SUBMIT_LIMIT` that is not a positive whole number is
   refused too.
5. It submits one job array per shape of job in `tasks.csv`, since an
   array has one memory request. Each array task runs its share of the
   plan, one process per core. Its log goes to `batches/<k>/<task>/log.txt`,
   beside its `run.json`.
6. It submits the next step with `--dependency afterany` on those
   arrays, so the next step starts however they ended. Its log goes to
   `batches/<k>/next_step.log`.

How the plan decides a point's pieces:

- **Batch one** gives each point one piece, since nothing is measured
  yet. A point with no target but a shot cap gets every piece up to the
  cap.
- **Later batches, a lost piece:** a piece a task never finished is
  planned again first, with its own seeds.
- **Later batches, a point still running:** it is extended to the
  shots its failure rate so far says the target needs, which is OpenMC's
  trigger rule. With no failure yet, its shots double.
- **Never past a cap.** The time cap is read at the point's measured
  seconds a shot, so a batch's last piece may be short.
- **An online point** gets all its pieces in one task, in seed order,
  since each starts from the calibrator the piece before it saved.

How the plan deals and sizes the tasks:

- Pieces go to tasks longest first, costed by each point's measured
  seconds a shot.
- A task asks for `--cores` cores (default 4) and runs that many pieces
  at once.
- A task asks for `--hours` hours of walltime (default 24).
- A task's memory is its cores times its points' largest measured peak,
  with a margin of one half. Where nothing was measured, it uses
  `--memory-mb` per piece (default 4096).

decsim writes no account, partition or QOS. `sbatch` reads them from
the environment variables SBATCH_ACCOUNT, SBATCH_PARTITION and
SBATCH_QOS, and every job of the loop inherits the environment it was
started from. On Della the QOS follows from the time limit: `short` up
to 24 hours, where a user may submit 1,000 jobs and use 1,400 cores,
`medium` up to 72 hours with 400 cores. So `--tasks` stays at or under
1,000, and 350 four-core tasks fill the short cores.

`--dry-run` runs the check and plans the batch, then prints the `sbatch`
lines and submits nothing. A dry-run batch's pieces are planned again
by the next real step, since no task saved them.

## 4. If a task dies

Nothing is lost but the pieces it was running. A piece folder appears
only once the piece is whole, so nothing is counted twice. The next
step finds each planned piece whose seeds no piece holds and plans it
again. A local `decsim run` of the same run file in the same folder
cuts its pieces where planned pieces begin and end, so a batch's task
run after it finds those seeds saved and runs nothing. Each seed is
saved once, whichever runs first.

If the loop itself stops, the same command starts it again from what
the folder holds.

## 5. Read the numbers

```bash
decsim status results/redo_window_switching
```

This folds every saved piece into the csv files at the top of
`results/redo_window_switching/`. It reads what each step recorded of
each point, not the run file, so a point the run file no longer sweeps
is still counted. It writes
`results/redo_window_switching/status.csv`, one row per point, with
these columns:

- its row of the run folder's `sweep.csv`, whole: its values, its state
  (`running`, `target`, `minimum`, `cap`, `time cap`, or `no data`), its
  shots, failures and unscored shots, and every estimate and exact
  interval of its contiguous prefix, per shot and per round
- the rounds and core seconds of all its pieces

Status can run while a batch runs. Each row reads the pieces once, so
a piece saved meanwhile is in all of a row or none of it.

The run folder's `sweep.csv` holds one row per point, its values and
its counts, and each point's `points/<name>/machine.json` holds every
setting it ran with ([How to compare two runs](compare_two_runs.md)).

## Read next

- [Your first sweep](../tutorials/first_sweep.md): a small sweep end to end, with the
  error bars explained.
- [The run folder](../reference/run_folder.md): what a piece, a plan and a run folder hold.
- [The commands](../reference/cli.md): every flag of `run` and `status`.
