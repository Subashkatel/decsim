[decsim docs](../README.md) › [How-to guides](README.md)

# How to run a sweep on Slurm

A real sweep is millions of shots. On a cluster it runs in rounds. Each
round is one written plan of pieces, and one Slurm array runs it. The
steps are:

1. `decsim plan` reads what the saved pieces say and writes the next round.
2. `slurm/round.sh` submits the round.
3. `decsim status` folds every saved piece into the numbers.

Plan again after each round, until the plan says every point has stopped.

## 1. Size the pieces and say when a point stops

A piece is one point's block of shots that one process runs from start
to finish. The yaml's `collection` section sets its size in QEC rounds
with `piece_rounds`, 20,000 unless it says otherwise
(`configs/reference.yaml`). So a piece takes about as long at any
history length.

The same section says when a point stops:

- `max_failures` is the target.
- `max_shots` and `max_core_seconds` are the caps.
- `min_shots` is the minimum.

A piece must finish inside one task's walltime, so size `piece_rounds`
from a point's seconds a shot.

## 2. Pin the code the rounds will run

A task imports the tree as it stands when it starts. A commit landing
between submitting and starting gives the task code nobody chose. So
submit from a worktree pinned at a commit, and leave the tree you work
in free:

```bash
git worktree add ../decsim-pinned <commit>
cd ../decsim-pinned
```

`slurm/round.sh` refuses a tree with uncommitted changes, or one git
cannot read. It checks once when you submit and again in every task.
`ALLOW_DIRTY=1` overrides both refusals.

Every piece's `piece.json` records the commit and the dirty flag, so a
result names its code. The interpreter decides which tree is imported,
not the directory you submit from. An environment installed with
`pip install -e` from another checkout imports that checkout, so a
worktree needs `PYTHONPATH` naming it.

## 3. Plan a round

```bash
decsim plan configs/experiments/switching/redo_window_switching.yaml \
  --out $PWD/results/redo_window_switching --tasks 300
```

This writes `results/redo_window_switching/round1/plan.csv`, with its
pieces dealt to at most 300 tasks, and `tasks.csv`, with each task's
cores, memory and hours.

Several yamls may share one experiment folder. Yamls of one
configuration, such as a grid split into one file per distance, are
planned as one. A point two configurations both reach is planned once,
and refused if they give it two collections.

How the plan decides a point's pieces:

- **Round one** gives each point one piece, since nothing is measured
  yet. A point with no target but a shot cap gets every piece up to the
  cap.
- **Later rounds, a lost piece:** a piece a task never finished is
  planned again first, with its own seeds.
- **Later rounds, a point still running:** it is extended to the
  shots its failure rate so far says the target needs, which is OpenMC's
  trigger rule. With no failure yet, its shots double.
- **Never past a cap.** The time cap is read at the point's measured
  seconds a shot, so a round's last piece may be short.

How the plan deals and sizes the tasks:

- Pieces go to tasks longest first, costed by each point's measured
  seconds a shot.
- A task asks for `--cores` cores (default 4) and runs that many pieces
  at once.
- A task asks for `--hours` hours of walltime (default 24).
- A task's memory is its cores times its points' largest measured peak,
  with a margin of one half. Where nothing was measured, it uses
  `--memory-mb` per piece.

On Della's `short` QOS, a user may submit 1,000 jobs, run 400 at once
and use 1,400 cores. So `--tasks` stays at or under 1,000, and 350
four-core tasks fill the cores.

## 4. Submit it

```bash
SBATCH_QOS=short slurm/round.sh results/redo_window_switching 1
```

The script submits one job array per shape of job in `tasks.csv`,
since an array has one memory request. Each array task runs `decsim
collect --plan results/redo_window_switching/round1/plan.csv --task
<id>`, one process per core. Its log goes to `round1/<id>/log.txt`,
beside its manifest.

The script writes no account, partition or QOS. `sbatch` reads them from
the environment variables SBATCH_ACCOUNT, SBATCH_PARTITION and
SBATCH_QOS.

`DRY_RUN=1` prints the `sbatch` lines and submits nothing.

Slurm counts each array task as a submitted job, and rejects a
submission past the limit. So before it submits any array, the script
adds the round's tasks to the jobs you have queued, and refuses the
round if they pass `SUBMIT_LIMIT` (1,000 unless you set it). Plan the
round again with fewer `--tasks`, or wait for queued jobs to end.
A `SUBMIT_LIMIT` that is not a positive whole number refuses the round
too.

## 5. If a task dies

Nothing is lost but the pieces it was running. A piece folder appears
only once the piece is whole, so nothing is counted twice. The next
`decsim plan` finds each planned piece whose seeds no piece holds and
plans it again. Running the same task again (`decsim collect --plan ...
--task <id>`) runs only the seeds no saved piece holds.

A plain `decsim collect` of the same yaml in the same folder cuts its
pieces where planned pieces begin and end, so a round's task run after
it finds those seeds saved and runs nothing. Each seed is saved once,
whichever runs first.

## 6. Read the numbers

```bash
decsim status results/redo_window_switching
```

This folds every saved piece into its configuration's run folder,
`results/redo_window_switching/combined/<name>-<id8>/`. It reads what
the plans and collects recorded of each point, not the yamls, so a
point a yaml no longer sweeps is still counted, and a point two
configurations reach is counted once. It writes
`results/redo_window_switching/status.csv`, one row per point, with
these columns:

- its configuration id
- its row of the run folder's `sweep.csv`, whole: its values, its state
  (`running`, `target`, `minimum`, `cap`, `time cap`, or `no data`), its
  shots, failures and unscored shots, and every estimate and exact
  interval of its contiguous prefix, per shot and per round
- the rounds and core seconds of all its pieces

Status can run while a round runs. Each row reads the pieces once, so
a piece saved meanwhile is in all of a row or none of it. Plan the next round when the last
one ends.

The run folder's `sweep.csv` holds one row per point, its values and
its counts, and `decsim.results.load` reads it beside every setting
([How to compare two runs](compare_two_runs.md)).

## Read next

- [Your first sweep](../tutorials/first_sweep.md): a small sweep end to end, with the
  error bars explained.
- [The run folder](../reference/run_folder.md): what a piece, a plan and a run folder hold.
- [The commands](../reference/cli.md): every flag of `plan`, `collect` and `status`.
