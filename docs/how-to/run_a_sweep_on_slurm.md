[decsim docs](../README.md) › [How-to guides](README.md)

# How to run a sweep on Slurm

A real sweep is millions of shots and does not fit in one job.
`slurm/slurm_run.sh` runs it as an array: each task takes a share of the
work units and writes its own folder, and one command folds the folders
back into one report.

## 1. Understand the two knobs

`--shard i/n` gives one array task the work units whose index modulo `n`
is `i`. `--shots-per-unit N` sets how many shots one work unit is: a
smaller unit trades the per-task window-model cache for shards that fit
a time limit.

Size them from the wall time. `configs/weak_ler.yaml`'s own header does
the arithmetic for its 35 points and 10,425,000 shots: at 0.22 seconds
per shot at distance 7, a 16 hour limit is about 262,000 shots, so a
task must stay under that. With `--shots-per-unit 50000` the sweep cuts
into 225 work units, and an array of 200 tasks gives twenty five tasks
two units and the other 175 one.

## 2. Pin the code the array will run

Every task imports the tree as it stands when that task starts, and a
500 task array takes minutes to launch. A commit landing in the middle
gives different tasks different code, and no folder can say which. So
submit from a worktree pinned at a commit, and leave the tree you work
in free:

```bash
git worktree add ../decsim-weak_ler <commit>
cd ../decsim-weak_ler
```

Submit from that directory. `slurm/slurm_run.sh` prints the tree it
imports from and that tree's commit, and refuses to start from a tree
with uncommitted changes unless `ALLOW_DIRTY` is set:

```
decsim tree: /scratch/.../decsim-weak_ler
decsim commit: 264853ada3..., dirty: 0
```

The commit and the dirty flag go into every folder's `manifest.json`, so
a result names its code. Two things to get right:

- **The interpreter decides which tree is imported, not the directory
  you submit from.** In this sandbox `.venv/bin/python` is a container
  wrapper that pins the main checkout on `PYTHONPATH`, so a worktree
  needs to come first: `DECSIM_PYTHON` pointing at the worktree's own
  wrapper, or `PYTHONPATH` naming the worktree. The line the runner
  prints is the check: it is the tree the interpreter actually imported.
- **`RUN` must be an absolute path.** The tasks `cd` to the directory
  the job was submitted from, which is the worktree, and a relative
  `RUN` would write the shards inside it.

## 3. Submit the array

```bash
RUN=results/weak_ler sbatch -a 0-199 slurm/slurm_run.sh \
  configs/weak_ler.yaml --shots-per-unit 50000
```

`RUN` is the folder the shards write into; an array task needs it, and
the script refuses without it. The script adds `--shard
$SLURM_ARRAY_TASK_ID/$SHARDS` and `--out $RUN/$SLURM_ARRAY_TASK_ID`
itself, and prints the shard it computed:

```
shard: 0 of 200
```

Everything after the config reaches `decsim collect` as written, so
`--processes` and `--out` pass through like anything else.

Without an array the same script runs the whole sweep in one job.

`slurm/campaign_run.sh` is the same script shaped for a sweep too large
for one array: it takes `SHARDS` and `OFFSET` from the environment, so
several arrays of at most 2,500 tasks can carry one sweep between them.
`configs/campaigns_2026_09/PLAN.md` has the submit lines of a sweep run
that way.

The script's own `#SBATCH` lines are the defaults: one node, one task,
two cpus, 16 gigabytes, 16 hours, on the `cpu` partition. Override them
on the `sbatch` command line, or edit the script for your cluster.

The interpreter is `${DECSIM_PYTHON:-.venv/bin/python}`, so a cluster
whose Python is elsewhere sets `DECSIM_PYTHON` in the environment.

## 4. Fold the shards

```bash
decsim combine results/weak_ler/*
```

`combine` reads every folder's additive files, adds them, recomputes
`sweep.csv` and `links.csv` from the sum, and writes one folder. Nothing
is double counted, because no summary is ever stored: a summary is
computed when it is read. That is why the shards can be added at all.

## 5. Plot

```bash
decsim plot results/<combined> --figure ler_vs_d --probability 0.001
```

`ler_vs_d` reads one physical error rate out of the sweep, so it asks
which one.

## If a task dies

Rerun those array indices, and name the count the sweep was cut into:

```bash
RUN=/scratch/.../results/weak_ler SHARDS=500 sbatch -a 447-499 \
  slurm/slurm_run.sh configs/weak_ler.yaml --shots-per-unit 50000
```

`SHARDS` is what makes that one line. Without it the count is the
array's own, `SLURM_ARRAY_TASK_COUNT`, which is 53 for `-a 447-499`, and
task 447 would compute shard 447 of 53: a share of the sweep no folder
of the first run holds. With it the printed line reads `shard: 447 of
500`, the same shard the first run gave that index. The folder is
written fresh, and `combine` reads whatever folders you hand it, so a
rerun shard replaces the old one simply by being the one you pass.

## Read next

- [Your first sweep](../tutorials/first_sweep.md): a small sweep end to end, with the
  error bars explained.
- [The run folder](../reference/run_folder.md): what each shard writes.
- [The commands](../reference/cli.md): every flag of `collect` and `combine`.
