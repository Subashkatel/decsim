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

## 2. Submit the array

```bash
RUN=results/weak_ler sbatch -a 0-199 slurm/slurm_run.sh \
  configs/weak_ler.yaml --shots-per-unit 50000
```

`RUN` is the folder the shards write into; an array task needs it, and
the script refuses without it. The script adds `--shard
$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT` and `--out
$RUN/$SLURM_ARRAY_TASK_ID` itself. Everything after the config reaches
`decsim collect` as written, so `--processes` and `--out` pass through
like anything else.

Without an array the same script runs the whole sweep in one job.

The script's own `#SBATCH` lines are the defaults: one node, one task,
two cpus, 16 gigabytes, 16 hours, on the `cpu` partition. Override them
on the `sbatch` command line, or edit the script for your cluster.

The interpreter is `${DECSIM_PYTHON:-.venv/bin/python}`, so a cluster
whose Python is elsewhere sets `DECSIM_PYTHON` in the environment.

## 3. Fold the shards

```bash
decsim combine results/weak_ler/*
```

`combine` reads every folder's additive files, adds them, recomputes
`sweep.csv` and `links.csv` from the sum, and writes one folder. Nothing
is double counted, because no summary is ever stored: a summary is
computed when it is read. That is why the shards can be added at all.

## 4. Plot

```bash
decsim plot results/<combined> --figure ler_vs_d --probability 0.001
```

`ler_vs_d` reads one physical error rate out of the sweep, so it asks
which one.

## If a task dies

Rerun that array index. Its folder is written fresh, and `combine` reads
whatever folders you hand it, so a rerun shard replaces the old one
simply by being the one you pass.

## Read next

- [`docs/tutorials/first_sweep.md`](../tutorials/first_sweep.md): a small sweep end to end, with the
  error bars explained.
- [`docs/reference/run_folder.md`](../reference/run_folder.md): what each shard writes.
- [`docs/reference/cli.md`](../reference/cli.md): every flag of `collect` and `combine`.
