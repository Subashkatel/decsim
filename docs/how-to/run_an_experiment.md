[decsim docs](../README.md) › [How-to guides](README.md)

# How to run an experiment

An experiment is one Python script, `experiments/<name>/run.py`. It
lists its points, and each point runs until it has enough errors, reaches
its shot limit, or its job's time runs out. The commands below use the decoder baseline,
`experiments/decoder_baseline/run.py`.

## Before you start

- A python environment with decsim's dependencies and sinter,
  relay-bp and tesseract-decoder, the packages the baseline's decoders
  come from. The `run` extra does not install the last three.
- The Union-Find library, built once in each checkout:
  `tools/build_union_find.sh`.
- Every command runs from the checkout's root with `PYTHONPATH=.`, so
  python imports this checkout's decsim and not another installed one.

## List the points

```bash
PYTHONPATH=. python experiments/decoder_baseline/run.py --list
```

Each line is a point's id, its decoder and its labels, for example
`0 decoder=union-find basis=x d=5 p=0.0005`. The ids are 0, 1, 2 and on,
in the order the script adds the points.

## Run one point

```bash
PYTHONPATH=. python experiments/decoder_baseline/run.py 7 --workers 4 --out results/baseline_test
```

This runs point 7 with 4 worker processes. Without `--out`, the results
go in `results/<today>_decoder_baseline/`, or under the folder the
DECSIM_RESULTS environment variable names. With no id, the script
runs every point, one after another.

The folder holds:

- `points/<id>.csv`, each point's counts in sinter's CSV format;
- `stats.csv`, every point's counts in one file, once you combine;
- a copy of `run.py` and `commit.txt`, the commit that ran and
  whether the tree had uncommitted changes.

On a node whose python has no git, as in the container, export
DECSIM_TREE_DIRTY (1 for uncommitted changes, 0 for none) before
running; without it `commit.txt` says `dirty None`. A folder compares
the commit always and the dirty flag only where both sides could read
it, so a gitless task and a git host agree on the same commit. With no
git program, the commit is read from the checkout's `.git` folder; a
tree with no readable commit at all, such as a copy without `.git`, is
refused.

A folder belongs to one script and one commit. Running a different
script or commit into it is refused; give `--out` a new folder.

## Run every point on Slurm

One array task runs one point. Pass the folder, so every task writes
into the same one:

```bash
sbatch --array 0-$(( $(PYTHONPATH=. python experiments/decoder_baseline/run.py --list | wc -l) - 1 )) \
  slurm/run.sbatch experiments/decoder_baseline/run.py results/2026-09-27_decoder_baseline
```

Each task imports the checkout the script sits in, two folders above
`experiments/<name>/run.py`, ahead of any PYTHONPATH the job already
has, so the job may be submitted from anywhere. A script outside a
decsim checkout is refused.

Each task has 16 cores and 24 hours (`slurm/run.sbatch`). The time
limit is the budget: `-t 3:00:00` on the `sbatch` line gives each point
3 hours, 48 core-hours. Add the account, partition or QOS your cluster
needs on the `sbatch` line too.

## Resume

Submit the same command again, with the same folder. A point reads its
own CSV and goes on from there; a point that already stopped takes no
new shots. A task its time limit stopped loses at most its workers'
last two minutes, since sinter saves each worker's counts at least that
often (`worker_flush_period`, sinter/_collection/_collection.py), and
sinter's collect is made to be killed and restarted
(sinter/_command/_main_collect.py, `--save_resume_filepath`). To spend
more on the points that have not reached their errors, submit again
with those ids. On a later day, `--out` or the sbatch folder must name the
original folder: the dated default would be a new, empty one.

## Combine

```bash
PYTHONPATH=. python experiments/decoder_baseline/run.py combine --out results/2026-09-27_decoder_baseline
```

This writes `stats.csv` from every point's CSV, with sinter's own
columns: shots, errors, discards, seconds, decoder, strong id and the
labels. A run of every point in one process writes it at the end.

## Plot and keep the results

```bash
PYTHONPATH=. python experiments/decoder_baseline/plot.py results/2026-09-27_decoder_baseline
```

This draws the folder's `plots/` from its `stats.csv`: the logical
error rate per round against the physical error rate, a figure per
decoder and one comparing the decoders for each basis. The repository
tracks a results folder's `stats.csv`, `commit.txt`, script copy and
`plots/`, so every experiment's results live beside the code that made
them; `points/` and `logs/` stay with the run (`.gitignore`).
