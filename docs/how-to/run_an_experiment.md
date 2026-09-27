[decsim docs](../README.md) › [How-to guides](README.md)

# How to run an experiment

An experiment is one Python script, `experiments/<name>/run.py`. It
lists its points, and each point runs until it has enough errors or
reaches its shot cap. The commands below use the decoder baseline,
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
it, so a gitless task and a git host agree on the same commit.

A folder belongs to one script and one commit. Running a different
script or commit into it is refused; give `--out` a new folder.

## Run every point on Slurm

One array task runs one point. Submit from the checkout's root, which
each task puts on its PYTHONPATH, and pass the folder, so every task
writes into the same one:

```bash
sbatch --array 0-$(( $(PYTHONPATH=. python experiments/decoder_baseline/run.py --list | wc -l) - 1 )) \
  slurm/run.sbatch experiments/decoder_baseline/run.py results/2026-09-27_decoder_baseline
```

Each task has 16 cores and 24 hours (`slurm/run.sbatch`). Add the
account, partition or QOS your cluster needs on the `sbatch` line.

## Resume

Submit the same command again, with the same folder. A point reads its
own CSV and goes on from there; a point that already stopped takes no
new shots. On a later day, `--out` or the sbatch folder must name the
original folder: the dated default would be a new, empty one.

## Combine

```bash
PYTHONPATH=. python experiments/decoder_baseline/run.py combine --out results/2026-09-27_decoder_baseline
```

This writes `stats.csv` from every point's CSV, with sinter's own
columns: shots, errors, discards, seconds, decoder, strong id and the
labels. A run of every point in one process writes it at the end.
