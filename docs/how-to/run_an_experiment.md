[decsim docs](../README.md) › [How-to guides](README.md)

# How to run an experiment

An offline experiment is one Python script, `experiments/<name>/run.py`,
that decodes with sinter or runs a function at each of its points,
without the machine. Each point runs until it has enough errors, reaches
its shot limit, or its job's time runs out. The commands below use the
decoder baseline, `experiments/decoder_baseline/run.py`. An experiment
on the machine runs with `decsim run` instead
([How to run a sweep on Slurm](run_a_sweep_on_slurm.md)).

## Before you start

- A python environment with decsim's dependencies and sinter,
  relay-bp and tesseract-decoder, the packages the baseline's decoders
  come from. The `run` extra installs sinter; it does not install the
  other two.
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
in the order the script builds the points.

## Run one point

```bash
PYTHONPATH=. python experiments/decoder_baseline/run.py 7 --workers 4 --out results/baseline_test
```

This runs point 7 with 4 worker processes. One point names its folder
with `--out`, since every task of an array writes the one folder the
points share. With no id the script runs every point, one after
another, then combines; without `--out` that run goes to a new
`results/<today>_decoder_baseline/`, and a second one the same day to
`_2`.

The folder holds:

- `points/<id>.csv`, each point's counts in sinter's CSV format;
- `stats.csv`, every point's counts in one file, once you combine;
- a copy of `run.py`, `run.json` (the commit that ran, whether the tree
  had uncommitted changes, the host and the package versions) and
  `code_state.patch` when the tree had changes, as every decsim results
  folder holds ([The run folder](../reference/run_folder.md)).

A folder belongs to one commit and one `run.py`. Running another
commit, or an edited `run.py`, into it is refused; give `--out` a new
folder. A tree whose commit cannot be read is refused too, unless
`ALLOW_DIRTY=1` is set.

## Run every point on Slurm

One array task runs one point. Pass the folder, so every task writes
into the same one:

```bash
sbatch --array 0-287 --cpus-per-task 16 --mem 32G --time 24:00:00 \
  --wrap "PYTHONPATH=$PWD python experiments/decoder_baseline/run.py \$SLURM_ARRAY_TASK_ID --workers \$SLURM_CPUS_PER_TASK --out results/2026-09-27_decoder_baseline"
```

The array's range is 0 to the last id `--list` prints. The time limit
is the budget: `--time 3:00:00` gives each point 3 hours, 48
core-hours. Add the account, partition or QOS your cluster needs on the
`sbatch` line too, or set them in SBATCH_ACCOUNT, SBATCH_PARTITION and
SBATCH_QOS.

## Resume

Submit the same command again, with the same folder. A point reads its
own CSV and goes on from there; a point that already stopped takes no
new shots. A task its time limit stopped loses at most its workers'
last two minutes, since sinter saves each worker's counts at least that
often (`worker_flush_period`, sinter/_collection/_collection.py), and
sinter's collect is made to be killed and restarted
(sinter/_command/_main_collect.py, `--save_resume_filepath`). To spend
more on the points that have not reached their errors, submit again
with those ids.

## Combine

```bash
PYTHONPATH=. python experiments/decoder_baseline/run.py combine --out results/2026-09-27_decoder_baseline
```

This writes `stats.csv` from every point's CSV, with sinter's own
columns: shots, errors, discards, seconds, decoder, strong id and the
labels. A run of every point in one process writes it at the end.

## Points that run a function

A point can be a Python function instead of a sinter task, for an
experiment that is not a decode, such as
`experiments/burst_detection/run.py`. Its script calls
`function(labels, seed, folder)`, the seed a hash of the labels so a
point draws the same shots in any grid, and writes the rows it returns,
a list of dicts, after the labels' columns into `points/<id>.csv`. The
CSV is written once the function returns, so a point with a CSV is done
and a resubmitted job runs only the rest. A point that needs another's
rows reads them from the results folder at `point_path(folder, id)`,
and fails when they are not there yet; on Slurm, submit that point with
`--dependency=afterok:<job>` on the job that runs the other. `combine`
writes each results file from the rows of the points that name it.

## Keep the results

The folder's `stats.csv` holds each point's sinter counts: `shots`,
`errors`, `discards`, its `decoder` and its labels in `json_metadata`.
Every baseline shot is a 100-round memory (`run.py`, `ROUNDS`), which
`stats.csv` does not carry. The logical error rate per round, the one
the Tesseract paper reports (Beni et al. 2503.10988, eq. 8), and its
band, each end converted the same way, are:

```python
rate = sinter.shot_error_rate_to_piece_error_rate(
    errors / (shots - discards), pieces=100
)
band = sinter.fit_binomial(
    num_shots=shots - discards, num_hits=errors, max_likelihood_factor=1e3
)
```

The repository tracks a results
folder's `stats.csv`, `run.json` and script copy, so every experiment's
results live beside the code that made them; `points/` and `logs/` stay
with the run (`.gitignore`).
