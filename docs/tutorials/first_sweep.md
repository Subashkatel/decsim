[decsim docs](../README.md) › [Tutorials](README.md)

# Your first sweep

In [Your first run](first_run.md) you ran 20 shots and got a logical
error rate with no estimate and an upper limit of 0.17, which says
almost nothing. This lesson runs a real, small sweep, explains that
interval, and shows how a sweep is cut into pieces and put back
together.

It takes about fifteen minutes, one or two of which are the machine
running.

## Step 1. Read the run file

This lesson's run file ships with decsim as `examples/my_first_sweep.py`.
Its loop:

```python examples/my_first_sweep.py
points = []
for distance in DISTANCES:
    base = machine_settings.weak_decoder_baseline(
        distance,
        PHYSICAL_ERROR_PROBABILITY,
        ROUND_PERIOD_MICROSECONDS,
    )
    # the baseline charges its decoder a number; this sweep decodes for real
    matching = minimum_weight_perfect_matching.PyMatchingDecoder.Settings()
    weak_decoder = dataclasses.replace(base.weak_decoder, algorithm=matching)
    machine = dataclasses.replace(base, weak_decoder=weak_decoder)
```

The decoder is PyMatching with no card, so decsim decodes every window
for real and charges each decode the wall clock it took on your
computer. The base prices its decoder with a number instead, which is
right for a timing study and wrong for an accuracy one.

`DISTANCES` is 3, 5 and 7, and the physical error probability is 0.003,
so the sweep is three points. Its `COLLECTION` stops a point at
`max_shots`, 400 shots, so 1,200 shots in all.

## Step 2. Run it, on four processes

```bash
decsim run examples/my_first_sweep.py --processes 4 --out results/first_sweep
```

`--processes` gives each worker one piece at a time. A piece is a block
of one point's shots that one process runs from start to finish and
saves whole the moment it ends; step 5 shows what that buys. Shots
inside a piece stay serial, which is what keeps a shot's result a
function of its seed alone.

It prints the folder it writes, then one line per point as the point
finishes:

```
run dir: results/first_sweep

d3: 400 shots done (cap)
d5: 400 shots done (cap)
d7: 400 shots done (cap)
```

Its `sweep.csv`, which step 3 reads, counts sixteen failures out of 400
at distance 3, twelve at distance 5 and eight at distance 7. Those
counts are yours too, since they follow from the seeds. The logical
error rate falls as the code gets bigger, which is what a code below its
threshold does: more physical qubits buy a better logical qubit.

Decoding in windows costs a little accuracy. A sliding window commits
its correction without the rounds a whole-circuit decode can see, so
windowed decoding approximates global decoding: Skoric et al.
(arXiv:2209.08552, Sec. I B) find a fidelity close to the global
decoder's when the window buffers a whole distance of rounds, which is
what decsim's default window does. To see it, add a point whose
`SlidingWindowScheme.Settings` has `commit_rounds` long enough to hold
the whole shot, so one window decodes the full history. At one seed the
two points draw the same shot (their `sample_digest` cells in
`shots.csv` are equal), so a seed where their `predictions` differ is a
shot the window's missing rounds decided. Expect a few in a large
sweep; a noticeable fraction of the shots is a broken machine.

## Step 3. Read the error bars

```bash
cut -d, -f3,6,8,11-14 results/first_sweep/sweep.csv
```

```
qpu.distance,shots,logical_failures,state,logical_error_rate_estimate,logical_error_rate_low,logical_error_rate_high
3,400,16,cap,0.04,0.023033395740548884,0.06414601093905078
5,400,12,cap,0.03,0.015595552999894341,0.05181721543058358
7,400,8,cap,0.02,0.00867317036911831,0.03902627671162005
```

The rows come in the run file's order, distance 3, 5 and 7. Each row's
first columns name its point: its id, then the point's settings as its
metadata names them.

`logical_error_rate_estimate` is the failures divided by the scored
shots. It is an estimate, and 16 out of 400 would have come out
differently with different seeds. The two limit columns say how
differently.

The limits are a 95 percent **confidence interval**: a range of true
failure probabilities that would plausibly produce the count you saw,
built so that 95 runs in 100 bracket the true rate. decsim computes it
exactly for the rule the point stopped by (`estimate` in
`decsim/experiments/failure_statistics.py`). Every point here stopped
at its shot cap, `state` `cap`, so its shot count was fixed and its
limits are Clopper and Pearson's. Read the distance 3 row as: the true
rate is somewhere between about 2.3 percent and about 6.4 percent, and
4 percent is the middle of the evidence.

Why not the textbook interval, the estimate plus or minus 1.96 times
the standard error? Because that one falls apart exactly where quantum
error correction lives. At a low failure count it gives an interval
that runs below zero, and at zero failures it gives an interval of zero
width, which would say a rate is known exactly from having seen no
failures at all. The exact interval stays inside 0 and 1, and at zero
failures it gives an upper limit alone, which is all zero failures can
say.

Now look at the rows together. Distance 5's interval runs from 1.6 to
5.2 percent and distance 7's from 0.9 to 3.9 percent. They overlap. On
400 shots this run has **not** shown that distance 7 is better than
distance 5, even though its estimate is lower. That is the honest
reading, and it is the reason a real sweep runs a million shots at its
lowest error rates.

The rule of thumb: aim for at least 100 failures at a point you want to
quote. A point can stop there by itself:
`decsim.CollectionSettings(max_failures=100, max_shots=...)` stops it at
its hundredth failure, `state` `target`, and its limits are then the
ones exact for a failure count fixed in advance. Below that, quote the
interval, or quote the point as an upper bound.

## Step 4. Draw it

decsim writes the numbers and leaves the figure to you, since only you
know what it should show. `sweep.csv` holds one row per point: its
`point_id`, one column per swept path, its counts, and the estimate
with its limits, `logical_error_rate_estimate`, `logical_error_rate_low`
and `logical_error_rate_high`. Read it with any csv reader and draw the
error bars from the limit columns you just read.

## Step 5. Stop it and pick it up again

A sweep that takes a minute is never stopped halfway. A sweep that
takes 350 core hours is, by a time limit or a node going down, and the
mechanism is worth seeing on something small.

`decsim run` saves each point's shots as pieces under
`results/first_sweep/pieces/`, one folder per point and one per piece,
named by its first and last seed. A piece holds a set number of QEC
rounds, 20,000 unless the point's `CollectionSettings` says otherwise
(`piece_rounds`), and a piece's folder appears only once the piece is
whole. Delete each point's first piece, as a killed job would leave
them missing, and run the same command again:

```bash
rm -r results/first_sweep/pieces/*/0-*
decsim run examples/my_first_sweep.py --processes 4 --out results/first_sweep
cut -d, -f3,6,8,12 results/first_sweep/sweep.csv
```

```
qpu.distance,shots,logical_failures,logical_error_rate_estimate
3,400,16,0.04
5,400,12,0.03
7,400,8,0.02
```

The second `decsim run` ran only the pieces that were missing, then
folded every piece again into the results folder. The counts are the
single run's. That is not luck: a shot's seed is derived from the run's
seed and the shot's position, so shot 173 of distance 5 is the same
shot whichever process runs it and whenever. Stopping a sweep and
picking it up changes nothing about the result. The tick columns do
move a little between the two, because this run is charged its
decoder's measured wall clock.

On a cluster the same pieces run as Slurm jobs. `decsim run
examples/my_first_sweep.py --slurm --out $PWD/results/first_sweep`
records every point, so a machine that cannot be built stops it before
anything is queued. It writes `run.sbatch`, a job array whose job `i`
runs point `i` until the point stops, and `fold.sbatch`, one job that
folds every saved piece after the array ends, however its jobs ended.
Then it submits both; `--dry-run` writes them and submits nothing. A
job asks for `--cores` cores (4), `--hours` hours (24) and
`--memory-mb` megabytes (16384), and logs to `logs/<i>.log`. A job
starts no piece past its point's stop, so a point with `max_failures`
stops near its target. If a job dies or runs out of time, submit the
same command again: each job starts from its point's saved pieces, and
a point that stopped runs nothing. `decsim run --fold --out
results/first_sweep` folds the saved pieces at any time, even while
jobs run. The launch refuses a checkout with uncommitted changes
unless `ALLOW_DIRTY=1` is set, and sbatch reads the account, partition
and QOS from SBATCH_ACCOUNT, SBATCH_PARTITION and SBATCH_QOS.

## What you learned

- A sweep is a set of points; a point is a machine; a shot is one run of
  it.
- A logical error rate is an estimate, and its exact interval is how
  much to trust it.
- Overlapping intervals mean the runs have not been shown to differ.
- Pieces add up exactly, because no summary is stored, so a stopped
  sweep picks up where it stopped.

## Read next

- [Two tiers](two_tiers.md): a run with two decoders and an
  escalation.
- [The run folder](../reference/run_folder.md): every column you did
  not read here.
