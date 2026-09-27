[decsim docs](../README.md) › [Tutorials](README.md)

# Your first sweep

In [Your first run](first_run.md) you ran two shots and got a logical
error rate of 0 with an interval running from 0 to 0.66, which says
almost nothing. This lesson runs a real, small sweep, explains that
interval, and shows how a sweep is cut into pieces and put back
together.

It takes about fifteen minutes, two of which are the machine running.

## Step 1. Copy a shipped config and make a sweep of it

Every shipped config is a starting point. `weak_decoder_baseline.yaml`
is the defaults a single-tier study begins from, with every key
documented in `configs/reference.yaml`.

This lesson's config ships with decsim, as
`configs/my_first_sweep.yaml`, so you can read it here rather than
type it:

```yaml configs/my_first_sweep.yaml
# My first sweep: three distances at one physical error rate.
extends: weak_decoder_baseline.yaml

weak_decoder:
  kind: pymatching
  units: 1
  unit_memory:
    bits: null
  engine:
    clock: fridge
    fetch_cycles_per_round: 1
    fetch_cycles_per_job: 0
    release_cycles_per_job: 10
    release_cycles_per_round: 0

sweep:
  - axes:
      workload.arguments.physical_error_probability: [0.003]
      qpu.distance: [3, 5, 7]
      qpu.round_period_microseconds: [1.0]
    collection: {max_shots: 400}
```

Three things are happening here.

`extends` reads `weak_decoder_baseline.yaml` from the same folder first
and applies this file's keys over it. A section written here replaces
the base's section **whole**, which is why the `weak_decoder` block
repeats `units`, `unit_memory` and `engine` even though the base
already had them. Leave `engine` out and the load fails.

The `weak_decoder` block names `pymatching`, so decsim decodes every
window for real. The baseline prices its decoder with a number instead,
which is right for a timing study and wrong for an accuracy one.

The `sweep` block is the set of points: every combination of its
`axes`, each a yaml path and the values the sweep sets there. The
error rate is an argument of the maker that builds the circuit, and the
distance and round period are the QPU's. The base file's workload reads
the distance back with `distance: ${qpu.distance}`, the value at that
path, so the code and the circuit share one number. Three distances
times one error rate times one round period is three points. The
block's `collection` says when a point stops, here at `max_shots`, 400
shots each, so 1,200 shots in all.

Check what it resolves to before running it:

```bash
decsim show configs/my_first_sweep.yaml
```

## Step 2. Run it, on four processes

```bash
decsim collect configs/my_first_sweep.yaml --processes 4 --out results/first_sweep
```

`--processes` gives each worker one piece at a time. A piece is a block
of one point's shots that one process runs from start to finish and
saves whole the moment it ends; step 5 shows what that buys. Shots
inside a piece stay serial, which is what keeps a shot's result a
function of its seed alone.

The summary, from a run of it. Its counts are yours too; its ticks are
that host's, because this config names a decoder rather than pricing
one:

```
{"qpu.distance": 3, "qpu.round_period_microseconds": 1.0, "workload.arguments.physical_error_probability": 0.003}: 400 shots done (cap)
{"qpu.distance": 5, "qpu.round_period_microseconds": 1.0, "workload.arguments.physical_error_probability": 0.003}: 400 shots done (cap)
{"qpu.distance": 7, "qpu.round_period_microseconds": 1.0, "workload.arguments.physical_error_probability": 0.003}: 400 shots done (cap)
qpu.distance: 3
qpu.round_period_microseconds: 1.0
workload.arguments.physical_error_probability: 0.003
algorithm: pymatching
load (service per window / window inter-arrival): 2.53
logical failures: 16 of 400 scored shots
logical error rate among scored shots: 0.04, 95% 0.023 to 0.0641 (cap)
unscored shots: 0 of 400 (0)
throughput: 0.412 rounds per us
queue wait, mean: 12.764 us
service time per window, mean: 7.593 us
ready to frame commit: median 26.504 us, p99 55.052 us

qpu.distance: 5
...
logical failures: 12 of 400 scored shots
logical error rate among scored shots: 0.03, 95% 0.0156 to 0.0518 (cap)
unscored shots: 0 of 400 (0)
...
qpu.distance: 7
...
logical failures: 8 of 400 scored shots
logical error rate among scored shots: 0.02, 95% 0.00867 to 0.039 (cap)
unscored shots: 0 of 400 (0)
...
```

Sixteen failures out of 400 at distance 3, twelve at distance 5, eight
at distance 7. The logical error rate falls as the code gets bigger,
which is what a code below its threshold does: more physical qubits buy
a better logical qubit.

To see what decoding in windows costs, add one more value to the sweep:
`windows.commit_rounds` long enough to hold the whole shot, so one window
decodes the full history the way a whole-circuit decode does. At one seed
the two points draw the same shot (their `sample_digest` cells in
`shots.csv` are equal), so a seed where their `predictions` differ is
a shot the window's missing rounds decided. This memory has one
observable, so there a differing prediction is a differing
`logical_failure`; with several observables or patches, two failures
can be two different answers, and `predictions` tells them apart.

Expect a few. A sliding window commits its correction without the rounds
the whole-circuit decode can see, so windowed decoding is an
approximation of global decoding: Skoric et al. (arXiv:2209.08552,
Sec. I B) put it as processing only a subset of the syndrome data
inevitably reducing the logical fidelity, with a fidelity close to the
global decoder's retained by buffering a whole distance, which is what
decsim's default window does. A run of 1800 shots at distances 3 and 5
and physical error rates 0.003 to 0.01 disagreed on 4 of them. A handful
in a larger sweep is the approximation showing, and a noticeable
fraction of the shots is a broken machine.

## Step 3. Read the error bars

```bash
cut -d, -f3,6,8,11-14 results/first_sweep/combined/*/sweep.csv
```

```
qpu.distance,shots,logical_failures,state,logical_error_rate_estimate,logical_error_rate_low,logical_error_rate_high
3,400,16,cap,0.04,0.023033395740548884,0.06414601093905078
5,400,12,cap,0.03,0.015595552999894341,0.05181721543058358
7,400,8,cap,0.02,0.00867317036911831,0.03902627671162005
```

The rows come in the sweep's order, distance 3, 5 and 7, the order the
summary printed them. Each row's first columns name its point: its id,
then one column per yaml path the sweep sets, here the error rate, the
distance and the round period.

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

Why not the textbook interval you may have met, the estimate plus or
minus 1.96 times the standard error? Because that one falls apart
exactly where quantum error correction lives. At a low failure count it
gives an interval that runs below zero, and at zero failures it gives
an interval of zero width, which would say a rate is known exactly from
having seen no failures at all. The exact interval stays inside 0 and 1,
and at zero failures it gives an upper limit alone, which is all zero
failures can say.

Now look at the rows together. Distance 5's interval runs from 1.6 to
5.2 percent and distance 7's from 0.9 to 3.9 percent. They overlap. On
400 shots this run has **not** shown that distance 7 is better than
distance 5, even though its estimate is lower. That is the honest
reading, and it is the reason a real sweep runs a million shots at its
lowest error rates.

The rule of thumb the shipped sweeps are sized by: aim for at least 100
failures at a point you want to quote. A point can stop there by
itself: `collection: {max_failures: 100, max_shots: ...}` stops it at
its hundredth failure, `state` `target`, and its limits are then the
ones exact for a failure count fixed in advance. Below that, quote the
interval, or quote the point as an upper bound.

## Step 4. Draw it

decsim writes the numbers and leaves the figure to you, since only you
know what it should show. `decsim.results.load` reads a run folder, the
one `collect` folded under the experiment folder's `combined/`, into
one row per point, its `sweep.csv` columns, one of them per swept path,
beside every setting it ran with (`settings.` and the setting's path):

```python
import glob

import matplotlib.pyplot as plt

import decsim.results as results

folders = glob.glob("results/first_sweep/combined/*")
rows = results.load(*folders)
error_rate = "workload.arguments.physical_error_probability"
kept = [row for row in rows if row[error_rate] == 0.003]
distances = [row["qpu.distance"] for row in kept]
estimate = "logical_error_rate_estimate"
rates = [row[estimate] for row in kept]
below = [row[estimate] - row["logical_error_rate_low"] for row in kept]
above = [row["logical_error_rate_high"] - row[estimate] for row in kept]
figure, ax = plt.subplots()
ax.errorbar(distances, rates, yerr=[below, above], fmt="o-")
ax.set_xlabel("code distance")
ax.set_ylabel("logical error rate")
results.save_figure(figure, "ler.png", rows, folders)
```

The error bars are the limit columns you just read. `save_figure`
writes `ler.png` and, beside it, the script that drew it, the rows it
drew and the folders they came from.

## Step 5. Stop it and pick it up again

A sweep that takes two minutes is never stopped halfway. A sweep that
takes 350 core hours is, by a time limit or a node going down, and the
mechanism is worth seeing on something small.

`collect` saves each point's shots as pieces under
`results/first_sweep/pieces/`, one folder per point and one per piece,
named by its first and last seed. A piece holds a set number of QEC
rounds, 20,000 unless the yaml's `collection` section says otherwise
(`configs/reference.yaml`), and a piece's folder appears only once the
piece is whole. Delete each point's first piece, as a killed job would
leave them missing, and run the same command again:

```bash
rm -r results/first_sweep/pieces/*/0-*
decsim collect configs/my_first_sweep.yaml --processes 4 --out results/first_sweep
cut -d, -f3,6,8,12 results/first_sweep/combined/*/sweep.csv
```

```
qpu.distance,shots,logical_failures,logical_error_rate_estimate
3,400,16,0.04
5,400,12,0.03
7,400,8,0.02
```

The second `collect` ran only the pieces that were missing, then folded
every piece again into the run folder. The counts are the single run's.
That is not luck: a shot's seed is derived from the run's seed and the
shot's position, so shot 173 of distance 5 is the same shot whichever
process runs it and whenever. Stopping a sweep and picking it up
changes nothing about the result.

The tick columns do move a little between the two, because this run is
charged its decoder's measured wall clock. See
[Time](../explanation/time.md).

[How to run a sweep on Slurm](../how-to/run_a_sweep_on_slurm.md) is the same mechanism on a
cluster.

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
- [How to run a sweep on Slurm](../how-to/run_a_sweep_on_slurm.md): the same sweep on a cluster.
- [The run folder](../reference/run_folder.md): every column you did not read here.
