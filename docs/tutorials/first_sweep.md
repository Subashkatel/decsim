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
    shots: 400
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
times one error rate times one round period is three points, 400 shots
each, so 1,200 shots in all.

Check what it resolves to before running it:

```bash
decsim show configs/my_first_sweep.yaml
```

## Step 2. Run it, on four processes

```bash
decsim collect configs/my_first_sweep.yaml --processes 4 --out results/first_sweep
```

`--processes` gives each worker one work unit at a time. A work unit is
a block of one point's shots that one process runs from start to finish;
step 5 shows how to size it. Shots inside a
unit stay serial, which is what keeps a shot's result a function of its
seed alone.

The summary, from a run of it. Its counts are yours too; its ticks are
that host's, because this config names a decoder rather than pricing
one:

```
{"qpu.distance": 3, "qpu.round_period_microseconds": 1.0, "workload.arguments.physical_error_probability": 0.003}: 400 shots done
{"qpu.distance": 5, "qpu.round_period_microseconds": 1.0, "workload.arguments.physical_error_probability": 0.003}: 400 shots done
{"qpu.distance": 7, "qpu.round_period_microseconds": 1.0, "workload.arguments.physical_error_probability": 0.003}: 400 shots done
qpu.distance: 3
qpu.round_period_microseconds: 1.0
workload.arguments.physical_error_probability: 0.003
algorithm: pymatching
load (service per window / window inter-arrival): 2.53
logical failures: 16 of 400 shots
mismatches vs direct PyMatching: 0
throughput: 0.412 rounds per us
queue wait, mean: 12.764 us
service time per window, mean: 7.593 us
ready to frame commit: median 26.504 us, p99 55.052 us

qpu.distance: 5
...
logical failures: 12 of 400 shots
mismatches vs direct PyMatching: 0
...
qpu.distance: 7
...
logical failures: 8 of 400 shots
mismatches vs direct PyMatching: 0
...
```

Sixteen failures out of 400 at distance 3, twelve at distance 5, eight
at distance 7. The logical error rate falls as the code gets bigger,
which is what a code below its threshold does: more physical qubits buy
a better logical qubit.

`mismatches vs direct PyMatching: 0` is a correctness check that runs on
every shot. decsim decodes the shot in windows, through the whole
machine, while the experiments layer decodes the same shot's detection
events in one piece with PyMatching outside the machine, and the two
predictions are compared.

Do not read zero as a promise. A sliding window commits its correction
without the rounds the whole-circuit decode can see, so windowed
decoding is an approximation of global decoding, and the two are
expected to disagree on a small fraction of shots: Skoric et al.
(arXiv:2209.08552, Sec. I B) put it as processing only a subset of the
syndrome data inevitably reducing the logical fidelity, with a fidelity
close to the global decoder's retained by buffering a whole distance,
which is what decsim's default window does. A run of 1800 shots at
distances 3 and 5 and physical error rates 0.003 to 0.01 disagreed on 4
of them. So zero here says these 400 shots had no disagreement; a
handful in a larger sweep is the approximation showing, and a noticeable
fraction of the shots is a broken machine.

## Step 3. Read the error bars

```bash
cut -d, -f3,6,8-11 results/first_sweep/sweep.csv
```

```
qpu.distance,shots,logical_failures,logical_error_rate,ler_wilson_low,ler_wilson_high
3,400,16,0.04,0.024768847722620668,0.06398278162908555
5,400,12,0.03,0.017242849034032177,0.05169903312966765
7,400,8,0.02,0.010168264597915496,0.038963870377777945
```

The rows come in the sweep's order, distance 3, 5 and 7, the order the
summary printed them. Each row's first columns name its point: its id,
then one column per yaml path the sweep sets, here the error rate, the
distance and the round period.

`logical_error_rate` is the failures divided by the shots. It is an
estimate, and 16 out of 400 would have come out differently with
different seeds. The two Wilson columns say how differently.

A **Wilson interval** is a range of true failure probabilities that
would plausibly produce the count you saw. decsim computes it at
`z = 1.96`, which is the conventional 95 percent (`wilson_interval` in
`decsim/experiments/report.py`). Read the distance 3 row as: the true rate is
somewhere between about 2.5 percent and about 6.4 percent, and 4
percent is the middle of the evidence.

Why Wilson and not the textbook interval you may have met, the estimate
plus or minus 1.96 times the standard error? Because that one falls
apart exactly where quantum error correction lives. At a low failure
count it gives an interval that runs below zero, and at zero failures it
gives an interval of zero width, which would say a rate is known
exactly from having seen no failures at all. The Wilson interval stays
inside 0 and 1 and stays sensible at zero counts, which is why it is the
one decsim reports.

Now look at the rows together. Distance 5's interval runs from 1.7 to
5.2 percent and distance 7's from 1.0 to 3.9 percent. They overlap. On
400 shots this run has **not** shown that distance 7 is better than
distance 5, even though its estimate is lower. That is the honest
reading, and it is the reason `configs/weak_ler.yaml` runs a million
shots at its lowest error rates.

The rule of thumb the shipped sweeps are sized by: aim for at least 100
failures at a point you want to quote. Below that, quote the interval,
or quote the point as an upper bound.

## Step 4. Draw it

decsim writes the numbers and leaves the figure to you, since only you
know what it should show. `decsim.results.load` reads a run folder into
one row per point, its `sweep.csv` columns, one of them per swept path,
beside every setting it ran with (`settings.` and the setting's path):

```python
import matplotlib.pyplot as plt

import decsim.results as results

folders = ["results/first_sweep"]
rows = results.load(*folders)
error_rate = "workload.arguments.physical_error_probability"
kept = [row for row in rows if row[error_rate] == 0.003]
distances = [row["qpu.distance"] for row in kept]
rates = [row["logical_error_rate"] for row in kept]
below = [row["logical_error_rate"] - row["ler_wilson_low"] for row in kept]
above = [row["ler_wilson_high"] - row["logical_error_rate"] for row in kept]
figure, ax = plt.subplots()
ax.errorbar(distances, rates, yerr=[below, above], fmt="o-")
ax.set_xlabel("code distance")
ax.set_ylabel("logical error rate")
results.save_figure(figure, "ler.png", rows, folders)
```

The error bars are the Wilson columns you just read. `save_figure`
writes `ler.png` and, beside it, the script that drew it, the rows it
drew and the folders they came from.

## Step 5. Cut it into shards and put it back together

A sweep that takes two minutes does not need cutting up. A sweep that
takes 350 core hours does, and the mechanism is worth seeing on
something small.

```bash
decsim collect configs/my_first_sweep.yaml --shard 0/2 --out results/shards/0 --shots-per-unit 100
decsim collect configs/my_first_sweep.yaml --shard 1/2 --out results/shards/1 --shots-per-unit 100
decsim combine results/shards/0 results/shards/1 --out results/combined
```

`--shots-per-unit 100` cuts each point's 400 shots into four work units.
`--shard i/n` runs the units whose index modulo `n` is `i`, so the two
commands between them run every unit exactly once. `combine` reads both
folders' additive files, adds them, and recomputes the summary.

The combined report's counts:

```bash
cut -d, -f3,6,8,9 results/combined/sweep.csv
```

```
qpu.distance,shots,logical_failures,logical_error_rate
3,400,16,0.04
5,400,12,0.03
7,400,8,0.02
```

The same three counts as the single run. That is not luck: a shot's seed
is derived from the run's seed and the shot's position, so shot 173 of
distance 5 is the same shot whichever process runs it. Cutting a sweep
into shards changes nothing about the result.

The tick columns do move a little between the two, because this run is
charged its decoder's measured wall clock. See
[Time](../explanation/time.md).

[How to run a sweep on Slurm](../how-to/run_a_sweep_on_slurm.md) is the same mechanism as a cluster
array job.

## What you learned

- A sweep is a set of points; a point is a machine; a shot is one run of
  it.
- A logical error rate is an estimate, and the Wilson interval is how
  much to trust it.
- Overlapping intervals mean the runs have not been shown to differ.
- Shards add up exactly, because no summary is stored.

## Read next

- [Two tiers](two_tiers.md): a run with two decoders and an
  escalation.
- [How to run a sweep on Slurm](../how-to/run_a_sweep_on_slurm.md): the same sweep on a cluster.
- [The run folder](../reference/run_folder.md): every column you did not read here.
