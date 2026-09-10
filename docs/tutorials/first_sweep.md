# Your first sweep

In `docs/tutorials/first_run.md` you ran two shots and got a logical
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

```yaml
# My first sweep: three distances at one physical error rate.
extends: weak_decoder_baseline.yaml

weak_decoder:
  kind: pymatching
  units: 1
  unit_memory_rounds: null
  engine:
    clock: fridge
    fetch_cycles_per_round: 1
    release_cycles_per_job: 10

sweep:
  - physical_error_probability: [0.003]
    distance: [3, 5, 7]
    round_period_us: [1.0]
    shots: 400
```

Three things are happening here.

`extends` reads `weak_decoder_baseline.yaml` from the same folder first
and applies this file's keys over it. A section written here replaces
the base's section **whole**, which is why the `weak_decoder` block
repeats `units`, `unit_memory_rounds` and `engine` even though the base
already had them. Leave `engine` out and the load fails.

The `weak_decoder` block names `pymatching`, so decsim decodes every
window for real. The baseline prices its decoder with a number instead,
which is right for a timing study and wrong for an accuracy one.

The `sweep` block is the set of points: every combination of the three
lists. Three distances times one error rate times one round period is
three points, 400 shots each, so 1,200 shots in all.

Check what it resolves to before running it:

```bash
decsim show configs/my_first_sweep.yaml
```

## Step 2. Run it, on four processes

```bash
decsim collect configs/my_first_sweep.yaml --processes 4
```

`--processes` gives each worker one work unit at a time. Shots inside a
unit stay serial, which is what keeps a shot's result a function of its
seed alone.

This took two minutes on the machine this page was written on. The
summary, from that run:

```
p 0.003, d 3, round period 1.0 us: 400 shots done
p 0.003, d 5, round period 1.0 us: 400 shots done
p 0.003, d 7, round period 1.0 us: 400 shots done
distance: 3
physical error rate: 0.003
algorithm: pymatching
round period: 1 us
load (service per window / window inter-arrival): 7.70
logical failures: 26 of 400 shots
mismatches vs direct PyMatching: 0
throughput: 0.259 rounds per us
queue wait, mean: 27.580 us
service time per window, mean: 23.108 us
ready to frame commit: median 50.063 us, p99 103.937 us

distance: 5
...
logical failures: 8 of 400 shots
...
distance: 7
...
logical failures: 5 of 400 shots
```

Twenty-six failures out of 400 at distance 3, eight at distance 5, five
at distance 7. The logical error rate falls as the code gets bigger,
which is what a code below its threshold does: more physical qubits buy
a better logical qubit.

`mismatches vs direct PyMatching: 0` is a correctness check that runs on
every shot. decsim decodes the shot in windows, through the whole
machine, while the front decodes the same shot's detection events in one
piece with PyMatching outside the machine, and the two predictions are
compared. Zero mismatches means the windowing did not change the answer
on any of these shots.

## Step 3. Read the error bars

```bash
cut -d, -f1,2,5,7,8,9,10 results/<run>/sweep.csv
```

```
distance,physical_error_probability,shots,logical_failures,logical_error_rate,ler_wilson_low,ler_wilson_high
3,0.003,400,26,0.065,0.04474017569726697,0.09353582162445521
5,0.003,400,8,0.02,0.010168264597915496,0.038963870377777945
7,0.003,400,5,0.0125,0.005350671853550634,0.02892415273113802
```

`logical_error_rate` is the failures divided by the shots. It is an
estimate, and 26 out of 400 would have come out differently with
different seeds. The two Wilson columns say how differently.

A **Wilson interval** is a range of true failure probabilities that
would plausibly produce the count you saw. decsim computes it at
`z = 1.96`, which is the conventional 95 percent (`wilson_interval` in
`decsim/front/report.py`). Read the distance 3 row as: the true rate is
somewhere between about 4.5 percent and about 9.4 percent, and 6.5
percent is the middle of the evidence.

Why Wilson and not the textbook interval you may have met, the estimate
plus or minus 1.96 times the standard error? Because that one falls
apart exactly where quantum error correction lives. At a low failure
count it gives an interval that runs below zero, and at zero failures it
gives an interval of zero width, which would say a rate is known
exactly from having seen no failures at all. The Wilson interval stays
inside 0 and 1 and stays sensible at zero counts, which is why it is the
one decsim reports.

Now look at the rows together. Distance 5's interval runs from 1.0 to
3.9 percent and distance 7's from 0.5 to 2.9 percent. They overlap. On
400 shots this run has **not** shown that distance 7 is better than
distance 5, even though its estimate is lower. That is the honest
reading, and it is the reason `configs/weak_ler.yaml` runs a million
shots at its lowest error rates.

The rule of thumb the shipped sweeps are sized by: aim for at least 100
failures at a point you want to quote. Below that, quote the interval,
or quote the point as an upper bound.

## Step 4. Draw it

```bash
decsim plot results/<run> --figure ler_vs_d --probability 0.003
```

```
results/<run>/ler_vs_distance.png
```

`ler_vs_d` plots the logical error rate against the code distance at one
physical error rate, so it asks which rate to read out of the sweep. The
error bars on it are the Wilson columns you just read.

## Step 5. Cut it into shards and put it back together

A sweep that takes two minutes does not need cutting up. A sweep that
takes 350 core hours does, and the mechanism is worth seeing on
something small.

```bash
decsim collect configs/my_first_sweep.yaml --shard 0/2 --out results/shards/0 --shots-per-unit 100
decsim collect configs/my_first_sweep.yaml --shard 1/2 --out results/shards/1 --shots-per-unit 100
decsim combine results/shards/0 results/shards/1
```

`--shots-per-unit 100` cuts each point's 400 shots into four work units.
`--shard i/n` runs the units whose index modulo `n` is `i`, so the two
commands between them run every unit exactly once. `combine` reads both
folders' additive files, adds them, and recomputes the summary.

The combined report says:

```
distance,shots,logical_failures,logical_error_rate
3,400,26,0.065
5,400,8,0.02
7,400,5,0.0125
```

The same three counts as the single run. That is not luck: a shot's seed
is derived from the run's seed and the shot's position, so shot 173 of
distance 5 is the same shot whichever process runs it. Cutting a sweep
into shards changes nothing about the result.

The tick columns do move a little between the two, because this run
names a decoder and is charged its measured wall clock. See
`docs/explanation/time.md`.

`docs/how-to/run_a_sweep_on_slurm.md` is the same mechanism as a cluster
array job.

## What you learned

- A sweep is a set of points; a point is a machine; a shot is one run of
  it.
- A logical error rate is an estimate, and the Wilson interval is how
  much to trust it.
- Overlapping intervals mean the runs have not been shown to differ.
- Shards add up exactly, because no summary is stored.

## Read next

- `docs/tutorials/two_tiers.md`: a run with two decoders and an
  escalation.
- `docs/how-to/run_a_sweep_on_slurm.md`: the same sweep on a cluster.
- `docs/reference/run_folder.md`: every column you did not read here.
