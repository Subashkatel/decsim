[decsim docs](../README.md) › [How-to guides](README.md)

# How to compare two runs

Two run folders can be compared several ways, and which one you want
depends on what the two runs are.

## If they are pieces of one sweep, they are already added

`decsim run` folds every piece of an experiment into the csv
files at the top of its results folder: it reads the pieces' additive
files, adds them, and recomputes `sweep.csv` and `links.csv` from the
sum. Two runs of different points are never added: the result would be
a single row that is neither.

## If they are different configurations, read the rows

Every folder's `sweep.csv` has one row per sweep point. Pick the point
the two runs share and the columns the question is about.

For example, the reference config's point at distance 3, decoded by
PyMatching and charged its measured wall clock. A row names its point
in its first columns, `point_id` and then one column per swept path,
left out here. Every column after `shots` is that host's, so yours will
differ:

```
algorithm,shots,load,queue_wait_mean_us,algorithm_mean_us,buffer0_ready_to_frame_median_us
pymatching,2,2.387,2.6029999999999998,7.12875,15.772
```

and the same point with the decoder priced at one microsecond by a
card, which is the same on every host:

```
algorithm,shots,load,queue_wait_mean_us,algorithm_mean_us,buffer0_ready_to_frame_median_us
1.0,20,0.35585185185185186,0.0,1.0,1.076
```

Read across. The card's algorithm time is smaller, so the load falls
under 1. Under a load of 1 the decoder keeps up, the queue wait falls to
nothing, and the reaction time falls with it. The reaction time here is
the median from a window having its rounds to its correction reaching
the frame.

That comparison is between a software wall clock and a hardware card,
which is a comparison of two questions rather than of two machines. Say
so when you report it. [Time](../explanation/time.md) says why.

## Let decsim say what differs

```bash
decsim diff results/<first> results/<second>
```

`diff` matches the two folders' points by their metadata, the values
their sweep set, and prints
three sections. `settings` lists every value that differs at a point,
the ones the build derived included (`machine.json`). `inputs` lists each
workload file whose sha256 differs (`points/<name>/inputs/hashes.json`).
`results` lists each `sweep.csv` column that differs, and says whether
the two values agree within their error bars: a logical error rate by
its exact interval (with no interval on either side, no statistical
comparison is possible, and `diff` says that), a mean over shots by the
standard error of its shots in `shots.csv`. A column with no error bar (a median, a p99, a
maximum, a count) is compared exactly. `sim_wall_seconds_per_shot` is
the host's own time and is never compared. `results` then gives a
point both folders hold its paired line (below) unless every pair
failed in both runs or in neither. A section with nothing to list says
`the same`.

Two runs of one yaml on one commit give the same numbers when every
decoder is priced by a card. A decoder charged its measured wall clock
moves the tick columns from run to run, and `diff` says for each mean
whether the move is within its error bars.

## If the two runs decoded the same shots, compare them shot by shot

Two configurations that differ only in their decoder draw the same
shot at the same seed, so most of their failures are the same shots.
What tells them apart is the shots where one failed and the other did
not. Two separate intervals cannot see that, and they overlap long
after the shots show a difference: 25 shots that only the first run
failed against 5 that only the second failed is a clear difference,
while the two intervals still overlap.

So `diff` also compares every point both folders hold shot by shot,
and prints the result on a line after the point's results, whether or
not its rate differs: two runs can print the same rate and still
differ on the shots they share. A point whose pairs all failed alike
gets no line, so two runs of one yaml still read `the same`. It never assumes the shots are the
same:

- It takes the seeds both runs hold from seed 0 up to where the shorter
  run stopped (`prefix_shots` in `sweep.csv`).
- Every one of those seeds must have the same `sample_digest` in both
  `shots.csv` files. If any differs, the two runs did not decode the
  same shots, and `diff` says `not paired`, with how many seeds
  differ; the interval line above it is then the only comparison.
- A shot either run left unscored has no failure to compare, so it is
  left out of the pairs and counted.

The pairs are the shots both runs scored among those shared seeds,
and every number on the line is about them alone. They are not each
run's whole failure rate: `sweep.csv` also counts a run's shots past
the shorter run's stop and the shots the other run left unscored, so
two rates in `sweep.csv` can differ by more or less than the pairs do.

The line gives two numbers, and they answer different questions:

- **The mixture test** (Robbins' beta-binomial mixture, Howard et al.,
  arXiv 1810.08240, Proposition 7) asks whether the two runs differ at
  all. It reads only the shots where exactly one failed and says
  `differ` or `no difference shown`. It never says the two are the
  same: a small difference needs many shots to show.
- **The difference interval** (eq. 24 of the same paper) bounds how
  much the first run's failure rate on the pairs minus the second's is,
  with 95 percent confidence.

They can disagree. With 25 shots against 5 in a million, the mixture
test says `differ`, because 25 against 5 is unlikely from a fair coin,
while the interval runs from about -3 to +7 in 100,000 and so still
holds zero: the difference is 2 in 100,000, and the interval needs
more shots to rule zero out. Both are right: one says the runs differ,
the other that the difference is not yet measured well enough to put
a size on it.

Both hold at any stop: a point that ran to a failure target and a
point that ran a fixed number of shots are read the same way, and so
are pairs cut at the shorter run's stop.

Two limits, in plain words:

- The 95 percent guarantee is for one pair of points. Compare twenty
  points of a sweep and about one of them may say `differ` by chance.
- The mixture test assumes that, if the two decoders are equally good,
  on a paired shot where only one failed either one is equally likely
  to be the one that failed, whatever happened on earlier shots. A
  decoder that adapts as it runs, such as an escalation threshold that
  learns online, can break that, because its behaviour on a shot
  depends on the shots before it.
- That assumption is about the pairs as they were selected. A shot
  either run left unscored is not among them, so when leaving a shot
  unscored goes with how hard it was to decode, the pairs that remain
  can favour one decoder, and the test cannot see the shots it lost.

`--out <file>` writes the same comparison as a csv, one row per point
both folders hold, for you to plot or test further. A row starts as a
`sweep.csv` row does, `point_id` and then one column per swept path,
before the comparison:

| Column | What it is |
| --- | --- |
| `point_id` | the point's id in the first folder, which joins that folder's `sweep.csv` |
| one column per swept path | the value the first folder's point ran with |
| `second_point_id` | the point's id in the second folder |
| `shared_shot_count` | the seeds both runs hold, from 0 to the shorter run's stop |
| `digest_mismatch_count` | shared seeds whose `sample_digest` differs; above 0, nothing is paired and the columns below are empty |
| `unscored_shot_count` | shared seeds either run left unscored, left out of the pairs |
| `scored_pair_count` | the shared seeds both runs scored |
| `first_only_failure_count`, `second_only_failure_count` | pairs that only the first run failed, and only the second |
| `is_mixture_difference` | whether the mixture test says the two differ |
| `difference_low`, `difference_high` | the 95 percent interval on the first run's failure rate on the pairs minus the second's; empty with no pair |

## Read the folders from Python

`decsim.results` reads run folders into one table, a row per point with
a column per result, one per swept path, and one per setting
(`settings.` and the setting's dotted path). The figure is yours to draw, in whatever form the
question needs, and `decsim.plots` saves it:

```python
import matplotlib.pyplot as plt

import decsim.plots as plots
import decsim.results as results

folders = ["results/first", "results/second"]
rows = results.load(*folders)
distance = "qpu.distance"
figure, ax = plt.subplots()
for folder in folders:
    kept = [row for row in rows if row["run_dir"] == folder]
    kept.sort(key=lambda row: row[distance])
    distances = [row[distance] for row in kept]
    rates = [row["logical_error_rate_estimate"] for row in kept]
    lows = [row["logical_error_rate_low"] for row in kept]
    highs = [row["logical_error_rate_high"] for row in kept]
    below = [rate - low for rate, low in zip(rates, lows)]
    above = [high - rate for rate, high in zip(rates, highs)]
    ax.errorbar(distances, rates, yerr=[below, above], fmt="o-", label=folder)
ax.set_yscale("log")
ax.legend()
plots.save(figure, "ler.png")
```

`plots.save` writes `ler.png` and closes the figure.

## What to check before you believe a difference

- **The shots.** Two shots say almost nothing about a logical error
  rate. Compare `logical_error_rate_low` and `logical_error_rate_high`,
  not just `logical_error_rate_estimate`; if the two intervals overlap,
  the runs have not been shown to differ that way. When the two runs
  decoded the same shots, read the paired line too. A point with no interval, no
  scored shot or a cap with no failure, has no comparison to make, and
  `diff` says so.
- **The run record.** `run.json` in each folder carries the git
  commit, whether the checkout was dirty, the library versions and the
  command line. Two runs on different commits are two experiments.
- **The run file.** Each folder holds a verbatim copy of its run file,
  a yaml chain in `config/`. Diff those two folders before diffing the numbers.
- **Whether either run measured a wall clock.** `latency_samples.csv`
  has rows only for a decoder named by a table row. If one run has that
  file populated and the other does not, their tick columns are not
  comparable.

## Read next

- [The run folder](../reference/run_folder.md): every file and column.
- [Your first sweep](../tutorials/first_sweep.md): what the exact interval is.
- [How to run a timing study whose numbers do not depend on your computer](run_a_timing_only_study.md): making the ticks comparable.
