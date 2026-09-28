[decsim docs](../README.md) › [How-to guides](README.md)

# How to compare two runs

Two run folders can be compared several ways, and which one you want
depends on what the two runs are.

## If they are pieces of one sweep, they are already added

`decsim collect` folds every piece of a configuration into its run
folder, `combined/<name>-<id8>/` under the experiment folder: it reads
the pieces' additive files, adds them, and recomputes `sweep.csv` and
`links.csv` from the sum. Two runs of different configurations are
never added: the result would be a single row that is neither.

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
decsim diff results/<first>/combined/<name>-<id8> results/<second>/combined/<name>-<id8>
```

`diff` matches the two folders' points by their metadata, the values
their sweep set, and prints
three sections. `settings` lists every value that differs at a point,
the ones the build derived included (`resolved/`). `inputs` lists each
workload file whose sha256 differs (`inputs/<id>/hashes.json`).
`results` lists each `sweep.csv` column that differs, and says whether
the two values agree within their error bars: a logical error rate by
its exact interval (with no interval on either side, no statistical
comparison is possible, and `diff` says that), a mean over shots by the
standard error of its shots in `shots.csv`. A column with no error bar (a median, a p99, a
maximum, a count) is compared exactly. `sim_wall_seconds_per_shot` is
the host's own time and is never compared. A section with nothing to
list says `the same`.

Two runs of one yaml on one commit give the same numbers when every
decoder is priced by a card. A decoder charged its measured wall clock
moves the tick columns from run to run, and `diff` says for each mean
whether the move is within its error bars.

## Read the folders from Python

`decsim.results` reads run folders into one table, a row per point with
a column per result, one per swept path, and one per setting
(`settings.` and the setting's dotted path). The figure is yours to draw, in whatever form the
question needs, and `decsim.plots` saves it:

```python
import glob

import matplotlib.pyplot as plt

import decsim.plots as plots
import decsim.results as results

first_folders = glob.glob("results/first/combined/*")
second_folders = glob.glob("results/second/combined/*")
folders = first_folders + second_folders
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

## If a burst made things worse: harder windows or an overloaded strong side

A burst does two things at once. It makes windows harder: more
detection events, a longer weak decode, more escalations. And when it
covers several patches, their escalations reach the strong decoder
together and wait for a unit. A queue splits every response into a
wait and a service; the paired run below reads the two apart.

Run the burst twice with the same seeds. The first run has the real
strong unit count. The second has enough strong units that no strong
decode waits: at least the patches the burst hits, times the strong
decodes one patch keeps in flight, and raised until
`strong_wait_max_us` reads 0. Both keep the switching records and the
backlog sampler. Four distance-5 patches on the two-tier machine, a
burst between patches 0 and 1 (each patch is 2d + 2 = 12 units wide on
the shared plane, so their centres are at x = 5 and 17):

```yaml
extends: examples/two_tiers.yaml
workload:
  kind: producer
  function: decsim.producers:memory_patches
  arguments:
    code_task: surface_code:rotated_memory_z
    rounds_per_shot: 40
    patch_count: 4
    distance: ${qpu.distance}
qpu:
  kind: burst_stim
  burst_onset_round: 15
  burst_decay_rounds: 8.0
  burst_radius: 7.0
  burst_center: [11.0, 5.0]
  burst_error_probability: 0.05
observation:
  record_switching_windows: true
  backlog_trace: true
sweep:
  - axes:
      workload.arguments.physical_error_probability: [0.001]
      qpu.distance: [5]
      qpu.round_period_microseconds: [1.0]
    collection: {max_shots: 100}
```

and the same file with `strong_decoder`'s `units` raised, in a copy of
two_tiers.yaml's `strong_decoder` section.

The syndromes and the escalation decisions are the same in both runs,
so check first that `escalated_windows` is equal in the two rows. Then:

- What stays equal is the difficulty: `weak_syndrome_weight_mean` and
  `_max`, `weak_service_mean_us` and `escalated_fraction`, each read
  against the same run with `burst_error_probability: 0`.
- What the extra units take away is the overload: `strong_wait_mean_us`
  and `_max` and `strong_held_in_units_max` fall to zero, and
  `strong_queue_max`,
  `backlog_peak_rounds` and the reaction-time points fall with them.
  The difference between the two rows is the overload alone.

A burst that leaves the weak decoder confident and wrong shows in
neither: it raises `logical_error_rate_estimate` with no rise in
`escalated_fraction`. `observation.check_windows_with: tesseract`
counts those windows in `referee_window_disagreements`.

## What to check before you believe a difference

- **The shots.** Two shots say almost nothing about a logical error
  rate. Compare `logical_error_rate_low` and `logical_error_rate_high`,
  not just `logical_error_rate_estimate`; if the two intervals overlap,
  the runs have not been shown to differ. A point with no interval, no
  scored shot or a cap with no failure, has no comparison to make, and
  `diff` says so.
- **The manifest.** `manifest.json` in each folder carries the git
  commit, whether the checkout was dirty, the library versions and the
  command line. Two runs on different commits are two experiments.
- **The config.** `config/` in each folder is a verbatim copy of the
  yaml chain. Diff those two folders before diffing the numbers.
- **Whether either run measured a wall clock.** `latency_samples.csv`
  has rows only for a decoder named by a table row. If one run has that
  file populated and the other does not, their tick columns are not
  comparable.

## Read next

- [The run folder](../reference/run_folder.md): every file and column.
- [Your first sweep](../tutorials/first_sweep.md): what the exact interval is.
- [How to run a timing study whose numbers do not depend on your computer](run_a_timing_only_study.md): making the ticks comparable.
