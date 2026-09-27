[decsim docs](../README.md) › [How-to guides](README.md)

# How to compare two runs

Two run folders can be compared several ways, and which one you want
depends on what the two runs are.

## If they are shards of one sweep, add them

```bash
decsim combine results/weak_ler/*
```

`combine` reads every folder's additive files, adds them, and recomputes
`sweep.csv` and `links.csv` from the sum. Use this only when the runs
are the same sweep cut into pieces. Two runs of different configurations
must not be added: the result would be a single row that is neither.

## If they are different configurations, read the rows

Every folder's `sweep.csv` has one row per sweep point. Pick the point
the two runs share and the columns the question is about.

For example, the reference config's point at distance 3, decoded by
PyMatching and charged its measured wall clock. A row names its point
in its last two columns, `point_id` and `metadata`, left out here. Every column
after `shots` is that host's, so yours will differ:

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
the ones the build derived included (`resolved/`). `inputs` lists each
workload file whose sha256 differs (`inputs/<id>/hashes.json`).
`results` lists each `sweep.csv` column that differs, and says whether
the two values agree within their error bars: a logical error rate by
its Wilson interval, a mean over shots by the standard error of its
shots in `shots.csv`. A column with no error bar (a median, a p99, a
maximum, a count) is compared exactly. `sim_wall_seconds_per_shot` is
the host's own time and is never compared. A section with nothing to
list says `the same`.

Two runs of one yaml on one commit give the same numbers when every
decoder is priced by a card. A decoder charged its measured wall clock
moves the tick columns from run to run, and `diff` says for each mean
whether the move is within its error bars.

## Read the folders from Python

`decsim.results` reads run folders into one table, a row per point with
a column per result and one per setting (`settings.` and the setting's
dotted path), and draws the logical error rate the way sinter's
`plot_error_rate` does:

```python
from matplotlib.figure import Figure

import decsim.results as results

folders = ["results/first", "results/second"]
rows = results.load(*folders)
figure = Figure()
ax = figure.subplots()
results.plot_error_rate(
    ax=ax,
    rows=rows,
    x="settings.workload.physical_error_probability",
    group="settings.qpu.distance",
    where={"settings.qpu.round_period_microseconds": 1.0},
)
results.save_figure(figure, "ler.png", rows, folders)
```

`save_figure` writes `ler.png` and, beside it, `ler.py` (a copy of the
script that drew it), `ler.csv` (the rows drawn) and `ler.json` (the
folders they came from).

## If you want a picture, hand `plot` both folders

```bash
decsim plot results/<first> results/<second> \
  --figure ler --x distance --where physical_error_probability=0.001
```

```
results/<first>/ler.png
```

`ler`, `latency` and `data_movement` read every folder given;
`timeline` and `stage_breakdown` read only the first. The file is
written next to the first folder unless `--out` says otherwise. Every
figure but the timeline is drawn against the swept setting `--x` names,
by the name the points' `metadata` gives it; `--group` draws a curve
per value of another, and `--where PATH=VALUE`, given as often as
needed, keeps the points whose sweep set that value, as sinter's plot
reads a point's metadata (`--x_func`, `--group_func`, `--filter_func`).

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
extends: two_tiers.yaml
workload:
  kind: producer
  function: decsim.producers:memory_patches
  arguments:
    code_task: surface_code:rotated_memory_z
    rounds_per_shot: 40
    patch_count: 4
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
  - physical_error_probability: [0.001]
    distance: [5]
    round_period_microseconds: [1.0]
    shots: 100
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
neither: it raises `logical_error_rate` with no rise in
`escalated_fraction`. `observation.check_windows_with: tesseract`
counts those windows in `tesseract_window_disagreements`.

## What to check before you believe a difference

- **The shots.** Two shots say almost nothing about a logical error
  rate. Compare `ler_wilson_low` and `ler_wilson_high`, not just
  `logical_error_rate`; if the two intervals overlap, the runs have not
  been shown to differ.
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
- [Your first sweep](../tutorials/first_sweep.md): what a Wilson interval is.
- [How to run a timing study whose numbers do not depend on your computer](run_a_timing_only_study.md): making the ticks comparable.
