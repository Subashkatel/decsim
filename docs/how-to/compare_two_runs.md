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

## If the two runs decoded the same shots, compare them shot by shot

Two configurations that differ only in their decoder draw the same
shot at the same seed, so most of their failures are the same shots.
What tells them apart is the shots where one failed and the other did
not. Two separate intervals cannot see that, and they overlap long
after the shots show a difference: 25 shots that only the first run
failed against 5 that only the second failed is a clear difference,
while the two intervals still overlap.

Every `shots.csv` row holds its point's `point_id`, the shot's `seed`,
its `sample_digest`, `is_scored` and `logical_failure`, so the pairs are
a join on the seed:

- Take the seeds both runs hold from seed 0 up to where the shorter
  run stopped (`prefix_shots` in `sweep.csv`).
- Check that every one of those seeds has the same `sample_digest` in
  both files. If any differs, the two runs did not decode the same
  shots and there is nothing to pair.
- Leave out a shot either run left unscored: it has no failure to
  compare.
- Count the pairs only the first run failed and the pairs only the
  second failed. Those two counts are what a paired test reads.

## Read the folders from Python

Each folder's `sweep.csv` is one row per point, its `point_id`, one
column per swept path, its counts, and each estimate with its exact
limits; the settings a point ran with are its
`points/<name>/machine.json`. Read them with any csv and json reader
and draw the figure the question needs.

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
