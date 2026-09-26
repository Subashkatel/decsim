[decsim docs](../README.md) › [How-to guides](README.md)

# How to compare two runs

Two run folders can be compared three ways, and which one you want
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

For example, the reference config at distance 3, decoded by PyMatching
and charged its measured wall clock. Every column after `shots` is that
host's, so yours will differ:

```
distance,algorithm,shots,load,queue_wait_mean_us,algorithm_mean_us,buffer0_ready_to_frame_median_us
3,pymatching,2,7.308208333333333,7.149500000000001,12.753125,32.91
```

and the same point with the decoder priced at one microsecond by a
card, which is the same on every host:

```
distance,algorithm,shots,load,queue_wait_mean_us,algorithm_mean_us,buffer0_ready_to_frame_median_us
3,1.0,20,0.35585185185185186,0.0,1.0,1.076
```

Read across. The card's algorithm time is smaller, so the load falls
under 1. Under a load of 1 the decoder keeps up, the queue wait falls to
nothing, and the reaction time falls with it. The reaction time here is
the median from a window having its rounds to its correction reaching
the frame.

That comparison is between a software wall clock and a hardware card,
which is a comparison of two questions rather than of two machines. Say
so when you report it. [Time](../explanation/time.md) says why.

## If you want a picture, hand `plot` both folders

```bash
decsim plot results/<first> results/<second> \
  --figure ler_vs_d --probability 0.001
```

```
results/<first>/ler_vs_distance.png
```

`ler_vs_d`, `latency` and `data_movement` read every folder given;
`timeline` and `stage_breakdown` read only the first. The file is
written next to the first folder unless `--out` says otherwise.
`ler_vs_d` reads one physical error rate out of the sweep, so it asks
which one; the other four do not.

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
  kind: memory_patches
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
  and `_max` fall to zero, and `strong_queue_max`,
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
