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
and charged its measured wall clock:

```
distance,algorithm,shots,load,queue_wait_mean_us,algorithm_mean_us,buffer0_ready_to_frame_median_us
3,pymatching,2,7.308208333333333,7.149500000000001,12.753125,32.91
```

and the same point with the decoder priced at one microsecond by a card:

```
distance,algorithm,shots,load,queue_wait_mean_us,algorithm_mean_us,buffer0_ready_to_frame_median_us
3,1.0,20,0.3571851851851852,0.0,1.0,1.076
```

Read across. The algorithm went from a measured 12.75 microseconds to a
stated 1.0. The load went from 7.3, where the decoder cannot keep up and
work queues, to 0.36, where it can. The queue wait went from 7.1
microseconds to nothing at all, which is the load crossing 1 showing up
as a delay. And the reaction time, the median from a window having its
rounds to its correction reaching the frame, went from 32.9 microseconds
to 1.08.

That comparison is between a software wall clock and a hardware card,
which is a comparison of two questions rather than of two machines. Say
so when you report it. [`docs/explanation/time.md`](../explanation/time.md) says why.

## If you want a picture, hand `plot` both folders

```bash
decsim plot results/<first> results/<second> \
  --figure ler_vs_d --probability 0.001
```

```
results/<first>/ler_vs_distance.png
```

Every figure takes several run folders, and the file is written next to
the first one unless `--out` says otherwise. `ler_vs_d` reads one
physical error rate out of the sweep, so it asks which one; `timeline`,
`stage_breakdown` and `latency` do not.

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

- [`docs/reference/run_folder.md`](../reference/run_folder.md): every file and column.
- [`docs/tutorials/first_sweep.md`](../tutorials/first_sweep.md): what a Wilson interval is.
- [`docs/how-to/run_a_timing_only_study.md`](run_a_timing_only_study.md): making the ticks comparable.
