# Experiment notes

What we are running and the figures the runs feed. Survey grounding:
SWIPER 2412.05115, Skoric 2209.08552, Toshio 2510.25222.

## Currently running

Nothing. The next approved run gets listed here with its results stamp.

## The plots

1. LER vs p, one curve per d, weak and strong on the SAME d set
   {3, 5, 7, 9}. Sweep p {0.5, 1, 2, 3, 5, 7, 10} x 1e-3, rounds 10d,
   paired seeds across tiers, >= 100 failures per plotted point.
   Status: machinery ready; shot plan not yet approved.
2. Decode latency vs d, weak and strong (SWIPER Fig 3 draws these as
   per-d distributions). Latency = the algorithm stage's measured wall
   clock the engine already charges for named algorithms; report median
   and p90 over windows. The weak tier runs algorithm: pymatching here,
   because the 0.028 LILLIPUT card is flat in d by construction; the
   card appears as a labeled reference line. Corrections are identical
   either way (the card wraps the same MWPM path), so weak LER stays
   consistent with plot 1. Label: software wall clock on this cluster,
   not hardware decoder latency; both tiers measured the same way.
   Status: measurement exists; needs a plotting pass and two small
   latency configs.
3. LER vs d at p = 1e-3, both tiers on one axes. Free from plot 1's
   data, no extra shots. With plot 2 this is the switching argument:
   strong accuracy at weak latency is the benefit to demonstrate.
4. Stage breakdown, syndrome generation -> commit, per d (the POINTS
   chain: transport, fetch, algorithm, release, output link, commit).
   Free from any run's shots.csv; plotting only.

Retired: latency vs p (no paper draws it).

## Next runs, awaiting go

- Latency vs d: both tiers, d {3, 5, 7, 9, 11}, p = 1e-3, rounds 10d,
  100 shots per d. Cheap: latency needs windows, not failures. One
  slurm job per tier; strong d = 11 last, its own job if slow.
- LER sweep: the two baseline configs, sharded one slurm job per
  (d, p) point. Shot counts sized for >= 100 failures at the small-p
  points before launch.

Methodology guards for every run: slurm node, never the login node;
check the first window for a warm-up spike; seeds 0..shots-1 so reruns
reproduce; every run lands in its own timestamped results folder.
