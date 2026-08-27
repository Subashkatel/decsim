# Experiment notes

What we are running and the figures the runs feed. Survey grounding:
SWIPER 2412.05115, Skoric 2209.08552, Toshio 2510.25222.

## Currently running

Nothing. Two pilot runs are planned and awaiting sbatch (see below).
The next approved run gets listed here with its results stamp.

## The plots

1. LER vs p, one curve per d, weak and strong on the SAME d set
   {3, 5, 7, 9, 11}. Sweep p {0.5, 1, 2, 3, 5, 7, 10} x 1e-3, rounds 10d,
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

## The offline LER lane

LER never depends on timing, so plot 1 runs through
experiments/offline_run.py: the exact windowed decode of the
closed-loop runner (same window models, boundary XOR, owned-fault
ledger, per-seed sampling; equivalence pinned shot for shot in
tests/test_offline_run.py) with window models built once per point
instead of once per shot. 14-22x faster per shot; parallelism comes
from a slurm array, one shards.tsv line per task:

    python -m experiments.offline_run plan configs/weak_ler.yaml
    sbatch --array=1-<N> <run_dir>/slurm_offline.sh <run_dir>
    python -m experiments.offline_run merge <run_dir>

plan copies slurm_offline.sh into the run dir and you submit that
copy, so every run folder records exactly how it ran. Worst-case
shard memory measured 0.46 GB (strong d=9); the script books 8G.

## Next runs

- LER pilot (awaiting sbatch): weak_ler + strong_ler at 100 shots per
  point, d {3, 5, 7, 9, 11}, p {0.5, 1, 2, 3, 5, 7, 10} x 1e-3. Purpose:
  smoke the array pipeline and measure per-shot cost per point.
- LER production: same configs with shots raised per point, sized from
  the pilot's cost table toward >= 100 failures per point under an
  approved core-hour budget; cap-limited points plot as bounds.
- Latency vs d (closed loop): both tiers, d {3, 5, 7, 9, 11},
  p = 1e-3, rounds 10d, 100 shots per d; needs two small latency
  configs (weak on algorithm: pymatching).

Methodology guards for every run: slurm node, never the login node;
check the first window for a warm-up spike; seeds 0..shots-1 so reruns
reproduce; every run lands in its own timestamped results folder.
