# The 2026-09 experiments: the plan

Sixteen experiments on one shot table, every run differing from every
other in its decoders and nothing else: weak alone, strong alone, and
switching, for pymatching, union find, Relay-BP and BP-OSD, distance 3
to 15. The families:

- weak tier alone: pymatching_weak, union_find_weak, relay_bp_weak,
  bposd_weak
- strong tier alone: pymatching_strong, union_find_strong,
  belief_matching_strong, relay_bp_strong, bposd_strong
- switching, weak pymatching reporting the complementary gap:
  pymatching_belief_matching_switching, pymatching_relay_bp_switching,
  pymatching_bposd_switching
- switching, weak union find reporting the cluster gap:
  union_find_pymatching_switching, union_find_belief_matching_switching,
  union_find_relay_bp_switching, union_find_bposd_switching

A weak tier must report a confidence, which is why Relay-BP and BP-OSD
appear as strong tiers and alone but never as the weak side of a
switching pair (reference.yaml, escalation.confidence).

Shot table: base.yaml's sweep, one table for all sixteen, each point
about 100 failures from the August strong-tier rates, floor 400, cap
200,000; four points skipped: d 11 to 15 at p 0.001 and d 15 at p
0.002. 1,792,115 shots an experiment.

Costs: seconds a shot at p 0.003 on the compiled union find (D17) and
the compiled cluster gap walk (D18), the model build included. A
switching shot costs many times its two tiers alone because every
escalated window builds a strong window model with its neighbour's
committed faults excluded (window_planner.py
strong_model_for_operation).

Arrays: one sbatch line per experiment, distance and slice of at most
150 tasks; one work unit per task, sized to about two and a half hours
so a task fits its 04:00:00 limit with room for a slow shot and can
backfill, and never more than 8,000 shots, since a run keeps every
shot's samples until its csv is written; OFFSET carries the shard index
from one line to the next and SHARDS is the distance's whole unit
count, and every distance has its own run folder because its shard
indices start at zero, so a task's folder is
results/experiments_2026_09/<experiment>/d<d>/<shard>.

Memory per task: a base by distance (d 3 to 7 4G, d 9 6G, d 11 8G,
d 13 12G, d 15 16G) plus what the unit's shots accumulate, 0.012 MB a
shot a unit of distance (0.06 MB a shot at d 5 for pymatching, union
find and BP-OSD, twice that for Relay-BP), with a margin of one half.

Lines go in cheapest experiment distance first, a few hundred tasks at
a time, so every single-tier family completes before the large switching
distances start.

| experiment | d | shots | s per shot | core-hours | shots per unit | tasks | sbatch lines | memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| union_find_weak | 3 | 25,159 | 0.06 | 0 | 8,000 | 8 | 1 | 5G |
| union_find_strong | 3 | 25,159 | 0.07 | 0 | 8,000 | 8 | 1 | 5G |
| bposd_strong | 3 | 25,159 | 0.07 | 1 | 8,000 | 8 | 1 | 5G |
| bposd_weak | 3 | 25,159 | 0.07 | 1 | 8,000 | 8 | 1 | 5G |
| pymatching_weak | 3 | 25,159 | 0.08 | 1 | 8,000 | 8 | 1 | 5G |
| pymatching_strong | 3 | 25,159 | 0.08 | 1 | 8,000 | 8 | 1 | 5G |
| relay_bp_weak | 3 | 25,159 | 0.10 | 1 | 8,000 | 8 | 1 | 5G |
| relay_bp_strong | 3 | 25,159 | 0.10 | 1 | 8,000 | 8 | 1 | 5G |
| belief_matching_strong | 3 | 25,159 | 0.15 | 1 | 8,000 | 8 | 1 | 5G |
| union_find_pymatching_switching | 3 | 25,159 | 0.30 | 2 | 8,000 | 8 | 1 | 5G |
| pymatching_relay_bp_switching | 3 | 25,159 | 0.43 | 3 | 8,000 | 8 | 1 | 5G |
| union_find_relay_bp_switching | 3 | 25,159 | 0.44 | 3 | 8,000 | 8 | 1 | 5G |
| union_find_bposd_switching | 3 | 25,159 | 0.45 | 3 | 8,000 | 8 | 1 | 5G |
| pymatching_bposd_switching | 3 | 25,159 | 0.45 | 3 | 8,000 | 8 | 1 | 5G |
| union_find_belief_matching_switching | 3 | 25,159 | 0.71 | 5 | 8,000 | 8 | 1 | 5G |
| pymatching_belief_matching_switching | 3 | 25,159 | 0.72 | 5 | 8,000 | 8 | 1 | 5G |
| union_find_strong | 5 | 117,672 | 0.26 | 8 | 8,000 | 19 | 1 | 5G |
| union_find_weak | 5 | 117,672 | 0.27 | 9 | 8,000 | 19 | 1 | 5G |
| pymatching_strong | 5 | 117,672 | 0.27 | 9 | 8,000 | 19 | 1 | 5G |
| pymatching_weak | 5 | 117,672 | 0.27 | 9 | 8,000 | 19 | 1 | 5G |
| relay_bp_strong | 5 | 117,672 | 0.42 | 14 | 8,000 | 19 | 1 | 6G |
| relay_bp_weak | 5 | 117,672 | 0.43 | 14 | 8,000 | 19 | 1 | 6G |
| bposd_strong | 5 | 117,672 | 0.49 | 16 | 8,000 | 19 | 1 | 5G |
| bposd_weak | 5 | 117,672 | 0.49 | 16 | 8,000 | 19 | 1 | 5G |
| belief_matching_strong | 5 | 117,672 | 0.81 | 26 | 8,000 | 19 | 1 | 5G |
| union_find_pymatching_switching | 5 | 117,672 | 1.57 | 51 | 5,732 | 24 | 1 | 5G |
| pymatching_strong | 7 | 289,389 | 0.73 | 59 | 8,000 | 40 | 1 | 6G |
| pymatching_weak | 7 | 289,389 | 0.77 | 62 | 8,000 | 40 | 1 | 6G |
| union_find_strong | 7 | 289,389 | 0.79 | 64 | 8,000 | 40 | 1 | 6G |
| union_find_weak | 7 | 289,389 | 0.84 | 68 | 8,000 | 40 | 1 | 6G |
| pymatching_relay_bp_switching | 5 | 117,672 | 2.40 | 78 | 3,756 | 35 | 1 | 5G |
| pymatching_bposd_switching | 5 | 117,672 | 2.47 | 81 | 3,637 | 36 | 1 | 5G |
| union_find_bposd_switching | 5 | 117,672 | 2.57 | 84 | 3,501 | 38 | 1 | 5G |
| union_find_relay_bp_switching | 5 | 117,672 | 2.63 | 86 | 3,422 | 39 | 1 | 5G |
| relay_bp_strong | 7 | 289,389 | 1.30 | 105 | 6,907 | 45 | 1 | 6G |
| relay_bp_weak | 7 | 289,389 | 1.50 | 120 | 6,012 | 53 | 1 | 6G |
| pymatching_belief_matching_switching | 5 | 117,672 | 3.96 | 129 | 2,273 | 56 | 1 | 5G |
| union_find_belief_matching_switching | 5 | 117,672 | 4.11 | 134 | 2,189 | 57 | 1 | 5G |
| bposd_strong | 7 | 289,389 | 1.68 | 135 | 5,363 | 58 | 1 | 5G |
| bposd_weak | 7 | 289,389 | 1.68 | 135 | 5,350 | 58 | 1 | 5G |
| belief_matching_strong | 7 | 289,389 | 2.36 | 190 | 3,811 | 80 | 1 | 5G |
| pymatching_strong | 11 | 300,603 | 2.47 | 206 | 3,643 | 86 | 1 | 9G |
| union_find_strong | 9 | 439,402 | 1.71 | 209 | 5,263 | 88 | 1 | 7G |
| union_find_weak | 9 | 439,402 | 1.71 | 209 | 5,263 | 88 | 1 | 7G |
| pymatching_strong | 9 | 439,402 | 1.74 | 212 | 5,172 | 88 | 1 | 7G |
| pymatching_weak | 9 | 439,402 | 1.86 | 227 | 4,838 | 95 | 1 | 7G |
| union_find_weak | 11 | 300,603 | 3.34 | 279 | 2,694 | 115 | 1 | 9G |
| union_find_strong | 11 | 300,603 | 3.52 | 294 | 2,556 | 121 | 1 | 9G |
| pymatching_weak | 11 | 300,603 | 3.67 | 306 | 2,452 | 125 | 1 | 9G |
| relay_bp_strong | 9 | 439,402 | 2.61 | 319 | 3,448 | 132 | 1 | 8G |
| bposd_strong | 9 | 439,402 | 2.88 | 352 | 3,125 | 143 | 1 | 7G |
| bposd_weak | 9 | 439,402 | 3.25 | 397 | 2,769 | 162 | 2 | 7G |
| bposd_weak | 11 | 300,603 | 5.34 | 446 | 1,685 | 181 | 2 | 9G |
| bposd_strong | 11 | 300,603 | 5.55 | 463 | 1,621 | 188 | 2 | 9G |
| relay_bp_weak | 9 | 439,402 | 4.69 | 572 | 1,918 | 233 | 2 | 7G |
| belief_matching_strong | 9 | 439,402 | 4.79 | 585 | 1,878 | 238 | 2 | 7G |
| relay_bp_strong | 11 | 300,603 | 7.27 | 607 | 1,237 | 245 | 2 | 9G |
| union_find_pymatching_switching | 7 | 289,389 | 7.70 | 619 | 1,168 | 251 | 2 | 5G |
| pymatching_strong | 15 | 211,941 | 11.67 | 687 | 771 | 277 | 2 | 17G |
| belief_matching_strong | 11 | 300,603 | 8.82 | 736 | 1,020 | 298 | 2 | 9G |
| union_find_strong | 15 | 211,941 | 13.12 | 772 | 685 | 311 | 3 | 17G |
| pymatching_bposd_switching | 7 | 289,389 | 9.83 | 790 | 915 | 320 | 3 | 5G |
| pymatching_relay_bp_switching | 7 | 289,389 | 9.94 | 799 | 905 | 323 | 3 | 5G |
| union_find_weak | 15 | 211,941 | 13.65 | 804 | 659 | 323 | 3 | 17G |
| pymatching_strong | 13 | 407,949 | 7.26 | 823 | 1,239 | 332 | 3 | 13G |
| union_find_weak | 13 | 407,949 | 8.34 | 945 | 1,079 | 381 | 3 | 13G |
| union_find_strong | 13 | 407,949 | 8.47 | 960 | 1,062 | 387 | 3 | 13G |
| pymatching_weak | 15 | 211,941 | 16.58 | 976 | 542 | 394 | 3 | 17G |
| union_find_relay_bp_switching | 7 | 289,389 | 13.02 | 1,047 | 691 | 421 | 3 | 5G |
| union_find_bposd_switching | 7 | 289,389 | 13.14 | 1,056 | 684 | 425 | 3 | 5G |
| relay_bp_weak | 11 | 300,603 | 13.03 | 1,088 | 690 | 439 | 3 | 9G |
| union_find_pymatching_switching | 9 | 439,402 | 9.01 | 1,100 | 998 | 443 | 3 | 7G |
| bposd_weak | 15 | 211,941 | 18.87 | 1,111 | 476 | 447 | 3 | 17G |
| pymatching_weak | 13 | 407,949 | 10.10 | 1,145 | 891 | 461 | 4 | 13G |
| bposd_strong | 15 | 211,941 | 20.51 | 1,207 | 438 | 485 | 4 | 17G |
| bposd_weak | 13 | 407,949 | 11.83 | 1,341 | 760 | 540 | 4 | 13G |
| bposd_strong | 13 | 407,949 | 12.25 | 1,388 | 734 | 559 | 4 | 13G |
| pymatching_bposd_switching | 9 | 439,402 | 11.42 | 1,394 | 788 | 560 | 4 | 7G |
| pymatching_relay_bp_switching | 9 | 439,402 | 11.48 | 1,401 | 783 | 564 | 4 | 7G |
| union_find_belief_matching_switching | 7 | 289,389 | 21.23 | 1,707 | 423 | 686 | 5 | 5G |
| union_find_relay_bp_switching | 9 | 439,402 | 15.02 | 1,833 | 599 | 735 | 5 | 7G |
| union_find_bposd_switching | 9 | 439,402 | 15.33 | 1,871 | 587 | 750 | 5 | 7G |
| relay_bp_strong | 13 | 407,949 | 19.24 | 2,180 | 467 | 876 | 6 | 13G |
| belief_matching_strong | 15 | 211,941 | 38.02 | 2,238 | 236 | 900 | 6 | 17G |
| relay_bp_strong | 15 | 211,941 | 39.71 | 2,338 | 226 | 939 | 7 | 17G |
| pymatching_belief_matching_switching | 9 | 439,402 | 20.46 | 2,497 | 439 | 1,004 | 7 | 7G |
| union_find_belief_matching_switching | 9 | 439,402 | 23.27 | 2,840 | 386 | 1,142 | 8 | 7G |
| pymatching_belief_matching_switching | 7 | 289,389 | 40.90 | 3,288 | 220 | 1,319 | 9 | 5G |
| union_find_pymatching_switching | 11 | 300,603 | 41.61 | 3,474 | 216 | 1,393 | 10 | 9G |
| relay_bp_weak | 13 | 407,949 | 30.86 | 3,497 | 291 | 1,405 | 10 | 13G |
| belief_matching_strong | 13 | 407,949 | 32.41 | 3,673 | 277 | 1,476 | 10 | 13G |
| relay_bp_weak | 15 | 211,941 | 66.46 | 3,913 | 135 | 1,572 | 11 | 17G |
| pymatching_bposd_switching | 11 | 300,603 | 46.93 | 3,919 | 191 | 1,577 | 11 | 9G |
| pymatching_relay_bp_switching | 11 | 300,603 | 50.38 | 4,207 | 178 | 1,692 | 12 | 9G |
| union_find_relay_bp_switching | 11 | 300,603 | 73.55 | 6,141 | 122 | 2,466 | 17 | 9G |
| union_find_bposd_switching | 11 | 300,603 | 74.33 | 6,207 | 121 | 2,488 | 17 | 9G |
| pymatching_bposd_switching | 15 | 211,941 | 107.39 | 6,322 | 83 | 2,555 | 18 | 17G |
| union_find_pymatching_switching | 15 | 211,941 | 107.65 | 6,338 | 83 | 2,555 | 18 | 17G |
| pymatching_relay_bp_switching | 15 | 211,941 | 107.87 | 6,351 | 83 | 2,555 | 18 | 17G |
| pymatching_belief_matching_switching | 11 | 300,603 | 83.27 | 6,953 | 108 | 2,785 | 19 | 9G |
| union_find_relay_bp_switching | 15 | 211,941 | 139.73 | 8,226 | 64 | 3,313 | 23 | 17G |
| union_find_pymatching_switching | 13 | 407,949 | 72.60 | 8,227 | 123 | 3,321 | 23 | 13G |
| union_find_bposd_switching | 15 | 211,941 | 157.87 | 9,294 | 57 | 3,720 | 25 | 17G |
| union_find_belief_matching_switching | 11 | 300,603 | 129.39 | 10,804 | 69 | 4,359 | 30 | 9G |
| pymatching_bposd_switching | 13 | 407,949 | 103.04 | 11,676 | 87 | 4,691 | 32 | 13G |
| pymatching_belief_matching_switching | 15 | 211,941 | 201.03 | 11,835 | 44 | 4,819 | 33 | 17G |
| pymatching_relay_bp_switching | 13 | 407,949 | 110.40 | 12,510 | 81 | 5,040 | 34 | 13G |
| union_find_bposd_switching | 13 | 407,949 | 117.51 | 13,316 | 76 | 5,370 | 36 | 13G |
| union_find_belief_matching_switching | 15 | 211,941 | 238.51 | 14,042 | 37 | 5,730 | 39 | 17G |
| union_find_relay_bp_switching | 13 | 407,949 | 130.25 | 14,760 | 69 | 5,915 | 40 | 13G |
| pymatching_belief_matching_switching | 13 | 407,949 | 194.34 | 22,022 | 46 | 8,870 | 60 | 13G |
| union_find_belief_matching_switching | 13 | 407,949 | 211.72 | 23,992 | 42 | 9,715 | 65 | 13G |
All sixteen: 272,096 core-hours, 110,040 tasks, 801 sbatch lines.

## The submit lines

Every line has this shape, from the decsim checkout, with SHARDS the
distance's unit count from the table, OFFSET stepping by 150 and the
array running to the smaller of 149 and the remaining units:

```bash
DECSIM_PYTHON=<interpreter> RUN=results/experiments_2026_09/<experiment>/d<d> SHARDS=<tasks> OFFSET=<o> sbatch -J <experiment>_d<d> -a 0-<n> --time=04:00:00 --mem=<memory from the table> -o results/experiments_2026_09/<experiment>/logs/%x_%A_%a.out slurm/experiment_run.sh configs/experiments_2026_09/<experiment>_d<d>.yaml --shots-per-unit <shots per unit>
```

DECSIM_PYTHON names the interpreter when the checkout has no venv of
its own (slurm/experiment_run.sh). The library the union find rows load
is built once on the host with tools/build_union_find.sh before the
first line is submitted.

## After an array finishes

```bash
decsim combine results/experiments_2026_09/<experiment>/*/*        # all distances of one experiment into one report
decsim plot results/experiments_2026_09/<experiment>/combined --figure ler_vs_d --probability 0.003
```

A task that died is rerun by resubmitting its shard index alone (`-a <i>` with the same SHARDS and OFFSET); its folder is rewritten and combine reads whatever folders it is handed.
