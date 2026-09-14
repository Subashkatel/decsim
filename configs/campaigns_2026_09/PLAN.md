# The 2026-09 campaigns: the plan

Sixteen campaigns on one shot table, every run differing from every
other in its decoders and nothing else (owner 2026-09-14: weak alone,
strong alone, and switching, for pymatching, union find, Relay-BP and
BP-OSD, distance 3 to 15). The families:

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
0.002. 1,792,115 shots a campaign.

Costs: seconds a shot measured 2026-09-14 at p 0.003 on the compiled
union find (D17) and the compiled cluster gap walk (D18), four shots a
point at d 3 to 9 and two at d 11 to 15, the model build included;
the union find pairs were timed on the tree at 70526eca, the rest on
81c69496's parent tree, which is the same code for them. A switching
shot costs many times its two tiers alone because every escalated
window builds a strong window model with its neighbour's committed
faults excluded (window_planner.py strong_model_for_operation).

Arrays: one sbatch line per campaign, distance and slice of at most 150
tasks; one work unit per task, sized to about two and a half hours so a
task fits its 04:00:00 limit with room for a slow shot and can
backfill; OFFSET carries the shard index from one line to the next and
SHARDS is the distance's whole unit count, so a task's folder is
results/campaigns_2026_09/<campaign>/<shard>.

Memory per task by distance: d 3 to 7 4G, d 9 6G, d 11 8G, d 13 12G,
d 15 16G. The pilots' largest resident set at d 15 was 7.6 GB
(relay_bp_strong); the union find pairs stayed under 2.3 GB.

Pacing (owner 2026-09-14: do not oversaturate the cluster): the lines
are fed to sbatch by a loop that submits the next line only while fewer
than 200 of our tasks are queued or running, so at most 350 are in the
queue at once against the QOS's 400 running and 1,000 submitted per
user; the cpu partition holds about 15,500 cores. Lines are ordered
cheapest campaign distance first, so every single-tier family completes
before the large switching distances start, and a line still in the
queue file can be removed before it is submitted.

The generator scripts and the queue live in the research sandbox,
tmp/architecture-study/campaign (make_campaign_configs.py,
make_submit_lines.py with costs.csv, feeder.sh); the queue as submitted
is copied to docs/rewrite/design_audit/campaign_2026_09 there.

| campaign | d | shots | s per shot | core-hours | shots per unit | tasks | sbatch lines |
| --- | --- | --- | --- | --- | --- | --- | --- |
| union_find_weak | 3 | 25,159 | 0.06 | 0 | 17,064 | 6 | 1 |
| union_find_strong | 3 | 25,159 | 0.07 | 0 | 17,064 | 6 | 1 |
| bposd_strong | 3 | 25,159 | 0.07 | 1 | 17,064 | 6 | 1 |
| bposd_weak | 3 | 25,159 | 0.07 | 1 | 17,064 | 6 | 1 |
| pymatching_weak | 3 | 25,159 | 0.08 | 1 | 17,064 | 6 | 1 |
| pymatching_strong | 3 | 25,159 | 0.08 | 1 | 17,064 | 6 | 1 |
| relay_bp_weak | 3 | 25,159 | 0.10 | 1 | 17,064 | 6 | 1 |
| relay_bp_strong | 3 | 25,159 | 0.10 | 1 | 17,064 | 6 | 1 |
| belief_matching_strong | 3 | 25,159 | 0.15 | 1 | 17,064 | 6 | 1 |
| union_find_pymatching_switching | 3 | 25,159 | 0.30 | 2 | 17,064 | 6 | 1 |
| pymatching_relay_bp_switching | 3 | 25,159 | 0.43 | 3 | 17,064 | 6 | 1 |
| union_find_relay_bp_switching | 3 | 25,159 | 0.44 | 3 | 17,064 | 6 | 1 |
| union_find_bposd_switching | 3 | 25,159 | 0.45 | 3 | 17,064 | 6 | 1 |
| pymatching_bposd_switching | 3 | 25,159 | 0.45 | 3 | 17,064 | 6 | 1 |
| union_find_belief_matching_switching | 3 | 25,159 | 0.71 | 5 | 12,676 | 7 | 1 |
| pymatching_belief_matching_switching | 3 | 25,159 | 0.72 | 5 | 12,569 | 7 | 1 |
| union_find_strong | 5 | 117,672 | 0.26 | 8 | 34,615 | 8 | 1 |
| union_find_weak | 5 | 117,672 | 0.27 | 9 | 33,333 | 8 | 1 |
| pymatching_strong | 5 | 117,672 | 0.27 | 9 | 33,210 | 8 | 1 |
| pymatching_weak | 5 | 117,672 | 0.27 | 9 | 32,967 | 8 | 1 |
| relay_bp_strong | 5 | 117,672 | 0.42 | 14 | 21,327 | 10 | 1 |
| relay_bp_weak | 5 | 117,672 | 0.43 | 14 | 20,979 | 10 | 1 |
| bposd_strong | 5 | 117,672 | 0.49 | 16 | 18,218 | 11 | 1 |
| bposd_weak | 5 | 117,672 | 0.49 | 16 | 18,181 | 11 | 1 |
| belief_matching_strong | 5 | 117,672 | 0.81 | 26 | 11,138 | 15 | 1 |
| union_find_pymatching_switching | 5 | 117,672 | 1.57 | 51 | 5,732 | 24 | 1 |
| pymatching_strong | 7 | 289,389 | 0.73 | 59 | 12,345 | 28 | 1 |
| pymatching_weak | 7 | 289,389 | 0.77 | 62 | 11,718 | 30 | 1 |
| union_find_strong | 7 | 289,389 | 0.79 | 64 | 11,392 | 30 | 1 |
| union_find_weak | 7 | 289,389 | 0.84 | 68 | 10,714 | 31 | 1 |
| pymatching_relay_bp_switching | 5 | 117,672 | 2.40 | 78 | 3,756 | 35 | 1 |
| pymatching_bposd_switching | 5 | 117,672 | 2.47 | 81 | 3,637 | 36 | 1 |
| union_find_bposd_switching | 5 | 117,672 | 2.57 | 84 | 3,501 | 38 | 1 |
| union_find_relay_bp_switching | 5 | 117,672 | 2.63 | 86 | 3,422 | 39 | 1 |
| relay_bp_strong | 7 | 289,389 | 1.30 | 105 | 6,907 | 45 | 1 |
| relay_bp_weak | 7 | 289,389 | 1.50 | 120 | 6,012 | 53 | 1 |
| pymatching_belief_matching_switching | 5 | 117,672 | 3.96 | 129 | 2,273 | 56 | 1 |
| union_find_belief_matching_switching | 5 | 117,672 | 4.11 | 134 | 2,189 | 57 | 1 |
| bposd_strong | 7 | 289,389 | 1.68 | 135 | 5,363 | 58 | 1 |
| bposd_weak | 7 | 289,389 | 1.68 | 135 | 5,350 | 58 | 1 |
| belief_matching_strong | 7 | 289,389 | 2.36 | 190 | 3,811 | 80 | 1 |
| pymatching_strong | 11 | 300,603 | 2.47 | 206 | 3,643 | 86 | 1 |
| union_find_strong | 9 | 439,402 | 1.71 | 209 | 5,263 | 88 | 1 |
| union_find_weak | 9 | 439,402 | 1.71 | 209 | 5,263 | 88 | 1 |
| pymatching_strong | 9 | 439,402 | 1.74 | 212 | 5,172 | 88 | 1 |
| pymatching_weak | 9 | 439,402 | 1.86 | 227 | 4,838 | 95 | 1 |
| union_find_weak | 11 | 300,603 | 3.34 | 279 | 2,694 | 115 | 1 |
| union_find_strong | 11 | 300,603 | 3.52 | 294 | 2,556 | 121 | 1 |
| pymatching_weak | 11 | 300,603 | 3.67 | 306 | 2,452 | 125 | 1 |
| relay_bp_strong | 9 | 439,402 | 2.61 | 319 | 3,448 | 132 | 1 |
| bposd_strong | 9 | 439,402 | 2.88 | 352 | 3,125 | 143 | 1 |
| bposd_weak | 9 | 439,402 | 3.25 | 397 | 2,769 | 162 | 2 |
| bposd_weak | 11 | 300,603 | 5.34 | 446 | 1,685 | 181 | 2 |
| bposd_strong | 11 | 300,603 | 5.55 | 463 | 1,621 | 188 | 2 |
| relay_bp_weak | 9 | 439,402 | 4.69 | 572 | 1,918 | 233 | 2 |
| belief_matching_strong | 9 | 439,402 | 4.79 | 585 | 1,878 | 238 | 2 |
| relay_bp_strong | 11 | 300,603 | 7.27 | 607 | 1,237 | 245 | 2 |
| union_find_pymatching_switching | 7 | 289,389 | 7.70 | 619 | 1,168 | 251 | 2 |
| pymatching_strong | 15 | 211,941 | 11.67 | 687 | 771 | 277 | 2 |
| belief_matching_strong | 11 | 300,603 | 8.82 | 736 | 1,020 | 298 | 2 |
| union_find_strong | 15 | 211,941 | 13.12 | 772 | 685 | 311 | 3 |
| pymatching_bposd_switching | 7 | 289,389 | 9.83 | 790 | 915 | 320 | 3 |
| pymatching_relay_bp_switching | 7 | 289,389 | 9.94 | 799 | 905 | 323 | 3 |
| union_find_weak | 15 | 211,941 | 13.65 | 804 | 659 | 323 | 3 |
| pymatching_strong | 13 | 407,949 | 7.26 | 823 | 1,239 | 332 | 3 |
| union_find_weak | 13 | 407,949 | 8.34 | 945 | 1,079 | 381 | 3 |
| union_find_strong | 13 | 407,949 | 8.47 | 960 | 1,062 | 387 | 3 |
| pymatching_weak | 15 | 211,941 | 16.58 | 976 | 542 | 394 | 3 |
| union_find_relay_bp_switching | 7 | 289,389 | 13.02 | 1,047 | 691 | 421 | 3 |
| union_find_bposd_switching | 7 | 289,389 | 13.14 | 1,056 | 684 | 425 | 3 |
| relay_bp_weak | 11 | 300,603 | 13.03 | 1,088 | 690 | 439 | 3 |
| union_find_pymatching_switching | 9 | 439,402 | 9.01 | 1,100 | 998 | 443 | 3 |
| bposd_weak | 15 | 211,941 | 18.87 | 1,111 | 476 | 447 | 3 |
| pymatching_weak | 13 | 407,949 | 10.10 | 1,145 | 891 | 461 | 4 |
| bposd_strong | 15 | 211,941 | 20.51 | 1,207 | 438 | 485 | 4 |
| bposd_weak | 13 | 407,949 | 11.83 | 1,341 | 760 | 540 | 4 |
| bposd_strong | 13 | 407,949 | 12.25 | 1,388 | 734 | 559 | 4 |
| pymatching_bposd_switching | 9 | 439,402 | 11.42 | 1,394 | 788 | 560 | 4 |
| pymatching_relay_bp_switching | 9 | 439,402 | 11.48 | 1,401 | 783 | 564 | 4 |
| union_find_belief_matching_switching | 7 | 289,389 | 21.23 | 1,707 | 423 | 686 | 5 |
| union_find_relay_bp_switching | 9 | 439,402 | 15.02 | 1,833 | 599 | 735 | 5 |
| union_find_bposd_switching | 9 | 439,402 | 15.33 | 1,871 | 587 | 750 | 5 |
| relay_bp_strong | 13 | 407,949 | 19.24 | 2,180 | 467 | 876 | 6 |
| belief_matching_strong | 15 | 211,941 | 38.02 | 2,238 | 236 | 900 | 6 |
| relay_bp_strong | 15 | 211,941 | 39.71 | 2,338 | 226 | 939 | 7 |
| pymatching_belief_matching_switching | 9 | 439,402 | 20.46 | 2,497 | 439 | 1,004 | 7 |
| union_find_belief_matching_switching | 9 | 439,402 | 23.27 | 2,840 | 386 | 1,142 | 8 |
| pymatching_belief_matching_switching | 7 | 289,389 | 40.90 | 3,288 | 220 | 1,319 | 9 |
| union_find_pymatching_switching | 11 | 300,603 | 41.61 | 3,474 | 216 | 1,393 | 10 |
| relay_bp_weak | 13 | 407,949 | 30.86 | 3,497 | 291 | 1,405 | 10 |
| belief_matching_strong | 13 | 407,949 | 32.41 | 3,673 | 277 | 1,476 | 10 |
| relay_bp_weak | 15 | 211,941 | 66.46 | 3,913 | 135 | 1,572 | 11 |
| pymatching_bposd_switching | 11 | 300,603 | 46.93 | 3,919 | 191 | 1,577 | 11 |
| pymatching_relay_bp_switching | 11 | 300,603 | 50.38 | 4,207 | 178 | 1,692 | 12 |
| union_find_relay_bp_switching | 11 | 300,603 | 73.55 | 6,141 | 122 | 2,466 | 17 |
| union_find_bposd_switching | 11 | 300,603 | 74.33 | 6,207 | 121 | 2,488 | 17 |
| pymatching_bposd_switching | 15 | 211,941 | 107.39 | 6,322 | 83 | 2,555 | 18 |
| union_find_pymatching_switching | 15 | 211,941 | 107.65 | 6,338 | 83 | 2,555 | 18 |
| pymatching_relay_bp_switching | 15 | 211,941 | 107.87 | 6,351 | 83 | 2,555 | 18 |
| pymatching_belief_matching_switching | 11 | 300,603 | 83.27 | 6,953 | 108 | 2,785 | 19 |
| union_find_relay_bp_switching | 15 | 211,941 | 139.73 | 8,226 | 64 | 3,313 | 23 |
| union_find_pymatching_switching | 13 | 407,949 | 72.60 | 8,227 | 123 | 3,321 | 23 |
| union_find_bposd_switching | 15 | 211,941 | 157.87 | 9,294 | 57 | 3,720 | 25 |
| union_find_belief_matching_switching | 11 | 300,603 | 129.39 | 10,804 | 69 | 4,359 | 30 |
| pymatching_bposd_switching | 13 | 407,949 | 103.04 | 11,676 | 87 | 4,691 | 32 |
| pymatching_belief_matching_switching | 15 | 211,941 | 201.03 | 11,835 | 44 | 4,819 | 33 |
| pymatching_relay_bp_switching | 13 | 407,949 | 110.40 | 12,510 | 81 | 5,040 | 34 |
| union_find_bposd_switching | 13 | 407,949 | 117.51 | 13,316 | 76 | 5,370 | 36 |
| union_find_belief_matching_switching | 15 | 211,941 | 238.51 | 14,042 | 37 | 5,730 | 39 |
| union_find_relay_bp_switching | 13 | 407,949 | 130.25 | 14,760 | 69 | 5,915 | 40 |
| pymatching_belief_matching_switching | 13 | 407,949 | 194.34 | 22,022 | 46 | 8,870 | 60 |
| union_find_belief_matching_switching | 13 | 407,949 | 211.72 | 23,992 | 42 | 9,715 | 65 |

Whole campaign: 272,096 core-hours, 109,887 tasks, 801 sbatch lines.

## The submit lines

Every line has this shape, from the decsim checkout, with SHARDS the
distance's unit count from the table, OFFSET stepping by 150 and the
array running to the smaller of 149 and the remaining units:

```bash
DECSIM_PYTHON=<interpreter> RUN=results/campaigns_2026_09/<campaign> SHARDS=<tasks> OFFSET=<o> sbatch -J <campaign>_d<d> -a 0-<n> --time=04:00:00 --mem=<memory by distance> -o results/campaigns_2026_09/<campaign>/logs/%x_%A_%a.out slurm/campaign_run.sh configs/campaigns_2026_09/<campaign>_d<d>.yaml --shots-per-unit <shots per unit>
```

DECSIM_PYTHON names the interpreter when the checkout has no venv of
its own (slurm/campaign_run.sh). The library the union find rows load
is built once on the host with tools/build_union_find.sh before the
first line is submitted.

## After an array finishes

```bash
decsim combine results/campaigns_2026_09/<campaign>/*          # all distances of one campaign into one report
decsim plot results/campaigns_2026_09/<campaign>/combined --figure ler_vs_d --probability 0.003
```

A task that died is rerun by resubmitting its shard index alone (`-a <i>` with the same SHARDS and OFFSET); its folder is rewritten and combine reads whatever folders it is handed.
