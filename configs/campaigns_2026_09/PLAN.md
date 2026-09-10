# The 2026-09 campaigns: the plan

Shot table: see base.yaml's sweep (one table for all six campaigns, each point about 100 failures from the August strong-tier rates, floor 400, cap 200,000, four points skipped: d 11 to 15 at p 0.001 and d 15 at p 0.002).
Costs: seconds per shot measured 2026-09-10 (tmp/after-merge/cost10d in the research sandbox), four shots per point at p 0.003, model build included.
Arrays: one per campaign and distance, one work unit per task, a unit sized to about 50 minutes so a task fits a 01:00:00 limit and can backfill; arrays are cut at the scheduler's MaxArraySize of 2500 with OFFSET carrying the shard index on.
Memory per task by distance: d 3 4G, d 5 4G, d 7 4G, d 9 6G, d 11 8G, d 13 12G, d 15 16G (peak resident set at d 15, two shots, p 0.003: PyMatching 1.9 GB, union find 0.9 GB, belief matching 1.9 GB, Relay-BP 7.6 GB, PyMatching plus belief matching switching 2.8 GB; the union find plus Relay-BP pair was not measured and is bounded by Relay-BP's figure plus the weak tier).

| campaign | d | shots | s per shot | core-hours | shots per unit | tasks | arrays |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pymatching_weak | 3 | 25,159 | 0.07 | 0 | 10,000 | 7 | 1 |
| pymatching_weak | 5 | 117,672 | 0.26 | 8 | 10,000 | 16 | 1 |
| pymatching_weak | 7 | 289,389 | 0.71 | 57 | 4,225 | 73 | 1 |
| pymatching_weak | 9 | 439,402 | 1.62 | 198 | 1,851 | 242 | 1 |
| pymatching_weak | 11 | 300,603 | 3.27 | 273 | 917 | 330 | 1 |
| pymatching_weak | 13 | 407,949 | 5.31 | 602 | 564 | 725 | 1 |
| pymatching_weak | 15 | 211,941 | 8.85 | 521 | 338 | 629 | 1 |
| belief_matching_strong | 3 | 25,159 | 0.14 | 1 | 10,000 | 7 | 1 |
| belief_matching_strong | 5 | 117,672 | 0.73 | 24 | 4,109 | 33 | 1 |
| belief_matching_strong | 7 | 289,389 | 2.36 | 190 | 1,271 | 231 | 1 |
| belief_matching_strong | 9 | 439,402 | 5.68 | 693 | 528 | 835 | 1 |
| belief_matching_strong | 11 | 300,603 | 10.72 | 895 | 279 | 1,080 | 1 |
| belief_matching_strong | 13 | 407,949 | 18.25 | 2,068 | 164 | 2,490 | 1 |
| belief_matching_strong | 15 | 211,941 | 28.53 | 1,680 | 105 | 2,020 | 1 |
| union_find_weak | 3 | 25,159 | 0.08 | 1 | 10,000 | 7 | 1 |
| union_find_weak | 5 | 117,672 | 0.48 | 16 | 6,250 | 23 | 1 |
| union_find_weak | 7 | 289,389 | 1.72 | 138 | 1,744 | 170 | 1 |
| union_find_weak | 9 | 439,402 | 5.29 | 646 | 567 | 777 | 1 |
| union_find_weak | 11 | 300,603 | 13.35 | 1,115 | 224 | 1,344 | 1 |
| union_find_weak | 13 | 407,949 | 23.50 | 2,663 | 127 | 3,215 | 2 |
| union_find_weak | 15 | 211,941 | 48.00 | 2,826 | 62 | 3,420 | 2 |
| relay_bp_strong | 3 | 25,159 | 0.09 | 1 | 10,000 | 7 | 1 |
| relay_bp_strong | 5 | 117,672 | 0.41 | 13 | 7,317 | 20 | 1 |
| relay_bp_strong | 7 | 289,389 | 1.17 | 94 | 2,564 | 117 | 1 |
| relay_bp_strong | 9 | 439,402 | 3.41 | 416 | 879 | 503 | 1 |
| relay_bp_strong | 11 | 300,603 | 8.92 | 745 | 336 | 898 | 1 |
| relay_bp_strong | 13 | 407,949 | 19.75 | 2,238 | 151 | 2,703 | 2 |
| relay_bp_strong | 15 | 211,941 | 77.77 | 4,579 | 38 | 5,580 | 3 |
| pymatching_belief_matching_switching | 3 | 25,159 | 0.68 | 5 | 4,411 | 10 | 1 |
| pymatching_belief_matching_switching | 5 | 117,672 | 3.88 | 127 | 773 | 155 | 1 |
| pymatching_belief_matching_switching | 7 | 289,389 | 15.93 | 1,281 | 188 | 1,543 | 1 |
| pymatching_belief_matching_switching | 9 | 439,402 | 20.24 | 2,470 | 148 | 2,972 | 2 |
| pymatching_belief_matching_switching | 11 | 300,603 | 80.05 | 6,684 | 37 | 8,126 | 4 |
| pymatching_belief_matching_switching | 13 | 407,949 | 157.12 | 17,805 | 19 | 21,474 | 9 |
| pymatching_belief_matching_switching | 15 | 211,941 | 196.42 | 11,564 | 15 | 14,132 | 6 |
| union_find_relay_bp_switching | 3 | 25,159 | 0.50 | 3 | 6,000 | 8 | 1 |
| union_find_relay_bp_switching | 5 | 117,672 | 3.90 | 127 | 769 | 156 | 1 |
| union_find_relay_bp_switching | 7 | 289,389 | 20.80 | 1,672 | 144 | 2,012 | 1 |
| union_find_relay_bp_switching | 9 | 439,402 | 51.38 | 6,271 | 58 | 7,579 | 4 |
| union_find_relay_bp_switching | 11 | 300,603 | 156.23 | 13,045 | 19 | 15,824 | 7 |
| union_find_relay_bp_switching | 13 | 407,949 | 387.70 | 43,934 | 7 | 58,281 | 24 |
| union_find_relay_bp_switching | 15 | 211,941 | 878.45 | 51,717 | 3 | 70,649 | 29 |

Stage 1 (the five campaigns without the Python union find under switching): 62,635 core-hours, 75,914 tasks. Stage 2 (union find with Relay-BP switching): 116,770 core-hours, 154,509 tasks; run after a compiled union find or a latency card, or cap it at d 11.

## Stage 1 submit lines (from the decsim checkout)

```bash
RUN=results/campaigns_2026_09/pymatching_weak SHARDS=7 OFFSET=0 sbatch -J pymatching_weak_d3 -a 0-6 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_weak_d3.yaml --shots-per-unit 10000
RUN=results/campaigns_2026_09/pymatching_weak SHARDS=16 OFFSET=0 sbatch -J pymatching_weak_d5 -a 0-15 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_weak_d5.yaml --shots-per-unit 10000
RUN=results/campaigns_2026_09/pymatching_weak SHARDS=73 OFFSET=0 sbatch -J pymatching_weak_d7 -a 0-72 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_weak_d7.yaml --shots-per-unit 4225
RUN=results/campaigns_2026_09/pymatching_weak SHARDS=242 OFFSET=0 sbatch -J pymatching_weak_d9 -a 0-241 --time=01:00:00 --mem=6G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_weak_d9.yaml --shots-per-unit 1851
RUN=results/campaigns_2026_09/pymatching_weak SHARDS=330 OFFSET=0 sbatch -J pymatching_weak_d11 -a 0-329 --time=01:00:00 --mem=8G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_weak_d11.yaml --shots-per-unit 917
RUN=results/campaigns_2026_09/pymatching_weak SHARDS=725 OFFSET=0 sbatch -J pymatching_weak_d13 -a 0-724 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_weak_d13.yaml --shots-per-unit 564
RUN=results/campaigns_2026_09/pymatching_weak SHARDS=629 OFFSET=0 sbatch -J pymatching_weak_d15 -a 0-628 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_weak_d15.yaml --shots-per-unit 338
RUN=results/campaigns_2026_09/belief_matching_strong SHARDS=7 OFFSET=0 sbatch -J belief_matching_strong_d3 -a 0-6 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/belief_matching_strong_d3.yaml --shots-per-unit 10000
RUN=results/campaigns_2026_09/belief_matching_strong SHARDS=33 OFFSET=0 sbatch -J belief_matching_strong_d5 -a 0-32 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/belief_matching_strong_d5.yaml --shots-per-unit 4109
RUN=results/campaigns_2026_09/belief_matching_strong SHARDS=231 OFFSET=0 sbatch -J belief_matching_strong_d7 -a 0-230 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/belief_matching_strong_d7.yaml --shots-per-unit 1271
RUN=results/campaigns_2026_09/belief_matching_strong SHARDS=835 OFFSET=0 sbatch -J belief_matching_strong_d9 -a 0-834 --time=01:00:00 --mem=6G slurm/campaign_run.sh configs/campaigns_2026_09/belief_matching_strong_d9.yaml --shots-per-unit 528
RUN=results/campaigns_2026_09/belief_matching_strong SHARDS=1080 OFFSET=0 sbatch -J belief_matching_strong_d11 -a 0-1079 --time=01:00:00 --mem=8G slurm/campaign_run.sh configs/campaigns_2026_09/belief_matching_strong_d11.yaml --shots-per-unit 279
RUN=results/campaigns_2026_09/belief_matching_strong SHARDS=2490 OFFSET=0 sbatch -J belief_matching_strong_d13 -a 0-2489 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/belief_matching_strong_d13.yaml --shots-per-unit 164
RUN=results/campaigns_2026_09/belief_matching_strong SHARDS=2020 OFFSET=0 sbatch -J belief_matching_strong_d15 -a 0-2019 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/belief_matching_strong_d15.yaml --shots-per-unit 105
RUN=results/campaigns_2026_09/union_find_weak SHARDS=7 OFFSET=0 sbatch -J union_find_weak_d3 -a 0-6 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_weak_d3.yaml --shots-per-unit 10000
RUN=results/campaigns_2026_09/union_find_weak SHARDS=23 OFFSET=0 sbatch -J union_find_weak_d5 -a 0-22 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_weak_d5.yaml --shots-per-unit 6250
RUN=results/campaigns_2026_09/union_find_weak SHARDS=170 OFFSET=0 sbatch -J union_find_weak_d7 -a 0-169 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_weak_d7.yaml --shots-per-unit 1744
RUN=results/campaigns_2026_09/union_find_weak SHARDS=777 OFFSET=0 sbatch -J union_find_weak_d9 -a 0-776 --time=01:00:00 --mem=6G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_weak_d9.yaml --shots-per-unit 567
RUN=results/campaigns_2026_09/union_find_weak SHARDS=1344 OFFSET=0 sbatch -J union_find_weak_d11 -a 0-1343 --time=01:00:00 --mem=8G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_weak_d11.yaml --shots-per-unit 224
RUN=results/campaigns_2026_09/union_find_weak SHARDS=3215 OFFSET=0 sbatch -J union_find_weak_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_weak_d13.yaml --shots-per-unit 127
RUN=results/campaigns_2026_09/union_find_weak SHARDS=3215 OFFSET=2500 sbatch -J union_find_weak_d13 -a 0-714 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_weak_d13.yaml --shots-per-unit 127
RUN=results/campaigns_2026_09/union_find_weak SHARDS=3420 OFFSET=0 sbatch -J union_find_weak_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_weak_d15.yaml --shots-per-unit 62
RUN=results/campaigns_2026_09/union_find_weak SHARDS=3420 OFFSET=2500 sbatch -J union_find_weak_d15 -a 0-919 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_weak_d15.yaml --shots-per-unit 62
RUN=results/campaigns_2026_09/relay_bp_strong SHARDS=7 OFFSET=0 sbatch -J relay_bp_strong_d3 -a 0-6 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/relay_bp_strong_d3.yaml --shots-per-unit 10000
RUN=results/campaigns_2026_09/relay_bp_strong SHARDS=20 OFFSET=0 sbatch -J relay_bp_strong_d5 -a 0-19 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/relay_bp_strong_d5.yaml --shots-per-unit 7317
RUN=results/campaigns_2026_09/relay_bp_strong SHARDS=117 OFFSET=0 sbatch -J relay_bp_strong_d7 -a 0-116 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/relay_bp_strong_d7.yaml --shots-per-unit 2564
RUN=results/campaigns_2026_09/relay_bp_strong SHARDS=503 OFFSET=0 sbatch -J relay_bp_strong_d9 -a 0-502 --time=01:00:00 --mem=6G slurm/campaign_run.sh configs/campaigns_2026_09/relay_bp_strong_d9.yaml --shots-per-unit 879
RUN=results/campaigns_2026_09/relay_bp_strong SHARDS=898 OFFSET=0 sbatch -J relay_bp_strong_d11 -a 0-897 --time=01:00:00 --mem=8G slurm/campaign_run.sh configs/campaigns_2026_09/relay_bp_strong_d11.yaml --shots-per-unit 336
RUN=results/campaigns_2026_09/relay_bp_strong SHARDS=2703 OFFSET=0 sbatch -J relay_bp_strong_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/relay_bp_strong_d13.yaml --shots-per-unit 151
RUN=results/campaigns_2026_09/relay_bp_strong SHARDS=2703 OFFSET=2500 sbatch -J relay_bp_strong_d13 -a 0-202 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/relay_bp_strong_d13.yaml --shots-per-unit 151
RUN=results/campaigns_2026_09/relay_bp_strong SHARDS=5580 OFFSET=0 sbatch -J relay_bp_strong_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/relay_bp_strong_d15.yaml --shots-per-unit 38
RUN=results/campaigns_2026_09/relay_bp_strong SHARDS=5580 OFFSET=2500 sbatch -J relay_bp_strong_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/relay_bp_strong_d15.yaml --shots-per-unit 38
RUN=results/campaigns_2026_09/relay_bp_strong SHARDS=5580 OFFSET=5000 sbatch -J relay_bp_strong_d15 -a 0-579 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/relay_bp_strong_d15.yaml --shots-per-unit 38
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=10 OFFSET=0 sbatch -J pymatching_belief_matching_switching_d3 -a 0-9 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d3.yaml --shots-per-unit 4411
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=155 OFFSET=0 sbatch -J pymatching_belief_matching_switching_d5 -a 0-154 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d5.yaml --shots-per-unit 773
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=1543 OFFSET=0 sbatch -J pymatching_belief_matching_switching_d7 -a 0-1542 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d7.yaml --shots-per-unit 188
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=2972 OFFSET=0 sbatch -J pymatching_belief_matching_switching_d9 -a 0-2499 --time=01:00:00 --mem=6G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d9.yaml --shots-per-unit 148
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=2972 OFFSET=2500 sbatch -J pymatching_belief_matching_switching_d9 -a 0-471 --time=01:00:00 --mem=6G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d9.yaml --shots-per-unit 148
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=8126 OFFSET=0 sbatch -J pymatching_belief_matching_switching_d11 -a 0-2499 --time=01:00:00 --mem=8G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d11.yaml --shots-per-unit 37
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=8126 OFFSET=2500 sbatch -J pymatching_belief_matching_switching_d11 -a 0-2499 --time=01:00:00 --mem=8G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d11.yaml --shots-per-unit 37
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=8126 OFFSET=5000 sbatch -J pymatching_belief_matching_switching_d11 -a 0-2499 --time=01:00:00 --mem=8G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d11.yaml --shots-per-unit 37
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=8126 OFFSET=7500 sbatch -J pymatching_belief_matching_switching_d11 -a 0-625 --time=01:00:00 --mem=8G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d11.yaml --shots-per-unit 37
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=21474 OFFSET=0 sbatch -J pymatching_belief_matching_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d13.yaml --shots-per-unit 19
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=21474 OFFSET=2500 sbatch -J pymatching_belief_matching_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d13.yaml --shots-per-unit 19
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=21474 OFFSET=5000 sbatch -J pymatching_belief_matching_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d13.yaml --shots-per-unit 19
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=21474 OFFSET=7500 sbatch -J pymatching_belief_matching_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d13.yaml --shots-per-unit 19
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=21474 OFFSET=10000 sbatch -J pymatching_belief_matching_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d13.yaml --shots-per-unit 19
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=21474 OFFSET=12500 sbatch -J pymatching_belief_matching_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d13.yaml --shots-per-unit 19
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=21474 OFFSET=15000 sbatch -J pymatching_belief_matching_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d13.yaml --shots-per-unit 19
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=21474 OFFSET=17500 sbatch -J pymatching_belief_matching_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d13.yaml --shots-per-unit 19
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=21474 OFFSET=20000 sbatch -J pymatching_belief_matching_switching_d13 -a 0-1473 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d13.yaml --shots-per-unit 19
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=14132 OFFSET=0 sbatch -J pymatching_belief_matching_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d15.yaml --shots-per-unit 15
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=14132 OFFSET=2500 sbatch -J pymatching_belief_matching_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d15.yaml --shots-per-unit 15
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=14132 OFFSET=5000 sbatch -J pymatching_belief_matching_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d15.yaml --shots-per-unit 15
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=14132 OFFSET=7500 sbatch -J pymatching_belief_matching_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d15.yaml --shots-per-unit 15
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=14132 OFFSET=10000 sbatch -J pymatching_belief_matching_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d15.yaml --shots-per-unit 15
RUN=results/campaigns_2026_09/pymatching_belief_matching_switching SHARDS=14132 OFFSET=12500 sbatch -J pymatching_belief_matching_switching_d15 -a 0-1631 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/pymatching_belief_matching_switching_d15.yaml --shots-per-unit 15
```

## Stage 2 submit lines (from the decsim checkout)

```bash
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=8 OFFSET=0 sbatch -J union_find_relay_bp_switching_d3 -a 0-7 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d3.yaml --shots-per-unit 6000
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=156 OFFSET=0 sbatch -J union_find_relay_bp_switching_d5 -a 0-155 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d5.yaml --shots-per-unit 769
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=2012 OFFSET=0 sbatch -J union_find_relay_bp_switching_d7 -a 0-2011 --time=01:00:00 --mem=4G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d7.yaml --shots-per-unit 144
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=7579 OFFSET=0 sbatch -J union_find_relay_bp_switching_d9 -a 0-2499 --time=01:00:00 --mem=6G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d9.yaml --shots-per-unit 58
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=7579 OFFSET=2500 sbatch -J union_find_relay_bp_switching_d9 -a 0-2499 --time=01:00:00 --mem=6G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d9.yaml --shots-per-unit 58
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=7579 OFFSET=5000 sbatch -J union_find_relay_bp_switching_d9 -a 0-2499 --time=01:00:00 --mem=6G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d9.yaml --shots-per-unit 58
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=7579 OFFSET=7500 sbatch -J union_find_relay_bp_switching_d9 -a 0-78 --time=01:00:00 --mem=6G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d9.yaml --shots-per-unit 58
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=15824 OFFSET=0 sbatch -J union_find_relay_bp_switching_d11 -a 0-2499 --time=01:00:00 --mem=8G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d11.yaml --shots-per-unit 19
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=15824 OFFSET=2500 sbatch -J union_find_relay_bp_switching_d11 -a 0-2499 --time=01:00:00 --mem=8G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d11.yaml --shots-per-unit 19
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=15824 OFFSET=5000 sbatch -J union_find_relay_bp_switching_d11 -a 0-2499 --time=01:00:00 --mem=8G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d11.yaml --shots-per-unit 19
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=15824 OFFSET=7500 sbatch -J union_find_relay_bp_switching_d11 -a 0-2499 --time=01:00:00 --mem=8G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d11.yaml --shots-per-unit 19
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=15824 OFFSET=10000 sbatch -J union_find_relay_bp_switching_d11 -a 0-2499 --time=01:00:00 --mem=8G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d11.yaml --shots-per-unit 19
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=15824 OFFSET=12500 sbatch -J union_find_relay_bp_switching_d11 -a 0-2499 --time=01:00:00 --mem=8G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d11.yaml --shots-per-unit 19
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=15824 OFFSET=15000 sbatch -J union_find_relay_bp_switching_d11 -a 0-823 --time=01:00:00 --mem=8G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d11.yaml --shots-per-unit 19
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=0 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=2500 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=5000 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=7500 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=10000 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=12500 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=15000 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=17500 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=20000 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=22500 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=25000 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=27500 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=30000 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=32500 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=35000 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=37500 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=40000 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=42500 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=45000 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=47500 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=50000 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=52500 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=55000 sbatch -J union_find_relay_bp_switching_d13 -a 0-2499 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=58281 OFFSET=57500 sbatch -J union_find_relay_bp_switching_d13 -a 0-780 --time=01:00:00 --mem=12G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d13.yaml --shots-per-unit 7
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=0 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=2500 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=5000 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=7500 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=10000 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=12500 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=15000 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=17500 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=20000 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=22500 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=25000 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=27500 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=30000 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=32500 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=35000 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=37500 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=40000 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=42500 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=45000 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=47500 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=50000 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=52500 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=55000 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=57500 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=60000 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=62500 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=65000 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=67500 sbatch -J union_find_relay_bp_switching_d15 -a 0-2499 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
RUN=results/campaigns_2026_09/union_find_relay_bp_switching SHARDS=70649 OFFSET=70000 sbatch -J union_find_relay_bp_switching_d15 -a 0-648 --time=01:00:00 --mem=16G slurm/campaign_run.sh configs/campaigns_2026_09/union_find_relay_bp_switching_d15.yaml --shots-per-unit 3
```

## After an array finishes

```bash
decsim combine results/campaigns_2026_09/<campaign>/*          # all distances of one campaign into one report
decsim plot results/campaigns_2026_09/<campaign>/combined --figure ler_vs_d --probability 0.003
```

A task that died is rerun by resubmitting its shard index alone (`-a <i>` with the same SHARDS and OFFSET); its folder is rewritten and combine reads whatever folders it is handed.

