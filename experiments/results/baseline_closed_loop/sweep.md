# Baseline closed loop, per-point latency and throughput

Plots: latency_stack.png (per-point stack vs input round period), throughput.png (decoded vs input rounds/us).

Circuit: surface_code:rotated_memory_z d=3, 60 rounds per shot, p=0.001, 10 shots per point. All latencies simulated, in microseconds, mean over decoded windows (max in the CSV).

| algo us | round us | LER | win/us | rounds/us | util | max q | c2b_per_round | buffer_fill | dep_block | queue_wait | cwd_per_window | fetch | algorithm | release | service | wdo_per_window | frame_commit | last_round_to_frame | reaction_first_round |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| measured | 0.02 | 0.00 | 0.1184 | 0.374 | 0.937 | 1 | 0.108 | 0.100 | 76.914 | 0.000 | 2.000 | 0.024 | 5.940 | 0.004 | 7.968 | 1.000 | 0.004 | 85.886 | 85.986 |
| measured | 0.05 | 0.00 | 0.1183 | 0.374 | 0.936 | 1 | 0.108 | 0.250 | 76.018 | 0.000 | 2.000 | 0.024 | 5.927 | 0.004 | 7.955 | 1.000 | 0.004 | 84.977 | 85.227 |
| measured | 0.1 | 0.00 | 0.1166 | 0.368 | 0.935 | 1 | 0.108 | 0.500 | 75.757 | 0.000 | 2.000 | 0.024 | 6.036 | 0.004 | 8.064 | 1.000 | 0.004 | 84.825 | 85.325 |
| measured | 0.2 | 0.00 | 0.1149 | 0.363 | 0.933 | 1 | 0.108 | 1.000 | 74.480 | 0.000 | 2.000 | 0.024 | 6.164 | 0.004 | 8.192 | 1.000 | 0.004 | 83.676 | 84.677 |
| measured | 0.5 | 0.00 | 0.1157 | 0.365 | 0.924 | 1 | 0.108 | 2.500 | 64.945 | 0.000 | 2.000 | 0.024 | 6.014 | 0.004 | 8.042 | 1.000 | 0.004 | 73.991 | 76.492 |
| measured | 1 | 0.00 | 0.1124 | 0.355 | 0.911 | 1 | 0.108 | 5.000 | 52.224 | 0.000 | 2.000 | 0.024 | 6.158 | 0.004 | 8.186 | 1.000 | 0.004 | 61.414 | 66.415 |
| 0.028 | 0.02 | 0.00 | 0.3864 | 1.220 | 0.794 | 1 | 0.108 | 0.100 | 22.464 | 0.000 | 2.000 | 0.024 | 0.028 | 0.004 | 2.056 | 1.000 | 0.004 | 25.524 | 25.624 |
| 0.028 | 0.05 | 0.00 | 0.3852 | 1.216 | 0.792 | 1 | 0.108 | 0.250 | 21.654 | 0.000 | 2.000 | 0.024 | 0.028 | 0.004 | 2.056 | 1.000 | 0.004 | 24.714 | 24.964 |
| 0.028 | 0.1 | 0.00 | 0.3833 | 1.210 | 0.788 | 1 | 0.108 | 0.500 | 20.304 | 0.000 | 2.000 | 0.024 | 0.028 | 0.004 | 2.056 | 1.000 | 0.004 | 23.364 | 23.864 |
| 0.028 | 0.2 | 0.00 | 0.3795 | 1.198 | 0.780 | 1 | 0.108 | 1.000 | 17.604 | 0.000 | 2.000 | 0.024 | 0.028 | 0.004 | 2.056 | 1.000 | 0.004 | 20.664 | 21.664 |
| 0.028 | 0.5 | 0.00 | 0.3684 | 1.163 | 0.757 | 1 | 0.108 | 2.500 | 9.504 | 0.000 | 2.000 | 0.024 | 0.028 | 0.004 | 2.056 | 1.000 | 0.004 | 12.564 | 15.064 |
| 0.028 | 1 | 0.00 | 0.3061 | 0.967 | 0.629 | 1 | 0.108 | 5.000 | 0.000 | 0.000 | 2.000 | 0.024 | 0.028 | 0.004 | 2.056 | 1.000 | 0.004 | 3.060 | 8.060 |
| 0.28 | 0.02 | 0.00 | 0.3521 | 1.112 | 0.813 | 1 | 0.108 | 0.100 | 24.732 | 0.000 | 2.000 | 0.024 | 0.280 | 0.004 | 2.308 | 1.000 | 0.004 | 28.044 | 28.144 |
| 0.28 | 0.05 | 0.00 | 0.3511 | 1.109 | 0.810 | 1 | 0.108 | 0.250 | 23.922 | 0.000 | 2.000 | 0.024 | 0.280 | 0.004 | 2.308 | 1.000 | 0.004 | 27.234 | 27.484 |
| 0.28 | 0.1 | 0.00 | 0.3495 | 1.104 | 0.807 | 1 | 0.108 | 0.500 | 22.572 | 0.000 | 2.000 | 0.024 | 0.280 | 0.004 | 2.308 | 1.000 | 0.004 | 25.884 | 26.384 |
| 0.28 | 0.2 | 0.00 | 0.3463 | 1.094 | 0.799 | 1 | 0.108 | 1.000 | 19.872 | 0.000 | 2.000 | 0.024 | 0.280 | 0.004 | 2.308 | 1.000 | 0.004 | 23.184 | 24.184 |
| 0.28 | 0.5 | 0.00 | 0.3371 | 1.065 | 0.778 | 1 | 0.108 | 2.500 | 11.772 | 0.000 | 2.000 | 0.024 | 0.280 | 0.004 | 2.308 | 1.000 | 0.004 | 15.084 | 17.584 |
| 0.28 | 1 | 0.00 | 0.3049 | 0.963 | 0.704 | 1 | 0.108 | 5.000 | 0.000 | 0.000 | 2.000 | 0.024 | 0.280 | 0.004 | 2.308 | 1.000 | 0.004 | 3.312 | 8.312 |

algorithm = 'measured' rows charge the wall clock of each real PyMatching call (software decoder on this host: AMD EPYC 9454 48-Core Processor, one thread, graph cached); numeric rows charge the stated modeled latency (an ASIC card).

Reading the table. Windows are sliding (commit d, buffer d) and serial: window k+1 starts only after window k's boundary arrives, so the loop's capacity is one window per serial chain = unit assigned, CWD transfer into its memory, decoder service, boundary handoff over DD at decode done (the WDO delivery and frame commit run downstream, off the chain). Measured chain at the fastest input, per algorithm: measured: 8.45 us per window, 0.36 rounds/us, knee near a 2.82 us round period; 0.028 us: 2.59 us per window, 1.16 rounds/us, knee near a 0.86 us round period; 0.28 us: 2.84 us per window, 1.06 rounds/us, knee near a 0.95 us round period. Faster input only grows dep_block (the wait for the previous window). A unit is held from assignment through its input transfer to the end of its decode, so utilization counts the CWD transfer; the decode itself is the fetch+algorithm+release columns. With these link cards the ASIC rows are link-bound (CWD 2 us + DD 0.5 us per window), not decoder-bound; the measured software row is decoder-bound.

Simulator wall clock per shot (host CPU, not a modeled latency): 0.25s, 0.25s, 0.25s, 0.25s, 0.25s, 0.25s ...

Anchor comparison against published numbers: anchor.md (experiments/baseline_anchor.py).
