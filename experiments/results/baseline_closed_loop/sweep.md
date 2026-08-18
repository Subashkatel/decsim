# Baseline closed loop, per-point latency and throughput

Plots: latency_stack.png (per-point stack vs input round period), throughput.png (decoded vs input rounds/us).

Circuit: surface_code:rotated_memory_z d=3, 60 rounds per shot, p=0.001, 10 shots per point. All latencies simulated, in microseconds, mean over decoded windows (max in the CSV).

| algo us | round us | LER | win/us | rounds/us | util | max q | c2b_per_round | buffer_fill | dep_block | queue_wait | cwd_per_window | fetch | algorithm | release | service | wdo_per_window | frame_commit | last_round_to_frame | reaction_first_round |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| measured | 0.02 | 0.00 | 0.1187 | 0.375 | 0.937 | 1 | 0.108 | 0.100 | 76.836 | 0.000 | 2.000 | 0.024 | 5.922 | 0.004 | 7.950 | 1.000 | 0.004 | 85.790 | 85.890 |
| measured | 0.05 | 0.00 | 0.1178 | 0.372 | 0.936 | 1 | 0.108 | 0.250 | 75.995 | 0.000 | 2.000 | 0.024 | 5.975 | 0.004 | 8.003 | 1.000 | 0.004 | 85.002 | 85.252 |
| measured | 0.1 | 0.00 | 0.1159 | 0.366 | 0.936 | 1 | 0.108 | 0.500 | 77.011 | 0.000 | 2.000 | 0.024 | 6.111 | 0.004 | 8.139 | 1.000 | 0.004 | 86.154 | 86.654 |
| measured | 0.2 | 0.00 | 0.1168 | 0.369 | 0.932 | 1 | 0.108 | 1.000 | 73.112 | 0.000 | 2.000 | 0.024 | 6.009 | 0.004 | 8.037 | 1.000 | 0.004 | 82.153 | 83.154 |
| measured | 0.5 | 0.00 | 0.1163 | 0.367 | 0.923 | 1 | 0.108 | 2.500 | 64.462 | 0.000 | 2.000 | 0.024 | 5.968 | 0.004 | 7.996 | 1.000 | 0.004 | 73.462 | 75.963 |
| measured | 1 | 0.00 | 0.1144 | 0.361 | 0.910 | 1 | 0.108 | 5.000 | 51.041 | 0.000 | 2.000 | 0.024 | 5.969 | 0.004 | 7.997 | 1.000 | 0.004 | 60.042 | 65.043 |
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

Reading the table. Windows are sliding (commit d, buffer d) and serial: window k+1 starts only after window k's boundary arrives, so the loop's capacity is one window per serial chain = unit assigned, CWD transfer into its memory, decoder service, boundary handoff over DD at decode done (the WDO delivery and frame commit run downstream, off the chain). Measured chain at the fastest input, per algorithm: measured: 8.43 us per window, 0.36 rounds/us, knee near a 2.81 us round period; 0.028 us: 2.59 us per window, 1.16 rounds/us, knee near a 0.86 us round period; 0.28 us: 2.84 us per window, 1.06 rounds/us, knee near a 0.95 us round period. Faster input only grows dep_block (the wait for the previous window). A unit is held from assignment through its input transfer to the end of its decode, so utilization counts the CWD transfer; the decode itself is the fetch+algorithm+release columns. With these link cards the ASIC rows are link-bound (CWD 2 us + DD 0.5 us per window), not decoder-bound; the measured software row is decoder-bound.

Simulator wall clock per shot (host CPU, not a modeled latency): 0.25s, 0.25s, 0.25s, 0.25s, 0.25s, 0.25s ...

Anchor comparison against published numbers: anchor.md (experiments/baseline_anchor.py).
