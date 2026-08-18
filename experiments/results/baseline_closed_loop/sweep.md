# Baseline closed loop, per-point latency and throughput

Plots: latency_stack.png (per-point stack vs input round period), throughput.png (decoded vs input rounds/us).

Circuit: surface_code:rotated_memory_z d=3, 60 rounds per shot, p=0.001, 10 shots per point. All latencies simulated, in microseconds, mean over decoded windows (max in the CSV).

| algo us | round us | LER | win/us | rounds/us | util | max q | c2b_per_round | buffer_fill | dep_block | queue_wait | cwd_per_window | fetch | algorithm | release | service | wdo_per_window | frame_commit | last_round_to_frame | reaction_first_round |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| measured | 0.02 | 0.00 | 0.1433 | 0.453 | 0.924 | 1 | 0.108 | 0.100 | 62.603 | 0.000 | 2.000 | 0.024 | 4.426 | 0.004 | 6.454 | 1.000 | 0.004 | 70.060 | 70.161 |
| measured | 0.05 | 0.00 | 0.1431 | 0.452 | 0.923 | 1 | 0.108 | 0.250 | 61.564 | 0.000 | 2.000 | 0.024 | 4.425 | 0.004 | 6.453 | 1.000 | 0.004 | 69.022 | 69.272 |
| measured | 0.1 | 0.00 | 0.1430 | 0.452 | 0.921 | 1 | 0.108 | 0.500 | 60.242 | 0.000 | 2.000 | 0.024 | 4.419 | 0.004 | 6.447 | 1.000 | 0.004 | 67.693 | 68.193 |
| measured | 0.2 | 0.00 | 0.1410 | 0.445 | 0.918 | 1 | 0.108 | 1.000 | 57.959 | 0.000 | 2.000 | 0.024 | 4.489 | 0.004 | 6.517 | 1.000 | 0.004 | 65.481 | 66.481 |
| measured | 0.5 | 0.00 | 0.1403 | 0.443 | 0.908 | 1 | 0.108 | 2.500 | 49.346 | 0.000 | 2.000 | 0.024 | 4.447 | 0.004 | 6.475 | 1.000 | 0.004 | 56.825 | 59.325 |
| measured | 1 | 0.00 | 0.1385 | 0.437 | 0.891 | 1 | 0.108 | 5.000 | 35.830 | 0.000 | 2.000 | 0.024 | 4.409 | 0.004 | 6.437 | 1.000 | 0.004 | 43.270 | 48.271 |
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

Reading the table. Windows are sliding (commit d, buffer d) and serial: window k+1 starts only after window k's boundary arrives, so the loop's capacity is one window per serial chain = unit assigned, CWD transfer into its memory, decoder service, boundary handoff over DD at decode done (the WDO delivery and frame commit run downstream, off the chain). Measured chain at the fastest input, per algorithm: measured: 6.98 us per window, 0.43 rounds/us, knee near a 2.33 us round period; 0.028 us: 2.59 us per window, 1.16 rounds/us, knee near a 0.86 us round period; 0.28 us: 2.84 us per window, 1.06 rounds/us, knee near a 0.95 us round period. Faster input only grows dep_block (the wait for the previous window). A unit is held from assignment through its input transfer to the end of its decode, so utilization counts the CWD transfer; the decode itself is the fetch+algorithm+release columns. With these link cards the ASIC rows are link-bound (CWD 2 us + DD 0.5 us per window), not decoder-bound; the measured software row is decoder-bound.

Simulator wall clock per shot (host CPU, not a modeled latency): 0.25s, 0.25s, 0.25s, 0.25s, 0.25s, 0.25s ...

Anchor comparison against published numbers: anchor.md (experiments/baseline_anchor.py).
