# Baseline closed loop, per-point latency and throughput

Plots: latency_stack.png (per-point stack vs input round period), throughput.png (decoded vs input rounds/us).

Circuit: surface_code:rotated_memory_z d=3, 60 rounds per shot, p=0.001, 10 shots per point. All latencies simulated, in microseconds, mean over decoded windows (max in the CSV).

| algo us | round us | LER | win/us | rounds/us | util | max q | c2b_per_round | buffer_fill | dep_block | queue_wait | cwd_per_window | fetch | algorithm | release | service | wdo_per_window | frame_commit | last_round_to_frame | reaction_first_round |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.028 | 0.02 | 0.00 | 0.2826 | 0.892 | 0.581 | 1 | 0.108 | 0.100 | 31.500 | 0.000 | 2.000 | 0.024 | 0.028 | 0.004 | 2.056 | 1.000 | 0.004 | 34.560 | 34.660 |
| 0.028 | 0.05 | 0.00 | 0.2819 | 0.890 | 0.580 | 1 | 0.108 | 0.250 | 30.690 | 0.000 | 2.000 | 0.024 | 0.028 | 0.004 | 2.056 | 1.000 | 0.004 | 33.750 | 34.000 |
| 0.028 | 0.1 | 0.00 | 0.2809 | 0.887 | 0.577 | 1 | 0.108 | 0.500 | 29.340 | 0.000 | 2.000 | 0.024 | 0.028 | 0.004 | 2.056 | 1.000 | 0.004 | 32.400 | 32.900 |
| 0.028 | 0.2 | 0.00 | 0.2788 | 0.880 | 0.573 | 1 | 0.108 | 1.000 | 26.640 | 0.000 | 2.000 | 0.024 | 0.028 | 0.004 | 2.056 | 1.000 | 0.004 | 29.700 | 30.700 |
| 0.028 | 0.5 | 0.00 | 0.2728 | 0.862 | 0.561 | 1 | 0.108 | 2.500 | 18.540 | 0.000 | 2.000 | 0.024 | 0.028 | 0.004 | 2.056 | 1.000 | 0.004 | 21.600 | 24.100 |
| 0.028 | 1 | 0.00 | 0.2634 | 0.832 | 0.541 | 1 | 0.108 | 5.000 | 5.040 | 0.000 | 2.000 | 0.024 | 0.028 | 0.004 | 2.056 | 1.000 | 0.004 | 8.100 | 13.100 |
| 0.28 | 0.02 | 0.00 | 0.2638 | 0.833 | 0.609 | 1 | 0.108 | 0.100 | 33.768 | 0.000 | 2.000 | 0.024 | 0.280 | 0.004 | 2.308 | 1.000 | 0.004 | 37.080 | 37.180 |
| 0.28 | 0.05 | 0.00 | 0.2632 | 0.831 | 0.608 | 1 | 0.108 | 0.250 | 32.958 | 0.000 | 2.000 | 0.024 | 0.280 | 0.004 | 2.308 | 1.000 | 0.004 | 36.270 | 36.520 |
| 0.28 | 0.1 | 0.00 | 0.2623 | 0.828 | 0.605 | 1 | 0.108 | 0.500 | 31.608 | 0.000 | 2.000 | 0.024 | 0.280 | 0.004 | 2.308 | 1.000 | 0.004 | 34.920 | 35.420 |
| 0.28 | 0.2 | 0.00 | 0.2605 | 0.823 | 0.601 | 1 | 0.108 | 1.000 | 28.908 | 0.000 | 2.000 | 0.024 | 0.280 | 0.004 | 2.308 | 1.000 | 0.004 | 32.220 | 33.220 |
| 0.28 | 0.5 | 0.00 | 0.2553 | 0.806 | 0.589 | 1 | 0.108 | 2.500 | 20.808 | 0.000 | 2.000 | 0.024 | 0.280 | 0.004 | 2.308 | 1.000 | 0.004 | 24.120 | 26.620 |
| 0.28 | 1 | 0.00 | 0.2470 | 0.780 | 0.570 | 1 | 0.108 | 5.000 | 7.308 | 0.000 | 2.000 | 0.024 | 0.280 | 0.004 | 2.308 | 1.000 | 0.004 | 10.620 | 15.620 |
| 2.8 | 0.02 | 0.00 | 0.1584 | 0.500 | 0.765 | 1 | 0.108 | 0.100 | 56.448 | 0.000 | 2.000 | 0.024 | 2.800 | 0.004 | 4.828 | 1.000 | 0.004 | 62.280 | 62.380 |
| 2.8 | 0.05 | 0.00 | 0.1583 | 0.500 | 0.764 | 1 | 0.108 | 0.250 | 55.638 | 0.000 | 2.000 | 0.024 | 2.800 | 0.004 | 4.828 | 1.000 | 0.004 | 61.470 | 61.720 |
| 2.8 | 0.1 | 0.00 | 0.1579 | 0.499 | 0.762 | 1 | 0.108 | 0.500 | 54.288 | 0.000 | 2.000 | 0.024 | 2.800 | 0.004 | 4.828 | 1.000 | 0.004 | 60.120 | 60.620 |
| 2.8 | 0.2 | 0.00 | 0.1573 | 0.497 | 0.759 | 1 | 0.108 | 1.000 | 51.588 | 0.000 | 2.000 | 0.024 | 2.800 | 0.004 | 4.828 | 1.000 | 0.004 | 57.420 | 58.420 |
| 2.8 | 0.5 | 0.00 | 0.1553 | 0.491 | 0.750 | 1 | 0.108 | 2.500 | 43.488 | 0.000 | 2.000 | 0.024 | 2.800 | 0.004 | 4.828 | 1.000 | 0.004 | 49.320 | 51.820 |
| 2.8 | 1 | 0.00 | 0.1522 | 0.481 | 0.735 | 1 | 0.108 | 5.000 | 29.988 | 0.000 | 2.000 | 0.024 | 2.800 | 0.004 | 4.828 | 1.000 | 0.004 | 35.820 | 40.820 |

Reading the table. Windows are sliding (commit d, buffer d) and serial: window k+1 starts only after window k's boundary arrives, so the loop's capacity is one window per serial chain. Measured chain at the fastest input: 3.54 us per window = unit assigned, CWD transfer into its memory, decoder service, WDO delivery, boundary handoff, i.e. 0.85 rounds/us sustainable, a knee at a round period of about 1.18 us. Faster input only grows dep_block (the wait for the previous window). A unit is held from assignment through its input transfer to the end of its decode, so utilization counts the CWD transfer; the decode itself is the fetch+algorithm+release columns. With these link cards the baseline is link-bound, not decoder-bound.

Simulator wall clock per shot (host CPU, not a modeled latency): 0.24s, 0.25s, 0.24s, 0.25s, 0.25s, 0.31s ...

Anchor comparison against published numbers: anchor.md (experiments/baseline_anchor.py).
