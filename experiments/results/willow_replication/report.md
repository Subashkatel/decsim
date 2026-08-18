# Willow data through the baseline loop

Data: Zenodo 13273331 (arXiv:2408.13687), patches d=3: d3_at_q4_5, d=5: d5_at_q6_5, d=7: d7_at_q6_7.
Plots: logical_error_vs_cycles.png, eps_per_cycle_vs_distance.png, realtime_latency.png, latency_vs_window.png.
Host: AMD EPYC 9454 48-Core Processor; load average at run start 9.2, 8.6, 8.8 (measured decode times are only meaningful on an otherwise idle core).

## 0. Correctness freeze: loop vs whole-shot PyMatching, shot for shot

| d | cycles | shots | agree | differ (window-boundary cases) |
|---|---|---|---|---|
| 3 | 30 | 2000 | 1999 | 1 |
| 5 | 30 | 1000 | 1000 | 0 |
| 7 | 30 | 500 | 500 | 0 |

## 1. Accuracy on the same recorded shots

| d | basis | cycles | loop LER (n shots) | whole-shot PyMatching, same n | Google corr. matching, same n | whole-shot, 50k | Google, 50k |
|---|---|---|---|---|---|---|---|
| 3 | X | 10 | 0.087 (600) | 0.087 | 0.063 | 0.0850 | 0.0694 |
| 3 | X | 30 | 0.222 (600) | 0.220 | 0.205 | 0.2166 | 0.1816 |
| 3 | X | 50 | 0.280 (600) | 0.280 | 0.270 | 0.3157 | 0.2695 |
| 3 | X | 90 | 0.435 (600) | 0.435 | 0.410 | 0.4348 | 0.4032 |
| 3 | Z | 10 | 0.092 (600) | 0.092 | 0.057 | 0.0737 | 0.0601 |
| 3 | Z | 30 | 0.178 (600) | 0.178 | 0.140 | 0.1965 | 0.1670 |
| 3 | Z | 50 | 0.257 (600) | 0.258 | 0.247 | 0.2785 | 0.2438 |
| 3 | Z | 90 | 0.422 (600) | 0.422 | 0.432 | 0.4214 | 0.3897 |
| 5 | X | 10 | 0.048 (400) | 0.048 | 0.048 | 0.0567 | 0.0381 |
| 5 | X | 30 | 0.198 (400) | 0.198 | 0.115 | 0.1879 | 0.1215 |
| 5 | X | 50 | 0.260 (400) | 0.258 | 0.177 | 0.2347 | 0.1659 |
| 5 | X | 90 | 0.365 (400) | 0.362 | 0.295 | 0.3755 | 0.2894 |
| 5 | Z | 10 | 0.045 (400) | 0.045 | 0.020 | 0.0464 | 0.0304 |
| 5 | Z | 30 | 0.155 (400) | 0.152 | 0.130 | 0.1689 | 0.1157 |
| 5 | Z | 50 | 0.182 (400) | 0.185 | 0.133 | 0.1962 | 0.1370 |
| 5 | Z | 90 | 0.370 (400) | 0.378 | 0.258 | 0.3596 | 0.2760 |
| 7 | X | 10 | 0.040 (150) | 0.040 | 0.033 | 0.0345 | 0.0197 |
| 7 | X | 30 | 0.187 (150) | 0.187 | 0.120 | 0.1210 | 0.0696 |
| 7 | X | 50 | 0.213 (150) | 0.213 | 0.107 | 0.1675 | 0.1009 |
| 7 | X | 90 | 0.300 (150) | 0.300 | 0.213 | 0.2832 | 0.1818 |
| 7 | Z | 10 | 0.047 (150) | 0.047 | 0.013 | 0.0275 | 0.0162 |
| 7 | Z | 30 | 0.093 (150) | 0.093 | 0.073 | 0.1109 | 0.0636 |
| 7 | Z | 50 | 0.220 (150) | 0.220 | 0.113 | 0.1539 | 0.0891 |
| 7 | Z | 90 | 0.300 (150) | 0.307 | 0.167 | 0.2985 | 0.1912 |

Logical error per cycle, fitted across cycle counts (bases pooled), percent:

| d | decsim loop | whole-shot PyMatching (50k) | Google corr. matching (50k) | paper (best decoders) |
|---|---|---|---|---|
| 3 | 1.086 | 1.101 | 0.896 | 0.655 |
| 5 | 0.748 | 0.741 | 0.466 | 0.306 |
| 7 | 0.509 | 0.497 | 0.266 | 0.143 |

Lambda(5->7): decsim loop 1.47, whole-shot PyMatching 1.49, Google pathway 1.76, paper 2.14.

Read across a row: the loop and whole-shot PyMatching share decoder and prior and decode the same n shots, so 'loop' and 'same n' agree shot for shot (up to the rare window-boundary difference between windowed and whole-shot matching); the 50k columns show the subset's sampling noise. The gap to Google's released pathway is decoder quality (correlated matching reweights Y-error correlations); the paper's headline adds optimized priors and ensembling. The paper's error bars come from all cycle counts up to 250; ours from four.

## 2. Real-time configuration (paper Sec. V: d = 5, 1.1 us cycle)

Software algorithm cost: the wall clock of each real PyMatching call on this host, measured per window inside the loop: median 16.0 us, max 42.3 us on these d=5 SI1000 windows (10 rounds, 240 detectors; the sparser synthetic p=0.001 window reference is 4.2 us); graph prebuilt, one thread; reference link cards (CWD 2 us, WDO 1 us, DD 0.5 us), C2B 0.1 us at 1 Gbit/s, frame commit 4 ns.

| quantity | this loop | paper |
|---|---|---|
| last cycle received -> correction committed, mean (sd, max) us | 361.3 (202.8, 804.3) | 63.0 +- 17.0 |
| sustained input, rounds/us | 0.26 | 1/1.1 = 0.91 |
| logical failures over 250 cycles, d=5 | loop 5/10, whole-shot PyMatching on the same shots 5/10 | eps_5 = 0.35 % per cycle real-time (0.269 offline NN); at 250 cycles that is P ~ 0.4 |

Deviations to state: the paper's latency includes Ethernet, shared-memory buffering and a multi-threaded streaming decoder on a workstation, none of which are our numbers; ours are the reference link cards plus one measured software decode per window. Windows here are sliding (commit 5, buffer 5) and serial; the paper's decoder streams and kept latency constant for a million cycles, ours grows over 250 cycles because a single-threaded PyMatching call per 5-round window (16 us median here; one Python matching.decode call costs 5-7 us before any matching work, the published sub-microsecond figures are decode_batch amortized over thousands of shots) cannot keep pace with 1.1 us cycles; the paper's decoder is a multi-worker streaming design. Their real-time run used the 72-qubit processor data (not in this archive); ours replays the 105-qubit d=5 patch.

## 3. What keeps the loop up with the QPU (d = 5, 1.1 us cycles, 250 recorded cycles)

| scheme | units | algorithm | sustained rounds/us (need 0.91) | latency first 10 windows us | last 10 windows us | max us | keeps up |
|---|---|---|---|---|---|---|---|
| serial sliding | 1 | software PyMatching, measured per call | 0.26 | 86.5 | 638.7 | 796.2 | NO |
| serial sliding | 1 | LILLIPUT-class ASIC 42 ns | 0.90 | 3.1 | 3.1 | 3.1 | yes |
| serial sliding | 2 | software PyMatching, measured per call | 0.25 | 90.5 | 651.7 | 813.3 | NO |
| serial sliding | 2 | LILLIPUT-class ASIC 42 ns | 0.90 | 3.1 | 3.1 | 3.1 | yes |
| parallel A/B (Skoric) | 1 | software PyMatching, measured per call | 0.36 | 151.2 | 327.6 | 499.2 | NO |
| parallel A/B (Skoric) | 1 | LILLIPUT-class ASIC 42 ns | 0.89 | 9.9 | 9.4 | 16.7 | yes |
| parallel A/B (Skoric) | 2 | software PyMatching, measured per call | 0.65 | 62.3 | 90.5 | 166.3 | yes |
| parallel A/B (Skoric) | 2 | LILLIPUT-class ASIC 42 ns | 0.89 | 9.9 | 9.4 | 16.7 | yes |

Serial sliding windows are bound by the per-window chain (unit assigned, CWD transfer, decode, boundary handoff over DD at decode done), so more units change nothing and latency grows with time; parallel A/B windows remove the chain and hold latency flat. Plot: latency_vs_window.png.
