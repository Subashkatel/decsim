# Willow data through the baseline loop

Data: Zenodo 13273331 (arXiv:2408.13687), patches d=3: d3_at_q4_5, d=5: d5_at_q6_5, d=7: d7_at_q6_7.

## 1. Accuracy on the same recorded shots (30 cycles)

| d | basis | loop LER (n shots) | whole-shot PyMatching, same n | Google corr. matching, same n | whole-shot, 50k | Google, 50k | Google eps/cycle % (50k) |
|---|---|---|---|---|---|---|---|
| 3 | X | 0.230 (300) | 0.230 | 0.200 | 0.2166 | 0.1816 | 0.747 |
| 3 | Z | 0.183 (300) | 0.183 | 0.147 | 0.1965 | 0.1670 | 0.673 |
| 5 | X | 0.200 (200) | 0.200 | 0.100 | 0.1879 | 0.1215 | 0.462 |
| 5 | Z | 0.160 (200) | 0.155 | 0.130 | 0.1689 | 0.1157 | 0.437 |
| 7 | X | 0.210 (100) | 0.210 | 0.130 | 0.1210 | 0.0696 | 0.249 |
| 7 | Z | 0.060 (100) | 0.060 | 0.060 | 0.1109 | 0.0636 | 0.226 |

Google pathway error per cycle averaged over bases (50k shots): d=3: 0.710 %; d=5: 0.449 %; d=7: 0.238 %
Lambda(5->7) of that pathway: 1.89; paper headline: eps_7 = 0.143 %, Lambda = 2.14 (best decoders, all cycle counts fitted).

Read across a row: the loop and whole-shot PyMatching share decoder and prior and decode the same n shots, so 'loop' and 'same n' must agree (they do, up to the rare window-boundary difference between windowed and whole-shot matching: one shot in 1200 here); the 50k columns show the subset's sampling noise. Google's released pathway adds correlation reweighting, so it is lower; the paper's headline uses further-optimized priors and ensembles. Per-cycle rates from the loop's small subsets are not quoted.

## 2. Real-time configuration (paper Sec. V: d = 5, 1.1 us cycle)

Software algorithm cost used: 4.1 us per d=5 window (median PyMatching wall clock on this host, graph cached); reference link cards (CWD 2 us, WDO 1 us, DD 0.5 us), C2B 0.1 us at 1 Gbit/s, frame commit 4 ns.

| quantity | this loop | paper |
|---|---|---|
| last cycle received -> correction committed, mean (sd, max) us | 58.7 (30.4, 110.2) | 63.0 +- 17.0 |
| sustained input, rounds/us | 0.65 | 1/1.1 = 0.91 |
| logical failures over 250 cycles, d=5 | loop 5/10, whole-shot PyMatching on the same shots 5/10 | eps_5 = 0.35 % per cycle real-time (0.269 offline NN); at 250 cycles that is P ~ 0.4 |

Deviations to state: the paper's latency includes Ethernet, shared-memory buffering and a multi-threaded streaming decoder on a workstation, none of which are our numbers; ours are the reference link cards plus one measured software decode per window. Windows here are sliding (commit 5, buffer 5) and serial; the paper's decoder streams. Their real-time run used the 72-qubit processor data (not in this archive); ours replays the 105-qubit d=5 patch.
