# Baseline parameters

The setup for the basic case, approved by the user on 2026-10-19. Every later
step adds to this one setup and compares against its three runs.

| Part | Setting |
| --- | --- |
| Code | Rotated surface code, Z memory |
| Distances | 5, 7, 9, 11 |
| Physical error rates | 0.002, 0.003, 0.005 |
| Noise | SD6 (needs the `deltakit` extra installed in `../venv`) |
| Shot length | 10d rounds |
| Round time | 1 µs per round |
| Start and end of shot | Noiseless preparation and final readout |
| Windows | Sliding: commit d rounds, look ahead d more |
| Last window | Same rule (`lookahead`) in all three runs |
| Weak decoder | Union find, 1 unit, Helios cycle model at 100 MHz |
| Confidence | Cluster gap, its computation charged 1 µs |
| Threshold | Fixed 20 dB: keep the weak answer if gap ≥ 20 dB, otherwise redo |
| Strong decoder | Belief matching, product-sum BP, 1 unit |
| BP iterations (max) | 20 |
| Strong decoder timing | Timing table filled from GPU measurements; test values until then |
| Redo rule | Belief matching redoes the same window; the next window waits for its answer |
| When belief matching starts | Only for windows below 20 dB, after union find's answer and gap are known |
| Link to belief matching | GPU link, 4.5 µs round trip; NVQLink, 3.8 µs, as a second point |
| Weak-side links, controller, frame | Sourced values already in the repo (below) |
| Shots | Exactly 1,000,000 per point, no early stop |
| Runs | Weak only, strong only, switching, on the same setup and seeds |
| Reported: accuracy | Logical error rate per shot; number of windows escalated |
| Reported: timing | Time from a window's data being complete to its correction in the frame (median, p99); whether the decoders keep up |

Every run decodes a saved dataset: 1,000,000 shots per (distance, p), sampled
once and fixed by its hash (slice 00), so the data never changes between runs.

36 production jobs: 4 distances × 3 error rates × 3 runs. Before them, a pilot
decodes the first 2,000 shots of every dataset with all three runs, to check
the config, window layout, run time and failure counts at every point. d = 13 is added
only if needed. If d = 11 at p = 0.002 gives too few failures to compare the
runs, that point is dropped.

## Where the sourced values come from

These are copied into the baseline config; the baseline imports nothing from
the older Experiment 1 preset (`decsim/presets/experiment_one.py`, which used
Relay-BP).

| Value | Setting | Source |
| --- | --- | --- |
| Union find | Weight step 0.5 (edges 6 to 16 ticks, Helios's finest w_max = 16, Sec. VI-C; 0.1 fails as often but is 4x slower and falls behind the rounds); `CycleCount` at 100 MHz, delay 3 cycles, 0 cycles per edge, no setup | Helios RTL, 2301.08419; 2406.08491 |
| Union-find engine | Fetch 1 cycle per job + 1 per round; release 3 cycles | Helios loading stage; 2603.16203 line 902 |
| Cluster gap walk | 1.0 µs on the weak unit | Chosen value, 2602.03336 |
| Verdict | 1 cycle compare, 1 cycle switch, 250 MHz | Estimate |
| Window issue | 5 cycles at 250 MHz (20 ns) | 2603.16203 lines 897-899 |
| Syndrome uplink | 40 cycles at 250 MHz, 4 lanes of 38.79 bits per cycle | 2603.16203 |
| Frame read | 1 cycle | 2603.16203 |
| Weak correction downlink | 41 cycles, same lanes | 2603.16203 |
| GPU link | RoCE v2 GPU echo, 4.5 µs median round trip | Backline 2609.09270 Table III |
| NVQLink | 3.839 µs round trip | 2510.25213 line 485 |
| Controller | Pack 8 cycles per round; decision to pulse 8 cycles; 250 MHz | 2603.16203; QubiC 2404.15260 |
| Detection events | 5 cycles, then 1 per round | 2605.04892 |
| Frame write | 1 cycle (4 ns) | 2605.04892 Table I |

## Still to settle with the GPU measurements

Whether the measured belief-matching times include copying data to and from
the GPU. If they do, the strong pool's engine charges no fetch or release
stages.
