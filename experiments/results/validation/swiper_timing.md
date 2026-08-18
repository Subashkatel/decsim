# Gate 2: decsim window timing vs SWIPER-SIM (sliding windows, fixed decode time)

Problem: one memory patch, d=7, sliding windows (commit 7, buffer 7), one decoder, fixed decode time 14 rounds, no speculation; SWIPER MemorySchedule(n) vs decsim FixedRounds(n) with every link at zero latency, 1 us rounds, decoder latency 14 us (1 us = 1 round). SWIPER pinned at tmp/references/code/swiper (1c00e09), its own test suite 17/17 passing in tmp/validation/external/swiper-venv.

| n rounds | windows SWIPER / decsim | interior windows identical (commit, data complete, start, done) | total SWIPER / decsim |
|---|---|---|---|
| 7 | 1 / 1 | 0/0 | 21 / 21.0 |
| 14 | 2 / 1 | 0/0 | 42 / 28.0 |
| 21 | 3 / 2 | 1/1 | 56 / 42.0 |
| 35 | 5 / 4 | 3/3 | 84 / 70.0 |
| 70 | 10 / 9 | 8/8 | 154 / 140.0 |
| 100 | 15 / 14 | 13/13 | 224 / 210.0 |

## Per-window timeline, n = 35 (rounds one-based, times in rounds)

| window | SWIPER commit | SWIPER data complete / start / done | decsim commit | decsim data complete / start / done |
|---|---|---|---|---|
| 0 | 1-7 | 14 / 14 / 28 | 1-7 | 14 / 14 / 28 |
| 1 | 8-14 | 21 / 28 / 42 | 8-14 | 21 / 28 / 42 |
| 2 | 15-21 | 28 / 42 / 56 | 15-21 | 28 / 42 / 56 |
| 3 | 22-28 | 35 / 56 / 70 | 22-35 | 35 / 56 / 70 |
| 4 | 29-35 | 35 / 70 / 84 | |  |

Every window but the last is identical in the two simulators: a window's data is complete when its buffer round arrives, it starts decoding at max(data complete, previous window done), and it holds the decoder for the fixed decode time; decsim's zero-link configuration adds nothing.

The tail differs by design. SWIPER cuts the final rounds into their own commit-only window (and uses them as the previous window's buffer), so it decodes one more window and its total is exactly 14 rounds longer. decsim's final window commits through the end of the operation because the final measurement layer closes the boundary; this is also qLDPC's SlidingWindowDecoder tail (Gate 1: equal window count and commit sets). n=7 is a single window in both and agrees fully.

Verdict: PASS on every interior window and on the single-window case; the tail difference is a documented window-planning convention, not a timing defect.

## Links restored one at a time (n = 35): shift of each window's decode start / done, in us

| link | latency | w0 | w1 | w2 | w3 |
|---|---|---|---|---|---|
| qc | 0.15 | 0.15 / 0.15 | 0.15 / 0.15 | 0.15 / 0.15 | 0.15 / 0.15 |
| cwd | 2 | 2 / 2 | 4 / 4 | 6 / 6 | 8 / 8 |
| dd | 0.5 | 0 / 0 | 0.5 / 0.5 | 1 / 1 | 1.5 / 1.5 |
| wdo | 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| do | 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| oc | 4 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| cq | 0.15 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| wsd | 0.5 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| csd | 2 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

QC delays every round and so every window by its latency once. CWD is paid per window on the serial chain: with one unit the next window is assigned only when the unit frees, and its input transfer then precedes its decode (assign-then-transfer). DD is paid once per boundary handoff, sent when the decode completes (Q-063). WDO carries the correction to the frame downstream and does not gate the next window; DO, OC, CQ, WSD, CSD are off the weak-only path. None of these shift a window.

## Feedback wait cadence (Q-064): the QPU keeps extracting syndromes while a patch waits on a decode

| quantity | SWIPER RegularTSchedule(1,0) | decsim merge -> blocked successor |
|---|---|---|
| operation ends at round | 14 | 7 |
| decode of that operation completes | 42 | 21 |
| idle rounds emitted while waiting | 28 (two serial 14-round windows) | 14 (one 14-round window) |
| conditional operation starts | 42 | 21 |
| device rounds total | 54 | 28 |

Same rule in both: the waiting patch emits one syndrome round every cycle from the end of its operation to the completion of the decode (SWIPER DECODE_IDLE rows in device_manager.py _generate_syndrome_round; decsim QPUDevice idle rounds), and the released operation starts on the next cycle boundary. SWIPER's merge spans two patches whose windows are serial, so it waits two decode times where decsim's single-patch merge waits one; the cadence, not the window count, is what this section checks. Every window count and time above is in rounds; idle rounds count exactly the wait (28 = 28 SWIPER, 14 = 14 decsim).
