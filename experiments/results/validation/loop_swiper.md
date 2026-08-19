# Gate 8: decsim vs SWIPER on the same program

## 1. Memory, sliding windows (commit d, buffer d), one patch
- d=3 rounds=30 latency=2 units=1: 32 checks, agree
- d=3 rounds=30 latency=2 units=2: 32 checks, agree
- d=3 rounds=30 latency=5 units=1: 32 checks, agree
- d=3 rounds=30 latency=5 units=2: 32 checks, agree
- d=3 rounds=30 latency=11 units=1: 32 checks, agree
- d=3 rounds=30 latency=11 units=2: 32 checks, agree
- d=3 rounds=47 latency=2 units=1: 50 checks, agree
- d=3 rounds=47 latency=2 units=2: 50 checks, agree
- d=3 rounds=47 latency=5 units=1: 50 checks, agree
- d=3 rounds=47 latency=5 units=2: 50 checks, agree
- d=3 rounds=47 latency=11 units=1: 50 checks, agree
- d=3 rounds=47 latency=11 units=2: 50 checks, agree
- d=5 rounds=30 latency=2 units=1: 20 checks, agree
- d=5 rounds=30 latency=2 units=2: 20 checks, agree
- d=5 rounds=30 latency=5 units=1: 20 checks, agree
- d=5 rounds=30 latency=5 units=2: 20 checks, agree
- d=5 rounds=30 latency=11 units=1: 20 checks, agree
- d=5 rounds=30 latency=11 units=2: 20 checks, agree
- d=5 rounds=47 latency=2 units=1: 32 checks, agree
- d=5 rounds=47 latency=2 units=2: 32 checks, agree
- d=5 rounds=47 latency=5 units=1: 32 checks, agree
- d=5 rounds=47 latency=5 units=2: 32 checks, agree
- d=5 rounds=47 latency=11 units=1: 32 checks, agree
- d=5 rounds=47 latency=11 units=2: 32 checks, agree

## 2. Memory, parallel windows
SWIPER's parallel construction (sources of one commit region, sinks of three) and decsim's Skoric block A/B differ as windows, so only the round the program is decoded is compared; the window lists are printed.
- SWIPER windows (lo, hi, buffers, start, completion): [(0, 2, [[3, 5]], 6, 9), (3, 11, [], 22, 25), (12, 14, [[9, 11], [15, 17]], 18, 21), (15, 23, [], 34, 37), (24, 26, [[21, 23], [27, 29]], 30, 33), (27, 29, [], 34, 37)]
- decsim windows (lo, hi, buffer_hi, dispatch, done), rounds from 0: [(0, 5, 8, 9.0, 13.0), (6, 14, 14, 25.0, 29.0), (15, 17, 20, 21.0, 25.0), (18, 26, 26, 34.0, 38.0), (27, 29, 29, 30.0, 34.0)]
- d=3 rounds=30 latency=4 units=3: program decoded at: 1 checks, agree

## 3. Conditional S on a patch stream: reaction wait
SWIPER: idle d, idle d, then S conditioned on the second idle's decode; the patch keeps emitting DECODE_IDLE rounds while it waits. decsim: two d-round operations on one dynamic stream, then an operation blocked by the second, ExtendStream idle policy. Compared: the rounds the conditional waits, and the stream's windows up to the release.
- d=3 latency=2: wait 5 rounds, 2 stream windows: 3 checks, agree
- d=3 latency=5: wait 10 rounds, 4 stream windows: 5 checks, agree
- d=3 latency=9: wait 18 rounds, 7 stream windows: 8 checks, agree
- d=5 latency=2: wait 7 rounds, 2 stream windows: 3 checks, agree
- d=5 latency=5: wait 10 rounds, 3 stream windows: 4 checks, agree
- d=5 latency=9: wait 18 rounds, 4 stream windows: 5 checks, agree

Verdict: PASS. 833 checks, 0 disagreements.
