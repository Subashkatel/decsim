# Gate 5: real syndrome data through a decode wait, vs Stim, qLDPC and PyMatching

Program: op1 (rounds 1-3), wait for its decode (window commit 3 buffer 3 completes at round 6, decode 4 us, release at 10), op2 (rounds 11-13); one finite Stim circuit rotated_memory_z d=3, 13 rounds, p=0.005, 300 recorded shots (Stim seed 23); QPU cycle 1 us, zero links, extend_stream idle policy so the waiting patch's rounds are the circuit's own rounds 4-10.

| check | result |
|---|---|
| every shot emits rounds 1..13 exactly once | 300/300 |
| every emitted round's bits equal the sampled detection events of that round (Stim oracle) | 300/300 |
| wait is 7 idle rounds and op2 starts on the boundary at 10 us | 300/300 |
| stream windows | decsim 4 / qLDPC 3 |
| per-window detector sets equal (qLDPC window 2d, stride d) | 2/4 |
| per-window commit sets equal | 2/4 |
| final logical prediction equal to qLDPC, shot for shot | 300/300 |
| final logical prediction equal to whole-circuit PyMatching, shot for shot | 300/300 |
| logical failures vs truth: qLDPC / PyMatching | 18 / 18 |

The wait rounds are not filler: they are decoded (windows 4-6 and 7-9 above), and op2's data starts at round 11 exactly where the circuit continues, so the always-on QPU is verified with data, not only timing.

Window tail: qLDPC knows the circuit is 13 rounds and merges the last 7 into one window; decsim's stream is cut in real time and cannot know at round 9 that the program ends at 13, so it commits 7-9 and then a final 10-13 window (the static planner, which knows the length, matches qLDPC's tail; Gate 1). The first windows are identical and every prediction agrees.

## Last-window rule across reference code (survey 2026-08-18)

| implementation | mode | last-window rule |
|---|---|---|
| ldpc (Roffe) base_overlapping_window_decoder.py:124-133 | offline | final decoding commits all of the last regular window |
| tesseract simplex (Google) simplex.cc:205-289 | offline | loop clamped at the end; the last solve reaches the end and commits its tail |
| Tan 2209.09219 L952-955 | streaming design | last window may be smaller, both time boundaries closed, the entire window is the core region |
| Skoric 2209.08552 L646-649 | streaming design | commit region of the last window is from the bottom of the regular commit region to the last round |
| qLDPC sinter.py:786-789 | offline | last window absorbs the remainder (window + stride - 1 wide), commits all |
| quits sliding_window.py:46-53, realtime_decoding_qldpc (Huang and Puri) | offline | ceil((T-W)/F) regular windows, then a last window of T - F*that rounds committing all: for 13 rounds, d=3 exactly 10-13, the same as decsim |
| SWIPER window_builder.py:320-353 | real-time | leftover flushed as its own smaller commit-only window |
| decsim (dynamic stream and static planner, schemes.py _finite_forward_window_geometries) | real-time | the QUITS/Skoric/Tan rule: regular d-round commits, the last window commits from its regular commit start to the last round (10-13 here) |

All agree that the final window commits everything up to the last round because the end boundary is closed. On the width of that final window decsim equals QUITS, realtime_decoding_qldpc, Skoric and Tan (checked on 13, 14, 15 and 60 rounds); qLDPC absorbs the tail into a wider last window (7-13) and SWIPER flushes it as an extra one-round window (13-13); the three rules coincide at 60 rounds, which is why Gate 1 matched window for window.
