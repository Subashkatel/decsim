# Strong-only stage timelines

The weak baseline's timeline figures (guide/walkthrough
small_run_timeline / pipelined_timeline), reproduced for the strong-only
mode: one row per pipeline stage, one color per window, alternating gray
shades for the round stream. Every bar is a stamp from a real executed run
(link ledger, decoder stage records, frame records); d=3, 9 rounds,
3 windows, staged decoder (fetch 4 cycles/round, algorithm 5.0 us,
release 8 cycles at 250 MHz), copy-out 0.5 us.

- `strong_only_small_timeline.png`: rounds every 3 us. The stride exceeds
  the chain, so windows wait only for their data; every stage is readable.
  The strong-only differences from the weak figure: the `copy-out link`
  row (each round's second write, out of the fridge), `sb1 fill` (the
  room-side store the strong lane reads), `transfer (csd)` and `do link`
  in place of cwd/wdo.
- `strong_only_pipelined_timeline.png`: rounds every 1 us. The chain
  (dd 0.5 + csd 2.0 + fetch + 5.0 + release = 7.6 us per window) exceeds
  the 3 us stride, so the wait bars grow window by window: that growth is
  the backlog the stability sweep shows diverging.

- `tan_staging_depth0_timeline.png` / `tan_staging_depth1_timeline.png`:
  Tan windows, 15 rounds, one unit, with the per-unit input staging knob
  off and on (`input_staging_depth`). At depth 0 each window's csd
  transfer starts only when the previous decode releases the unit; at
  depth 1 the next transfer runs under the current algorithm bar (the
  ping-pong: TI EDMA SPRAAN4A, Smith 1982 depth-1 DAE, gem5-Aladdin ready
  bits at whole-buffer granularity). This pair uses the HAND-CHECK
  configuration (plain 5.0 us decoder, zero fetch/release, free copy-out)
  so every bar edge equals a cell in
  ../strong_only_hand_check/staging_window_events_hand_vs_sim.csv;
  completions tick every 7.0 us at depth 0 and 5.0 us at depth 1.

- `sliding_staging_depth1_timeline.png`: the null result: staging alone
  changes NOTHING for sliding (bit-identical stamps to the depth-0
  pipelined figure), because sliding's wait is the boundary gate, not the
  unit hold.
- `sliding_boundary_decoder_timeline.png`: the boundary fix
  (`boundary_application=DECODER`: raw rounds ship at data-complete, the
  seam mask is XORed into the landed input at the decoder, the rule every
  reference uses: qLDPC net_error, cudaq-x syndrome_mods, LILLIPUT state
  register, Skoric artificial defects). Combined with staging, each
  window's transfer runs under the previous decode and the chain drops
  from dd+csd+decode to dd+max(csd, decode); same config as the pipelined
  figure, so the two are directly comparable.

Generator: tmp/strong-prototype/gen_timeline_strong.py (rerun to
regenerate all).
