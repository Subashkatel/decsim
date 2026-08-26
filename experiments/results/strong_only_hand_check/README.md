# Strong-only sliding: hand trace vs simulator

The weak baseline's discipline applied to the strong-only path: every
per-window event time computed by pencil formula from the declared
constants, then compared cell by cell against the simulator's stamps and
link ledger. Produced by `hand_check_strong_only.py` (durable copy in
`tmp/strong-prototype/`).

## Setup

d=3 rotated memory, 27 rounds at 1.0 us, sliding windows commit 3 buffer 3
(9 windows), StrongOnly policy, one strong unit, decode 5.0 us, reference
links (qc 0.15, csd 2.0, dd 0.5, do 1.0), copy-out 0.5 us, frame 0.1 us.
All links unbounded bandwidth: no serialization terms anywhere.

## The pencil formulas

    stored(r)   = r*1.0 + 0.15 + 0.5          (QPU emit + qc + copy-out)
    data(k)     = stored(min(3k+6, 27))
    dep(k)      = done(k-1) + 0.5             (boundary over dd)
    queued(k)   = max(data, dep)
    csd(k)      = queued(k) + 2.0             (one unit; the chain's dep
    done(k)     = csd(k) + 5.0                 always exceeds unit-free)
    do(k)       = done(k) + 1.0
    frame(k)    = do(k) + 0.1

## The sheets

- `window_events_hand.csv`: the pure hand trace, every number from the
  pencil formulas above, no simulator values and no tags.
- `window_events_hand_vs_sim.csv`: the verification record: the same
  events with the simulator's stamp beside each hand value and an
  OK/MISMATCH flag per cell.

## Two rows by literal arithmetic

W0: reads 1..6. stored(6) = 6 + 0.65 = 6.65. No predecessor, so
queued = 6.65. csd = 8.65. done = 13.65. do = 14.65. frame = 14.75.

W8 (terminal): reads 25..27 (30 clamped to 27). stored(27) = 27.65.
The chain: done(k) = 13.65 + 7.5k for k <= 7 (each link dd 0.5 + csd 2.0 +
decode 5.0 = 7.5), so done(7) = 66.15 and dep(8) = 66.65 > 27.65.
queued = 66.65. csd = 68.65. done = 73.65. do = 74.65. frame = 74.75.

Both rows, and every other cell, equal the simulator's stamps exactly.

## Count, honestly

9 windows x 9 event columns = 81 cells. 79 are real hand-vs-sim
comparisons and all 79 match to the tick. The remaining 2 are structural
n/a: W0 has no dependency arrival, and W8 ships no dd boundary (no
successor); the sheet marks them OK by convention.

Note also what this check proves that the acceptance MATCH did not: the
acceptance compared strong-only against another simulator run; this sheet
compares it against arithmetic done outside the simulator entirely.

## Staging hand check (both depths)

`staging_window_events_hand_vs_sim.csv`, produced by
`tmp/strong-prototype/hand_check_staging.py`: an independent reference
implementation of both rule sets (depth 0, transfer under the unit hold;
depth 1, ping-pong: slot-free transfer start, swap at compute end, idle
until landing) replays Tan 1-unit runs and compares three events per
window against the simulator.

- 15 rounds, depths 0 and 1: 15/15 cells each. These rows match the
  timeline figures in ../strong_only_timelines/
  (tan_staging_depth0/1_timeline.png) BAR FOR BAR: same configuration
  (plain 5.0 us decoder, free copy-out), so every bar edge in the plot is
  a number in this sheet. Completion gaps: depth 0 exactly {7.0} =
  csd + decode; depth 1 exactly {5.0} = max(csd, decode).
- 27 rounds, depth 1 (the recorded lock configuration): 39/39 cells,
  gaps exactly {5.0}.

## Sliding with DECODER boundary application (phase 2)

`sliding_decoder_boundary_hand_vs_sim.csv`: the reference model extended
with the decoder-side boundary rules (submission at data-complete; the
staging slot frees at the SWAP, i.e. the previous compute's end, even when
the promoted job then parks waiting for its boundary; service starts at
max(promotion, boundary arrival)). 9 windows x 3 events = 27/27 cells
match; completion gaps exactly {5.5 us} = dd + max(csd, decode), down from
{7.5} in assembly mode. Results are bit-identical across modes (pinned by
tests/13_orchestrators/test_boundary_application.py). One modeling lesson
the mismatch taught before it was fixed: promotion and service are
distinct events, and the simulator had it right.
