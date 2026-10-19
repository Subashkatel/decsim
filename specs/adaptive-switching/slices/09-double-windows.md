# 09 Double windows

## Question

With belief matching as the strong decoder, what does the paper's double
window change in correctness, work and latency compared with ordinary redo?

## Current state

- The double window exists (`escalation/strong_window_shapes.py`): on
  escalation the strong region covers r_com + 2 r_buf rounds from the escalated
  commit, reads only its commit rounds with both faces pinned (the far face on
  the restart window's weak commit), absorbs the next two weak windows even
  if they were confident, and uses eager boundaries so the weak chain keeps
  committing.
- It is not the default; Experiment 1 (`experiments/switching_baseline`,
  `decsim/presets/experiment_one.py`) used it with the Relay-BP measured table
  and kept unconverged answers. Those results are not reused.
- Belief matching always runs matching after BP, so every decode returns a
  valid answer; BP non-convergence costs time, not answers.
- A scratch check found the double window with every window escalated differs
  from strong-only (11 of 150 shots at d = 3, 8 of 60 at d = 5): it is a
  different decoding problem, not a re-decode of the same one.

## Work

1. Re-run the slice 01 fixtures and the slice 03 oracle for the double window
   with belief matching.
2. Extend the timing table for 3d-round strong windows; do not price them as
   ordinary redo windows.
3. Count trigger events, strong jobs, covered rounds and service work
   separately; a 1% trigger fraction is not a 1% work fraction.
4. Check Theorem 1 (Eqs. 14–15, 18) in stream mode under its linear-service
   assumption, then with real belief-matching service.
5. Compare whole-shot, ordinary-redo and double-window LER on the same
   physical shots, and the static and adaptive policies from 04–06.

## Gate and evidence

Unique ownership, valid pins, no deadlock, reconciled coverage, all costs
charged. Theorem 1's boundary reproduced under its assumptions.

## Scope boundary

No reuse of the ordinary-window calibration table or denominator. A late
strong result can still sit on a gate's critical path even though the weak
chain advances.
