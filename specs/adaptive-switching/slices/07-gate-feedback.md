# 07 Gate feedback

## Question

When an operation must wait for a decode result, as the S correction after a
T-gate teleportation does (paper, Fig. 2), how long does it wait, and does the
backlog over a sequence of such gates match the paper's model?

## Current state

- `Operation.blocked_by` names the operation whose decoded result releases
  it (`records/program.py`). Release waits for every window of that operation
  to be final, including earlier escalated windows
  (`windows/operation_results.py`), then charges the frame write,
  frame→controller, the controller pulse and controller→QPU; the known-answer
  test checks each hop (`tests/test_known_answers.py`).
- Release does not read the outcome bits: the consumer starts whether or not
  a correction is needed.
- `RuntimeStamps` records only the magic-state wait. Nothing records when the
  consumer was otherwise ready, so stall time is not measured.
- Two hangs: `blocked_by` pointing at an operation that emits no detector data
  passes validation and never releases; `blocked_by` cycles are not checked.

## Work

1. Fix both hangs with refusals at planning time.
2. Add the stall record per released operation: blocker's last round,
   blocker's final commit, release at the controller, otherwise-ready time,
   start. Stall = start − max(otherwise-ready, next round boundary). Expose
   it as a trace the slice 06 port can read and in the run result.
3. Build the paper's gate chain on one patch: a sequence of gates r_op rounds
   apart, each consumer released by the previous measurement, with the patch
   kept alive (protected) while it waits. Verify that the protected-region
   rules in `controller/feedback_streams.py` support a chain of N releases on
   one stream; if not, that support is part of this slice.
4. Measure r_i − r_op per gate and compare with the paper's recursion
   (Eqs. 9–13) under its assumptions (linear service, synthetic i.i.d.
   escalation), then with real decodes and the slice 02 table.
5. Vary r_op between d and 9d (the paper's range for Litinski's compilation).

## Gate and evidence

Hand-calculated release schedules match to the tick. The gate-chain backlog
matches the paper's recursion under matched assumptions. Stall
distributions are reported with sample counts.

## Scope boundary

Timing through gates is not a T-gate LER claim; Stim memory correctness does
not simulate a non-Clifford gate. Synthetic gate positions are labeled.
