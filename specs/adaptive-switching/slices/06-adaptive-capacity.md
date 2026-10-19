# 06 Throughput and latency loop

## Question

Can a slower loop set the escalation target from what the pipeline reports,
so the system keeps up and corrections arrive in time, and does it beat the
best static threshold when conditions change?

## Why adaptive at all

For fixed, known conditions a static threshold at the q\* quantile already
keeps up (slice 03, and the paper's Eq. 18 for double windows). Adaptation
only earns its place where that static choice fails: belief-matching service
time varies with p and window content; p drifts; gate spacing r_op ranges
from d to 9d rounds (paper, Sec. III.3); several patches share one strong
unit (the paper's γ → γ·N_window). The experiments below are built around
those cases.

## Deliverable and ownership

An outer loop that adjusts the slice 05 tracker's target. It reads live
state through a new port on the threshold source, bound the way the deadline
row binds `source_queue` (`escalation/deadlines.py`, `build/escalation.py`).
The tracker and cap stay separate responsibilities.

## Work

1. **Stage A, memory stream.** Inputs: lag, strong queue depth and outstanding
   work, recent reaction p99. Rule: when lag or reaction exceeds its budget,
   multiply the target by β < 1; after a sustained interval under a low
   watermark, add α; hold otherwise. Fix units, cadence, step sizes and
   hysteresis before implementing.
2. **Stage B, gate stalls** (after 07 and 08): the input becomes the stall at
   each released operation, the collaborator's design. A stall raises the
   threshold-side pressure (fewer escalations); several consecutive reads
   with zero stall lower it.
3. Only information available at the update instant: a service time is known
   after completion, not at admission.
4. Lowering the target cannot undo admitted work; report it.

## Experiments

On the same workload and hardware, compare fixed 20 dB, the best static
threshold calibrated at the design point, a static threshold at the q\*
quantile, and the adaptive loop, under:
- steady conditions (adaptive should match the best static);
- p rising mid-stream;
- strong service slowing mid-stream (scale the timing table);
- gate spacing changing (stage B);
- deliberately infeasible weak-only capacity (no controller can fix it; it
  must say so rather than pretend).

Metrics: lag and reaction p50/p99/max, gate stall distribution, escalation
fraction over time, settling time, overshoot, LER.

## Gate and evidence

Stable behavior on steady input; recovery after each change; the
infeasible case reported as infeasible. Results repeated over seeds and
held-out distances.

## Scope boundary

No online LER guarantee and no correctness inferred from decoder agreement.
A missed deadline stalls the consumer; admitted strong work is never silently
cancelled or replaced.
