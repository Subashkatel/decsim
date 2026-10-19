# 10 Speculation

Status: deferred until the baseline and gate-feedback gates pass.

## Question

Does a defined speculative hybrid policy improve useful completion and gate
response after paying for verification, invalidated work and recovery?

## Deliverable and ownership

A separate proposal and study on the verified runtime. First distinguish
parallel weak/strong execution from downstream-window speculation using a
provisional boundary. They have different work and recovery costs.

## Work

1. State exactly what is provisional: boundary payload, logical frame update,
   or a quantum operation. Begin with classical-window speculation only.
2. Define the architectural verifier result without claiming it is perfect
   ground truth. Label both-decoder failures in separate simulation evaluation.
3. Specify dependent-window invalidation, checkpoints, re-decoding, result
   replacement and finalization. Distinguish logical-only revisions from
   boundary changes that affect downstream work.
4. Include verification traffic, duplicated compute, cancellation semantics,
   queue contention, recovery work and the last unresolved result at a gate.
5. Compare with non-speculative hybrid routing and an applicable SWIPER
   baseline under matched schedules/resources, after reading its actual method.

## Gate and evidence

Use injected boundary mismatches and logical-only revisions to verify recovery
and final outcomes. Require no consumer to use unresolved speculative state
under the initial model. Report useful throughput, LER where supported, wasted
work, recovery frequency, tail response and gate stalls. A fast provisional
answer is not an early finalized correction.

## Scope boundary

No speculative execution of irreversible quantum operations in this first
study. Double-window speculation additionally depends on slice 09. Retain a
negative result if strong verification or recovery erases the proposed gain.
