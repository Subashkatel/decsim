# 09 Double windows

## Question

What does double-window recovery change in correctness, useful work, boundary
dependencies and late-result behavior compared with the verified sliding case?

## Deliverable and ownership

A separate experiment using the existing strong-window planner and ownership
ledger. Keep the ordinary sliding baseline available for comparison.

## Work

1. Derive actual strong read/commit extents and both boundary dependencies from
   the chosen scheme. Audit absorbed/restarted weak windows and retained rounds.
2. Repeat tiny correctness and deterministic timing fixtures before any sweep.
3. Extend the supplied timing profile for the actual strong job geometries;
   do not price a larger strong region as an ordinary redo window by default.
4. Define separate counts for trigger events, strong jobs, covered rounds and
   service work. A 1% trigger fraction is not a 1% work fraction.
5. Evaluate fixed threshold, calibrated static routing and only then adaptation.
   Revisit late results and gate-consumer dependencies explicitly.

## Gate and evidence

Require unique logical ownership, valid boundary pins, no deadlock, reconciled
coverage and all costs charged. Match a small example to the paper's geometry
before comparing its backlog equations under their stated linear-service
assumption. Compare whole-shot, sliding and double-window LER on the same
physical workload without assuming their boundary-conditioned inputs are equal.

## Scope boundary

No inherited audit-based controller assumptions and no automatic reuse of the
ordinary-window denominator or calibration table. A later response can remain
on the critical path even if the weak decoder advances independently.
