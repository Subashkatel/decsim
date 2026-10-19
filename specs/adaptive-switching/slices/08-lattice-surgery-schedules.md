# 08 Lattice surgery schedules

## Question

Can an external logical schedule drive the verified timing path without losing
patch ownership, operation durations or classical feedforward dependencies?

## Deliverable and ownership

One importer at the existing frontend/program boundary. Select a compiler by
its actual exported semantics before implementation; do not build a compiler
inside the simulator. Candidate resources are linked in the [plan](../README.md).

## Work

1. Audit supported exports for operation IDs, patch/ancilla placement,
   merge/split intervals, QEC-round durations, outcome identities and dependency
   edges. Keep unavailable semantics explicit rather than invent them.
2. Validate a published small lattice-surgery CNOT schedule as a geometry and
   timing fixture. Then add a supported injection/feedforward schedule or an
   independently documented explicit dependency overlay.
3. Map resources and syndrome streams through joint patch operations; a merge
   is not automatically three independent decoder problems.
4. Preserve protected memory during schedule stalls and distinguish scheduled
   starts from starts shifted by classical feedback.
5. Test a small algorithmic schedule only after the local fixtures agree.

## Gate and evidence

Record compiler/revision and source schedule. Match patch occupancy, rounds,
dependency graph and expected completion in an ideal-classical-cost reference.
Then add supplied hardware costs and explain every additional wait. Compare
like-for-like with an external schedule or published timing model; different
decoder, noise and hardware assumptions prevent a claim of paper replication.

## Scope boundary

An importer that lacks outcome dependencies cannot establish T-gate timing.
Do not promise arbitrary algorithm correctness or infer complete T support
from a CNOT example or the ability to export a Stim circuit.
