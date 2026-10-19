# 07 Gate feedback

## Question

Can a timing-only workload exercise the actual measurement-to-decoder-to-frame
to-control path and enforce the right dependent-operation release?

## Deliverable and ownership

A small explicit gate-teleportation/feedforward schedule using the existing
operation dependency, feedback-stream and decision-dispatch owners. Inspect
`records/program.py`, `frontends/execution_runtime.py`,
`pauli_frame/decision_dispatch.py` and the existing feedback tests.

## Work

1. Define the consumed measurement outcome, its patch/stream round coverage,
   desired consumer start and control return path. Do not require unrelated
   patches to finish or equate a T label with a complete dependency definition.
2. Use supplied costs and synthetic or recorded score/service sequences. Test
   weak completion, escalation, one earlier straggler and transfer contention.
3. Verify that a consumer waits for every required correction and return hop.
   Later ready results cannot release it past an earlier unresolved result.
4. Continue memory-protection rounds on live patches during waits and account
   for their additional workload. Separate magic-state availability initially
   so factory delay is not mistaken for decoder delay.
5. Add consumer slack/stall feedback to slice 06 only after fixed-policy
   consumer timing is verified.

## Gate and evidence

Compare a separately calculated dependency schedule with exact simulator
timestamps. Vary gate spacing and worker contention, then validate at least one
published gate-teleportation dependency pattern. Record readout, frame commit,
control return and actual consumer start. Model memory reaction budgets and
gate deadlines as distinct contracts. Report stalls and excess protected rounds.

## Scope boundary

Timing-only means no T-circuit LER claim. Stim memory correctness is not a
simulation of a non-Clifford T gate. Random feedback flags are explicitly
synthetic stress tests, not compiler validation or full-algorithm evidence.
