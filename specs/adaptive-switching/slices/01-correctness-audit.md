# 01 Correctness

## Question

Does switching decode correctly, and does each piece match an independent
reference?

## Deliverable and ownership

Repo tests that pin the switching contract, plus a short report from the pilot.
Existing anchors: `tests/escalation/test_switching_mode.py` (never
escalate equals weak alone), `tests/observe/test_window_outcomes.py`,
`tests/escalation/test_strong_window_shapes.py`.

## Work

1. **Equivalence tests (new, the main deliverable).** Under ordinary redo
   on the slice 00 geometry, shot for shot on fixed seeds:
   - threshold unreachable (every window escalates) equals strong-only;
   - threshold zero (no window escalates) equals weak-only.
   A scratch run already found 0 of 215 differing shots at d = 3 and 5. The
   existing `test_unreachable_threshold_escalates_every_window` checks only
   transfer counts, not predictions.
2. **Belief matching reference.** On identical window inputs, decsim's
   adapter matches the pinned `beliefmatching` package's predictions.
3. **Cluster gap reference.** On small hand-built and sampled windows, the gap
   equals an independent Dijkstra on the quotient graph G/C between
   inequivalent boundaries (the paper's Sec. II.3 definition), in the same
   units (`cluster.py` reports nats; 20 dB ≈ 4.605 nats).
4. **Count reconciliation.** At every pilot point, eligible windows, escalated
   windows, strong commits and shots reconcile with the manifest.
5. **Outcome classes.** For escalated windows, count fixed, unneeded, broken
   and both-wrong from the per-window records (`record_window_outcomes`).

## Gate and evidence

Equivalence tests pass in CI. Belief matching and gap agree with their
references exactly on deterministic fixtures. Any disagreement is diagnosed
before sample sizes grow.

## Scope boundary

Window labels are local and do not replace shot failure. Independently
windowed weak-only and strong-only runs see different predecessor boundaries,
so per-shot union bounds across them are not claimed. Whole-shot decoding is
the paper's accuracy reference, not an equality oracle for windows.
