# Buffer readiness audit

The commissioned questions about Buffer 0 and syndrome buffer 1, with
evidence. The contract lives in
`validation/responsibility_audit_2026_08_30/buffer_contract.md`; the
per-mode diagrams in `docs/architecture/BUFFER_0_AND_BUFFER_1.md`; the
pinned behaviors in `tests/21_stabilization/test_buffer_readiness.py`.

## The callback contract, documented exactly

`SyndromeBuffer1.write` crosses CSB, `_store` retains the round, THEN
`on_round_stored(operation_id)` fires (syndrome_buffer_1.py:91-110).
The callback carries the operation id only; WindowManager reads
`syndrome_buffer_1.rounds_arrived[op]` as "stored through". It was not
changed into a packet callback.

## The ordered-arrival question

Proof A (strict in-order arrival) holds for every supported
configuration, by this chain:

1. one QPU cycle clock; per-operation round index strictly increases,
   all fragments of a round emitted at one boundary in fragment order
   (qpu/cycle_clock.py:80-157);
2. links are strict FIFOs: `Link.reserve` rejects a send tick earlier
   than the prior reservation and serializes on a monotone
   next-free-tick; an unbounded channel prices constant propagation
   (links/links.py:417-474);
3. the engine fires same-tick events in insertion order
   (engine.py:14-47);
4. controller binary availability is one constant per-fragment delay
   (syndrome_packing.py:172-179);
5. per-operation fragment counts are constant across rounds in every
   shipped device model, so the uniform t_pack (applied only when a
   round has more than one fragment) preserves completion order;
6. Buffer 0 CWB keeps explicit round order under backpressure
   (syndrome_packing.py:341-352, 385-401); SB1 writes issue in
   completion order and land FIFO over CSB.

Boundary of Proof A, recorded: a device model emitting varying
fragment counts per round of one operation with t_pack > 0, or the
opt-in DROP_ROUND overflow policy, can produce a gap under the max
counters. No shipped configuration does either.

Proof B (exact readiness makes any gap harmless) also holds, and is
what actually protects correctness: the counters only decide WHEN a
submission is attempted; every consumer verifies exact rounds:

- `SyndromeBuffer1.ready_tick` raises for any unstored round and runs
  inside every strong `reserve_transfer` (strong_escalation.py:507-514);
- the deferred strong build checks covered == needed exactly
  (strong_escalation.py, `_build_pending_strong_job`);
- the immediate serial build raises "csb lag beyond the escalation
  margin" for context at Buffer 0 but missing in SB1;
- `_require_retained_payloads` refuses consumers of released rounds.

Pinned: `test_sb1_gap_cannot_be_served` (a written round 3 over a
missing round 2 advances the counter to 3, and the exact read of round
2 refuses); mutation M08 (counter advanced across a gap) is detected.

Given both proofs, the max counter STAYS (no replacement with a
contiguous-prefix tracker); the decision is recorded here, and the
fail-loud checks are pinned by tests.

## CSB lag behavior, by switching mode

- Serial: the immediate strong build FAILS LOUDLY when context reached
  Buffer 0 but is not stored in SB1 ("csb lag beyond the escalation
  margin"). Under the declared ticks the weak decode (wbd 5 + weak 10)
  covers the csb margin (7 vs 4), so serial runs never trip it.
- Parallel (`run_both_at_once`): the strong job is built AT weak-window
  readiness, so the two-sided context must already be stored then:
  csb lag beyond the cwb path trips the SAME fail-loud check at build
  time. This is stricter than a DMA-time wait and is pinned by
  `test_parallel_requires_the_csb_margin`. The `ready_tick` clamp in
  `reserve_transfer` remains as the belt-and-braces gate after a
  successful build. Neither check was converted into waiting; a
  wait-based design would need its own note and timing validation.
- Double-window terminal: submission waits for stored-through to reach
  context_hi; the exact round-set check still guards the build.
- Double-window far boundary: submission waits for the far weak
  commit; SBD still gates on `ready_tick`.

## Findings that need the owner's eye

1. Far-boundary deferral assumes the restart window's Buffer 0 hold is
   still live. In a backlog regime (data ahead of the weak chain, e.g.
   1.0 us rounds with the declared ticks and rounds >= 15), the weak
   chain has already consumed the restart window's rounds by escalation
   time, and `defer_strong_escalation` fails: first with the contract
   message "strong-region plan ... requires retained payload rounds
   that are no longer available" (rounds released after landing), and
   in a narrower configuration with a bare KeyError from
   `hold_round_identities(restart_key)` (strong_escalation.py:716)
   when the restart window's own hold is gone. The refusal is correct;
   the bare KeyError deserves the contract message. No change was made
   (it is behavior); flagged for a decision.
2. Parallel switching is only usable on cards where the CSB margin
   holds (csb latency at most the Buffer 0 publication path). Worth a
   one-line note in the parameter reference when parallel runs are
   next configured.

## Retention and release, verified

Buffer 0 rounds stay held until the decoder input lands (the staging
`land()` releases the hold), WBD delivery releases feedback-memory
rounds, and unheld rounds drop on arrival. SB1 rounds stay held from
the plan's potential holds through PendingStrong to CsdInput, released
at SBD landing or idempotent cancel. Both stores refuse to end a run
non-empty (`check_settled`, `test_stores_settle_empty`).
