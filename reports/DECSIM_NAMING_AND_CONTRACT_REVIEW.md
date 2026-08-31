# Naming and contract review

What the stabilization pass found worth flagging about names and
contracts. Nothing here was changed; renames are the owner's call
(standing rule: flag, never rename unasked).

## Names flagged for the owner

1. `RequestProcessingOutcome.WEAK_FORWARDED_FOR_DELIVERY`
   (decoder_manager.py:22). Carried over from the audit: the value
   reads as a transport state but records a terminal outcome ("the
   weak result was accepted as final and left for delivery"). If a
   rename is ever wanted, something like WEAK_ACCEPTED_FINAL would say
   it; cosmetic only.
2. `PauliFrame.commit_weak_correction` (pauli_frame.py) also commits
   STRONG finals (window_manager._commit_strong_decode_done calls it
   with a strong request key; the record's tier says so). The method
   name undersells its contract ("commit the window's FINAL
   correction"). Behavior is correct; the docstring inside the strong
   path explains it. Flag only.
3. `SyndromeBuffer1.rounds_arrived` and `WindowManager.rounds_arrived`
   share a name and a max() convention but different authorities
   (room-side stored-through vs Buffer 0 publication count). The
   terminology page now defines "stored-through"; no rename proposed.

## Contract sharp edges, recorded

1. Bare KeyError in far-boundary deferral when the restart window's
   Buffer 0 hold is no longer live (strong_escalation.py:716); the
   sibling condition a few lines earlier raises the proper
   "requires retained payload rounds that are no longer available"
   message. Suggest extending the contract message to this case when
   the double-window lane is next touched; not a behavior change.
2. Parallel switching (`run_both_at_once`) requires the CSB margin at
   weak-window readiness; on cards where csb is slower than the
   Buffer 0 publication path it refuses loudly at strong-job build.
   Pinned by `test_parallel_requires_the_csb_margin`; worth one line in
   guide/parameter-reference.md when parallel is next configured.
3. `PackingOverflowPolicy.DROP_ROUND` (opt-in) creates permanent gaps
   under the stored-through counters; downstream exact checks fail
   loudly rather than silently, but a windowed run with drops will
   crash at decode admission rather than skip. FAIL_STOP stays the
   right default.
4. The `NotImplementedError` refusal for crossing strong windows
   (strong_escalation.py:791) stays; its unreachable body was deleted
   in the cleanup pass, so the refusal is now the whole method.

## Contracts confirmed clean

- The two-store notification contracts (id-only callbacks, publication
  before notification) match their documentation exactly.
- The strong-start gate is a true conjunction; no single signal starts
  compute (mutation matrix M06 to M18).
- DecodeJob's shared-record writes are phase-flagged and single-writer
  per phase; no contradictory owner found (see
  DECSIM_RUNTIME_STATE_OWNERSHIP.md).
