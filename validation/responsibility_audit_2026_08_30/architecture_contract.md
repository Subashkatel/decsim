# Architecture contract: the eight owners at runtime

The full runtime walkthrough lives in
`docs/architecture/RUNTIME_FLOW.md`; the strong-request state machine in
`docs/architecture/STRONG_REQUEST_LIFECYCLE.md`. This file records the
ownership rules a refactor may not break.

## Ownership

- ExecutionRuntime owns the program DAG, readiness, timestamps;
  ResourceLedger owns one-holder-per-resource.
- Controller owns the op-to-QPU command seam, readout-to-packing relay,
  and the OC then CQ instruction relay. It never decides readiness.
- SyndromePacking owns fragment reassembly, detector formation, the
  dual write into the stores, and route arbitration (CWB or WBD).
- Buffer 0 (SyndromeBuffer) owns upstream retention and refcounted
  holds; SyndromeBuffer1 owns the CSB crossing, room-side retention, and
  the stored-round arrival gate.
- WindowManager owns window lifecycle, readiness interpretation, commit
  and buffer regions, boundary couriering, retention holds, and results.
- StrongEscalation (constructed with the WindowManager, the one object
  allowed to work its tables directly) owns strong-job building,
  deferred submission, and the StrongRequestLedger.
- DecoderManager owns pools, two-slot decoupled access-execute units,
  DMA staging, parking, and dispatch. It never bypasses the Controller
  for feedback.
- PauliFrame owns the priced-write XOR frame; exactly one authoritative
  correction per window reaches it.
- ConditionalRelease owns the SWIPER release rule: a blocked successor
  releases on the final decode of its blocker, over OC then CQ through
  the Controller.

## The strong-start gate

Strong compute begins only when ALL hold: request live and of the
current generation (StrongRequestLedger; stale results raise), WSD
selection reserved where the mode requires it, every required SB1 round
stored (three exact checks, see buffer_contract.md), a strong unit
assigned (DecoderManager), SBD transfer landed in that unit's memory
(input_landed), and the window boundary gate satisfied (service_gate,
seam mask applied exactly once at true service start). No single
"Buffer 1 ready" signal may start compute.

## Messaging rules

- Notifications announce availability; they never carry the payload
  (SB1's on_round_stored carries an operation id only).
- Readiness interpretation is WindowManager's and StrongEscalation's;
  hardware assignment is DecoderManager's; payload movement is SBD or
  CWB via the priced links.
- Feedback travels ConditionalRelease to Controller (OC) to QPU or
  runtime (CQ); DecoderManager cannot deliver a release itself.
