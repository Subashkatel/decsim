# Strong request lifecycle

Every state of one strong request, with its owner, data structure,
transition, link, and buffer hold. Anchors verified at commit 65660a0.
Files: `decsim/decoders/strong_escalation.py` (SE),
`decsim/decoders/decoder_manager.py` (DM),
`decsim/decoders/decoder_memory_transfer.py` (DMT),
`decsim/windows/window_manager.py` (WM).

| State | Owner | Data structure | Enters via | Link | Buffer hold |
|---|---|---|---|---|---|
| potential strong context | WM plan load | SB1 hold `PotentialStrong(window_key)` | `_register_planned_holds` (WM:277-281) or dynamic window creation (WM:328) | CSB retains context as it lands | PotentialStrong on SB1 |
| pending strong request (double-window) | SE registry | `_PendingEscalation` in `_EscalationRegistry` (SE:288-340) | `register_terminal` / `register_far` | none yet | `PendingStrong(request_key)` on SB1 (SE:751-757) |
| WSD selection in flight | SE + DM | `wsd_arrival_ticks` on the pending record; `_waiting_selection` in the ledger | `prepare_strong_selection` reserves WSD (SE:528-537, 545-556); `begin_selection` (DM:951) | WSD | unchanged |
| strong request admitted | StrongRequestLedger | `_running[destination] = LiveStrongRequest` | `DM.enqueue` -> `admit_strong` (DM:222-223, SE ledger:70-78); duplicate destination raises | none | PotentialStrong or PendingStrong transferred to `CsdInput(request_key)` at submission (SE:489-501) |
| queued | DM | pool ready queue | `_enqueue_now` (DM:242-258); a request cancelled across the link is dropped here (DM:244-249) | none | CsdInput on SB1 |
| unit assigned | DM | `_unit_residents[(pool, unit)]` | `_dispatch_pool` -> `_start_job` (DM:461-488, 585-616); unit assigned at DMA start | none | CsdInput on SB1 |
| SBD in flight | DMT staging | transport delivery | `staging.stage` calls the job's `reserve_transfer`: max(SBD arrival, WSD arrival, `sb1.ready_tick(context)`) (SE:507-514; DMT:20-38) | SBD | CsdInput until landing |
| input landed | DM | `job.input_landed`; rounds in `DecoderMemory` | `landed()` when every member landed (DM:625-652); hold released in `land()` (DMT:27-34) | none | released; rounds now in unit memory |
| running | DM | `_computing[(pool, unit)]`; `job.service_started` | `_begin_service`: service_gate may park; seam mask applied exactly once (DM:654-674) | none | unit memory |
| completion held | StrongRequestLedger | `_completed[destination] = HeldStrongCompletion` | `ledger.complete` when the destination's demand is not yet selected (:158-177); an unconsumable result RAISES | none | none |
| completion selected | StrongRequestLedger | `_waiting_result` | `_select_strong_result` after WSD delivery (DM:969-974); `ledger.select` matches the request key (:146-156) | (WSD already paid) | none |
| final delivered | WM | `on_strong_window_decoded` -> `WM.on_strong_decode_done` | `_complete_strong_result` (DM:999-1007) | DO toward the Pauli frame | none |
| cancelled | DM | counters + records | `cancel_strong(key)`: queued, crossing the link, running, or held all cancel idempotently (DM:288-360; DMT:39-47) | none | released idempotently |

## Settledness

The run refuses to end with strong work in flight:
`window_manager.escalation.pending_escalations` must be empty
(run_spec.py:347-350), `decoder_manager.check_decode_work_settled()`
runs the ledger's `unsettled()` categories (waiting for a result,
waiting for selection, holding an unclaimed result, still holding a
request, decoding with no outcome), and `syndrome_buffer_1.check_settled()`
refuses in-flight CSB writes, live rounds, or unresolved holds.

## Generation correctness

- One destination has at most one unconsumed strong result:
  `admit_strong` raises on a duplicate (ledger:70-78).
- One destination decodes once at a time: `admit_weak` raises on a
  second unresolved weak decode (ledger:80-89).
- A strong result arriving after a newer request took the destination
  raises ("nothing would consume this one", ledger:166-170); a result
  with no registered demand raises (ledger:171-175).
- The escalation registry take is single-shot and phase-checked
  (SE:392-410), so a terminal or far submission can never fire twice.

## The strong-start gate, in one line

RequestLive AND CorrectRequestGeneration AND SelectionReady (where the
mode requires WSD) AND RequiredBuffer1RoundsStored (three exact checks:
`ready_tick`, the exact round-set equality at build, the serial csb-lag
check) AND StrongUnitAssigned AND SBDComplete AND DecoderInputLanded
AND WindowBoundaryGateSatisfied. No single "Buffer 1 ready" signal may
start strong compute; `after_arrival` only triggers submission
attempts.
