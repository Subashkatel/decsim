# Runtime flow

Verified against decsim at commit 65660a0. Link segment names:
`TERMINOLOGY.md`. Store details: `BUFFER_0_AND_BUFFER_1.md`. Strong
request states: `STRONG_REQUEST_LIFECYCLE.md`.

```
ExecutionRuntime (program sequencer)
        |  operation ready (_attempt_start -> _maybe_begin)
        v
Controller.issue_operation
        |  RunOperationBody
        v
QPUDevice (one cycle clock; every live patch emits every cycle)
        |  QPUReadout fragments            [QC]
        v
Controller.accept_qpu_readout (+ t_binary_availability)
        |  SyndromePayload
        v
SyndromePacking (PARTIAL -> t_pack -> complete)
        |  merge fragments, form detection events, then the DUAL WRITE
        +----------------------------------------------+
        |                                              |
        v                                              v
Buffer 0 (SyndromeBuffer)                    SyndromeBuffer1
  accept_packed_round at completion            write(): [CSB] then _store
        |                                              |
        |  route arbitration                           |  on_round_stored(op_id)
        |  [CWB] window input, in round order          |  (strictly after storage)
        |  [WBD] feedback-memory round                 v
        v                                     WindowManager._on_room_round_stored
WindowManager.accept_window_input               escalation.after_arrival always;
  publication verified, then                    readiness advance only when the
  _count_arrival (weak-primary)                 strong tier is primary
        |
        v
check_window: data complete? deps? -> escalation policy submits
        |
        v
DecoderManager.enqueue (StrongRequestLedger admits weak/strong)
        |  dispatch: startable first, unit assigned at DMA start
        v
DecoderInputStaging.stage: reserve_transfer() -> transport -> deposit
        |  input_landed (Buffer 0 / SB1 hold released here)
        v
_begin_service: service_gate may PARK (boundary owed); seam mask XORed
exactly once at true start; then the tier's decoder runs
        |
        +-- weak result trusted --------------------------------+
        |                                                       |
        |  weak result untrusted: strong request                |
        |  (serial: build now, fail-loud csb-lag check;         |
        |   parallel: submitted alongside; double-window:       |
        |   deferred to terminal data or the far boundary)      |
        |          [WSD] selection, [SBD] strong input DMA      |
        v                                                       v
StrongRequestLedger matches completion to the live request generation
        |  stale or unconsumable strong results raise           |
        v                                                       v
WindowManager.on_decode_done / on_strong_decode_done: commit windows,
courier boundaries [DD], release retention, record results
        |  authoritative correction (one per window)   [WDO/DO]
        v
PauliFrame (XOR accumulate; priced write)
        |
        v
ConditionalRelease: blocked successor? release on FINAL decode
        |  Decision                                   [OC]
        v
Controller.relay_instruction
        |                                             [CQ]
        v
ExecutionRuntime.on_decision -> decode_release_time -> _maybe_begin
(the blocked successor starts; QPU commanded on the next boundary)
```

## Notes that keep this honest

- Syndrome extraction never stops: idle patches emit every cycle; the
  idle policy routes those rounds ([WBD] feedback-memory rounds, and
  the default policy charges one load-only decode per commit region).
- Neither store's notification performs decoding. WindowManager and
  StrongEscalation interpret readiness; DecoderManager assigns
  hardware; SBD moves the selected SB1 context into decoder-local
  memory; only then may strong decoding begin (the full gate is the
  conjunction in `STRONG_REQUEST_LIFECYCLE.md`).
- DecoderManager cannot bypass the Controller: every release travels
  ConditionalRelease to Controller over OC, then CQ to the QPU side.
- Units are depth-1 decoupled access-execute machines: two input
  slots so the next window's DMA overlaps the current compute; compute
  is claimed separately (a parked job keeps its slot, never the
  compute), so a dependent can never deadlock a unit against its own
  predecessor.
