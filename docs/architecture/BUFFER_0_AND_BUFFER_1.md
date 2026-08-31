# Buffer 0 and syndrome buffer 1

Two stores, two readiness authorities. The invariant form of this
document is `validation/responsibility_audit_2026_08_30/buffer_contract.md`.
All anchors verified at commit 65660a0.

## Buffer 0 (upstream store, weak lane)

| Question | Answer |
|---|---|
| When written | `SyndromePacking._finish_packing` at round completion (syndrome_packing.py:270-271) |
| Which link charged | CWB when priced (reserve at :372, publication marked at CWB landing :382-384); WBD for feedback-memory rounds |
| When readable | publication tick: completion tick, or CWB landing when CWB is priced (:258-259, :383) |
| Notification | `accept_window_input(packet)` to the WindowManager (:388) |
| Who receives | `WindowManager.on_syndrome_arrival` (window_manager.py:528-548) |
| Readiness change | `_count_arrival` advances `rounds_arrived[op]`; ONLY when the primary tier is not STRONG (:537-542) |
| Publication before notification | ENFORCED: `_store_payload` raises unless the packet is retained and its publication tick is now (:584-591) |
| Who retains | refcounted holds; planned weak holds registered at plan load (:277-281) |
| Who releases | decoder input landing (`DecoderInputStaging.stage` land(), decoder_memory_transfer.py:27-34), WBD delivery (syndrome_packing.py:419-423), or drop-on-arrival when unheld (window_manager.py:547) |
| To decoder memory | CWB-published rounds are read at decode admission and DMAd per `reserve_transfer` at dispatch |
| Round order | kept explicitly on CWB: FIFO route queue, a refused round blocks those behind it (:341-352, :385-401) |

## Syndrome buffer 1 (room-side store, strong tier)

| Question | Answer |
|---|---|
| When written | dual write in `_finish_packing` (syndrome_packing.py:291-296), exactly once per round |
| Which link charged | CSB when priced; capacity checked before the reservation, counting in-flight writes (syndrome_buffer_1.py:59-67) |
| When readable | after `_store` (accept at landing, publication tick = landing tick, :91-97) |
| Notification | `on_round_stored(operation_id)` strictly AFTER storage (:109-110); the payload is never carried |
| Who receives | `WindowManager._on_room_round_stored` (window_manager.py:608-618) |
| Readiness change | `escalation.after_arrival(op_id)` in every mode; `_count_arrival` from `sb1.rounds_arrived` only when the strong tier is primary |
| Who retains | PotentialStrong / PendingStrong holds, transferred to CsdInput at strong submission (strong_escalation.py:489-501) |
| Who releases | SBD landing releases the CsdInput hold; cancels release idempotently (decoder_memory_transfer.py:39-47) |
| To decoder memory | `reserve_transfer` clamps the SBD DMA to max(link arrival, WSD arrival, `sb1.ready_tick(context)`) (strong_escalation.py:507-514) |

`rounds_arrived` on both stores advances with max(previous,
round_index). The ordered-arrival argument (Proof A) and the fail-loud
exactness backstop (Proof B) are in `buffer_contract.md`; the
deterministic tests pin both
(`tests/16_stabilization/test_buffer_readiness.py`).

## Mode diagrams

### Weak-only baseline

```
packing --CWB--> Buffer 0 --accept_window_input--> WindowManager --> weak decode
                 (SB1 is not constructed: run_spec.py:209-215)
```

### Strong-only baseline

```
packing --+--> Buffer 0 (drop-on-arrival: no weak consumers hold rounds)
          |
          +--CSB--> SB1 --on_round_stored--> WindowManager readiness
                     |                        (primary_store = SB1)
                     +------SBD (DMA at dispatch)-----> strong unit memory
```
Buffer 0 arrival does NOT advance readiness (window_manager.py:537).
Primary window jobs read SB1 and price SBD.

### Weak-primary serial switching

```
packing --+--CWB--> Buffer 0 --> WindowManager --> weak decode --> trusted? done
          |                                            |
          |                                       untrusted
          |                                            v
          +--CSB--> SB1 <---------------- make_strong_decode_job reads SB1 NOW
                     |     context at Buffer 0 but not in SB1?
                     |     RuntimeError "csb lag beyond the escalation margin"
                     |     (strong_escalation.py:598-607)  FAIL LOUD, never wait
                     +--SBD--> strong unit memory
```
Replacing the fail-loud check with waiting is an architectural change
that needs its own design note and timing validation.

### Parallel switching

```
weak job ----> Buffer 0 path (readiness, weak decode)
strong job --> submitted alongside (submit_strong); SBD DMA start
               clamps to sb1.ready_tick, so Buffer 0 readiness ahead of
               Buffer 1 delays the DMA, never reads unstored data.
trusted weak -> cancel_strong: queued, crossing, running, or held
               requests all cancel idempotently.
```

### Double-window terminal escalation

```
register_terminal(pending)          every SB1 store wakes after_arrival
        |                           (strong_escalation.py:1437-1447)
        v
stored-through >= context_hi  -->  _submit_terminal_strong, exactly once
                                   (single-shot registry take :380-410;
                                    exact round-set check :1380-1388)
```
A second trigger inside `prepare_strong_selection` (:539-543) covers the
WSD reservation arriving after the data was already complete.

### Double-window far-boundary escalation

```
after_weak_commit(far key) --> _submit_far_strong
```
The trigger is the far weak boundary, not an SB1 callback
(:469-474); the WSD reservation must already exist (:1405-1406), and
the SBD DMA still gates on `ready_tick`.
