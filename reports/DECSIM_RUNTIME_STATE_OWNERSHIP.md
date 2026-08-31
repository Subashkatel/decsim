# Runtime state ownership

Every mutable field of the runtime owners, with who creates, writes,
and reads it, the intended owner, the invariant, and a verdict.
Verified at the stabilization commits following 65660a0 (after the five
safe cleanups). Verdicts: KEEP, DOCUMENT DELEGATION (a deliberate
cross-owner access, recorded, not a defect), DELETE (already executed
in the cleanup pass where noted). No RENAME, MOVE, SPLIT, or MERGE is
recommended; no contradictory-owner defect was found.

Shorthand: WM = WindowManager, DM = DecoderManager, SE =
StrongEscalation, SP = SyndromePacking, ER = ExecutionRuntime,
SB0 = SyndromeBuffer, SB1 = SyndromeBuffer1, CR = ConditionalRelease,
PF = PauliFrame.

## Window (decsim/message.py:245-291)

A Window is created by the planner (static plan) or DynamicWindows
(streams) and thereafter owned by WM; SE builds private strong Windows.

| Field | Created by | Writers | Readers | Owner | Invariant | Verdict |
|---|---|---|---|---|---|---|
| op_id, k, commit_lo, buffer_lo | planner / DynamicWindows / SE | set at creation; commit_hi, buffer_hi, n_rounds re-sliced by WM for dynamic tails (trim_dynamic_window_tail) and SE restarts | WM, DM, SE, metrics | WM | ranges 1-based inclusive; reads span [start_round, buffer_hi] | KEEP |
| deps, dependents, deps_remaining | planner / WM | WM (courier merges), SE (absorb) | WM, DM (_startable) | WM | deps_remaining is 0 exactly when every dep boundary merged | KEEP |
| queued, committed, service_began | Window default | WM (submit/commit), DM (service) | WM, DM | WM | a window is queued once and committed once | KEEP |
| boundary_in | WM at install/creation | WindowInteraction via WM | WM, DM (apply_service_boundary) | the configured WindowInteraction | seam mask applied exactly once, at true service start | DOCUMENT DELEGATION |
| decode_status | Window default | WM at commit | frozen suite, views | WM | None means the decode succeeded | KEEP |
| t_first_round, t_data_complete, t_queued, t_dispatch, t_done | Window default | WM (t_first_round, t_data_complete), DM (t_dispatch), WM/DM at done | metrics, frozen suite | WM for data ticks, DM for service ticks | each stamped at most once | DOCUMENT DELEGATION (t_dispatch written by DM at :607-608) |
| buffer_filled_by_memory, batched_preceding_idle_round_count, blocked_logged | Window default | WM | WM, metrics | WM | memory-filled release is flagged, never hidden | KEEP |

## DecodeJob (decsim/message.py:529-570)

A DecodeJob is a shared work record: WM/SE create it, DM and the
staging layer advance it. The phase flags make the shared writes safe.

| Field | Created by | Writers | Readers | Owner | Invariant | Verdict |
|---|---|---|---|---|---|---|
| identity (op_id, window_id, n_rounds, dem, code, spatial_nodes, hint, label) | WM / SE builder | set at build | DM, decoders | creator | immutable after build | KEEP |
| payloads -> decoder_input, memory | WM/SE (payloads) | staging land(): deposit clears payloads, sets decoder_input, memory | decoders | DecoderInputStaging | a decoder reads only its unit's memory | KEEP |
| input_hold | WM (_bind_decoder_input_hold) | staging (released at landing or cancel, idempotent) | staging | the store hold system | the upstream rounds outlive the transfer | KEEP |
| reserve_transfer | SE/WM at submit | staging consumes it once (set to None) | staging | submitter | called after unit assignment, never before | KEEP |
| unit, pool | None | DM at dispatch | DM, records | DM | assigned at DMA start | KEEP |
| submitted, cancelled, completed, input_landed, service_started, awaiting_strong_result | defaults | DM (and cancel paths) | DM, ledger, WM | DM | _reject_spent_job refuses reuse; cancelled is checked before service | KEEP |
| request_key, request_created_ticks, request_admitted_ticks, service_key, service_original_request_keys, service_dispatch_ticks | WM/SE at build, DM at admit/dispatch | DM | ledger, records | DM + ledger | one request key per admission; service keys stamped at dispatch | KEEP |

## SyndromeBuffer / Buffer 0 (syndrome_buffer.py:128-155)

| Field | Created by | Writers | Readers | Owner | Invariant | Verdict |
|---|---|---|---|---|---|---|
| _slots, _free_slot_indices, _rounds, _publication_ticks | ctor | SB0 only | via has/retained/publication accessors | SB0 | one allocation per retained round; publication tick set once | KEEP |
| _open_operations, _closed_operations, _tombstones | ctor | SB0 (open_operation, close_operation) | SB0 | SB0 | a closed identity never reopens | KEEP |
| _live_holds, _released_holds, _holders_by_round | ctor | SB0 (register/replace/transfer/release) | SB0 | SB0 | holds reference open operations only; duplicate holder tokens refused | KEEP |
| payloads_held, peak_payloads, counters | ctor | SB0 | metrics | SB0 | monotone peaks | KEEP |

## SyndromeBuffer1 (syndrome_buffer_1.py:25-110)

| Field | Created by | Writers | Readers | Owner | Invariant | Verdict |
|---|---|---|---|---|---|---|
| store (composed SyndromeBuffer) | ctor | SB1 methods only | via SB1 delegation | SB1 | buffer semantics have one owner: the composed store | KEEP |
| rounds_arrived | ctor | SB1._store only | WM (_on_room_round_stored, :615), SE (after_arrival, prepare_strong_selection) | SB1 | stored-through: max counter, sound under ordered arrival; every consumer re-verifies exact rounds (ready_tick, exact set checks) | KEEP; DOCUMENT DELEGATION for the WM/SE reads (the documented callback contract) |
| _in_flight_writes, _written | ctor | SB1.write / land | check_settled | SB1 | capacity counts in-flight writes; one write per round identity ever | KEEP |
| on_round_stored | ctor (None) | WM wires it once (window_manager.py:99) | SB1 fires after storage | WM supplies, SB1 invokes | fires strictly after storage; carries the id only | KEEP |
| copied_bits_total | ctor | SB1 | metrics | SB1 | counts every csb payload once | KEEP |

## WindowManager (window_manager.py:37-132)

| Field | Created by | Writers | Readers | Owner | Invariant | Verdict |
|---|---|---|---|---|---|---|
| _ops, rounds_arrived, memory_rounds | ctor | WM (register_op, _count_arrival, on_memory_round) | WM, SE | WM | registration is idempotent (guard :143); rounds_arrived is Buffer 0's stored-through in weak-primary runs | KEEP |
| windows, op_windows, window_count, successors, total_windows, _windowed_by_operation, _batch_preceding_idle_rounds_by_operation, _protocol_by_operation | plan (load_execution_plan) / DynamicWindows | WM; SE re-slices restart windows | WM, DM via services | WM | installed once (_windows_built latch) | KEEP |
| committed_windows, _committed_per_op | ctor | WM; SE (absorb/rollback paths) | WM, DynamicWindows | WM | one commit per window | DOCUMENT DELEGATION (SE is the named exception) |
| blocking_ops, op_results, segment_results_sent | ctor | WM | run_spec capture, run_views | WM | op_results written once per op | KEEP |
| _required_stream_end_by_operation_id, _stream_binding_by_operation_id | ctor | WM (stream close bookkeeping) | WM | WM | | KEEP |
| ledger (LogicalLedger), courier (BoundaryCourier), lifecycle (DynamicWindows) | ctor | their own methods; SE swaps ledger.contributions in its atomic sections | WM, SE | each sub-owner | courier ships a boundary once; ledger contributions form a partition | KEEP |
| _pending_strong_windows, _pending_strong_per_op, absorbed_windows, op_strong_commit_time, _finished_ops, _workload_complete_sent | ctor | WM (SE via its documented access) | WM, boundaries, run_views | WM | absorbed windows are skipped by the weak chain exactly once | KEEP |
| window_models | ctor | WM (_build_window_error_models, dynamic creation) | routers/devices via services | WM | one model per windowed key | KEEP |
| memory_rounds_total, memory_filled_buffer_windows | ctor | WM | metrics | WM | counters only | KEEP |
| submit_fn, withdraw_decode, release_service, _idle_decode_demand_receiver, on_workload_complete | run wiring | run_spec wires once | WM calls | run composition root | the WM/DM seam is these closures | DOCUMENT DELEGATION |
| escalation (StrongEscalation or NoStrongTier) | ctor | SE owns its registry | WM, DM (services) | SE | SE is the one object allowed to work WM tables directly (class docstring) | DOCUMENT DELEGATION |
| _next_decoder_request_sequence | ctor | WM (_new_request_key) | request keys | WM | strictly increasing run sequence | KEEP |
| _code_geometry | ctor | none | tests (observability handle) | WM | resolved plan geometry | KEEP |

## DecoderManager (decoder_manager.py:91-179)

| Field | Created by | Writers | Readers | Owner | Invariant | Verdict |
|---|---|---|---|---|---|---|
| ready, pool_ready, queue_log | ctor | DM | metrics, frozen suite | DM | queue_log appends at every transition | KEEP |
| unit_totals, pool_free, _free_units, decoder_memories | ctor | DM | DM | DM | pool_free mirrors _free_units lengths | KEEP |
| _unit_residents, _computing, _parked_service | ctor | DM | DM | DM | at most two residents; compute held by at most one; a parked job keeps its slot, never compute | KEEP |
| staging (DecoderInputStaging) | ctor | staging | DM | staging | cancel and release are idempotent | KEEP |
| strong (StrongRequestLedger) | ctor | ledger methods | DM, snapshots | ledger | see below | KEEP |
| _gap_joins, gap_split_enabled | ctor | DM | DM | DM | a join concludes its window exactly once | KEEP |
| _terminal_request_records, _terminal_service_records | ctor (capture only) | DM | capture views | DM | None when capture is off | KEEP |

## StrongRequestLedger (strong_escalation.py:58-118)

| Field | Created by | Writers | Readers | Owner | Invariant | Verdict |
|---|---|---|---|---|---|---|
| _running | ctor | admit_strong, register_batch, take_live, finish_service | DM, snapshots | ledger | at most one unconsumed strong result per destination | KEEP |
| _unresolved_weak | ctor | admit_weak, resolve_weak | destination_may_consume | ledger | a destination decodes once at a time | KEEP |
| _waiting_selection, _waiting_result | ctor | begin_selection, select, complete | ledger | ledger | select matches the exact request key; stale completions raise | KEEP |
| _completed | ctor | complete, take_held | select | ledger | held results have a registered demand or the run fails | KEEP |
| strong_needed, strong_cancelled | ctor | ledger, cancel paths | frozen suite | ledger | counters | KEEP |

## StrongEscalation (strong_escalation.py:399-...)

| Field | Created by | Writers | Readers | Owner | Invariant | Verdict |
|---|---|---|---|---|---|---|
| wm | ctor | never | SE | WM reference | SE is constructed with the manager; the single documented cross-owner access | DOCUMENT DELEGATION |
| _escalations (_EscalationRegistry) | ctor | register_far/terminal, update_wsd_arrival, take_* | after_arrival, after_weak_commit, prepare_strong_selection | SE | one escalation per window; takes are phase-checked and single-shot; duplicate WSD reservation refused | KEEP |

## PauliFrame (pauli_frame.py:118-129)

| Field | Created by | Writers | Readers | Owner | Invariant | Verdict |
|---|---|---|---|---|---|---|
| _accepted_window_keys | ctor | commit_weak_correction | duplicate gate | PF | one accepted write per window key | KEEP |
| _pending_by_window_key | ctor | commit / _install_accepted_write | install | PF | every accepted write installs exactly once after its priced delay | KEEP |
| _entry_by_window_key, _window_keys_by_stream, _records | ctor | install / commit | frame_for, snapshot | PF | records in acceptance order | KEEP |
| _duplicate_drops | ctor | commit | snapshot | PF | drops charge no write cost | KEEP |
| PauliFrameSnapshot.window_count | | | none | | duplicated commit_count | DELETE (executed in cleanup 4) |

## ConditionalRelease (conditional_release.py:17-31)

| Field | Created by | Writers | Readers | Owner | Invariant | Verdict |
|---|---|---|---|---|---|---|
| blocked_by_index | ctor | register_blocked_operation, popped by on_result | on_result | CR | each blocker's list is consumed exactly once, at its final result | KEEP |
| controller, decision_sink | connect() | run wiring | integrate | CR | decisions travel controller OC then CQ; never direct | KEEP |

## Controller (controller/controller.py:18-38, post-cleanup)

| Field | Created by | Writers | Readers | Owner | Invariant | Verdict |
|---|---|---|---|---|---|---|
| runtime | connect_runtime | run wiring | controller relays | Controller | wired once before load_program | KEEP |
| idle_rounds_emitted | ctor | emit_idle_round | frozen suite | Controller | counts every relayed idle round | KEEP |
| round_ticks, _code_geometry | | | none in core | | never read | DELETE (executed in cleanup 2; tests re-pointed at qpu.cycle_ticks and the WM geometry) |
| _resolved_operations, _resolved_patches | ctor | never (MappingProxyType) | round/count lookups, idle decode demand | Controller | immutable views | KEEP |

## ExecutionRuntime + ResourceLedger (execution_runtime.py:16-93)

| Field | Created by | Writers | Readers | Owner | Invariant | Verdict |
|---|---|---|---|---|---|---|
| operations, dependencies_remaining, successors | load_program | ER | ER, controller | ER | dependencies_remaining hits zero exactly once per op | KEEP |
| schedule_released, requested, state_ready | ctor | ER | ER | ER | requested guards double resource claims | KEEP |
| op_start_time | ctor | _maybe_begin (provisional stamp, then the QPU boundary) | body_done guard, metrics, frozen suite | ER | the provisional stamp closes the reentry window during issue_operation | KEEP (comment added in cleanup 5) |
| body_done_time, decode_release_time, result_return_time_by_operation, last_finish_time | ctor | ER (body_done, on_decision) | workload_complete, capture | ER | one release per blocked op (double release raises) | KEEP |
| idle_rounds_by_patch | ctor | record_idle_round, consume_idle_rounds (pop) | ER | ER | consumed by the next operation on the patch | KEEP |
| ResourceLedger.busy_claims | ctor | claim, release | claim conflicts | ledger | one holder per resource; conflicting claims raise with the missing-edge message | KEEP |

## Findings

- No contradictory state owner was found. The three deliberate
  delegations (SE over WM tables; DM stamping Window.t_dispatch; the
  run-wired WM/DM closures) are documented above and in
  `docs/architecture/STATE_OWNERSHIP.md`.
- The audit's "never read" claim for Controller.round_ticks and
  Controller._code_geometry was corrected during cleanup: core never
  read them, but three test files used them as observability handles;
  those tests now read the owning components (qpu.cycle_ticks,
  window_manager._code_geometry) with unchanged assertion strength.
- No new component is proposed; file length alone is not a reason to
  split (WindowManager stays one owner).
