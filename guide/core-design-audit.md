# Core design audit and refactor plan

Date 2026-08-18. Read-only audit of every core module against three sources
and one debate: Ousterhout, A Philosophy of Software Design; Rhodes,
python-patterns.guide; tef, "Write code that is easy to delete"; and the
Ousterhout vs Martin exchange (aposd-vs-clean-code). Nothing in this document
has been applied. Every finding cites file:line at HEAD 5299966.

## 1. Principles as applied here

- Deep modules, not many modules. Split only where a lump hides its own state
  behind a small interface. Never split for length. (Ousterhout ch. 4; both
  sides of aposd-vs-clean-code agree over-decomposition is real.)
- Entanglement is the smell: a reader who must flip between implementations
  to follow one event. Fix by inlining shallow pass-throughs and by moving a
  whole feature behind one seam, not by adding layers.
- Comments are the interface: each module and public method states what it
  hides and its invariants, present tense; no history, ticket ids or defect
  narration. Long "megasyllabic" names do not replace comments.
- Define errors out of existence: internal type-check ceremony on values the
  caller just built is cost without value; keep the checks that guard a real
  invariant.
- Composition root (Rhodes): run_spec wires objects; policy defaults and
  compatibility rules do not belong in the wiring.
- Easy to delete (tef): put each feature the baseline does not run behind a
  seam of 3 to 6 methods so it can be removed whole; boilerplate at the
  edges, the core stays boring.
- Tests as a lock, not a driver: 772 tests, the smoke snapshot, the baseline
  sweep, Gates 1 to 5 stay byte-identical after every commit; the sweep wall
  clock column and the smoke guard against silent performance regressions
  (the PrimeGenerator lesson).
- Owner rules: names kept, no shims or aliases, no new file without approval
  (this document is the request), one lump per commit.

## 2. Verdicts per module

Legend: keep, extract (a collaborator with the interface named in section 6),
inline (shallow methods folded back), delete (dead), leave (deliberately not
touched).

| module | lines | verdict | why |
|---|---|---|---|
| window_manager.py | 2738 | extract 5 lumps, inline 12 shallow, delete 3 dead | 47 percent is strong escalation the baseline never runs; boundary courier, retention, logical ledger and link attribution are separable state |
| decoder_manager.py | 958 | extract 4 lumps, inline StrategyServicesImpl | strong-request ledger, batching, input staging and terminal records are separable; the weak service core stays |
| controller.py | 521 | extract 3 lumps, delete 1 dead, inline 1 | protected streams (38 percent) are a baseline-off feature; instruction relay and idle relay are self-contained |
| execution_runtime.py | 211 | extract resource ledger | claims code is self-contained (80-128) |
| syndrome_ingress.py | 559 | keep, remove ceremony, one owner for slots | guard lines outnumber modeling lines in 3 of 5 public methods; slot state duplicates the buffer |
| syndrome_buffer.py | 618 | leave, add a named round key | cleanest module of the front path |
| qpu.py, planner.py, orchestrators.py, rounds.py, devices.py, policies.py, config.py, codes.py, layouts.py, schedulers.py, decoder_memory_transfer.py, pauli_frame.py, seeding.py, engine.py, link_profiles.py, factories.py | | keep | small strategy objects or clean single-purpose modules |
| decoder_memory.py | 254 | keep; move _check_detector_row_layout (96-150) to the DEM package | model validation, not memory |
| decoder_engine.py | 190 | keep; replace the _completed dict with an on_result callback | result travels through a private dict keyed by request or service key (104,148,183) |
| decoders.py | 380 | keep; delete SwitchingDecoder (158-208, zero hooks) | dead end |
| switching.py | 352 | extract SwitchingPreconditions (243-323) | validators are a distinct responsibility; class docstring narrates a paper comparison |
| message.py | 1033 | leave as one vocabulary; move 6 hold tokens to syndrome_buffer, bring Submission/Directive/OutcomeDirective in from protocols | one responsibility, imported jointly by 20 modules; splitting adds imports and hides nothing |
| protocols.py | 540 | keep as seams only after the three values move | 18 of 24 protocols are documentation only; fine as port docs |
| links.py | 972 | split reporting out (828-972 to link_reports.py); trim _validate_attribution_shape | mechanism vs JSON reporting; 56 lines re-check what the caller constructs |
| run_spec.py | 561 | move defaults and compatibility rules to defaults.py; replace 8 post-construction installs with constructor args | not a clean composition root today |
| metrics.py | 742 | keep; decide BurstEscalationDetector (410-502, never wired) | delete from core or move to experiments |
| views.py | 404 | keep; move capture_primary_result next to PrimaryRunResult; replace private reads with accessors | views reach into _ops, _selected_request_keys, engine internals, device._truth |
| speculative_recovery.py | 419 | keep the seam, rewrite the body against the new collaborators | 30 private touches into window_manager, including 10 writes |
| dynamic_windows.py | 313 | keep; invert seal callbacks | mostly deep; seal calls five manager mutators and admits it cannot roll back |
| window_interactions.py | 197 | leave | the model collaborator: injected, no back reference, value objects in and out |
| schemes.py | 399 | keep; replace the scheme inheritance with a protocol plus a shared data_complete function | NaiveOnlineScheme.validate_buffer is an empty override |
| frontends/qlx.py | 592 | keep; extract _prove_detector_routing (137-239) | 100-line validator inside a lowering module |
| detector_error_model/* | 1827 | leave | declared L0 to L3 import lattice, each header names its seam |

## 3. Entanglement chains (reader must flip)

1. Commit to boundary to receive to check to submit: window_manager
   _commit_decode_done 2061 to _hand_on_boundary 2080 to _send_boundary 2392
   to scheduled _receive_boundary 2479 to _propose_boundary_update 2523 to
   window_interactions.merge_boundary 60 back to check_window 721 and
   _submit_window_decode 847. Seven implementations, three files.
2. Strong deferral: switching 228 to decoder_manager 945 to
   defer_strong_escalation 1169 to _resolve_strong_region_plan 1733 to
   _absorb_window 1886 or _reslice_restart_window 1869, later
   _check_deferred_strong_after_commit 1996 to _submit_far_strong 1957 to
   _build_pending_strong_job 1921 to _submit_strong_with_csd 1035. Ten hops,
   three files, _EscalationRegistry as a fourth state holder.
3. Strong selection reentry: decoder_manager 616 to StrategyServicesImpl 951
   to window_manager 1068 to _submit_terminal_strong 1095 to submit_fn (bound
   at run_spec 351) to decoder_manager.enqueue 240. Re-enters the manager
   mid-completion.
4. Result side channel: decoder_manager._begin_service 487 (getattr run) to
   decoder_engine.run 124 to _enter 140 stores into _completed 148; then
   _decode_and_validate_result 660 calls decoder_engine.decode 181 which pops
   it. The result should ride the callback.
5. Landing: _start_job 454 to _transfer_into_decoder_memory 456 to
   transfer.deliver 32 to receive_once 462 to memory.deposit 230 to
   materialize_decoder_input 152 to landed 448 to _begin_service 481. Four
   files for one arrival.
6. Start gating cycle: execution_runtime._maybe_begin 140 to
   controller.can_start 145 to _must_wait_for_round_boundary 154 to
   controller._open_protected_boundary 276 to runtime.retry_ready_operations
   180 to _maybe_begin. Neither file alone says when an operation starts.
7. Completion round trip: qpu._body_done 133 to controller._body_done 372
   (pass-through) to runtime.body_done 146 to controller.before and
   after_successor_release 160,169 to runtime.workload_complete 33 to
   qpu.finish 382.
8. Readout re-wrapping: qpu._emit 166 to controller.accept_qpu_readout 462
   builds SyndromePayload, builds RetainedSyndromeFragment 484 only to
   validate, ingress.relay_qpu_readout 191 builds the same fragment again;
   syndrome_buffer 342 builds SyndromeRoundPacket. Three shapes, two
   normalizations, one round.
9. Ingress and buffer packing: ingress _receive_fragment 229 to
   buffer.accept_fragment 254 to ingress _finish_packing 313 to
   buffer.finish_packing 318; the packet is stored a second time on the
   ingress slot 320; round keys rebuilt by tuple slicing 315,474,484. Two
   objects allocate and free a slot table for the same round.
10. Speculative replay: _hand_on_boundary 2085 to speculative_recovery.begin
    43, complete 82, _repair 153 re-enters _send_boundary,
    _finish_operation_if_ready, _resolve_strong_wait,
    release_stream_segments_at_commit.
11. Link attribution: window_manager 958,972,981,998,2441 build
    TrafficAttribution and RequestTransferRelation, LinkModel.reserve 717,
    _attribution_snapshot 481, _relation_snapshot 460,
    _validate_attribution_shape 764-819 re-derives what the caller built.

## 4. Information hiding leaks (both sides cited)

- speculative_recovery writes window_manager private state:
  _committed_boundaries (167,300 vs 277), _boundary_versions (274 vs 278),
  _boundary_delivery_versions (278 vs 279), _released_boundary_dependencies
  (280,320 vs 280), logical_contributions.pop (281 vs 281), op_results.pop
  (282 vs 274), committed_windows and _committed_per_op (284-289 vs
  272,273), _held_boundary.pop (301 vs 282), window.committed, queued, t_*
  (291-296); reads _window_infos, _pending_strong_windows, _ops,
  _finished_ops; calls nine private methods.
- dynamic_windows reads wm.rounds_arrived (96,152,279 vs 267), wm.windows
  and committed_windows (306-307 vs 271,272).
- DecodeJob is a shared mutable bag with five writers in four modules:
  payloads (wm 1148, dm 471,491, engine 156), decoder_input (dm 470,520),
  input_hold (wm 845, cleared dm 476,512), reserve_transfer (dm 192,459),
  unit and pool (dm 429,366,415), memory (dm 472,519), awaiting_strong_result
  (dm 627, read wm 2029-2119), service_* keys (dm 322,398-427),
  strong_label and dem (wm 870,1140-1150). Only dem has one owner.
- run_spec installs 8 attributes after construction: syndrome_ingress
  .syndrome_buffer 319; window_manager .strategy .services .submit_fn
  .on_workload_complete 349-355,406; decoder_manager .strategy .services
  .on_window_decoded .on_strong_window_decoded 349-355; and calls private
  controller._body_done 381, window_manager._register_dynamic_stream 415,
  engine._invalidate _start_running _begin_finalization _complete
  145,419,430,435.
- views reads window_manager._ops 206,212, _selected_request_keys 353,
  controller._round_count_for 251, engine._invoke_metric_callback and
  _event_queue 391,394, device._truth 272; metrics 734 calls a concrete
  factory API.
- Front path: ingress reads buffer private slot dict (478 vs 192) and
  hand-unwinds slot state on buffer exhaustion (257-273 vs 305-325);
  controller indexes runtime.operations 436,447,501 and reads
  runtime.workload_complete 381,416; stream binding map kept in controller 64
  and window_manager 2692-2698; getattr back-channel ingress 173-176 vs
  window_manager 644.
- getattr dispatch a protocol would remove: decoder_manager 487 (run), 332
  (cancel), 100 (engine), 516 (memory); decoder_engine 119; decoders 271;
  decoder_memory 104-108; switching 213,232,336; speculative_recovery 45
  (boundary_policy.speculative); window_manager 2686.

## 5. Deletable lumps the baseline never runs

| lump | where | size | hooks into core | seam |
|---|---|---|---|---|
| strong escalation and double window | window_manager 1035-2019, 2314-2386, 2671-2677, 2728-2730, _EscalationRegistry 80-231 | about 1280 lines, 38 methods | 14 | 6 methods |
| strong request lifecycle, cancellation, batching | decoder_manager 206-238, 277-342, 375-412, 640-656, 699-841; switching Switching 118-352; StrategyServicesImpl 929-958 | about 640 lines | 14 | ledger 6 methods, batch 3 |
| speculative recovery (Eager replay) | speculative_recovery.py | 419 lines | 8 hooks, 30 private touches | 5 methods, already named |
| dynamic streams | dynamic_windows.py plus 10 manager methods | about 430 | 9 | 6, already this shape |
| protected regions and feedback streams | controller 75-96, 105-127, 154-222, 253-335, 384-399 | about 200, 9 methods | 6 | 6 |
| feedback boundary mode measurement_closed | controller 401-413, runtime 171-178, wm 784 | about 30 | 1 | 2 |
| idle policies extend_stream, separate_decode_jobs | controller 444-459, 490-501; policies 40-57; wm 2683-2690 | about 40 | 1 | 2 |
| multi-fragment rounds and t_pack | ingress 282-289, buffer 138-170, qpu 148-168 | about 60 | 2 | 2 (only the qlx frontend emits two fragments) |
| ingress overflow DROP_ROUND and reassembly timeout | ingress 33-40, 73-104, 249, 257-279, 305-310, 467-492 | about 90 | 2 | 2 |
| terminal record capture | decoder_manager 31-71, 843-895 plus 7 sites | about 115 | 10 (all capture_enabled) | 5 |
| soft output | decoders 30-38, 283-380; switching 22-82, 327-345 | about 200 | 1 | tiny |
| SwitchingDecoder | decoders 158-208 | 51 | 0 | delete |
| BurstEscalationDetector | metrics 410-502 | 93 | 0 | delete or move |
| offline shot statistics | adapters/window_decode_results 404-501 | 98 | 0 | move |
| dead accessors | window_manager speculative_replays 2389, peak_payloads 2733, payloads_held 2737; controller _patch_for_operation 365-370 | 12 | 0 | delete |
| link JSON reporting | links 828-972 | 145 | 2 | move to link_reports.py |
| provenance-as-data in core | links payload_selection 641, relation types 136-163, 406-448, SoftOutputSource.references 658-676 | about 120 plus 56 lines of validation | read only by traffic_json_value | retire or move to reporting |

## 6. The cut plan

Order chosen so every step reduces a leak, a dead symbol, or a baseline-off
feature; each is one commit; the lock (section 1) is green after each. Steps
inside a phase are independent; phases are sequential because later
extractions depend on earlier collaborators.

Phase 0, zero-risk deletions and inlines (one commit)
- delete: window_manager speculative_replays, peak_payloads, payloads_held;
  controller _patch_for_operation; decoders SwitchingDecoder;
  controller.py 483-484 (fragment built only to validate).
- inline: controller round_ticks_for at runtime 50; window_manager
  _spatial_nodes, _planning_view, _window_has_closed_boundary,
  _round_count_for_window, _transfer_pending_to_csd,
  _transfer_potential_to_csd, _release_hold_if_live, validate_stream_length,
  _check_deferred_strong_after_commit, check_windows_for_operation; the four
  lifecycle pass-throughs 2716-2726 by letting the controller hold
  window_manager.lifecycle; decoder_manager decoder_for, free_units,
  queue_for, _handle_strong_decode_result; StrategyServicesImpl into
  window_manager.
- validation ceremony: delete the __post_init__ blocks and type guards that
  re-prove values another decsim module just built (message.py 45 raises /
  23 type checks, links.py 44, ingress 12, buffer 8, engine 18, seeding 14,
  qlx 46; roughly 300 lines). Keep only the six real invariants: qpu.issue
  cadence equals the cycle; ExecutionRuntime._claim_resources no double
  allocation; SyndromeBuffer duplicate and late-fragment checks; decoder
  memory capacity exhaustion (fail-stop); RunSpec compatibility rules on
  user config (moved to defaults.py); link card completeness.
- comments: convert the 10 history comments to present-tense invariants;
  add one-line invariants to the 14 undocumented message.py classes and the
  public methods listed in the audits.

Phase 1, front path
- InstructionRelay from controller.relay_instruction 503-521. Interface:
  relay_instruction(decision, deliver). Removes links from Controller.
- ResourceLedger from execution_runtime 80-128. Interface: claim(operation),
  release(operation), holder_of(kind, resource_id).
- Collapse chain 8: accept_qpu_readout hands the built
  RetainedSyndromeFragment to ingress; delete the duplicate construction and
  the four guards at ingress 184-189.
- ProtectedStreamRegions from controller (about 200 lines). Interface:
  index(program), blocks_start(operation) -> bool, activate(operation),
  request_close(operation), seal_finished(), binding_for(operation_id).
  Controller keeps five delegating lines; the collaborator is deletable whole
  for baseline runs.
- IdleRoundRelay from controller 439-459, 490-501. Interface:
  relay(op_id, patch, round_index), account(operation).
- SyndromeBuffer: named RoundIdentity and public
  assembling_fragment_count(identity); accept_fragment returns a rejection
  outcome instead of raising so ingress 257-273 disappears; ingress stops
  reading buffer 192 and slicing identity tuples.
- Ceremony trim in ingress relay_syndrome 212-219, relay_qpu_readout
  184-189, controller 466-471, buffer 261-266, and the nine
  _validated_round_identity re-runs; keep qpu.issue cadence, _claim_resources,
  and the buffer duplicate and late-fragment checks.

Phase 2, decoder side
- TerminalRecordLedger from decoder_manager 31-71, 843-895. Interface:
  record_request, record_service, request_snapshot, service_snapshot,
  enabled. Removes the capture_enabled branch from 7 sites.
- DecoderInputStaging from decoder_manager 456-539. Interface:
  stage(job, memory, on_landed), release(job), cancel(job),
  release_service_members(service_job). One owner for job.memory,
  decoder_input, input_hold; kills getattr(job, "memory").
- StagedDecoder protocol with on_result: run(job, engine, on_result),
  cancel(job), latency(job); delete DecoderEngine._completed and the
  duplicated payload materialization at decoder_engine 155-157.
- StrongRequestLedger from decoder_manager 206-238, 640-656, 699-841 and
  most of cancel_strong. Interface: admit(job), resolve_weak(key),
  cancel(key), select(key, request_key), complete(held), snapshot(). About
  250 lines and 5 dicts leave the manager; _on_decode_done shrinks as a
  consequence, not by splitting.
- StrongBatch behind the ledger (375-377, 380-412, 772-798): merge(queue,
  scheduler), members_of(service_job), split_result(result, requests); or
  delete if the batching axis is not needed for thrust 2.
- WindowCompletionSink injected by constructor replacing run_spec 349-355:
  on_window_decoded, on_strong_window_decoded, make_strong_job,
  prepare_strong_selection, defer_strong_escalation. Removes the four None
  fields (decoder_manager 110-113) and both "has no callback" guards.
- SwitchingPreconditions from switching 243-323; move
  _check_detector_row_layout to detector_error_model.

Phase 3, window manager (order matters: 3a to 3c before 3d)
- 3a WindowRetention from cluster 453-552. Interface: register_window(key,
  window), replace_window(key, window), read_keys(op_id, lo, hi, window),
  transfer(previous_owner, new_owner), require_retained(keys, purpose).
- 3b BoundaryCourier from 2392-2567, owning _committed_boundaries,
  _boundary_versions, _boundary_delivery_versions,
  _released_boundary_dependencies, _held_boundary. Interface: send(window,
  op, boundary, source_request_key), merge_available(source_key,
  destination, boundary), hold(key, held), take_held(key), invalidate(key,
  deps). Converts 9 of speculative_recovery's private writes into calls.
- 3c LogicalLedger from 2122-2312 with logical_contributions and
  _observable_arity_by_stream. Interface: install(contribution),
  replace_prediction(owner_key, observables), observables_for_interval(
  stream_id, lo, hi, policy), drop(owner_key), slab_candidate(key, region).
- 3d StrongEscalation from cluster 8 (about 1280 lines), built on 3a to 3c.
  Interface: defer(weak_job), make_job(weak_job, round_count, label),
  prepare_selection(weak_job, request_key, serial_job, deferred),
  note_arrival(op_id), note_weak_commit(key), pending_snapshot(). Baseline
  constructs the manager without it. window_manager.py ends near 1400 lines.
- 3e WindowTraffic from 941-1033 as functions plus the two links.reserve
  calls: job_payload_bits, attribution(window, op, request_key),
  link_arrival(path, window, op, request_key).
- 3f SuffixRephase inside StrongEscalation: plan(...) -> RephaseProposal,
  apply(proposal), replacing the hand-written snapshot and restore at
  1620-1673.
- 3g speculative_recovery rewritten against 3a to 3c through their
  interfaces; DynamicWindows.seal returns a plan the manager applies,
  removing the FAIL_STOP admission at dynamic_windows 187-192.
- 3h WindowManager constructor takes strategy, services, submit_fn,
  on_workload_complete; no post-construction installs remain.

Phase 4, wiring and vocabulary
- protocols.py: move Submission, Directive, OutcomeDirective into message.py;
  message.py: move the six hold tokens (135-152) to syndrome_buffer.
- links.py: reporting to link_reports.py; trim _validate_attribution_shape
  to caller-facing rules; retire payload_selection (derive from
  payload_source) and the relation types out of core mechanism, per the
  no-provenance-as-data rule; keep source strings on config cards.
- run_spec.py: defaults.py holds the ten default selections and five
  compatibility rules (174-271, 550-557); _build_once becomes wiring plus
  bind_run_seed; the engine phase machine gets a public run(); views and
  metrics use small public accessors instead of private reads;
  capture_primary_result moves next to PrimaryRunResult (breaks the
  run_spec to views import cycle).
- schemes.py: WindowScheme protocol plus shared data_complete; drop the
  inheritance chain and the type(scheme) is checks in switching 186,275.
- metrics.py: BurstEscalationDetector deleted from core or moved to
  experiments; frontends/qlx.py: _prove_detector_routing extracted.

## 7. What not to do

- Do not split message.py by domain, links.py mechanism, or
  detector_error_model; they are one responsibility each.
- Do not split _on_decode_done, Switching.on_decode_outcome,
  _defer_crossing_strong_escalation or the DecoderEngine stage walk for
  length; each is a single stated ordering that reads top to bottom.
- Do not add layers, base classes or registries; every extraction above is a
  plain object injected by constructor with value objects in and out, the
  DefaultWindowInteraction shape.
- Do not chase method length in tests; test files change only where an
  import path moved.

## 8. Effort and order of value

Phase 0 and Phase 1 are a day and remove the most-read entanglement on the
baseline path (front path chains 6 to 9). Phase 2 is a day. Phase 3 is two to
three days and is where the baseline stops paying for the strong tier. Phase 4
is a day. Each phase ends with the lock green and a status note.
