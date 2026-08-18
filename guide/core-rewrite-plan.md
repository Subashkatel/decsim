# Core rewrite plan, file by file

Companion to guide/core-design-audit.md (findings with file:line) and
guide/core-layout-and-names.md (folders and names; the file names below use
that layout). Behavior
does not change: the lock is 772 tests, the smoke snapshot, the baseline
sweep and Gates 1 to 5, byte-identical after every commit. Module count is
not the goal, depth is: only six new modules, each hiding real state behind
3 to 6 methods; everything else is deletion, inlining and reordering inside
the existing file. New names are proposals until the owner approves them.

## A. Every core file: broken down or not

| file | today | decision | leaves to | arrives from | deleted | file after |
|---|---|---|---|---|---|---|
| window_manager.py | 2738 | broken down | decoders/strong_escalation.py (escalation, ~1280 lines), windows/window_boundaries.py (~180), windows/committed_rounds.py (~190) | StrategyServicesImpl from decoder_manager (inlined) | 3 dead accessors, 12 shallow forwarders, 4 lifecycle pass-throughs | ~1000 lines: registry and plan; retention (section); arrival; readiness and submission; traffic functions (section); commit and results |
| decoder_manager.py | 958 | broken down | decoders/strong_escalation.py (strong ledger, cancellation, batching, ~340) ; decoder_memory_transfer.py (input staging, ~80) | | StrategyServicesImpl, 4 shallow forwarders | ~450 lines: pool and queue; dispatch; service; completion; TerminalRecords small class at the bottom |
| controller.py | 521 | broken down | controller/feedback_streams.py (~200 incl. stream reservation) ; controller/idle_policies.py (idle relay branches, ~30) | | _patch_for_operation, throwaway fragment 483-484, round_ticks_for | ~200 lines: load, issue, readout relay, idle accounting, instruction relay |
| execution_runtime.py | 211 | not broken down | | | boundary-flag remnants | same file, resource claims as their own small class in the file |
| qpu.py | 192 | not broken down (just rewritten, Q-064) | | | | unchanged; header states the six cadence invariants |
| syndrome_ingress.py | 559 | not broken down, slimmed | slot ownership to syndrome_buffer | | own slot table, exhaustion unwind 257-273, duplicate packet store 320, tuple slicing, guards 184-189/212-219 | ~350 lines: arrival, packing, C2B arbitration, transmit, snapshot |
| syndrome_buffer.py | 618 | not broken down | | 6 hold tokens from message.py | | same plus RoundIdentity, assembling_fragment_count, accept_fragment returns an outcome |
| decoder_engine.py | 190 | not broken down | run_seed_children to decoders.py | | _completed dict, duplicate payload materialization 155-157 | run(job, engine, on_result) delivers the result by callback |
| decoder_memory.py | 254 | not broken down | _check_detector_row_layout to detector_error_model | | | config, input, per-unit memory, snapshot |
| decoder_memory_transfer.py | 64 | grows | | input staging from decoder_manager | | transfer, landing into unit memory, release, cancel: one concept |
| decoders.py | 380 | not broken down | | run_seed_children | SwitchingDecoder (dead), two subclass-to-bind-lambda cases | router, latency models, soft output |
| switching.py | 352 | not broken down | validators 243-323 to run_defaults.py | | paper narration in the class docstring (to guide/) | threshold register, Baseline, Switching hooks |
| schemes.py | 399 | not broken down | | | inheritance chain (protocol + shared data_complete instead), empty override | same file |
| speculative_recovery.py | 419 | not broken down, rewritten on the seams | | | 30 private touches into window_manager | same file, calls only BoundaryCourier, LogicalLedger, retention and public commit methods |
| dynamic_windows.py | 313 | not broken down | | | FAIL_STOP admission 187-192 | seal returns a plan the manager applies |
| window_interactions.py | 197 | leave | | | | the model collaborator |
| pauli_frame.py | 238 | leave | | | one history comment | |
| controller/idle_policies.py | 57 | grows | | idle relay branches from controller | | Ignore, ExtendStream, SeparateDecodeJobs own their relay behavior |
| planner.py, orchestrators.py, rounds.py, devices.py, config.py, codes.py, layouts.py, schedulers.py | small | leave | | | | |
| run_spec.py | 561 | broken down | run_defaults.py (10 defaults, 5 compatibility rules, factory decode-service check) | capture_primary_result from views.py | 8 post-construction installs, private engine calls | ~350 lines: RunSpec config, _build_once as wiring, results |
| factories.py | 569 | leave | _check_factory_decode_service to run_defaults.py | | 3 history comments | |
| links.py | 972 | broken down | links/link_traffic_report.py (JSON reporting 828-972) | | payload_selection, relation types, cross-check 816-819 | ~750 lines: vocabulary, rules, cards, Link, LinkModel.reserve, ledger |
| link_profiles.py | 300 | leave | | | one history comment | |
| message.py | 1033 | not broken down | 6 hold tokens to syndrome_buffer | Submission, Directive, OutcomeDirective from protocols | __post_init__ ceremony (~150 lines), 2 history comments | one vocabulary module, ~850 lines, every class with a one-line invariant |
| protocols.py | 540 | not broken down | 3 value classes to message | | | seams only |
| metrics.py | 742 | not broken down | BurstEscalationDetector out of core | | 2 history comments | |
| views.py | 404 | not broken down | capture_primary_result to run_spec | | private reads (each owner publishes one snapshot()) | |
| seeding.py, engine.py | 224, 192 | leave | | | guard ceremony beyond real invariants; engine gets a public run | |
| frontends/qlx.py | 592 | not broken down | | | | _prove_detector_routing becomes a labeled section |
| detector_error_model/* | 1827 | leave | | _check_detector_row_layout | | |
| adapters/window_decode_results.py | 596 | not broken down | offline shot statistics 404-501 to experiments | | | |
| adapters/stim_device.py | 432 | leave | | | | |
| NEW decoders/strong_escalation.py | | | | window_manager escalation, decoder_manager strong ledger | | ~1650 lines: StrongEscalation, StrongRequestLedger, StrongBatch, SuffixRephase, the registry types |
| NEW windows/window_boundaries.py | | | | window_manager 2392-2567 and boundary state | | ~200: BoundaryCourier |
| NEW windows/committed_rounds.py | | | | window_manager 2122-2312 | | ~200: LogicalLedger |
| NEW controller/feedback_streams.py | | | | controller protected regions and stream reservation | | ~230: ProtectedStreamRegions |
| NEW run_defaults.py | | | | run_spec defaults and rules, switching validators, factory check | | ~150 |
| NEW links/link_traffic_report.py | | | | links.py 828-972 | | ~150 |

Net: core goes from about 22600 lines to about 20500 (ceremony, dead code,
duplicates gone), the largest file from 2738 to about 1000, and the baseline
path no longer imports the strong tier.

## B. Per file: what it looks like after

### window_manager.py
Header comment: "Owns the window life cycle of every operation: which rounds
each window needs, when it is ready, when its result commits. Boundaries
between windows are BoundaryCourier's, ownership of committed rounds is
LogicalLedger's, the strong tier is StrongEscalation's and may be absent."
Sections, in reading order:
1. registry and plan: register_op, load_execution_plan, rounds_for.
2. retention (kept in file): which Buffer 0 rounds each window holds; the
   read-key helpers.
3. arrival: on_syndrome_arrival, on_memory_round, prepend_idle_rounds.
4. readiness and submission: check_window, _window_readiness,
   _submit_window_decode; traffic helpers (payload bits, attribution, link
   arrival) as module functions right below.
5. completion: on_decode_done (hand boundary to courier, schedule WDO
   publish), _commit_window (ledger.install, results), finish_workload.
6. stream surface: bind_stream_operation, seal via lifecycle.
Constructor takes: engine, scheme, geometry, plans, links, orchestrator,
boundary_policy, window_interaction, syndrome_buffer, courier, ledger,
escalation (or NoStrongTier), lifecycle, recovery, strategy, services, submit_fn,
on_workload_complete. Nothing is installed after construction.
Reading path for one round: 3 -> 4 -> 5, with courier.send and
courier.deliver in between; three methods you can name.

### decoders/strong_escalation.py (new)
Header: "The strong decoder tier: when a weak window escalates, how its
strong job is built and selected, how a strong result replaces the weak one
and how the two managers account for it. Absent in the baseline."
Classes: StrongEscalation (window side: defer, make_job, prepare_selection,
note_arrival, note_weak_commit, pending_snapshot; contains SuffixRephase
plan/apply and the escalation registry types), StrongRequestLedger (decoder
side: admit, resolve_weak, cancel, select, complete, snapshot; contains
StrongBatch). Both are built by run_spec only when the strategy is Switching.

### windows/window_boundaries.py (new)
Header: "A boundary is the residual defects at a window's commit edge,
produced by the decoder at decode done and delivered to dependent windows
over DD; a held boundary waits for a final result. Versions make late
deliveries harmless." BoundaryCourier: send, merge_available, hold,
take_held, invalidate; on_boundary_received callable given at construction.

### windows/committed_rounds.py (new)
Header: "Which window owns which committed rounds of a stream, and the
logical observables that result; a strong slab may replace ordinary windows;
contributions tile a stream without gap or overlap." LogicalLedger: install,
replace_prediction, observables_for_interval, drop, slab_candidate.

### decoder_manager.py
Header: "Assigns decoder units to ready windows: queue per pool, unit
assignment, input staged into that unit's memory, service, completion. Strong
requests are the ledger's business." Sections: pool and queue (enqueue,
submit_decode); dispatch (try_dispatch, _start_job); service
(_begin_service, result validation); completion (_on_decode_done, about 40
lines); TerminalRecords small class at the bottom. Constructor takes router,
scheduler, units, memory config, transfer (which now stages), records
capture flag, sink (WindowCompletionSink), strong ledger or None. Decoders
implement StagedDecoder: run(job, engine, on_result), cancel(job),
latency(job).

### decoder_memory_transfer.py
Header: "Moves a job's rounds from Buffer 0 into the assigned unit's memory:
the transfer delay, the landing, the release; the sole writer of
job.decoder_input, job.memory, job.input_hold." Methods: stage(job, memory,
on_landed), release(job), cancel(job), release_service_members(service_job).

### controller.py
Header: "Turns admitted operations into QPU commands and QPU readouts into
controller-side binary handed to ingress; the QEC cycle itself is the QPU's."
Sections: load_program; issue_operation and can_start (asks
protected_streams, or True); accept_qpu_readout (one fragment, to ingress);
emit_idle_round (accounting, then idle_policy.relay); relay_instruction.
Constructor: engine, qpu, window_manager, syndrome_ingress, links,
resolved plan, idle_policy, protected (ProtectedStreamRegions or None).

### controller/feedback_streams.py (new)
Header: "Per-patch feedback stream regions: one live stream per patch,
boundary ticks on the cadence, seal only after the final round; the
stream_next_round ledger." ProtectedStreamRegions: index, blocks_start,
activate, request_close, seal_finished, binding_for.

### syndrome_ingress.py and syndrome_buffer.py
Buffer header unchanged (it already says what it does not do); adds
RoundIdentity, assembling_fragment_count, and accept_fragment returning
accepted / complete / refused. Ingress header: "Controller-side arrival of a
readout, packing into a round, arbitration onto C2B, delivery to the window
manager; the buffer owns the round slots." Ingress sections: arrival,
packing, arbitration, transmit, snapshot.

### run_spec.py and run_defaults.py (new)
run_spec: RunSpec dataclass, _build_once as construction in dependency
order (each object gets everything by constructor, then bind_run_seed,
engine.run(), finalize), CompletedRun and PrimaryRunResult with
capture_primary_result. run_defaults.py: default_strategy, default_scheme,
default_rounds, default_boundary_policy, default_idle_policy,
default_scheduler, default_links, default_code, and the compatibility rules
(dynamic streams need SlidingWindowScheme; binary availability vs QC
exclusion; device code identity; router exclusivity; factory decode service;
switching preconditions), each a function with a one-line reason.

### links.py and links/link_traffic_report.py (new)
links keeps the closed vocabulary, rules, cards, Link FIFO, LinkModel.reserve
and the ledger; _validate_attribution_shape keeps only caller-facing rules.
link_reports holds topology_json_value, traffic_json_value, _transfer_json.

### message.py and protocols.py
message: one vocabulary; hold tokens move to the buffer; Submission,
Directive, OutcomeDirective move in; ceremony __post_init__ blocks go
(SyndromeRoundPacket 268-293, OperationWindowPlan 504-556,
RetainedSyndromeFragment 239-245 and kin); every class gets a one-line
invariant. protocols: seams only, each with the promise implementers make.

### views.py and metrics.py
views: private reads replaced by one snapshot() per owner (window_manager,
controller, engine, device); capture_primary_result leaves. metrics:
BurstEscalationDetector leaves core; factory access through the protocol.

### speculative_recovery.py, dynamic_windows.py, schemes.py, decoders.py, switching.py
As in section A: same files, bodies rewritten against the new seams
(recovery), seal returns a plan (dynamic windows), protocol instead of
inheritance (schemes), dead class and lambda subclasses gone (decoders),
validators to defaults and narration to guide/ (switching).

## C. Order and lock
Phase 0: deletions, inlines, ceremony, comments (all files, one commit per
file group). Phase 1: controller, protected_streams, policies, ingress and
buffer, execution_runtime. Phase 2: decoder_manager, decoder_memory_transfer,
decoder_engine, strong ledger into strong_tier. Phase 3: boundaries,
logical_ledger, then escalation into strong_tier, then speculative_recovery
and dynamic_windows on the seams, then the window_manager constructor.
Phase 4: run_spec and defaults, links and link_reports, message and
protocols, views and metrics, schemes, decoders, switching. After every
commit: pytest, smoke; after every phase: sweep and Gates 1 to 5;
guide/baseline-status.md records the phase.
