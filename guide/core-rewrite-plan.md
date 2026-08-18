# Core rewrite plan, module by module

Companion to guide/core-design-audit.md (findings and file:line citations).
This document says, for each module, what it does today, what it will look
like after the rewrite, what moves where, what is deleted, and how a reader
follows the main path afterwards. Behavior does not change: the lock is 772
tests, the smoke snapshot, the baseline sweep and Gates 1 to 5, byte-identical
after every commit. Names below marked "proposed" are new module or class
names that need the owner's approval before they exist.

Conventions used throughout
- A collaborator is a plain object built by the composition root and passed
  in by constructor; it holds its own state, exposes 3 to 6 methods, takes and
  returns value objects, and never keeps a back reference to its caller
  (the DefaultWindowInteraction shape). No base classes, no registries.
- Every module opens with a comment that says what it hides and its
  invariants, in the present tense. Every public method has a one-line
  contract. Nothing narrates history.
- Internal validation ceremony is deleted; the six invariants kept are listed
  in the audit, Phase 0.
- Long methods that read as one ordering stay long. Shallow 3-line
  forwarders are inlined at their call site.

---------------------------------------------------------------------------

## 1. window_manager.py (2738 lines today)

Today. One class carrying twelve responsibilities: op and plan registry,
payload retention holds, dynamic-stream surface, arrival ingestion, readiness
and weak submission, link attribution, strong escalation and double window
(about 1280 lines, 47 percent), commit and logical ownership, boundary
delivery, result delivery, stream binding, metric accessors. Four fields are
None until run_spec installs them. speculative_recovery writes ten of its
private containers.

Target. window_manager.py keeps its name and becomes the orchestrator of the
window life cycle only:

    WindowManager
      register_op(op) / load_execution_plan(plan, buffering_plan)
      on_syndrome_arrival(packet), on_memory_round(op_id), prepend_idle_rounds(op_id, n)
      check_window(key)                       readiness, then submit
      on_decode_done(job, result)             hand boundary on, publish after WDO
      on_strong_decode_done(completion)       delegated to escalation
      finish_workload_if_ready()

    state it still owns: windows, op_windows, rounds_arrived, memory_rounds,
    committed_windows, op_results, the t_* stamps on Window.

    collaborators (constructor arguments, all proposed names):
      retention: WindowRetention        which rounds each window holds in Buffer 0
      courier:   BoundaryCourier        boundaries between windows
      ledger:    LogicalLedger          who owns which committed rounds, observables
      traffic:   window_traffic         payload sizes, attribution, link arrival (functions)
      escalation: StrongEscalation | None   the strong tier; None in the baseline
      lifecycle: DynamicWindows         (exists) real-time streams
      recovery:  SpeculativeRecovery    (exists) Eager replay, rewritten on the seams
      window_interaction, boundary_policy, links, orchestrator, syndrome_buffer (exist)

Constructor: everything above is passed in; strategy, services, submit_fn
and on_workload_complete become constructor arguments (no post-construction
installs). Expected size after Phase 3: about 1400 lines, then about 900
once escalation is out.

What moves where
- 453-552 retention holds -> WindowRetention (proposed decsim/window_retention.py):
  register_window(key, window), replace_window(key, window),
  read_keys(op_id, lo, hi, window), transfer(previous_owner, new_owner),
  require_retained(keys, purpose). Owns the hold owners and typed keys.
- 2392-2567 boundary send/receive/merge/validate plus _committed_boundaries,
  _boundary_versions, _boundary_delivery_versions,
  _released_boundary_dependencies, _held_boundary -> BoundaryCourier
  (proposed decsim/boundaries.py): send(window, op, boundary, source_request_key),
  merge_available(source_key, destination, boundary), hold(key, held),
  take_held(key), invalidate(key, deps). It schedules the DD delivery and
  calls back check_window through a small callable it is given at
  construction (on_boundary_received), the only outbound call.
- 2122-2312 logical contributions and observables -> LogicalLedger (proposed
  decsim/logical_ledger.py): install(contribution), replace_prediction(owner_key,
  observables), observables_for_interval(stream_id, lo, hi, policy),
  drop(owner_key), slab_candidate(key, region). Pure data, no engine.
- 941-1033 attribution and link arrival -> window_traffic (proposed
  decsim/window_traffic.py) as functions: job_payload_bits(job),
  attribution(window, op, request_key), link_arrival(links, path, window,
  op, request_key). No state.
- 1035-2019, 2314-2386, 2671-2677, _EscalationRegistry, _PendingEscalation,
  _ResolvedStrongRegion, _EscalationPhase -> StrongEscalation (proposed
  decsim/escalation.py): defer(weak_job), make_job(weak_job, round_count,
  label), prepare_selection(weak_job, request_key, serial_job, deferred),
  note_arrival(op_id), note_weak_commit(key), pending_snapshot(). Built on
  retention, courier and ledger, so it never touches manager internals. The
  369-line _defer_crossing_strong_escalation becomes SuffixRephase inside it:
  plan(...) -> RephaseProposal, apply(proposal), replacing the hand-written
  snapshot and restore at 1620-1673. Baseline runs pass escalation=None; the
  eleven hook sites become "if self.escalation is not None".
- StrategyServicesImpl (decoder_manager 929-958) is inlined here: the window
  manager already implements every method it forwards.

Deleted: speculative_replays, peak_payloads, payloads_held (no callers);
the four lifecycle pass-throughs 2716-2726 (the controller holds
window_manager.lifecycle directly); the twelve shallow forwarders listed in
the audit Phase 0.

Reading path afterwards. To follow one round: on_syndrome_arrival stores it
(retention says which windows read it), check_window decides readiness and
submits (traffic prices the payload), on_decode_done hands the boundary to
courier.send and schedules the WDO publish, courier delivers and calls
check_window on the dependent, _commit_window installs into ledger and
delivers results. Five methods in one file plus two collaborators whose names
say what they hold; the strong tier is one import you can ignore.

---------------------------------------------------------------------------

## 2. decoder_manager.py (958 lines today)

Today. Weak service core interleaved with strong request lifecycle, strong
cancellation, bulk batching, terminal record capture, decoder-input staging,
and a StrategyServices adapter of pass-throughs. Four None fields installed
by run_spec. DecodeJob fields written from four modules.

Target.

    DecoderManager
      enqueue(job, reserve_transfer)      admission into the pool queue
      submit_decode(round_count, on_done, ...)  external load-only jobs
      try_dispatch()                       assign a free unit, stage input, begin service
      cancel_strong(key)                   delegated to the strong ledger
      check_decode_work_settled()          terminal assertion
      snapshots for views (free units, queue, memories)

    collaborators (constructor arguments):
      staging:   DecoderInputStaging (proposed decsim/decoder_input_staging.py)
                 stage(job, memory, on_landed), release(job), cancel(job),
                 release_service_members(service_job).
                 Sole owner of job.decoder_input, job.memory, job.input_hold.
      records:   TerminalRecordLedger (proposed decsim/decode_records.py)
                 record_request, record_service, request_snapshot,
                 service_snapshot, enabled. Removes the capture_enabled
                 branch from seven sites.
      strong:    StrongRequestLedger | None (proposed decsim/strong_requests.py)
                 admit(job), resolve_weak(key), cancel(key),
                 select(key, request_key), complete(held), snapshot();
                 StrongBatch inside it: merge(queue, scheduler),
                 members_of(service_job), split_result(result, requests)
                 (or deleted if the batching axis is dropped).
      sink:      WindowCompletionSink, injected: on_window_decoded,
                 on_strong_window_decoded, make_strong_job,
                 prepare_strong_selection, defer_strong_escalation.
                 Replaces the None fields and both "has no callback" guards.
      decoders implement StagedDecoder: run(job, engine, on_result),
                 cancel(job), latency(job). No getattr probing.

_on_decode_done stays one method and shrinks to about 40 lines because the
strong branch becomes one call into the ledger; it is not split for length.
Expected size: about 450 lines.

Reading path afterwards. enqueue puts a job in the pool queue; try_dispatch
pops one, assigns a numbered unit, staging.stage moves its rounds from
Buffer 0 into that unit's memory and calls back; _begin_service runs the
decoder; _on_decode_done frees the unit, releases the input, and hands the
result to the sink. One file, no side channels.

---------------------------------------------------------------------------

## 3. decoder_engine.py (190) and decoder_memory.py (254)

decoder_engine keeps DecoderStage, DecoderTiming, DecoderStageRecord and the
stage walk. Changes: run(job, engine, on_result) delivers the DecodeResult
through the callback; the _completed dict, decode() pop and
stage_records_for lookups by side channel go; the duplicated payload
materialization (155-157) goes because staging already materialized the
input. run_seed_children moves next to the decoders it seeds.

decoder_memory keeps DecoderMemoryConfig, DecoderInput, DecoderMemory,
snapshot. _check_detector_row_layout (96-150) moves to
detector_error_model where the model it checks lives.

---------------------------------------------------------------------------

## 4. controller.py (521 lines today)

Today. Op issue and QPU command; protected regions and feedback streams
(38 percent); stream reservation; feedback boundary modes; idle round relay
with two policy branches; readout relay with a throwaway fragment; the
instruction relay, which is pure link code; one dead method.

Target.

    Controller
      load_program(program), connect_runtime(runtime)
      issue_operation(operation, idle_rounds) -> start tick
      can_start(operation)                     asks protected streams
      accept_qpu_readout(readout, route)       builds one fragment, hands to ingress
      emit_idle_round(op_id, patch, round_index)  accounts, hands to idle relay
      before_successor_release / after_successor_release  hooks kept

    collaborators:
      protected: ProtectedStreamRegions | None (proposed decsim/protected_streams.py)
                 index(program), blocks_start(operation) -> bool,
                 activate(operation), request_close(operation),
                 seal_finished(), binding_for(operation_id).
                 Owns stream_next_round, region and boundary state; deletable
                 whole for the baseline.
      idle:      IdleRoundRelay (proposed decsim/idle_rounds.py)
                 relay(op_id, patch, round_index), account(operation);
                 the extend_stream and separate_decode_jobs branches live here.
      relay:     InstructionRelay (proposed decsim/instruction_relay.py)
                 relay_instruction(decision, deliver); links leave Controller.

Deleted: _patch_for_operation; the fragment built only to validate
(483-484); round_ticks_for inlined at execution_runtime.
Expected size: about 200 lines.

Reading path afterwards. issue_operation reserves stream rounds and commands
the QPU; the QPU calls back _body_done and emit_idle_round; readouts go
straight to ingress. Whether an operation may start now is one call,
protected.blocks_start.

---------------------------------------------------------------------------

## 5. execution_runtime.py (211) and qpu.py (192)

execution_runtime keeps DAG readiness, start gating, completion, feedback
decisions. Resource claims (80-128) move to ResourceLedger (proposed
decsim/resource_ledger.py): claim(operation), release(operation),
holder_of(kind, resource_id). op_start_time is set once, to the cycle
boundary the QPU returns (already so after Q-064).

qpu.py stays as rewritten for Q-064 (one cycle clock, idle rounds, starts on
boundaries). Only its comment gets the six invariants stated in one place.

---------------------------------------------------------------------------

## 6. syndrome_ingress.py (559) and syndrome_buffer.py (618)

Today. Ingress and buffer both allocate and free a slot for the same round;
ingress reads the buffer's private slot dict and unwinds buffer state by
hand on exhaustion; the round key is a tuple sliced positionally; guard
lines outnumber modeling lines in three of five public methods; the packet
is stored twice.

Target. SyndromeBuffer is the single owner of round slots. It gets a named
RoundIdentity value and two public methods, assembling_fragment_count(
identity) and an accept_fragment that returns an admission outcome
(accepted, complete, refused) instead of raising on capacity. Its holds
(410-523) stay; its module comment already says what it does not do.

SyndromeIngress keeps: relay_qpu_readout (arrival after QC, controller
processing time), packing (t_pack), arbitration onto C2B, transmit to the
window manager or the feedback-memory port, snapshot. It loses its own slot
table, the exhaustion unwind (257-273), the duplicate packet store (320),
the tuple slicing (315,474,484), and the type guards on values the
controller built (184-189, 212-219). Overflow policy DROP_ROUND and the
reassembly timeout stay but as one clearly bounded block. Expected size:
about 350 lines.

Reading path afterwards. A readout enters relay_qpu_readout, becomes one
fragment, buffer.accept_fragment says whether the round is complete, the
complete packet is queued for C2B and delivered to the window manager. Two
files, one slot owner.

---------------------------------------------------------------------------

## 7. run_spec.py (561), factories.py (569)

Today. RunSpec is config, composition root, result records and policy: ten
default selections, five compatibility rules, a metric decision, eight
post-construction attribute installs, calls into private engine phases.

Target.
- defaults.py (proposed): the ten defaults (Baseline, SlidingWindowScheme,
  GateRounds, Eager, Ignore, FifoScheduler, logical_reference_profile,
  num_units, distance-3 code selection, factory) and the five compatibility
  rules as functions with one-line reasons. run_spec imports them.
- run_spec._build_once becomes wiring in construction order, each
  collaborator built once with everything it needs, then bind_run_seed,
  engine.run(), finalize. No attribute installs after construction; no
  private engine calls (engine gets a public run_to_quiescence()).
- PrimaryRunResult and CompletedRun stay here; capture_primary_result moves
  in from views.py, breaking the import cycle.
- factories.py unchanged except _check_factory_decode_service moves to
  defaults.py as a compatibility rule.

---------------------------------------------------------------------------

## 8. links.py (972), link_profiles.py (300)

links.py keeps the closed path vocabulary, path rules, LinkConfig and card
types, the FIFO Link, LinkModel.reserve and the ledger. The JSON reporting
(topology_json_value, traffic_json_value, _transfer_json, 828-972) moves to
link_reports.py (proposed). _validate_attribution_shape shrinks to the rules
a caller can violate; the request/attribution cross-check that the window
manager satisfies by construction goes. Provenance-as-data inside the
mechanism (payload_selection, the two relation types, SoftOutputSource.
references) is retired or moved to reporting; source strings on config
cards stay. link_profiles.py unchanged (it is the exemplar of data separate
from mechanism).

---------------------------------------------------------------------------

## 9. message.py (1033), protocols.py (540)

message.py stays one vocabulary module. Changes: the six buffer hold tokens
(135-152) move to syndrome_buffer, which owns them; Submission, Directive,
OutcomeDirective come in from protocols.py because they are values, not
seams; the 14 classes without a docstring get one line each; the
__post_init__ blocks that re-prove internally built values (SyndromeRoundPacket
268-293, OperationWindowPlan 504-556, RetainedSyndromeFragment 239-245 and
their kin) go. protocols.py becomes seams only: the 6 load-bearing protocols
plus the 18 documentation protocols, each with the invariant its
implementers promise.

---------------------------------------------------------------------------

## 10. views.py (404), metrics.py (742)

views keeps the frozen views. Every private read (window_manager._ops,
_selected_request_keys, controller._round_count_for, engine internals,
device._truth) is replaced by a small public accessor on the owner.
capture_primary_result moves to run_spec. metrics keeps the observers;
BurstEscalationDetector is deleted from core (never wired) or moved to
experiments; the concrete factory call at 734 goes through the factory
protocol; the thin rows() re-labelling stays (it is the reporting edge).

---------------------------------------------------------------------------

## 11. speculative_recovery.py (419), dynamic_windows.py (313)

speculative_recovery keeps its five-method seam (begin, complete,
after_commit, blocks_finality, blocks_stream_segment) and is rewritten so
that _repair and _reset_window act only through BoundaryCourier,
LogicalLedger, WindowRetention and the manager's public commit methods; the
thirty private touches go. dynamic_windows keeps its six-method seam; seal
returns a trim-and-recheck plan the manager applies, removing the FAIL_STOP
admission at 187-192.

---------------------------------------------------------------------------

## 12. schemes.py (399), decoders.py (380), switching.py (352)

schemes: a WindowScheme protocol (plan_operation, data_complete,
validate_buffer) plus one shared default_data_complete function replaces the
inheritance chain; NaiveOnlineScheme's empty override disappears; the two
type(scheme) is checks in switching become a scheme property.
decoders: SwitchingDecoder deleted; the two subclass-to-bind-a-lambda cases
become PresetLatencyDecoder(fn); soft output stays behind DecodeResult.
switching: the three validators (243-323) become SwitchingPreconditions
(proposed, may live in defaults.py as compatibility rules); the class
docstring becomes the invariant of on_decode_outcome; the paper comparison
moves to guide/.

---------------------------------------------------------------------------

## 13. Order of work and the lock

Phase 0 (deletions, inlines, ceremony, comments) -> Phase 1 (sections 4, 5,
6) -> Phase 2 (sections 2, 3) -> Phase 3 (sections 1, 11) -> Phase 4
(sections 7 to 10, 12). Each bullet above is one commit. After every commit:
python -m pytest tests -q, python tmp/validation/smoke_run.py, and after
every phase python -m experiments.baseline_closed_loop and the five gates;
all byte-identical to HEAD 5299966 (measured software rows may vary in wall
clock only). guide/baseline-status.md records each phase.
