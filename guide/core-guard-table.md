# Guard table: every raise and __post_init__ in the core, keep or delete

Written 2026-08-18 for Phase 0 of guide/core-rewrite-plan.md, before any
deletion. One row per `raise` in decsim/*.py and decsim/adapters/*.py at
commit 7eda59b (598 rows). Backends, detector_error_model, frontends and
stimcircuits are outside the table (external boundaries and sealed packages).

## Rules

Delete (Phase 1, guard by guard, lock green after each file):
- every `type(x) is not T`, `isinstance`, `is_stable_identity`, `callable(...)`
  guard on an internal path;
- every dataclass `__post_init__` on internal vocabulary (message.py, links
  relation types, ingress and buffer records) that only re-proves values
  another decsim module built; a `__post_init__` that normalizes a value
  keeps the assignment and loses the raise;
- every check that re-proves wiring run_spec makes by construction ("is not
  connected", "already loaded", "has no callback", "different engine"): gone
  when Phase 2 injects everything by constructor;
- every check that re-proves an identity another module just built (request
  key mismatch, result identity equals job identity, "requires a request key");
- the engine's phase and stable-boundary ceremony (Phase 2 gives the engine
  one public run()).

Keep:
- the six invariants of the plan plus the two engine clock checks;
- modeled failure semantics: overflow fail-stop, reassembly timeout, buffer
  and decoder memory capacity exhaustion, FAIL_STOP seal, decode work
  unsettled at the end of a run, pending escalations at the end;
- user config rules: RunSpec compatibility, config card value ranges and
  normalization (LinkConfig, TimingConfig, DecoderMemoryConfig, ...),
  strategy and scheme option conflicts, the operation graph checks in the
  planner, factory and metric arguments, explicit seed conflicts;
- external boundaries: Stim circuits and sampling in stim_device, decoder
  backend outcomes in adapters/window_decode_results;
- state-machine invariants whose violation would silently corrupt results:
  strong tier registry and rephase, boundary courier, logical ledger
  coverage, buffer slot states and holds, protected stream regions,
  resource claims, seed path uniqueness. These are the algorithm's own
  invariants, not validation of someone else's values; the owner may cut
  further later, one row at a time.

## Counts per file

| file | keep | delete |
|---|---|---|
| decsim/adapters/stim_device.py | 16 | 6 |
| decsim/adapters/window_decode_results.py | 42 | 4 |
| decsim/codes.py | 1 | 6 |
| decsim/config.py | 2 | 0 |
| decsim/controller.py | 23 | 6 |
| decsim/decoder_engine.py | 4 | 0 |
| decsim/decoder_manager.py | 16 | 22 |
| decsim/decoder_memory.py | 3 | 6 |
| decsim/decoder_memory_transfer.py | 2 | 2 |
| decsim/decoders.py | 3 | 6 |
| decsim/devices.py | 2 | 0 |
| decsim/dynamic_windows.py | 6 | 0 |
| decsim/engine.py | 4 | 14 |
| decsim/execution_runtime.py | 9 | 1 |
| decsim/factories.py | 5 | 7 |
| decsim/links.py | 29 | 15 |
| decsim/message.py | 0 | 45 |
| decsim/metrics.py | 12 | 0 |
| decsim/pauli_frame.py | 5 | 1 |
| decsim/planner.py | 18 | 3 |
| decsim/qpu.py | 5 | 7 |
| decsim/rounds.py | 2 | 0 |
| decsim/run_spec.py | 20 | 6 |
| decsim/schemes.py | 7 | 1 |
| decsim/seeding.py | 9 | 6 |
| decsim/speculative_recovery.py | 10 | 0 |
| decsim/switching.py | 13 | 9 |
| decsim/syndrome_buffer.py | 26 | 8 |
| decsim/syndrome_ingress.py | 6 | 12 |
| decsim/views.py | 5 | 1 |
| decsim/window_manager.py | 79 | 20 |
| total | 384 | 214 |

## Rows

### decsim/adapters/stim_device.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 85 | StimDevice._validate_sample_key | `sample_key_type not in (int, str)` -> TypeError( f"stream_id must be an int or str so the run's sa | keep | external boundary: Stim circuits and sampling contract |
| 93 | StimDevice._validated_root_seed | `not isinstance(self._seed, Integral)` -> ValueError( f"seed must be None or a 64-bit unsigned integer | delete | type guard |
| 98 | StimDevice._validated_root_seed | `not 0 <= root_seed < (1 << 64)` -> ValueError( f"seed must be None or a 64-bit unsigned integer | keep | external boundary: Stim circuits and sampling contract |
| 133 | StimDevice.begin_operation | `type(value) is not int or value < 1` -> ValueError(f"{name} must be a positive built-in int") | delete | type guard |
| 135 | StimDevice.begin_operation | `op.circuit is None` -> ValueError("StimDevice operations require a circuit") | keep | external boundary: Stim circuits and sampling contract |
| 138 | StimDevice.begin_operation | `op.stream_offset is not None or segment_round_count != source_round_count` -> ValueError("standalone duration must equal its source durati | keep | external boundary: Stim circuits and sampling contract |
| 141 | StimDevice.begin_operation | `type(op.stream_offset) is not int or op.stream_offset < 0` -> ValueError("stream_offset must be a nonnegative built-in int | delete | type guard |
| 143 | StimDevice.begin_operation | `op.stream_offset + segment_round_count > source_round_count` -> ValueError("stream segment extends beyond its finite source" | keep | external boundary: Stim circuits and sampling contract |
| 194 | StimDevice.finalize_stream_round | `key not in self._dets` -> RuntimeError("terminal finalizer requires a sampled stream") | keep | external boundary: Stim circuits and sampling contract |
| 197 | StimDevice.finalize_stream_round | `binding is None` -> RuntimeError("sampled stream has no source binding") | keep | external boundary: Stim circuits and sampling contract |
| 200 | StimDevice.finalize_stream_round | `type(source_round_count) is not int or source_round_count < 1` -> ValueError("source_round_count must be a positive built-in i | delete | type guard |
| 202 | StimDevice.finalize_stream_round | `source_round_count != bound_round_count` -> ValueError("finalizer source duration differs from its bindi | keep | external boundary: Stim circuits and sampling contract |
| 204 | StimDevice.finalize_stream_round | `op.circuit is None or str(op.circuit) != circuit_text` -> ValueError("finalizer circuit differs from its source bindin | keep | external boundary: Stim circuits and sampling contract |
| 206 | StimDevice.finalize_stream_round | `type(op.stream_offset) is not int or op.stream_offset < 0` -> ValueError("finalizer offset must be a nonnegative built-in  | delete | type guard |
| 208 | StimDevice.finalize_stream_round | `op.stream_offset + 1 != source_round_count` -> ValueError("finalizer is not at the final source round") | keep | external boundary: Stim circuits and sampling contract |
| 211 | StimDevice.finalize_stream_round | `detector_ids is None` -> ValueError("terminal finalizer has no declared detector ids" | keep | external boundary: Stim circuits and sampling contract |
| 213 | StimDevice.finalize_stream_round | `key not in self._terminal_data_bits` -> ValueError("terminal finalizer has no raw data-bit size") | keep | external boundary: Stim circuits and sampling contract |
| 228 | StimDevice.idle_round_payloads | `stream_id not in self._dets or binding is None` -> RuntimeError("idle emission requires a sampled bound stream" | keep | external boundary: Stim circuits and sampling contract |
| 230 | StimDevice.idle_round_payloads | `type(global_round) is not int or not 1 <= global_round <= binding[1]` -> ValueError("idle round is outside the finite source") | delete | type guard |
| 254 | StimDevice._source_rounds | `binding[0] != circuit_text` -> ValueError("circuit differs from the bound finite source") | keep | external boundary: Stim circuits and sampling contract |
| 256 | StimDevice._source_rounds | `binding[1] != source_round_count` -> ValueError("source duration differs from the bound finite so | keep | external boundary: Stim circuits and sampling contract |
| 290 | StimDevice.validate_stream_length | `(unconditional)` -> RuntimeError( f"{stream_op.name} sealed at {stream_round_cou | keep | external boundary: Stim circuits and sampling contract |

### decsim/adapters/window_decode_results.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 66 | _binary_tuple | `value is None` -> ValueError(f"{name} is required") | keep | external boundary: outputs of third-party decoder backends |
| 70 | _binary_tuple | `(unconditional)` -> TypeError(f"{name} must be a one-dimensional iterable") | keep | external boundary: outputs of third-party decoder backends |
| 74 | _binary_tuple | `not isinstance(bit, Integral) or int(bit) not in (0, 1)` -> ValueError(f"{name}[{index}] must be binary") | keep | external boundary: outputs of third-party decoder backends |
| 85 | _float_tuple | `(unconditional)` -> TypeError(f"{name} must be a one-dimensional iterable") | keep | external boundary: outputs of third-party decoder backends |
| 89 | _float_tuple | `isinstance(item, bool) or not isinstance(item, Real)` -> TypeError(f"{name}[{index}] must be a real number") | keep | external boundary: outputs of third-party decoder backends |
| 92 | _float_tuple | `math.isnan(number)` -> ValueError(f"{name}[{index}] cannot be NaN") | keep | external boundary: outputs of third-party decoder backends |
| 101 | _nonnegative_integer | `isinstance(value, bool) or not isinstance(value, Integral)` -> TypeError(f"{name} must be an integer or None") | keep | external boundary: outputs of third-party decoder backends |
| 104 | _nonnegative_integer | `normalized < 0` -> ValueError(f"{name} must be nonnegative") | keep | external boundary: outputs of third-party decoder backends |
| 110 | _validate_fingerprint | `not isinstance(value, str) or len(value) != 64` -> ValueError(f"{name} must be a lowercase SHA-256 hexadecimal  | keep | external boundary: outputs of third-party decoder backends |
| 112 | _validate_fingerprint | `value != value.lower()` -> ValueError(f"{name} must be a lowercase SHA-256 hexadecimal  | keep | external boundary: outputs of third-party decoder backends |
| 116 | _validate_fingerprint | `(unconditional)` -> ValueError( f"{name} must be a lowercase SHA-256 hexadecimal | keep | external boundary: outputs of third-party decoder backends |
| 138 | BackendDecodeOutcome.__post_init__ | `not isinstance(self.status, BackendDecodeStatus)` -> TypeError("status must be a BackendDecodeStatus") | delete | type-exactness inside a config-card __post_init__ |
| 142 | BackendDecodeOutcome.__post_init__ | `self.failure_reason is not None and not isinstance(             self.failure_rea` -> TypeError("failure_reason must be a BackendFailureReason or  | delete | type-exactness inside a config-card __post_init__ |
| 145 | BackendDecodeOutcome.__post_init__ | `self.failure_reason is not None` -> ValueError("a successful outcome cannot have a failure reaso | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 148 | BackendDecodeOutcome.__post_init__ | `self.failure_reason is None` -> ValueError("a failed outcome requires a typed failure reason | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 150 | BackendDecodeOutcome.__post_init__ | `self.failure_reason not in _STATUS_REASONS[self.status]` -> ValueError( f"{self.failure_reason.value} is not valid for " | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 213 | DecoderAttemptFailed.__init__ | `not isinstance(outcome, BackendDecodeOutcome)` -> TypeError("outcome must be a BackendDecodeOutcome") | keep | external boundary: outputs of third-party decoder backends |
| 215 | DecoderAttemptFailed.__init__ | `outcome.succeeded` -> ValueError("a successful outcome cannot fail a decoder attem | keep | external boundary: outputs of third-party decoder backends |
| 240 | _canonical_bytes | `math.isnan(number)` -> ValueError("fingerprinted values cannot contain NaN") | keep | external boundary: outputs of third-party decoder backends |
| 249 | _canonical_bytes | `value.dtype.hasobject` -> TypeError("object arrays cannot be fingerprinted") | keep | external boundary: outputs of third-party decoder backends |
| 271 | _canonical_bytes | `(unconditional)` -> TypeError( f"cannot fingerprint value of type {type(value)._ | keep | external boundary: outputs of third-party decoder backends |
| 309 | validate_backend_outcome | `not isinstance(outcome, BackendDecodeOutcome)` -> TypeError("backend must return a BackendDecodeOutcome") | keep | external boundary: outputs of third-party decoder backends |
| 312 | validate_backend_outcome | `outcome.fault_model_fingerprint != expected_fingerprint` -> ValueError("backend outcome belongs to a different fault mod | keep | external boundary: outputs of third-party decoder backends |
| 316 | validate_backend_outcome | `syndrome.ndim != 1 or syndrome.shape[0] != placed_faults.check.shape[0]` -> ValueError("syndrome arity does not match the placed fault m | keep | external boundary: outputs of third-party decoder backends |
| 318 | validate_backend_outcome | `not np.all((syndrome == 0) \| (syndrome == 1))` -> ValueError("syndrome must contain only binary values") | keep | external boundary: outputs of third-party decoder backends |
| 324 | validate_backend_outcome | `correction.shape != (placed_faults.check.shape[1],)` -> ValueError("backend correction has the wrong fault-model ari | keep | external boundary: outputs of third-party decoder backends |
| 332 | validate_backend_outcome | `outcome.reconstructed_syndrome is not None and tuple(             int(bit) for b` -> ValueError( "backend reconstructed syndrome does not match i | keep | external boundary: outputs of third-party decoder backends |
| 339 | validate_backend_outcome | `projection is None or correction is None` -> ValueError( "component correction requires a physical correc | keep | external boundary: outputs of third-party decoder backends |
| 348 | validate_backend_outcome | `tuple(int(bit) for bit in component) != outcome.component_correction` -> ValueError("component correction does not match the local pr | keep | external boundary: outputs of third-party decoder backends |
| 354 | validate_backend_outcome | `outcome.succeeded and not np.array_equal(         reconstructed,         syndrom` -> ValueError("successful correction does not match the syndrom | keep | external boundary: outputs of third-party decoder backends |
| 367 | empty_fault_model_outcome | `placed_faults.check.shape[1] != 0` -> ValueError("empty-fault outcome requires a model with zero c | keep | external boundary: outputs of third-party decoder backends |
| 370 | empty_fault_model_outcome | `syndrome.ndim != 1 or syndrome.shape[0] != placed_faults.check.shape[0]` -> ValueError("syndrome arity does not match the empty fault mo | keep | external boundary: outputs of third-party decoder backends |
| 372 | empty_fault_model_outcome | `not np.all((syndrome == 0) \| (syndrome == 1))` -> ValueError("syndrome must contain only binary values") | keep | external boundary: outputs of third-party decoder backends |
| 418 | OfflineShotRecord.__post_init__ | `isinstance(self.shot_index, bool) or not isinstance(             self.shot_index` -> TypeError("shot_index must be an integer") | delete | type-exactness inside a config-card __post_init__ |
| 420 | OfflineShotRecord.__post_init__ | `self.shot_index < 0` -> ValueError("shot_index must be nonnegative") | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 422 | OfflineShotRecord.__post_init__ | `self.attempted is not True` -> ValueError("every offline shot record must be an attempted s | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 430 | OfflineShotRecord.__post_init__ | `not outcomes` -> ValueError("window_outcomes must contain at least one attemp | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 432 | OfflineShotRecord.__post_init__ | `not all(isinstance(outcome, BackendDecodeOutcome) for outcome in outcomes)` -> TypeError("window_outcomes must contain BackendDecodeOutcome | delete | type-exactness inside a config-card __post_init__ |
| 439 | OfflineShotRecord.__post_init__ | `accepted and prediction is None` -> ValueError("an all-success shot requires a logical predictio | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 441 | OfflineShotRecord.__post_init__ | `not accepted and prediction is not None` -> ValueError("a failed shot cannot carry an accepted predictio | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 443 | OfflineShotRecord.__post_init__ | `prediction is not None and len(prediction) != len(truth)` -> ValueError("logical prediction and sampled truth have differ | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 486 | summarize_offline_shots | `not records` -> ValueError("at least one attempted shot is required") | keep | external boundary: outputs of third-party decoder backends |
| 488 | summarize_offline_shots | `not all(isinstance(record, OfflineShotRecord) for record in records)` -> TypeError("records must contain OfflineShotRecord values") | keep | external boundary: outputs of third-party decoder backends |
| 491 | summarize_offline_shots | `len(set(shot_indices)) != len(shot_indices)` -> ValueError("shot_index values must be unique") | keep | external boundary: outputs of third-party decoder backends |
| 521 | check_syndrome_size | `(unconditional)` -> ValueError( f"{job.label}: payload bits ({syndrome.size}) do | keep | external boundary: outputs of third-party decoder backends |
| 539 | result_from_selected_faults | `selected.ndim != 1 or selected.shape[0] != placed_faults.check.shape[1]` -> ValueError( f"{job.label}: selected correction has shape {se | keep | external boundary: outputs of third-party decoder backends |

### decsim/codes.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 18 | _require_positive_int | `type(value) is not int` -> TypeError(f"{field_name} must be a built-in int; got {value! | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 20 | _require_positive_int | `value <= 0` -> ValueError(f"{field_name} must be positive; got {value!r}") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 107 | BBCodeModel.__post_init__ | `self.n % 2` -> ValueError(f"n must be even; got {self.n!r}") | delete | __post_init__ re-proves values another decsim module built |
| 109 | BBCodeModel.__post_init__ | `self.k > self.n` -> ValueError(f"k must not exceed n; got k={self.k!r}, n={self. | delete | __post_init__ re-proves values another decsim module built |
| 111 | BBCodeModel.__post_init__ | `self.d > self.n` -> ValueError(f"d must not exceed n; got d={self.d!r}, n={self. | delete | __post_init__ re-proves values another decsim module built |
| 119 | BBCodeModel.__post_init__ | `type(value) is not int` -> TypeError("buffer_rounds_override must be a built-in int") | delete | __post_init__ re-proves values another decsim module built |
| 121 | BBCodeModel.__post_init__ | `value < 0` -> ValueError("buffer_rounds_override must be nonnegative") | delete | __post_init__ re-proves values another decsim module built |

### decsim/config.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 35 | TimingConfig.__post_init__ | `not math.isfinite(value) or value < 0` -> ValueError(f"{name} must be a finite nonnegative number") | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 37 | TimingConfig.__post_init__ | `value > 0 and us(value) == 0` -> ValueError(f"{name} is positive but rounds to zero ticks") | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |

### decsim/controller.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 35 | Controller.__init__ | `type(binary_availability_ticks) is not int or binary_availability_ticks < 0` -> TypeError("binary_availability_ticks must be a nonnegative e | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 81 | Controller._index_protected_regions | `stream_id in self._stream_owner_by_id` -> ValueError(f"duplicate protected stream {stream_id}") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 84 | Controller._index_protected_regions | `owner is None or tuple(owner.patches) != (region.patch_id,)` -> ValueError(f"protected stream {stream_id} owner/patch mismat | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 91 | Controller._index_protected_regions | `operation is None or region.patch_id not in operation.patches` -> ValueError( f"protected stream {stream_id} invalid {endpoint | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 100 | Controller.load_program | `hasattr(self, "_loaded_program")` -> RuntimeError("controller sequencer program is already loaded | delete | re-proves wiring that run_spec makes by construction (Phase 2 constructor injection) |
| 123 | Controller.load_program | `(unconditional)` -> ValueError( f"external source {source.id} from {source_role} | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 139 | Controller.connect_runtime | `self.runtime is not None` -> RuntimeError("controller runtime is already connected") | delete | re-proves wiring that run_spec makes by construction (Phase 2 constructor injection) |
| 178 | Controller._protected_feedback_stream | `len(ordered_stream_ids) > 1` -> ValueError( f"feedback source {operation.id} spans protected | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 188 | Controller._protected_feedback_stream | `declared_stream_id not in (None, stream_id)` -> ValueError( f"feedback source {operation.id} has conflicting | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 192 | Controller._protected_feedback_stream | `declared_stream_offset not in (None, current_round)` -> ValueError( f"feedback source {operation.id} has conflicting | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 201 | Controller._activate_protected_regions | `protected_patches and operation.emits_detector_data` -> ValueError( f"operation {operation.id} duplicates protected  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 207 | Controller._activate_protected_regions | `region.patch_id in self._active_stream_id_by_patch                     or region` -> RuntimeError( f"protected patch {region.patch_id!r} already  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 266 | Controller._open_protected_boundary | `region is None` -> RuntimeError(f"protected stream {stream_id} is not active") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 269 | Controller._open_protected_boundary | `expected_tick != self.engine.now` -> RuntimeError( f"protected stream {stream_id} boundary tick m | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 273 | Controller._open_protected_boundary | `region.patch_id in self._boundary_open_patches` -> RuntimeError( f"protected stream {stream_id} boundary alread | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 289 | Controller._emit_protected_round | `region is None` -> RuntimeError(f"protected stream {stream_id} is not active") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 291 | Controller._emit_protected_round | `region.patch_id not in self._boundary_open_patches` -> RuntimeError( f"protected stream {stream_id} boundary is not | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 322 | Controller._seal_protected_region | `region is None` -> RuntimeError(f"protected stream {stream_id} is not active") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 324 | Controller._seal_protected_region | `stream_id not in self._close_requested_stream_ids` -> RuntimeError(f"protected stream {stream_id} was not closed") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 326 | Controller._seal_protected_region | `stream_id in self._next_boundary_tick_by_stream_id` -> RuntimeError(f"protected stream {stream_id} has a pending bo | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 328 | Controller._seal_protected_region | `self._last_emission_tick_by_stream_id.get(stream_id) != self.engine.now` -> RuntimeError( f"protected stream {stream_id} lacks final-rou | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 347 | Controller._reserve_stream_rounds | `stream_offset != next_round - 1` -> RuntimeError( f"{operation.name} must finalize stream round  | keep | user workload rule: stream segments declare consecutive offsets |
| 356 | Controller._reserve_stream_rounds | `stream_offset < next_round` -> RuntimeError( f"{operation.name} starts at stream round {str | keep | user workload rule: stream segments declare consecutive offsets |
| 389 | Controller._request_protected_region_closes | `self._active_region_by_stream_id.get(stream_id) is not region` -> RuntimeError( f"protected stream {stream_id} ended while ina | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 392 | Controller._request_protected_region_closes | `self._active_stream_id_by_patch.get(region.patch_id) != stream_id` -> RuntimeError( f"protected stream {stream_id} lost patch owne | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 395 | Controller._request_protected_region_closes | `self._next_boundary_tick_by_stream_id.get(stream_id) != self.engine.now` -> RuntimeError( f"protected stream {stream_id} ended off bound | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 467 | Controller.accept_qpu_readout | `type(readout) is not QPUReadout` -> TypeError("controller accepts only exact QPUReadout values") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 469 | Controller.accept_qpu_readout | `type(route) is not SyndromePacketRoute` -> TypeError("controller requires a typed packet route") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 471 | Controller.accept_qpu_readout | `self.syndrome_ingress is None` -> RuntimeError("controller syndrome ingress is not connected") | delete | re-proves wiring that run_spec makes by construction (Phase 2 constructor injection) |

### decsim/decoder_engine.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 41 | DecoderStage.__post_init__ | `self.cycles_per_job < 0 or self.cycles_per_round < 0` -> ValueError(f"stage {self.name!r} cycles must be nonnegative" | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 57 | DecoderTiming.__post_init__ | `not math.isfinite(self.frequency_mhz) or self.frequency_mhz <= 0` -> ValueError("frequency_mhz must be finite and positive") | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 59 | DecoderTiming.__post_init__ | `any(s.name == ALGORITHM_STAGE for s in self.before + self.after)` -> ValueError(f"{ALGORITHM_STAGE!r} names the decoder itself, n | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 185 | DecoderEngine.decode | `result is None` -> RuntimeError(f"{job.label!r} has not completed") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |

### decsim/decoder_manager.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 96 | DecoderManager.__init__ | `not isinstance(self.decoder_memory_transfer, DecoderMemoryTransfer)             ` -> TypeError( "decoder_memory_transfer must implement DecoderMe | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 102 | DecoderManager.__init__ | `transport_engine is not engine` -> ValueError("decoder_memory_transfer uses a different engine" | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 118 | DecoderManager.__init__ | `"default" not in unit_pools` -> ValueError(f'unit_pools must include a "default" pool ' f'(g | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 122 | DecoderManager.__init__ | `units < 1` -> ValueError( f"pool {pool_name!r} needs at least 1 unit (got  | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 185 | DecoderManager.enqueue | `reserve_transfer is not None and not callable(reserve_transfer)` -> TypeError("reserve_transfer must be callable or None") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 201 | DecoderManager._reject_spent_job | `job.cancelled or job.completed or job.submitted` -> RuntimeError( f"decode job {job.label!r} for window " f"({jo | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 210 | DecoderManager._admit_strong_request | `job.request_key is None` -> RuntimeError("built-in strong decode requires a request key" | delete | re-proves an identity another module just built |
| 213 | DecoderManager._admit_strong_request | `key in self._running_strong_decodes                 or key in self._completed_st` -> RuntimeError( f"duplicate strong decode for window {key}: a  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 223 | DecoderManager._admit_weak_decode | `job.request_key is None` -> RuntimeError("built-in weak decode requires a request key") | delete | re-proves an identity another module just built |
| 225 | DecoderManager._admit_weak_decode | `key in self._unresolved_weak_decodes` -> RuntimeError( f"second weak decode for window {key} while th | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 281 | DecoderManager.check_strong_route | `self.decoder_for(strong_job) is self.decoder_for(weak_job)` -> RuntimeError( "Strong job routes to the same decoder as the  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 391 | DecoderManager._merge_strong_batch | `has_model or has_bits` -> RuntimeError( "bulk_strong only merges timing-only strong re | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 464 | DecoderManager.receive_once | `delivered_job is not job` -> RuntimeError("decoder memory transport delivered a different | delete | re-proves wiring that run_spec makes by construction (Phase 2 constructor injection) |
| 466 | DecoderManager.receive_once | `self.engine.now != expected_delivery_tick` -> RuntimeError("decoder memory transport delivered at the wron | delete | re-proves wiring that run_spec makes by construction (Phase 2 constructor injection) |
| 468 | DecoderManager.receive_once | `job.decoder_input is not None` -> RuntimeError("decoder memory transport materialized an input | delete | re-proves wiring that run_spec makes by construction (Phase 2 constructor injection) |
| 591 | DecoderManager._on_decode_done | `not awaiting and (directive.extra is not None                              or di` -> RuntimeError(f"{name} directive cannot carry strong identity | delete | re-proves an identity another module just built |
| 605 | DecoderManager._on_decode_done | `strong_request_key is None or carriers                         or serial_job.req` -> RuntimeError("serial directive request key mismatch") | delete | re-proves an identity another module just built |
| 608 | DecoderManager._on_decode_done | `carriers` -> RuntimeError( "parallel directive cannot provide an explicit | delete | re-proves an identity another module just built |
| 612 | DecoderManager._on_decode_done | `len(carriers) != 1` -> RuntimeError( "parallel strong selection needs exactly one c | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 629 | DecoderManager._on_decode_done | `self.on_window_decoded is None` -> RuntimeError("DecoderManager has no window completion callba | delete | re-proves wiring that run_spec makes by construction (Phase 2 constructor injection) |
| 662 | DecoderManager._decode_and_validate_result | `not isinstance(result, DecodeResult)` -> TypeError( f"decoder for job ({job.op_id}, {job.window_id})  | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 668 | DecoderManager._decode_and_validate_result | `actual != expected` -> RuntimeError( f"decoder result identity {actual} does not ma | delete | re-proves an identity another module just built |
| 683 | DecoderManager._validate_logical_observables | `type(logical_observables) is not tuple` -> TypeError( f"job ({job.op_id}, {job.window_id}) logical_obse | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 689 | DecoderManager._validate_logical_observables | `type(bit) is not int` -> TypeError( f"job ({job.op_id}, {job.window_id}) " f"logical_ | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 694 | DecoderManager._validate_logical_observables | `bit not in (0, 1)` -> ValueError( f"job ({job.op_id}, {job.window_id}) " f"logical | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 729 | DecoderManager.admitted_strong_work_snapshot | `any(candidate is not job for candidate in queued_matches)` -> RuntimeError("strong-work identity collision in ready queues | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 731 | DecoderManager.admitted_strong_work_snapshot | `len(queued_matches) > 1` -> RuntimeError("one strong job appears in multiple ready queue | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 734 | DecoderManager.admitted_strong_work_snapshot | `queued_matches` -> RuntimeError("running strong job also remains queued") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 787 | DecoderManager._prepare_strong_result_deliveries | `populated_field_names` -> RuntimeError( "merged strong decode returned accuracy-bearin | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 813 | DecoderManager._complete_strong_result | `self.on_strong_window_decoded is None` -> RuntimeError( "DecoderManager has no strong completion callb | delete | re-proves wiring that run_spec makes by construction (Phase 2 constructor injection) |
| 822 | DecoderManager._complete_strong_result | `key in self._running_strong_decodes` -> RuntimeError( f"strong result for window {key} arrived after | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 827 | DecoderManager._complete_strong_result | `not self._destination_may_consume_strong(key)` -> RuntimeError( f"strong result for window {key} has no destin | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 849 | DecoderManager._record_request | `job.request_key is None or job.request_created_ticks is None` -> RuntimeError("terminal built-in request has no identity") | delete | re-proves an identity another module just built |
| 852 | DecoderManager._record_request | `window is None` -> RuntimeError("terminal built-in request has no window") | delete | re-proves an identity another module just built |
| 882 | DecoderManager._record_service | `dispatch is None or job.pool is None` -> RuntimeError("terminal decoder service has no dispatch") | delete | re-proves an identity another module just built |
| 889 | DecoderManager.terminal_request_records_snapshot | `self._terminal_request_records is None` -> RuntimeError("switching record capture is disabled") | delete | re-proves an identity another module just built |
| 894 | DecoderManager.terminal_service_records_snapshot | `self._terminal_service_records is None` -> RuntimeError("switching record capture is disabled") | delete | re-proves an identity another module just built |
| 924 | DecoderManager.check_decode_work_settled | `unsettled` -> RuntimeError( f"the run ended with decode work unsettled ({d | keep | modeled failure semantics (fail-stop, timeout, exhaustion, unsettled work) |

### decsim/decoder_memory.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 55 | DecoderMemoryConfig.__post_init__ | `type(pool) is not str` -> TypeError("decoder memory pool names must be str") | delete | type-exactness inside a config-card __post_init__ |
| 57 | DecoderMemoryConfig.__post_init__ | `type(capacity) is not int or capacity < 1` -> ValueError(f"pool {pool!r} needs a positive int round capaci | delete | type-exactness inside a config-card __post_init__ |
| 75 | MaterializedSyndromeRound.__post_init__ | `type(self.round_index) is not int` -> TypeError("round_index must be an exact built-in int") | delete | __post_init__ re-proves values another decsim module built |
| 79 | MaterializedSyndromeRound.__post_init__ | `not same_stable_identity(                     fragment.operation_id, self.operat` -> ValueError( "materialized fragments must share operation ide | delete | __post_init__ re-proves values another decsim module built |
| 118 | _check_detector_row_layout | `not same_stable_identity(round_input.operation_id, job.op_id)` -> ValueError( f"{getattr(job, 'label', '')}: model-backed deco | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 146 | _check_detector_row_layout | `input_row_identities != model_row_identities` -> ValueError( f"{getattr(job, 'label', '')}: canonical decoder | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 157 | materialize_decoder_input | `type(payload) is not RetainedSyndromeFragment` -> TypeError( "every job payload must be a RetainedSyndromeFrag | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 234 | DecoderMemory.deposit | `key in self._inputs` -> RuntimeError(f"unit {self.pool!r}#{self.unit} already holds  | delete | re-proves wiring that run_spec makes by construction (Phase 2 constructor injection) |
| 238 | DecoderMemory.deposit | `self.capacity_rounds is not None and needed > self.capacity_rounds` -> DecoderMemoryCapacityExhaustion( pool=self.pool, unit=self.u | keep | modeled failure semantics (fail-stop, timeout, exhaustion, unsettled work) |

### decsim/decoder_memory_transfer.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 37 | FixedLatencyDecoderMemoryTransfer.deliver | `type(delay_ticks) is not int` -> TypeError("delay_ticks must be an exact int") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 39 | FixedLatencyDecoderMemoryTransfer.deliver | `delay_ticks < 0` -> ValueError("delay_ticks must be nonnegative") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 41 | FixedLatencyDecoderMemoryTransfer.deliver | `not callable(receiver)` -> TypeError("decoder input receiver must be callable") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 44 | FixedLatencyDecoderMemoryTransfer.deliver | `key in self._in_flight_keys` -> RuntimeError( f"decoder input for {job.label!r} is already i | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |

### decsim/decoders.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 47 | _check_probability | `not math.isfinite(normalized) or not 0 <= normalized <= 1` -> ValueError(f"{field_name} must be finite and in [0, 1]") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 62 | CodeRouter.__init__ | `invalid_keys` -> TypeError( "CodeRouter keys must be exact built-in str or No | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 255 | _decoder_fault_model_requirement | `(unconditional)` -> TypeError( f"{type(decoder).__name__} must declare fault_mod | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 259 | _decoder_fault_model_requirement | `not isinstance(requirement, DecoderFaultModelRequirement)` -> TypeError( f"{type(decoder).__name__}.fault_model_requiremen | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 275 | _fault_model_requirement_for | `not isinstance(requirement, DecoderFaultModelRequirement)` -> TypeError( f"{type(decoder_or_router).__name__}.fault_model_ | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 359 | switch_probability_per_round | `type(d) is not int` -> TypeError(f"d must be a built-in int; got {d!r}") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 361 | switch_probability_per_round | `d <= 0` -> ValueError(f"d must be positive; got {d!r}") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 368 | probability | `type(commit_rounds) is not int` -> TypeError( "commit_rounds must be a built-in int; " f"got {c | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 373 | probability | `commit_rounds <= 0` -> ValueError( f"commit_rounds must be positive; got {commit_ro | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |

### decsim/devices.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 50 | TimingOnlyDevice.finalize_stream_round | `(unconditional)` -> ValueError("TimingOnlyDevice cannot finalize a physical stre | keep | modeled failure semantics (fail-stop, timeout, exhaustion, unsettled work) |
| 145 | SyndromeBitDevice.finalize_stream_round | `(unconditional)` -> ValueError("SyndromeBitDevice cannot finalize a physical str | keep | modeled failure semantics (fail-stop, timeout, exhaustion, unsettled work) |

### decsim/dynamic_windows.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 187 | DynamicWindows.seal | `(unconditional)` -> RuntimeError( f"FAIL_STOP: sealing stream {stream_id!r} at " | keep | modeled failure semantics (fail-stop, timeout, exhaustion, unsettled work) |
| 199 | DynamicWindows.seal | `(unconditional)` -> RuntimeError( f"FAIL_STOP: stream {stream_id!r} sealed at "  | keep | modeled failure semantics (fail-stop, timeout, exhaustion, unsettled work) |
| 207 | DynamicWindows.seal | `(unconditional)` -> RuntimeError( f"FAIL_STOP: stream {stream_id!r} sealed at "  | keep | modeled failure semantics (fail-stop, timeout, exhaustion, unsettled work) |
| 227 | DynamicWindows.close_boundary | `stream_round_count < 1` -> ValueError( f"stream_round_count must be >= 1 (got {stream_r | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 232 | DynamicWindows.close_boundary | `stream_round_count in closed_rounds` -> RuntimeError( f"stream {stream_id!r} already closed a feedba | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 249 | DynamicWindows._reject_unsupported_boundary | `(unconditional)` -> RuntimeError( "measurement_closed live-stream boundaries ins | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |

### decsim/engine.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 46 | Engine._start_running | `self._phase != "construction"` -> RuntimeError( f"engine cannot start running from phase {self | delete | engine phase ceremony; run_spec drives the phases in order and Phase 2 gives the engine one public run() |
| 54 | Engine._begin_finalization | `self._phase != "running" or self._event_queue` -> RuntimeError( "engine finalization requires a running engine | delete | engine phase ceremony; run_spec drives the phases in order and Phase 2 gives the engine one public run() |
| 63 | Engine._complete | `self._phase != "finalizing" or self._event_queue` -> RuntimeError( "engine completion requires sealed empty final | delete | engine phase ceremony; run_spec drives the phases in order and Phase 2 gives the engine one public run() |
| 81 | Engine._raise_if_failed | `(unconditional)` -> failure | keep | SimulationFailed after a failed run (modeled) |
| 91 | Engine.schedule | `self._phase in ("finalizing", "completed")` -> RuntimeError( f"engine cannot schedule events while {self._p | delete | engine phase ceremony; run_spec drives the phases in order and Phase 2 gives the engine one public run() |
| 95 | Engine.schedule | `type(delay) is not int` -> TypeError("event delay must be a built-in int") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 97 | Engine.schedule | `delay < 0` -> ValueError( f"Cannot schedule an event in the past delay={de | keep | engine clock: simulated time stays monotone |
| 114 | Engine.add_metric | `self._phase in ("finalizing", "completed")` -> RuntimeError( f"engine cannot register metrics while {self._ | delete | engine phase ceremony; run_spec drives the phases in order and Phase 2 gives the engine one public run() |
| 118 | Engine.add_metric | `self._event_action_in_progress or self._metric_callback_in_progress` -> RuntimeError("metrics may be registered only at a stable bou | delete | stable-boundary ceremony for metric callbacks |
| 122 | Engine.add_metric | `not is_stable_string(name) or not name` -> TypeError("metric name must be a nonempty Unicode scalar str | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 124 | Engine.add_metric | `type(version) is not int or version < 1` -> TypeError("metric result_schema_version must be a positive b | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 126 | Engine.add_metric | `any(existing.name == name for existing in self.metrics)` -> ValueError(f"metric name {name!r} is already registered") | keep | user config: duplicate metric name |
| 138 | Engine.add_metric | `not is_stable_string(current_name)             or current_name != name          ` -> RuntimeError("metric identity changed during initial observa | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 144 | Engine._invoke_metric_callback | `self._event_action_in_progress or self._metric_callback_in_progress` -> RuntimeError( f"metric {callback_kind} requires a stable eng | delete | stable-boundary ceremony for metric callbacks |
| 173 | Engine.run | `self._phase == "construction"` -> RuntimeError("engine cannot run during construction") | delete | engine phase ceremony; run_spec drives the phases in order and Phase 2 gives the engine one public run() |
| 175 | Engine.run | `self._phase in ("finalizing", "completed")` -> RuntimeError( f"engine cannot run while {self._phase}" ) | delete | engine phase ceremony; run_spec drives the phases in order and Phase 2 gives the engine one public run() |
| 179 | Engine.run | `self._event_action_in_progress or self._metric_callback_in_progress` -> RuntimeError("engine may run only at a stable boundary") | delete | stable-boundary ceremony for metric callbacks |
| 184 | Engine.run | `event.time < self.now` -> ValueError(f"Event scheduled in the past: {event} " f"(now={ | keep | engine clock: simulated time stays monotone |

### decsim/execution_runtime.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 39 | ExecutionRuntime.load_program | `self.program is not None` -> RuntimeError("execution program is already loaded") | delete | re-proves wiring that run_spec makes by construction (Phase 2 constructor injection) |
| 82 | ExecutionRuntime._claim_resources | `len(set(operation.qubits)) != len(operation.qubits)` -> RuntimeError( f"{operation.name} lists a qubit more than onc | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 98 | ExecutionRuntime._claim_resources | `(unconditional)` -> RuntimeError( f"{operation.name} and {holder} share {claim.k | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 120 | ExecutionRuntime._free_resources | `holder_id is missing_holder` -> RuntimeError( f"{operation.name} cannot release unclaimed {k | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 124 | ExecutionRuntime._free_resources | `holder_id != operation.id` -> RuntimeError( f"{operation.name} cannot release {kind} resou | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 148 | ExecutionRuntime.body_done | `operation.id not in self.operations` -> RuntimeError( f"cannot complete unindexed operation id {oper | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 151 | ExecutionRuntime.body_done | `operation.id not in self.op_start_time` -> RuntimeError( f"cannot complete operation {operation.name} b | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 154 | ExecutionRuntime.body_done | `operation.id in self.body_done_time` -> RuntimeError( f"operation {operation.name} body is already c | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 202 | ExecutionRuntime.on_decision | `operation.blocked_by is None` -> RuntimeError( f"release decision targets {operation.name}, " | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 206 | ExecutionRuntime.on_decision | `operation_id in self.decode_release_time` -> RuntimeError( f"{operation.name} was already released by an  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |

### decsim/factories.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 21 | _validate_production_mode | `type(production) is not str` -> TypeError("production must be a built-in str") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 23 | _validate_production_mode | `production not in ("demand", "continuous")` -> ValueError( f"production must be 'demand' or 'continuous' (g | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 28 | _validate_production_mode | `buffer_capacity is not None and (         type(buffer_capacity) is not int or bu` -> TypeError("buffer_capacity must be a positive built-in int o | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 30 | _validate_production_mode | `production == "continuous" and buffer_capacity is None` -> ValueError("continuous production needs buffer_capacity >= 1 | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 36 | _validate_exact_integer | `type(value) is not int or value < minimum` -> TypeError(f"{name} must be a {relation} built-in int") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 42 | _validate_probability | `type(value) not in (int, float)` -> TypeError(f"{name} must be a built-in int or float") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 45 | _validate_probability | `not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0` -> ValueError(f"{name} must be finite and in [0, 1]") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 52 | _validate_correction_decode_service | `type(n_corr) is not int or n_corr < 0` -> TypeError("n_corr must be a nonnegative built-in int") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 54 | _validate_correction_decode_service | `n_corr == 0 and decode_service is not None` -> ValueError("decode_service must be None when n_corr is zero" | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 56 | _validate_correction_decode_service | `n_corr > 0 and decode_service is None` -> ValueError("decode_service is required when n_corr is positi | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 328 | MultiLevelDistillationFactory.__init__ | `type(levels) is not list or not levels` -> TypeError("levels must be a nonempty built-in list") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 332 | MultiLevelDistillationFactory.__init__ | `type(level) is not DistillLevel` -> TypeError( f"levels[{index}] must be an exact DistillLevel"  | delete | type / exact-type / stable-identity / callable guard on an internal path |

### decsim/links.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 115 | _whole | `(unconditional)` -> ValueError(f"{name} must be a finite whole number") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 117 | _whole | `normalized != value` -> ValueError(f"{name} must be a finite whole number") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 132 | _finite | `not finite` -> ValueError(f"{name} must be a finite number") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 155 | BoundaryTransferRelation.__post_init__ | `self.source_window_key != expected_source` -> ValueError("boundary source does not match its request key") | delete | __post_init__ re-proves values another decsim module built |
| 163 | BoundaryTransferRelation.__post_init__ | `min(self.source_revision, self.delivery_revision) < 1` -> ValueError("boundary revisions must be positive") | delete | __post_init__ re-proves values another decsim module built |
| 177 | _require_known_basis | `basis is not LinkQuantityBasis.DIRECT_AGGREGATE             and basis is not Lin` -> ValueError(f"unknown link quantity basis {basis!r}") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 287 | LinkCapacityConfig.__post_init__ | `self.input_bits_per_us <= 0` -> ValueError("input_bits_per_us must be positive") | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 295 | LinkCapacityConfig._validate_channel_count | `self.channel_count is not None` -> ValueError( "direct aggregate capacity requires channel_coun | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 303 | LinkCapacityConfig._validate_channel_count | `self.channel_count <= 0` -> ValueError("per-channel capacity count must be positive") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 334 | PayloadSizeConfig.__post_init__ | `self.input_bits < 0` -> ValueError("input_bits must be nonnegative") | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 341 | PayloadSizeConfig._validate_channel_count | `self.channel_count is not None` -> ValueError( "direct aggregate payload requires channel_count | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 349 | PayloadSizeConfig._validate_channel_count | `self.channel_count <= 0` -> ValueError("per-channel payload count must be positive") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 381 | LinkConfig.__post_init__ | `self.propagation_latency_ticks < 0` -> ValueError("propagation_latency_ticks must be nonnegative") | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 394 | LinkEdgeConfig.__post_init__ | `self.default_payload is None and self.actual_payload_source is None` -> ValueError( "an edge requires a configured default or actual | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 401 | LinkEdgeConfig.__post_init__ | `capacity.basis is not default.basis` -> ValueError("capacity and payload bases must match") | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 403 | LinkEdgeConfig.__post_init__ | `capacity.channel_count != default.channel_count` -> ValueError("capacity and payload channel counts must match") | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 421 | TrafficAttribution.__post_init__ | `not is_stable_identity(self.operation_id)` -> TypeError("operation_id must be a stable identity") | delete | __post_init__ re-proves values another decsim module built |
| 423 | TrafficAttribution.__post_init__ | `not all(is_stable_identity(patch_id) for patch_id in self.patch_ids)` -> TypeError("patch_ids must contain stable identities") | delete | __post_init__ re-proves values another decsim module built |
| 428 | TrafficAttribution.__post_init__ | `tuple(map(stable_identity_order_key, self.patch_ids)) != tuple(             map(` -> ValueError("patch_ids must use stable structural order") | delete | __post_init__ re-proves values another decsim module built |
| 430 | TrafficAttribution.__post_init__ | `(self.round_lo is None) != (self.round_hi is None)` -> ValueError("round endpoints are present together") | delete | __post_init__ re-proves values another decsim module built |
| 435 | TrafficAttribution.__post_init__ | `self.window_id < 0` -> ValueError("window_id must be a nonnegative window index") | delete | __post_init__ re-proves values another decsim module built |
| 437 | TrafficAttribution.__post_init__ | `self.round_lo is None` -> ValueError("window attribution requires a round range") | delete | __post_init__ re-proves values another decsim module built |
| 445 | TrafficAttribution.__post_init__ | `self.round_lo < 1` -> ValueError("round_lo must be a positive round index") | delete | __post_init__ re-proves values another decsim module built |
| 447 | TrafficAttribution.__post_init__ | `self.round_hi < self.round_lo` -> ValueError("round_hi must be at least round_lo") | delete | __post_init__ re-proves values another decsim module built |
| 476 | _relation_snapshot | `(unconditional)` -> TypeError( "a transfer relation is a request relation or a b | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 592 | Link.reserve | `payload_bits < 0` -> ValueError("payload_bits must be nonnegative") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 595 | Link.reserve | `now_ticks < 0` -> ValueError("now_ticks must be nonnegative") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 597 | Link.reserve | `self._last_send_tick is not None and now_ticks < self._last_send_tick` -> ValueError("now_ticks must not precede the prior reservation | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 675 | LinkModelConfig.wired_paths | `_PATH_RULES[path].required` -> ValueError(f"{path.value} is a required link path") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 726 | LinkModel.reserve | `path not in self._bindings` -> ValueError(f"{path.value} is not wired in this link fabric") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 731 | LinkModel.reserve | `payload_bits is not None and edge.actual_payload_source is None` -> ValueError( f"{path.value} does not declare an actual payloa | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 781 | LinkModel._validate_attribution_shape | `not valid` -> ValueError(f"{path.value} requires {rule.scope_description}" | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 790 | LinkModel._validate_attribution_shape | `needs_request and type(relation) is not RequestTransferRelation` -> ValueError(f"{path.value} requires a request relation") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 792 | LinkModel._validate_attribution_shape | `needs_boundary and type(relation) is not BoundaryTransferRelation` -> ValueError(f"{path.value} requires a boundary relation") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 794 | LinkModel._validate_attribution_shape | `not needs_request and not needs_boundary and relation is not None` -> ValueError(f"{path.value} does not accept a relation") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 800 | LinkModel._validate_attribution_shape | `not is_stable_identity(request_key.operation_id)` -> TypeError("request operation_id must be a stable identity") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 802 | LinkModel._validate_attribution_shape | `request_key.window_id < 0` -> ValueError( "request window_id must be a nonnegative window  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 806 | LinkModel._validate_attribution_shape | `request_key.run_sequence < 0` -> ValueError( "request run_sequence must be a nonnegative requ | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 811 | LinkModel._validate_attribution_shape | `not is_stable_identity(relation.source_window_key)` -> TypeError("boundary source_window_key must be stable") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 813 | LinkModel._validate_attribution_shape | `not is_stable_identity(relation.destination_window_key)` -> TypeError("boundary destination_window_key must be stable") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 815 | LinkModel._validate_attribution_shape | `needs_request and request_key.tier is not rule.tier` -> ValueError(f"{path.value} requires the {rule.tier.value} tie | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 819 | LinkModel._validate_attribution_shape | `request_key is not None and (                 request_key.operation_id != attrib` -> ValueError("transfer relation does not match attribution") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 837 | LinkModel.topology_json_value | `controller_link_integration_assurance not in (             "shipped_controller",` -> ValueError("unknown controller link integration assurance") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 896 | LinkModel.traffic_json_value | `semantic_sum != physical_counters` -> RuntimeError(f"traffic counters do not reconcile for {alias} | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |

### decsim/message.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 52 | stable_identity_bytes | `not is_stable_identity(identity)` -> TypeError( "stable identities are exact int, Unicode scalar  | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 95 | RunSeedPathSegment.__post_init__ | `self.kind not in ("field", "string_key", "none_key", "integer_key")` -> ValueError(f"unknown run-seed path segment kind {self.kind!r | delete | __post_init__ re-proves values another decsim module built |
| 163 | SyndromePacketRoute.__post_init__ | `self.kind is not SyndromePacketRouteKind.WINDOW_INPUT             and not is_sta` -> TypeError("feedback route needs a stable source operation id | delete | __post_init__ re-proves values another decsim module built |
| 210 | normalize_binary_bits | `not all(             type(bit) is bool or (type(bit) is int and bit in (0, 1))  ` -> TypeError("syndrome bits must contain only exact binary valu | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 221 | normalize_binary_bits | `(unconditional)` -> TypeError( "syndrome bits must be None, an exact binary list | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 245 | RetainedSyndromeFragment.__post_init__ | `self.bits is not None and (             type(self.bits) is not tuple            ` -> TypeError("retained bits must be an exact tuple of binary in | delete | __post_init__ re-proves values another decsim module built |
| 275 | SyndromeRoundPacket.__post_init__ | `type(self.fragments) is not tuple             or not self.fragments             ` -> TypeError( "packet fragments must be a nonempty tuple of ret | delete | __post_init__ re-proves values another decsim module built |
| 282 | SyndromeRoundPacket.__post_init__ | `not same_stable_identity(fragment.operation_id, self.operation_id)` -> ValueError("packet fragments must share operation identity") | delete | __post_init__ re-proves values another decsim module built |
| 284 | SyndromeRoundPacket.__post_init__ | `not same_stable_identity(fragment.round_index, self.round_index)` -> ValueError("packet fragments must share round_index") | delete | __post_init__ re-proves values another decsim module built |
| 286 | SyndromeRoundPacket.__post_init__ | `fragment.fragment_index in seen_fragment_indices` -> ValueError("packet fragment indices must be distinct") | delete | __post_init__ re-proves values another decsim module built |
| 292 | SyndromeRoundPacket.__post_init__ | `any(                 same_stable_identity(fragment.patch_id, patch_id)          ` -> ValueError("packet patch identities must be distinct") | delete | __post_init__ re-proves values another decsim module built |
| 391 | _exact_positive_int | `type(value) is not int or value < 1` -> TypeError(f"{label} must be an exact positive int") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 396 | _exact_nonnegative_int | `type(value) is not int or value < 0` -> TypeError(f"{label} must be an exact nonnegative int") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 484 | WindowGeometry.__post_init__ | `not (             self.buffer_lo             <= self.commit_lo             <= se` -> ValueError("window geometry bounds are not ordered") | delete | __post_init__ re-proves values another decsim module built |
| 509 | OperationWindowPlan.__post_init__ | `not self.windows             or any(type(window) is not WindowGeometry for windo` -> TypeError("windows must be nonempty WindowGeometry values") | delete | __post_init__ re-proves values another decsim module built |
| 516 | OperationWindowPlan.__post_init__ | `any(type(index) is not int for index in edge)` -> TypeError("dependency edge indices must be exact ints") | delete | __post_init__ re-proves values another decsim module built |
| 524 | OperationWindowPlan.__post_init__ | `source < 0                 or destination < 0                 or source >= windo` -> ValueError("dependency edge is out of range") | delete | __post_init__ re-proves values another decsim module built |
| 526 | OperationWindowPlan.__post_init__ | `edge in edge_set` -> ValueError("dependency edges must be unique") | delete | __post_init__ re-proves values another decsim module built |
| 539 | OperationWindowPlan.__post_init__ | `not same_stable_identity(self.entry_window_indices, expected_entries)` -> ValueError("entry_window_indices must equal all graph roots" | delete | __post_init__ re-proves values another decsim module built |
| 541 | OperationWindowPlan.__post_init__ | `not same_stable_identity(self.exit_window_indices, expected_exits)` -> ValueError("exit_window_indices must equal all graph sinks") | delete | __post_init__ re-proves values another decsim module built |
| 554 | OperationWindowPlan.__post_init__ | `visited != window_count` -> ValueError("operation window graph must be acyclic") | delete | __post_init__ re-proves values another decsim module built |
| 556 | OperationWindowPlan.__post_init__ | `type(self.batch_preceding_idle_rounds) is not bool` -> TypeError("batch_preceding_idle_rounds must be an exact bool | delete | __post_init__ re-proves values another decsim module built |
| 587 | DependencyResidual.__post_init__ | `any(type(detector_id) is not int or detector_id < 0                for detector_` -> TypeError( "dependency residual detector_ids must be nonnega | delete | __post_init__ re-proves values another decsim module built |
| 591 | DependencyResidual.__post_init__ | `len(set(self.detector_ids)) != len(self.detector_ids)` -> ValueError("dependency residual detector_ids must be unique" | delete | __post_init__ re-proves values another decsim module built |
| 647 | StrongRegionPlan.__post_init__ | `not 1 <= self.context_lo <= self.commit_lo \                 <= self.commit_hi <` -> ValueError( "strong-region bounds must satisfy 1 <= context_ | delete | __post_init__ re-proves values another decsim module built |
| 651 | StrongRegionPlan.__post_init__ | `self.restart_buffer_lo is not None and self.restart_buffer_lo < 1` -> ValueError( "strong-region restart_buffer_lo must be at leas | delete | __post_init__ re-proves values another decsim module built |
| 673 | SoftOutputSource.__post_init__ | `not math.isfinite(normalized) or normalized <= 0.0` -> ValueError( "soft-output source weight step must be finite a | delete | __post_init__ re-proves values another decsim module built |
| 689 | SoftOutput.__post_init__ | `isinstance(self.gap, bool) or not isinstance(self.gap, Real)` -> TypeError("soft output gap must be a real number") | delete | __post_init__ re-proves values another decsim module built |
| 692 | SoftOutput.__post_init__ | `math.isnan(normalized_gap) or normalized_gap < 0` -> ValueError( "soft output gap must be nonnegative or positive | delete | __post_init__ re-proves values another decsim module built |
| 701 | SoftOutput.__post_init__ | `isinstance(value, bool) or not isinstance(value, Real)` -> TypeError( f"soft output {field_name} must be a real number  | delete | __post_init__ re-proves values another decsim module built |
| 706 | SoftOutput.__post_init__ | `math.isnan(normalized_value)` -> ValueError(f"soft output {field_name} cannot be NaN") | delete | __post_init__ re-proves values another decsim module built |
| 792 | StrongDecodeCompletion.__post_init__ | `(type(self.request_key), type(self.result)) != (                 DecoderRequestK` -> TypeError("strong completion requires exact key and result t | delete | __post_init__ re-proves values another decsim module built |
| 797 | StrongDecodeCompletion.__post_init__ | `self.request_key.tier is not DecoderTier.STRONG                 or not same_stab` -> ValueError("strong completion identity must match its strong | delete | __post_init__ re-proves values another decsim module built |
| 845 | StreamBinding.__post_init__ | `self.stream_offset < 0` -> ValueError("stream_offset must be nonnegative") | delete | __post_init__ re-proves values another decsim module built |
| 861 | RunOperationBody.__post_init__ | `self.round_ticks <= 0` -> ValueError("round_ticks must be positive") | delete | __post_init__ re-proves values another decsim module built |
| 863 | RunOperationBody.__post_init__ | `self.round_count < 0 or self.source_round_count < 0` -> ValueError("round counts must be nonnegative") | delete | __post_init__ re-proves values another decsim module built |
| 937 | Operation.__post_init__ | `self.scheduled_start_round < 0` -> ValueError("scheduled_start_round must be nonnegative") | delete | __post_init__ re-proves values another decsim module built |
| 943 | Operation.__post_init__ | `(fragment_fields[0] is None) != (fragment_fields[1] is None)` -> ValueError( "syndrome fragment index and count must be set t | delete | __post_init__ re-proves values another decsim module built |
| 947 | Operation.__post_init__ | `any(type(value) is not int for value in fragment_fields)` -> TypeError("syndrome fragment fields must be exact ints") | delete | __post_init__ re-proves values another decsim module built |
| 949 | Operation.__post_init__ | `not 0 <= fragment_fields[0] < fragment_fields[1]` -> ValueError( "syndrome fragment index must be within fragment | delete | __post_init__ re-proves values another decsim module built |
| 952 | Operation.__post_init__ | `not self.emits_detector_data` -> ValueError("syndrome fragment slots require an emitter") | delete | __post_init__ re-proves values another decsim module built |
| 955 | Operation.__post_init__ | `not self.emits_detector_data` -> ValueError("stream finalizers must emit detector data") | delete | __post_init__ re-proves values another decsim module built |
| 957 | Operation.__post_init__ | `self.stream_id is None` -> ValueError("stream finalizers require an explicit stream_id" | delete | __post_init__ re-proves values another decsim module built |
| 959 | Operation.__post_init__ | `type(self.stream_offset) is not int or self.stream_offset < 0` -> ValueError( "stream finalizers require a nonnegative stream_ | delete | __post_init__ re-proves values another decsim module built |
| 962 | Operation.__post_init__ | `fragment_fields[0] is None` -> ValueError( "stream finalizers require an explicit fragment  | delete | __post_init__ re-proves values another decsim module built |

### decsim/metrics.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 72 | _StepIntegral.observe | `tick < self.last_tick` -> ValueError("metric observations must be monotone") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 107 | DecoderUtilization.observe | `topology != self._topology` -> RuntimeError("decoder pool topology changed during measureme | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 318 | BacklogEarlyWarning.__init__ | `round_ticks <= 0` -> ValueError("round_ticks must be positive") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 320 | BacklogEarlyWarning.__init__ | `window_ticks <= 0` -> ValueError("window_ticks must be positive") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 322 | BacklogEarlyWarning.__init__ | `not math.isfinite(threshold_f) or threshold_f < 0` -> ValueError("threshold_f must be finite and nonnegative") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 324 | BacklogEarlyWarning.__init__ | `consecutive <= 0` -> ValueError("consecutive must be positive") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 449 | BurstEscalationDetector.__init__ | `not patches` -> ValueError("patches must be nonempty") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 451 | BurstEscalationDetector.__init__ | `len(set(patches)) != len(patches)` -> ValueError("patches must not contain duplicates") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 453 | BurstEscalationDetector.__init__ | `not math.isfinite(z) or z < 0` -> ValueError("z must be finite and nonnegative") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 455 | BurstEscalationDetector.__init__ | `baseline_bins <= 0` -> ValueError("baseline_bins must be positive") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 457 | BurstEscalationDetector.__init__ | `warmup_bins <= 0` -> ValueError("warmup_bins must be positive") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 459 | BurstEscalationDetector.__init__ | `not 1 <= patch_quorum <= len(patches)` -> ValueError("patch_quorum must select configured patches") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |

### decsim/pauli_frame.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 43 | PauliFrameConfig.__post_init__ | `not math.isfinite(self.commit_us) or self.commit_us < 0` -> ValueError("commit_us must be a finite nonnegative number") | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 45 | PauliFrameConfig.__post_init__ | `self.commit_us > 0 and us(self.commit_us) == 0` -> ValueError("commit_us is positive but rounds to zero ticks") | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 47 | PauliFrameConfig.__post_init__ | `self.commit_us == 0 and not self.zero_commit_cost_justification` -> ValueError( "zero commit_us requires zero_commit_cost_justif | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 54 | PauliFrameConfig.__post_init__ | `self.commit_us > 0             and self.zero_commit_cost_justification is not No` -> ValueError( "zero_commit_cost_justification is only valid fo | keep | config card / backend outcome __post_init__: value rules and normalization stay; type-exactness lines inside it go |
| 141 | PauliFrame.commit_weak_correction | `not is_stable_identity(window_key)` -> TypeError("window_key must be a stable identity") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 206 | PauliFrame.frame_for | `len(logical_observables) != observable_arity` -> RuntimeError( f"Pauli frame stream {stream_id!r} changed obs | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |

### decsim/planner.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 117 | _validate_operation_graph | `operation.id in by_id` -> ValueError( f"duplicate operation id {operation.id}: " f"{by | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 129 | _validate_operation_graph | `predecessor_id in seen_predecessors` -> ValueError( f"operation {operation.id} lists predecessor " f | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 134 | _validate_operation_graph | `predecessor_id == operation.id` -> ValueError(f"operation {operation.id} depends on itself") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 136 | _validate_operation_graph | `predecessor_id not in by_id` -> ValueError( f"operation {operation.id} has unknown predecess | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 145 | _validate_operation_graph | `blocker == operation.id` -> ValueError( f"operation {operation.id} is blocked by itself" | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 148 | _validate_operation_graph | `blocker not in valid_blocker_ids` -> ValueError( f"operation {operation.id} has unknown blocking  | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 166 | _validate_operation_graph | `visited != len(by_id)` -> ValueError(f"operation dependency cycle involving IDs {cycle | keep | modeled failure semantics (fail-stop, timeout, exhaustion, unsettled work) |
| 179 | _validate_workload_identity | `type(operation) is not Operation` -> TypeError(f"{role} entries must be exact Operation values") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 181 | _validate_workload_identity | `operation.id in seen_ids` -> ValueError( f"operation id {operation.id} appears more than  | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 186 | _validate_workload_identity | `prior is not None and prior is not operation` -> ValueError( f"operation id {operation.id} belongs to distinc | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 194 | _validate_workload_identity | `"dynamic_streams" in roles and len(roles) > 1` -> ValueError( f"operation id {operation.id} cannot appear in b | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 206 | _validate_workload_identity | `decode_ops and not any(                 owner is operation for owner in static_o` -> ValueError( f"operation {operation.id} must share static dec | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 215 | _validate_workload_identity | `owner is None` -> ValueError( f"operation {operation.id} stream_id {operation. | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 239 | _plan_execution | `isinstance(round_us, bool) or not isinstance(round_us, numbers.Real)` -> ValueError("resolved round_us must be a finite real number") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 242 | _plan_execution | `not math.isfinite(round_us)` -> ValueError("resolved round_us must be a finite real number") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 245 | _plan_execution | `round_ticks < 1` -> ValueError("resolved round cadence must be at least one tick | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 275 | _plan_execution | `layout.code_for_op(operation) is not code` -> ValueError( f"layout operation {operation.id} selected a cod | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 298 | _plan_execution | `layout.code_for_patch(patch_id) is not code` -> ValueError( f"layout patch {patch_id!r} selected a code diff | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 322 | _plan_execution | `(unconditional)` -> ValueError(f"unknown planned operation id {error.args[0]}") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 324 | _plan_execution | `any(operation.round_count < 1 for operation in planned_resolved)` -> ValueError("decode owners must have at least one round") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 378 | _materialize_execution_plan | `operation.id != operation_plan.operation_id` -> ValueError("operation planning inputs must match by position | delete | re-proves an identity another module just built |

### decsim/qpu.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 25 | QPUDevice.__init__ | `type(cycle_ticks) is not int or cycle_ticks <= 0` -> ValueError("cycle_ticks must be a positive exact int") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 54 | QPUDevice.issue | `self.completion_receiver is None` -> RuntimeError("QPU completion receiver is not connected") | delete | re-proves wiring that run_spec makes by construction (Phase 2 constructor injection) |
| 56 | QPUDevice.issue | `type(command) is not RunOperationBody` -> TypeError("QPUDevice accepts only RunOperationBody commands" | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 58 | QPUDevice.issue | `command.round_ticks != self.cycle_ticks` -> ValueError("operation cadence must equal the QPU cycle") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 61 | QPUDevice.issue | `command.round_count == 0 and command.emits_detector_data \                 and n` -> ValueError("zero-duration detector emitters must finalize a  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 145 | QPUDevice._emit | `type(payloads) not in (list, tuple)` -> TypeError("QPU model payloads must be an exact list or tuple | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 147 | QPUDevice._emit | `not payloads` -> ValueError("a detector-emitting round must emit at least one | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 149 | QPUDevice._emit | `operation.syndrome_fragment_index is not None and len(payloads) != 1` -> ValueError("an explicit syndrome fragment slot must emit one | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 158 | QPUDevice._emit | `operation.syndrome_fragment_index is None             and operation.syndrome_fra` -> ValueError( "declared syndrome fragment count must match emi | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 162 | QPUDevice._emit | `type(payload) is not QPUReadout` -> TypeError("QPU model must emit exact QPUReadout values") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 165 | QPUDevice._emit | `self.readout_receiver is None` -> RuntimeError("QPU readout receiver is not connected") | delete | re-proves wiring that run_spec makes by construction (Phase 2 constructor injection) |
| 182 | QPUDevice.emit_feedback_memory_round | `self.readout_receiver is None` -> RuntimeError("QPU readout receiver is not connected") | delete | re-proves wiring that run_spec makes by construction (Phase 2 constructor injection) |

### decsim/rounds.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 10 | _validated | `value < 1` -> ValueError(f"{source} must give >= 1 round (got {value})") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 39 | PerOpRounds._validated | `value < 0` -> ValueError( f"PerOpRounds[{operation_id}] must give >= 0 rou | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |

### decsim/run_spec.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 137 | RunSpec.build | `self._build_state != "unstarted"` -> RuntimeError(f"RunSpec build is already {self._build_state}" | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 147 | RunSpec.build | `(unconditional)` ->  | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 184 | RunSpec._build_once | `type(value) is not bool` -> TypeError(f"strategy capability {name} must be an exact bool | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 187 | RunSpec._build_once | `(self.ops is None) == (self.frontend is None)` -> ValueError("provide exactly one of ops= or frontend=") | keep | user config rule: exactly one of ops= or frontend= |
| 196 | RunSpec._build_once | `self.feedback_boundary_mode not in (             "trailing_buffer", "measurement` -> ValueError("invalid feedback_boundary_mode") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 198 | RunSpec._build_once | `type(self.record_switching_windows) is not bool` -> TypeError("record_switching_windows must be an exact bool") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 213 | RunSpec._build_once | `dynamic_streams and type(scheme) is not SlidingWindowScheme` -> ValueError("dynamic streams require SlidingWindowScheme") | keep | user config rule: dynamic streams need SlidingWindowScheme |
| 247 | RunSpec._build_once | `type(device) is SyndromeBitDevice and device.code is not code` -> ValueError( "SyndromeBitDevice.code must be the exact resolv | keep | user config rule: SyndromeBitDevice code identity |
| 250 | RunSpec._build_once | `self.router is not None and (self.decoder is not None or self.decoders)` -> ValueError("router is exclusive with decoder and decoders") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 252 | RunSpec._build_once | `self.router is None and self.decoder is None and planned_operations` -> ValueError("decoder is required when router is omitted") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 268 | RunSpec._build_once | `self.timing.ticks("t_binary_availability") > 0             and not link_config.q` -> ValueError( "a separate controller readout cost requires a l | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 275 | RunSpec._build_once | `type(buffering) is not SyndromeBufferingConfig` -> TypeError("syndrome_buffering must be SyndromeBufferingConfi | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 284 | RunSpec._build_once | `self.pauli_frame is not None and pauli_frame is None` -> TypeError("pauli_frame.resolve must return a PauliFrame") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 302 | RunSpec._build_once | `self.make_syndrome_ingress is not None and self.syndrome_ingress_policy is not N` -> ValueError("syndrome_ingress_policy cannot be combined with  | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 334 | RunSpec._build_once | `self.make_decoder_memory_transfer is not None             and decoder_memory_tra` -> TypeError( "make_decoder_memory_transfer must return a " "De | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 360 | RunSpec._build_once | `factory.engine is not engine` -> ValueError( f"{type(factory).__name__} uses a different engi | keep | user config rule: factory built on the run engine |
| 422 | RunSpec._build_once | `window_manager.pending_escalations` -> RuntimeError( f"the run ended with pending strong escalation | keep | modeled failure semantics (fail-stop, timeout, exhaustion, unsettled work) |
| 449 | _select_code | `sum(value is not None for value in (distance, code, layout)) > 1` -> ValueError(f"multiple code sources supplied: {', '.join(supp | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 453 | _select_code | `len(codes) != 1` -> ValueError( f"layout must declare exactly one code (got {len | keep | user config rule: a layout declares one code |
| 481 | _install_device_circuits | `scope != "per_operation"` -> ValueError("device operation_circuit_scope must be none or p | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 487 | _install_device_circuits | `type(operation.circuit) is not stim.Circuit` -> TypeError("active operation circuit is not an exact stim.Cir | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 516 | _metric_bindings | `not is_stable_string(metric.name) or not metric.name` -> ValueError("metric names must be nonempty Unicode strings") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 518 | _metric_bindings | `metric.name in names` -> ValueError(f"duplicate metric name {metric.name!r}") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 528 | _root_seed | `type(value) is bool or not isinstance(value, Integral)` -> TypeError("seed must be a 64-bit unsigned integer or None") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 531 | _root_seed | `not 0 <= value < 2**64` -> ValueError("seed must be in [0, 2**64)") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 556 | _check_factory_decode_service | `factory.decode_service is not expected` -> ValueError( f"{type(factory).__name__} decode_service must b | keep | user config rule: factory decode service is the run decoder manager |

### decsim/schemes.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 74 | SlidingWindowScheme.__init__ | `type(terminal_policy) is not SlidingTerminalPolicy` -> TypeError("terminal_policy must be an exact SlidingTerminalP | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 135 | SlidingWindowScheme.validate_buffer | `geometry.buffer_round_count             < geometry.minimum_trailing_buffer_round` -> ValueError( f"buffer_rounds={geometry.buffer_round_count} is | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 222 | ParallelWindowScheme.plan_operation | `commit_round_count != buffer_round_count` -> ValueError( "parallel A/B decoding requires commit_round_cou | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 312 | ParallelWindowScheme.validate_buffer | `geometry.buffer_round_count < required` -> ValueError( f"buffer_rounds={geometry.buffer_round_count} is | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 344 | TanSandwichScheme.plan_operation | `step < 2` -> ValueError("Tan sandwich decoding requires step size s >= 2" | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 346 | TanSandwichScheme.plan_operation | `buffer < 1` -> ValueError("Tan sandwich decoding requires overlapping windo | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 397 | TanSandwichScheme.validate_buffer | `geometry.commit_round_count < 2` -> ValueError("Tan sandwich decoding requires step size s >= 2" | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 399 | TanSandwichScheme.validate_buffer | `geometry.buffer_round_count < 1` -> ValueError("Tan sandwich decoding requires b >= 1") | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |

### decsim/seeding.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 35 | _AtomicRunSeedConsumer._install_run_seed_state | `(unconditional)` -> NotImplementedError | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 42 | _AtomicRunSeedConsumer.reserve_run_seed | `seed is not None and (             type(seed) is not int or not 0 <= seed < (1 <` -> TypeError( f"{component_name} run root must be an unsigned 6 | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 48 | _AtomicRunSeedConsumer.reserve_run_seed | `self._stochastic_use_started` -> ValueError( f"{component_name} was already used and cannot b | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 52 | _AtomicRunSeedConsumer.reserve_run_seed | `self._run_seed_claimed` -> ValueError( f"{component_name} is already claimed by a built | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 56 | _AtomicRunSeedConsumer.reserve_run_seed | `self._pending_run_seed is not None` -> ValueError( f"{component_name} already has a pending run-see | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 60 | _AtomicRunSeedConsumer.reserve_run_seed | `seed is not None and self._explicit_seed is not None` -> ValueError( f"{component_name} has an explicit " f"{self._ex | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 78 | _AtomicRunSeedConsumer.reserve_run_seed | `effective_seed is not None and (                 type(effective_seed) is not int` -> TypeError( f"{component_name} explicit seed must be an unsig | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 98 | _AtomicRunSeedConsumer.commit_run_seed | `self._pending_run_seed is not reservation` -> ValueError( f"{type(self).__name__} can commit only its exac | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 109 | _AtomicRunSeedConsumer._mark_stochastic_use | `self._pending_run_seed is not None` -> RuntimeError( f"{type(self).__name__} cannot draw while a ru | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 133 | derive_component_seed | `type(root_seed) is not int or not 0 <= root_seed < (1 << 64)` -> TypeError( "root seed must be an unsigned 64-bit built-in in | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 157 | walk | `path_bytes in paths` -> ValueError(f"duplicate seed path {_render(path)}") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 161 | walk | `identity in active` -> ValueError( f"seed cycle from {_render(path)} to " f"{_rende | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 202 | bind_run_seed | `seed is not None and (                 reservation.proposed_seed_source != "deri` -> ValueError( f"{type(component).__name__} disagrees with the  | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 209 | bind_run_seed | `seed is None and reservation.proposed_seed_source not in (                 "expl` -> ValueError("unseeded components must report their seed sourc | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 213 | bind_run_seed | `(unconditional)` ->  | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |

### decsim/speculative_recovery.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 55 | SpeculativeRecovery.begin | `(unconditional)` -> TypeError( f"window interaction invalidation for {key} must  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 64 | SpeculativeRecovery.begin | `key in self._records` -> RuntimeError(f"recovery for {key} is already live") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 178 | SpeculativeRecovery._repair | `not set(runtime.syndrome_buffer.hold_round_identities(                     owner` -> RuntimeError( f"replacement Replay for {key} does not cover  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 330 | SpeculativeRecovery._causal_closure | `key in seen` -> RuntimeError( f"window interaction invalidation for {root} s | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 353 | SpeculativeRecovery._causal_closure | `not ready` -> RuntimeError( f"window interaction invalidation for {root} c | keep | modeled failure semantics (fail-stop, timeout, exhaustion, unsettled work) |
| 372 | SpeculativeRecovery._validate_descendants | `key == root` -> RuntimeError( f"window interaction invalidation for {root} i | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 376 | SpeculativeRecovery._validate_descendants | `key not in self.runtime.windows` -> RuntimeError( f"window interaction invalidation for {root} s | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 380 | SpeculativeRecovery._validate_descendants | `key[0] in self.runtime._finished_ops` -> RuntimeError( f"window interaction invalidation for {root} s | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 384 | SpeculativeRecovery._validate_descendants | `key not in reachable` -> RuntimeError( f"window interaction invalidation for {root} s | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 389 | SpeculativeRecovery._validate_descendants | `window.queued or window.committed` -> RuntimeError( f"window interaction invalidation for {root} s | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |

### decsim/switching.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 38 | ThresholdRegister.__init__ | `not isinstance(expected_source, SoftOutputSource)` -> TypeError( "ThresholdRegister expected_source must be a Soft | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 44 | ThresholdRegister.__init__ | `type(code) is not str` -> TypeError( "threshold-register code identities must be exact | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 48 | ThresholdRegister.__init__ | `not code` -> ValueError( "threshold-register code identities must be none | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 64 | ThresholdRegister.set | `type(code) is not str` -> TypeError( "threshold-register code identities must be exact | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 68 | ThresholdRegister.set | `not code` -> ValueError( "threshold-register code identities must be none | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 158 | Switching.__init__ | `weak_keepup_ratio is not None and not 0 < weak_keepup_ratio < 1` -> ValueError(f"weak_keepup_ratio must be between 0 and 1 " f"( | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 161 | Switching.__init__ | `bulk_strong and run_both_at_once` -> ValueError("bulk_strong is only meaningful in serial mode "  | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 164 | Switching.__init__ | `double_window and run_both_at_once` -> ValueError( "double_window defers the strong start until the | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 169 | Switching.__init__ | `double_window and bulk_strong` -> ValueError( "double_window + bulk_strong is not supported: d | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 173 | Switching.__init__ | `not isinstance(expected_source, SoftOutputSource)` -> TypeError( "Switching expected_source must be a SoftOutputSo | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 178 | Switching.__init__ | `confidence_threshold != threshold_register.default` -> ValueError( "Switching confidence threshold must equal the t | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 183 | Switching.__init__ | `expected_source != threshold_register.expected_source` -> ValueError( "Switching expected source must equal the thresh | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 260 | Switching.validate_declared_run | `type(scheme) is SlidingWindowScheme             and scheme.terminal_policy      ` -> ValueError( "switching and strong-slab recovery require the  | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 268 | Switching.validate_declared_run | `self.weak_keepup_ratio is not None and (             type(scheme) is not Sliding` -> ValueError( "weak_keepup_ratio implements the exact shipped  | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 274 | Switching.validate_declared_run | `has_dynamic_streams and isinstance(boundary_policy, Eager)` -> ValueError( "Eager speculative recovery needs a statically p | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 281 | Switching.validate_declared_run | `type(scheme) is not SlidingWindowScheme` -> ValueError( "double_window requires the exact shipped serial | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 286 | Switching.validate_declared_run | `isinstance(boundary_policy, Held)` -> ValueError( "double_window requires the weak chain to keep c | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 292 | Switching.validate_declared_run | `has_dynamic_streams or static_decode_plan_selected` -> ValueError( "double_window skips statically planned windows  | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 297 | Switching.validate_declared_run | `has_frontend` -> ValueError( "double_window is validated for explicit ops= wo | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 306 | Switching.validate_operations | `self.double_window and any(             operation.decoder_boundary_predecessors ` -> ValueError( "double_window supports one single-patch stream  | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 319 | Switching.validate_code_geometry | `ratio > commit_rounds / (commit_rounds + buffer_rounds)` -> ValueError( f"commit region of {commit_rounds} rounds too sh | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 341 | Switching.keep_weak_result | `result.soft_output.source != self.expected_source` -> ValueError( "decoder confidence source does not match the sw | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |

### decsim/syndrome_buffer.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 42 | SyndromeBufferingConfig.__post_init__ | `self.upstream_packet_slots is not None and (                 type(self.upstream_` -> TypeError( "upstream_packet_slots must be a positive int or  | delete | type-exactness inside a config-card __post_init__ |
| 131 | _validated_round_identity | `type(identity) is not tuple         or len(identity) != 2         or not is_stab` -> TypeError( "round identities are (stable operation_id, round | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 180 | SyndromeBuffer.__init__ | `capacity is not None and (type(capacity) is not int or capacity < 1)` -> TypeError("capacity must be a positive int or None") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 209 | SyndromeBuffer.open_operation | `not is_stable_identity(operation_id)` -> TypeError("operation_id must be a stable identity") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 211 | SyndromeBuffer.open_operation | `operation_id in self._closed_operations` -> RuntimeError("closed operation identities cannot be reused") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 225 | SyndromeBuffer.close_operation | `live_rounds` -> RuntimeError( f"operation {operation_id!r} has live buffer r | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 234 | SyndromeBuffer.close_operation | `live_holders` -> RuntimeError( f"operation {operation_id!r} has live consumer | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 262 | SyndromeBuffer.accept_fragment | `type(fragment) is not RetainedSyndromeFragment` -> TypeError( "accept_fragment requires a RetainedSyndromeFragm | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 266 | SyndromeBuffer.accept_fragment | `type(expected_fragments) is not int or expected_fragments < 1` -> TypeError("expected_fragments must be a positive int") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 268 | SyndromeBuffer.accept_fragment | `fragment.operation_id not in self._open_operations` -> RuntimeError( f"operation {fragment.operation_id!r} is not o | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 273 | SyndromeBuffer.accept_fragment | `identity in self._tombstones` -> ValueError( f"late fragment: round {identity!r} was already  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 280 | SyndromeBuffer.accept_fragment | `slot.state is not SyndromeBufferRoundState.ASSEMBLING` -> ValueError( f"fragment admission for round {identity!r} was  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 285 | SyndromeBuffer.accept_fragment | `expected_fragments != slot.expected_fragments` -> ValueError("all fragments must declare the same count") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 287 | SyndromeBuffer.accept_fragment | `fragment.fragment_index >= slot.expected_fragments` -> ValueError("fragment index exceeds the declared count") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 292 | SyndromeBuffer.accept_fragment | `any(             fragment.fragment_index == held.fragment_index             for ` -> ValueError("duplicate syndrome fragment index") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 307 | SyndromeBuffer._allocate | `self.capacity is not None and not self._free_slot_indices` -> SyndromeBufferCapacityExhaustion( incoming_identity=identity | keep | modeled failure semantics (fail-stop, timeout, exhaustion, unsettled work) |
| 335 | SyndromeBuffer.finish_packing | `slot is None` -> RuntimeError( f"round {round_identity!r} holds no live alloc | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 339 | SyndromeBuffer.finish_packing | `slot.state is not SyndromeBufferRoundState.PACKING` -> RuntimeError( f"round {round_identity!r} is {slot.state.name | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 360 | SyndromeBuffer.finish_packing | `(unconditional)` ->  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 377 | SyndromeBuffer.read_retained_round | `slot is None or slot.state is not (             SyndromeBufferRoundState.PACKED_` -> RuntimeError( f"round {round_identity!r} is not packed and r | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 394 | SyndromeBuffer.mark_publication_tick | `slot is None or slot.state is not SyndromeBufferRoundState.PACKED_RETAINED` -> RuntimeError(f"round {identity!r} is not packed and retained | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 396 | SyndromeBuffer.mark_publication_tick | `type(publication_tick) is not int or publication_tick < 0` -> TypeError("publication_tick must be a nonnegative exact int" | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 398 | SyndromeBuffer.mark_publication_tick | `self._publication_ticks[identity] is not None` -> RuntimeError(f"round {identity!r} was already published") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 415 | SyndromeBuffer._hold_record | `identity[0] not in self._open_operations` -> RuntimeError( f"hold references closed operation {identity[0 | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 419 | SyndromeBuffer._hold_record | `identity in self._tombstones` -> ValueError( f"hold references released round {identity!r}" ) | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 430 | SyndromeBuffer._hold_record | `closed` -> RuntimeError( f"hold {holder!r} references closed operation  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 443 | SyndromeBuffer._validate_new_holder | `holder is None` -> TypeError("consumer hold tokens cannot be None") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 445 | SyndromeBuffer._validate_new_holder | `holder in self._live_holds or holder in self._released_holds` -> ValueError(f"duplicate consumer hold token {holder!r}") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 450 | SyndromeBuffer._validate_new_holder | `type(holder) is Replay and (             type(holder.boundary_generation) is not` -> TypeError("Replay generation must be a nonnegative int") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 464 | SyndromeBuffer.replace_hold | `(unconditional)` -> RuntimeError("consumer hold is not live") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 482 | SyndromeBuffer.transfer_hold | `(unconditional)` -> RuntimeError("consumer hold is not live") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 496 | SyndromeBuffer.release_hold | `(unconditional)` -> RuntimeError( "consumer hold was never registered" ) | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 541 | SyndromeBuffer.release_round | `slot is None` -> RuntimeError( f"round {round_identity!r} holds no live alloc | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 545 | SyndromeBuffer.release_round | `self._holders_by_round.get(identity)` -> RuntimeError( f"round {round_identity!r} has live consumer h | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |

### decsim/syndrome_ingress.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 52 | SyndromeIngressPolicy.__post_init__ | `type(self.queue_admission) is not ReassemblyQueueAdmission` -> TypeError("queue_admission must be ReassemblyQueueAdmission" | delete | type-exactness inside a config-card __post_init__ |
| 54 | SyndromeIngressPolicy.__post_init__ | `type(self.overflow) is not IngressOverflowPolicy` -> TypeError("overflow must be IngressOverflowPolicy") | delete | type-exactness inside a config-card __post_init__ |
| 58 | SyndromeIngressPolicy.__post_init__ | `self.reassembly_timeout_ticks is not None and                 (type(self.reassem` -> TypeError("reassembly_timeout_ticks must be positive or None | delete | type-exactness inside a config-card __post_init__ |
| 142 | SyndromeIngress.__init__ | `ingress_context_capacity is not None and (             type(ingress_context_capa` -> TypeError("ingress_context_capacity must be a positive int o | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 144 | SyndromeIngress.__init__ | `type(policy) is not SyndromeIngressPolicy` -> TypeError("policy must be an exact SyndromeIngressPolicy") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 185 | SyndromeIngress.relay_qpu_readout | `type(processing_ticks) is not int or processing_ticks < 0` -> TypeError("processing_ticks must be a nonnegative exact int" | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 187 | SyndromeIngress.relay_qpu_readout | `type(payload) is not SyndromePayload` -> TypeError("relay_qpu_readout requires exact SyndromePayload" | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 189 | SyndromeIngress.relay_qpu_readout | `type(route) is not SyndromePacketRoute` -> TypeError("relay_qpu_readout requires a typed packet route") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 213 | SyndromeIngress.relay_syndrome | `type(payload) is not SyndromePayload` -> TypeError("relay_syndrome accepts only exact SyndromePayload | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 215 | SyndromeIngress.relay_syndrome | `type(route) is not SyndromePacketRoute` -> TypeError("relay_syndrome requires a typed packet route") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 217 | SyndromeIngress.relay_syndrome | `type(payload.n_fragments) is not int` -> TypeError("n_fragments must be an exact built-in int") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 219 | SyndromeIngress.relay_syndrome | `payload.n_fragments < 1` -> ValueError("n_fragments must be at least one") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 246 | SyndromeIngress._receive_fragment | `any(live_identity[-2:] == round_key                    for live_identity in self` -> ValueError("all fragments must share one typed route") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 274 | SyndromeIngress._receive_fragment | `(unconditional)` -> SyndromeIngressOverflow( tick=self.engine.now, route=route,  | keep | modeled failure semantics (fail-stop, timeout, exhaustion, unsettled work) |
| 417 | SyndromeIngress._deliver_window_input_round | `type(accepted) is not bool` -> TypeError("window input receiver must return an exact bool") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 487 | SyndromeIngress._expire_reassembly | `(unconditional)` -> SyndromeReassemblyTimeout( tick=self.engine.now, identity=id | keep | modeled failure semantics (fail-stop, timeout, exhaustion, unsettled work) |
| 498 | SyndromeIngress._release_slot | `not route_queue or route_queue.pop(0) != slot_index` -> RuntimeError("controller released a non-head route slot") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 511 | SyndromeIngress.check_work_settled | `snapshot.ingress_contexts` -> RuntimeError( "run ended with incomplete syndrome ingress co | keep | modeled failure semantics (fail-stop, timeout, exhaustion, unsettled work) |

### decsim/views.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 291 | strong_work_view | `overlap` -> RuntimeError(f"strong work has overlapping owners for {overl | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 342 | switching_records_view | `len(owners) != 1` -> RuntimeError(f"absorbed window {key} has no unique owner") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 345 | switching_records_view | `contribution is None` -> RuntimeError(f"final window {key} has no logical contributio | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 381 | capture_primary_result | `len(bits) != len(actual)` -> RuntimeError( f"operation {operation_id} predicted {len(bits | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 395 | capture_primary_result | `engine._event_queue or not execution_runtime.workload_complete` -> RuntimeError("primary run ended before workload completed") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 403 | _logical_bit | `type(value) is not int or value not in (0, 1)` -> TypeError(f"logical observables must contain bits; got {valu | delete | type / exact-type / stable-identity / callable guard on an internal path |

### decsim/window_manager.py

| line | function | check | decision | reason |
|---|---|---|---|---|
| 121 | _EscalationRegistry._register | `type(pending) is not _PendingEscalation` -> TypeError("pending escalation must use the exact record type | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 123 | _EscalationRegistry._register | `not is_stable_identity(pending.key)` -> TypeError("pending escalation key must be a stable identity" | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 125 | _EscalationRegistry._register | `not is_stable_identity(readiness_key)` -> TypeError("readiness key must be a stable identity") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 127 | _EscalationRegistry._register | `pending.phase is not expected_phase` -> RuntimeError( f"pending escalation {pending.key} has phase " | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 131 | _EscalationRegistry._register | `pending.key in self._by_key` -> RuntimeError( f"duplicate strong escalation for window {pend | keep | state-machine invariant: one escalation per window |
| 135 | _EscalationRegistry._register | `readiness_key in readiness_index` -> RuntimeError( f"readiness index collision for {readiness_key | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 147 | _EscalationRegistry.update_wsd_arrival | `self._by_key.get(expected.key) is not expected` -> RuntimeError( f"stale escalation timing update for {expected | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 150 | _EscalationRegistry.update_wsd_arrival | `expected.wsd_arrival_ticks is not None` -> RuntimeError(f"duplicate WSD reservation for {expected.key}" | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 199 | _EscalationRegistry._take | `type(expected) is not _PendingEscalation` -> TypeError("expected escalation must use the exact record typ | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 201 | _EscalationRegistry._take | `expected.phase is not expected_phase` -> RuntimeError( f"wrong-phase take for escalation {expected.ke | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 206 | _EscalationRegistry._take | `primary is not expected or indexed_key != expected.key` -> RuntimeError( f"stale escalation take for readiness key {rea | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 268 | WindowManager.__init__ | `not callable(fault_model_requirement_for)` -> TypeError("fault_model_requirement_for must be callable") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 398 | WindowManager.rounds_for | `(unconditional)` -> ValueError( f"operation {op.id} has no resolved planning rec | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 441 | WindowManager.load_execution_plan | `capacity is not None and capacity < len(minimum)` -> ValueError( f"upstream syndrome buffer needs {len(minimum)}  | keep | user config rule (RunSpec, cards, strategy and scheme options, operation graph) |
| 550 | WindowManager._require_retained_payloads | `missing` -> RuntimeError( f"{purpose} requires retained payload rounds t | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 630 | WindowManager.on_syndrome_arrival | `type(packet) is not SyndromeRoundPacket` -> TypeError("window manager requires a SyndromeRoundPacket") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 634 | WindowManager.on_syndrome_arrival | `(unconditional)` -> ValueError( f"unknown syndrome operation {packet.operation_i | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 665 | WindowManager._store_payload | `not self.syndrome_buffer.has_operation(op.id)` -> RuntimeError( f"round {packet.round_index} of {op.name} arri | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 674 | WindowManager._store_payload | `round_limit is not None and packet.round_index > round_limit` -> ValueError( f"round {packet.round_index} of {op.name} exceed | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 683 | WindowManager._store_payload | `self.syndrome_buffer.retained_fragments((op.id, packet.round_index)) != packet.f` -> RuntimeError( "syndrome packet was not published from the re | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 834 | WindowManager._bind_decoder_input_hold | `job.request_key is None` -> RuntimeError("decoder input hold requires a request key") | delete | re-proves an identity another module just built |
| 897 | WindowManager._submit_window_decode | `submission.delay_ticks != 0` -> ValueError( "strong transport delay is owned by the link fab | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 957 | WindowManager._job_attribution | `window is None` -> RuntimeError("window-scoped transport requires a DecodeJob w | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1026 | WindowManager._link_arrival | `relation_key is None` -> RuntimeError("window transfer requires a request key") | delete | re-proves an identity another module just built |
| 1081 | WindowManager.prepare_strong_selection | `pending is None or pending.strong_request_key != strong_request_key` -> RuntimeError("deferred directive key has no matching pending | delete | re-proves an identity another module just built |
| 1099 | WindowManager.prepare_strong_selection | `serial_strong_job.request_key != strong_request_key` -> RuntimeError("serial strong selection request key mismatch") | delete | re-proves an identity another module just built |
| 1112 | WindowManager.prepare_strong_selection | `pending is not None` -> RuntimeError("deferred pending request needs an explicit key | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1184 | WindowManager.defer_strong_escalation | `self._escalations.peek_key(key) is not None             or (                 exi` -> RuntimeError( f"duplicate strong escalation for window {key} | keep | state-machine invariant: one escalation per window |
| 1188 | WindowManager.defer_strong_escalation | `weak_job.strong_label is None` -> RuntimeError( f"double-window escalation {key} needs a decla | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1207 | WindowManager.defer_strong_escalation | `not isinstance(plan, StrongRegionPlan)` -> TypeError( f"window interaction must return StrongRegionPlan | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 1231 | WindowManager.defer_strong_escalation | `readiness_collision is not None` -> RuntimeError( f"readiness index collision for {readiness_key | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1341 | WindowManager._defer_crossing_strong_escalation | `not (             1 <= plan.context_lo <= plan.commit_lo             <= weak_win` -> RuntimeError( f"invalid strong-region bounds for {key}: cont | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1346 | WindowManager._defer_crossing_strong_escalation | `plan.commit_lo != weak_window.commit_lo` -> RuntimeError( f"strong-region commit for {key} must start at | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1354 | WindowManager._defer_crossing_strong_escalation | `plan.restart_buffer_lo is None             or not 1 <= plan.restart_buffer_lo <=` -> RuntimeError( f"strong-region restart after {key} needs an e | keep | state-machine invariant: strong-region restart geometry |
| 1374 | WindowManager._defer_crossing_strong_escalation | `not suffix_plan.windows` -> RuntimeError( f"crossing strong-region plan for {key} produc | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1377 | WindowManager._defer_crossing_strong_escalation | `len(suffix_plan.windows) > len(reusable_keys)` -> RuntimeError( f"crossing strong-region plan for {key} needs  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1401 | WindowManager._defer_crossing_strong_escalation | `not (                 1 <= buffer_lo <= commit_lo <= commit_hi                 <` -> RuntimeError( f"invalid rephased suffix geometry for {window | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1424 | WindowManager._defer_crossing_strong_escalation | `replacement_windows[0].commit_lo != plan.commit_hi + 1             or replacemen` -> RuntimeError( f"rephased suffix for {key} must tile rounds " | keep | state-machine invariant: rephased suffix tiles the rounds |
| 1430 | WindowManager._defer_crossing_strong_escalation | `op_id in self._finished_ops or op_id in self.op_results` -> RuntimeError( f"cannot rephase suffix for completed operatio | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1446 | WindowManager._defer_crossing_strong_escalation | `window.queued or window.committed` -> RuntimeError( f"cannot rephase window {affected_key}: decode | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1450 | WindowManager._defer_crossing_strong_escalation | `any(affected_key in values for values in historical_sets)` -> RuntimeError( f"cannot rephase historical window {affected_k | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1453 | WindowManager._defer_crossing_strong_escalation | `any(affected_key in values for values in historical_maps)` -> RuntimeError( f"cannot rephase window {affected_key} with pu | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1456 | WindowManager._defer_crossing_strong_escalation | `self._escalations.peek_key(affected_key) is not None` -> RuntimeError( f"cannot rephase pending escalation {affected_ | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1460 | WindowManager._defer_crossing_strong_escalation | `readiness_owner is not None` -> RuntimeError( f"cannot rephase readiness key {affected_key}: | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1470 | WindowManager._defer_crossing_strong_escalation | `any(             source in affected_keys or destination in affected_keys        ` -> RuntimeError( f"cannot rephase suffix for {key} after bounda | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1476 | WindowManager._defer_crossing_strong_escalation | `source.dependents != [destination.key]` -> RuntimeError( f"cannot rephase suffix with external or non-s | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1480 | WindowManager._defer_crossing_strong_escalation | `destination.deps != [source.key] or destination.deps_remaining != 1` -> RuntimeError( f"cannot rephase suffix with released or non-s | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1484 | WindowManager._defer_crossing_strong_escalation | `later_windows[-1].dependents` -> RuntimeError( f"cannot rephase suffix with external edge fro | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1521 | WindowManager._defer_crossing_strong_escalation | `suffix_models and len(suffix_models) != len(replacement_windows)` -> RuntimeError( f"device returned {len(suffix_models)} models  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1548 | WindowManager._defer_crossing_strong_escalation | `self._escalations.peek_far(restart_window.key) is not None` -> RuntimeError( f"readiness index collision for {restart_windo | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1578 | WindowManager._defer_crossing_strong_escalation | `owner_snapshot[strong_token] is absent` -> RuntimeError(f"strong owner for {key} is not live") | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1676 | WindowManager._defer_crossing_strong_escalation | `(unconditional)` ->  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1719 | WindowManager._strong_slab_ownership_candidate | `contribution.commit_lo <= plan.commit_hi                 and plan.commit_lo <= c` -> RuntimeError( f"strong slab {key} extent {plan.commit_lo}-"  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1743 | WindowManager._resolve_strong_region_plan | `not (             1 <= plan.context_lo <= plan.commit_lo             <= weak_win` -> RuntimeError( f"invalid strong-region bounds for {key}: cont | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1748 | WindowManager._resolve_strong_region_plan | `plan.commit_lo != weak_window.commit_lo` -> RuntimeError( f"strong-region commit for {key} must start at | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1761 | WindowManager._resolve_strong_region_plan | `crossing` -> RuntimeError( f"window {window.key} commits {window.commit_l | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1768 | WindowManager._resolve_strong_region_plan | `absorbed_window.queued or absorbed_window.committed` -> RuntimeError( f"cannot absorb window {absorbed_key}: already | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1780 | WindowManager._resolve_strong_region_plan | `plan.restart_buffer_lo is not None                     or plan.restart_seam_faul` -> RuntimeError( f"terminal strong-region plan for {key} cannot | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1786 | WindowManager._resolve_strong_region_plan | `restart.commit_lo != plan.commit_hi + 1` -> RuntimeError( f"strong-region plan for {key} ends at " f"{pl | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1793 | WindowManager._resolve_strong_region_plan | `plan.restart_buffer_lo is None                     or not 1 <= plan.restart_buff` -> RuntimeError( f"strong-region restart {expected_restart} nee | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1801 | WindowManager._resolve_strong_region_plan | `not isinstance(                     plan.restart_seam_fault_owner, SeamFaultOwne` -> RuntimeError( f"strong-region plan for {key} must select a v | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 1853 | WindowManager._build_strong_window_model | `not isinstance(             self.error_model_provider, MultiFaultExclusionSyndro` -> TypeError( f"device {type(self.error_model_provider).__name_ | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 1893 | WindowManager._absorb_window | `window.queued or window.committed` -> RuntimeError(f"cannot absorb window {key}: already " f"{'que | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1915 | WindowManager._absorb_window | `not needed <= replacements` -> RuntimeError("absorption replacement does not cover packets" | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1940 | WindowManager._build_pending_strong_job | `covered != needed` -> RuntimeError( f"{pending.label}: slab submitted with rounds  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1963 | WindowManager._submit_far_strong | `pending.wsd_arrival_ticks is None` -> RuntimeError("far strong submission requires WSD reservation | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 1983 | WindowManager._submit_terminal_strong | `pending.wsd_arrival_ticks is None` -> RuntimeError("terminal strong submission requires WSD reserv | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2128 | WindowManager._install_logical_contribution | `type(contribution.owner_key) is not tuple \                 or len(contribution.` -> TypeError( "logical contribution owner_key must be a two-ite | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 2134 | WindowManager._install_logical_contribution | `contribution.ownership_kind not in (             "ordinary_window",             ` -> ValueError( "logical contribution ownership_kind must be " " | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2141 | WindowManager._install_logical_contribution | `type(contribution.commit_lo) is not int             or type(contribution.commit_` -> TypeError( "logical contribution bounds must be exact ints") | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 2147 | WindowManager._install_logical_contribution | `contribution.commit_lo < 1             or contribution.commit_hi < contribution.` -> ValueError( f"logical contribution {contribution.owner_key}  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2154 | WindowManager._install_logical_contribution | `type(logical_observables) is not tuple` -> TypeError( f"logical contribution {contribution.owner_key} " | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 2165 | WindowManager._install_logical_contribution | `previous is not None and (             previous.commit_lo != contribution.commit` -> RuntimeError( f"logical contribution {contribution.owner_key | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2180 | WindowManager._install_logical_contribution | `contribution.commit_lo <= other.commit_hi                 and other.commit_lo <=` -> RuntimeError( f"logical contribution {contribution.owner_key | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2193 | WindowManager._install_logical_contribution | `expected_arity is not None                 and observed_arity != expected_arity` -> ValueError( f"logical contribution {contribution.owner_key}  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2211 | WindowManager._logical_observables_for_interval | `boundary_policy not in ("strict", "stream_segment")` -> ValueError( f"unknown logical contribution boundary policy " | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2215 | WindowManager._logical_observables_for_interval | `commit_lo < 1 or commit_hi < commit_lo` -> ValueError( f"invalid logical prediction interval " f"{commi | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2234 | WindowManager._logical_observables_for_interval | `not contributions` -> RuntimeError( f"logical prediction interval {stream_id!r} "  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2245 | WindowManager._logical_observables_for_interval | `covered_lo != cursor` -> RuntimeError( f"logical prediction interval {stream_id!r} "  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2251 | WindowManager._logical_observables_for_interval | `cursor != commit_hi + 1` -> RuntimeError( f"logical prediction interval {stream_id!r} "  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2269 | WindowManager._logical_observables_for_interval | `boundary_policy == "stream_segment"` -> RuntimeError( f"functional logical contribution " f"{contrib | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2273 | WindowManager._logical_observables_for_interval | `(unconditional)` -> RuntimeError( f"logical contribution {contribution.owner_key | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2288 | WindowManager._logical_observables_for_interval | `len(logical_observables) != arity` -> RuntimeError( f"logical prediction interval {stream_id!r} ch | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2302 | WindowManager._replace_contribution_prediction | `contribution is None` -> RuntimeError( f"result for {owner_key} has no logical contri | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2399 | WindowManager._send_boundary | `len(set(selected_targets)) != len(selected_targets)` -> RuntimeError( f"window interaction selected duplicate bounda | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2408 | WindowManager._send_boundary | `dep_key not in self.windows` -> RuntimeError( f"window interaction selected unknown boundary | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2413 | WindowManager._send_boundary | `source_key not in target.deps` -> RuntimeError( f"window interaction selected boundary target  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2418 | WindowManager._send_boundary | `target.queued or target.committed` -> RuntimeError( f"window interaction selected boundary target  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2473 | WindowManager._merge_available_boundary | `update.release_dependency` -> RuntimeError( f"window interaction released boundary depende | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2485 | WindowManager._receive_boundary | `source_key is None or version is None or delivery_version is None` -> RuntimeError("boundary delivery is missing source provenance | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2504 | WindowManager._receive_boundary | `update.accepted and (w.queued or w.committed)` -> RuntimeError( f"accepted boundary delivery {delivery_key} re | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2509 | WindowManager._receive_boundary | `dependency_released` -> RuntimeError( f"window interaction released boundary depende | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2513 | WindowManager._receive_boundary | `source_key not in w.deps or w.deps_remaining <= 0` -> RuntimeError( f"window interaction released unresolved edge  | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2530 | WindowManager._propose_boundary_update | `(unconditional)` -> TypeError( f"boundary state for {delivery.destination_key} m | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2554 | WindowManager._validate_boundary_update | `not isinstance(update, BoundaryUpdate)` -> TypeError( f"window interaction merge_boundary for " f"{deli | delete | type / exact-type / stable-identity / callable guard on an internal path |
| 2559 | WindowManager._validate_boundary_update | `not update.accepted and update.release_dependency` -> RuntimeError( f"rejected boundary {delivery.source_key}->" f | keep | state-machine invariant: a violation would silently corrupt results (strong tier, boundaries, ledger, buffer holds, protected streams, resource claims, seeds) |
| 2688 | WindowManager.accept_idle_decode_demand | `receiver is None` -> RuntimeError("idle decode demand receiver is not connected") | delete | re-proves wiring that run_spec makes by construction (Phase 2 constructor injection) |
| 2697 | WindowManager.bind_stream_operation | `previous is not None and previous != binding` -> RuntimeError("operation stream binding is already fixed") | delete | re-proves wiring that run_spec makes by construction (Phase 2 constructor injection) |
| 2703 | WindowManager.bind_required_stream_end | `operation_id in self._required_stream_end_by_operation_id` -> RuntimeError("protected feedback stream end is already bound | delete | re-proves wiring that run_spec makes by construction (Phase 2 constructor injection) |

## Applied (Phase 1, commits 14db92d..HEAD)

Every row marked delete is gone, with these adjustments found while
applying, each recorded here as the plan requires:

- planner._plan_execution gained one user config check with its reason:
  distance >= 1, commit_round_count >= 1, buffer_round_count >= 0 on the
  code card. The ResolvedCodeGeometry __post_init__ that used to reject
  d = 0 was internal ceremony by the table, but without it a zero geometry
  never terminates (the first lock run hung).
- links _validate_attribution_shape keeps "path requires a request / a
  boundary relation" (rows 790, 792 moved to keep): they are the link
  contract callers rely on; the stable-identity re-proving around them is
  gone.
- switching.validate_declared_run rows 260-286 moved to keep: switching
  options versus scheme and workload are user config rules (the "exact
  shipped serial SlidingWindowScheme" wording had matched the type-guard
  pattern).
- codes.py BB card value rules (n even, k <= n, d <= n, buffer override
  nonnegative) and decoder_memory rows 57 (capacity positive) and 234 (one
  job per unit memory) moved to keep.
- RunSeedPathSegment.kind: no __post_init__; canonical_bytes looks the tag
  up, so an unknown kind fails on its own.
- Engine: the whole phase machine, the stable-boundary flags and
  SimulationFailed are gone; delay >= 0, monotone event time and unique
  metric names stay. RunSpec.build is one call.
- Tests that asserted the removed guards were dropped (about 70), and
  tests that used engine._start_running were trimmed.
