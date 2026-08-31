# decsim credibility audit (NASA-STD-7009B framing)

Date: 2026-08-30. Audited tree: branch `reorganize` at 8e5cf56 (local).
The public GitHub `main` (155a3e9) is an older snapshot, 441 commits behind
this line at merge-base b67896f; everything cited below is the local tree.
Read-only audit: no code was modified.

Vocabulary used throughout, per NASA-STD-7009B: **verification** asks
whether the simulator implements its own specification correctly;
**validation** asks whether the model reproduces real hardware well enough
for the intended use. Claims are tagged FACT (checked in source this pass),
MODEL CHOICE (a deliberate abstraction), ASSUMPTION, or UNVALIDATED
PARAMETER.

---

## 1. Architecture map

| Responsibility | Owner (file / class) | Key anchors |
|---|---|---|
| Event scheduling | `decsim/engine.py` `Engine`, `Event` | ordering key (time, priority, seq) engine.py:14-22; schedule engine.py:36-47; run loop engine.py:84-94 |
| Tick/time units | `decsim/config.py` (`us`, `fmt`) | integer ticks; all latencies priced through `us()` |
| QPU cycle clock | `decsim/qpu/cycle_clock.py` `QPUDevice` | boundary math cycle_clock.py:65-69; per-cycle tick cycle_clock.py:80-93; idle emission cycle_clock.py:95-98 |
| Rounds per operation | `decsim/qpu/round_policies.py` (Fixed/PerOp/Code/Gate/Temporal) | GateRounds cites Horsman 1111.4022 and Litinski 1808.02892, round_policies.py:56-67 |
| Syndrome generation | device models `decsim/qpu/stim_device.py`, `syndrome_devices.py`; emission `cycle_clock.py:137-157` | fragment-count contract enforced at emission cycle_clock.py:140-151 |
| Detector formation | `decsim/controller/syndrome_packing.py:303`; window DEMs in `decsim/detector_error_model/` | |
| Packetization | `SyndromePacking` (controller/syndrome_packing.py) | qc leg + controller processing syndrome_packing.py:159-179; `t_pack` syndrome_packing.py:192-194; CWB arbitration/pipelining syndrome_packing.py:318-401 |
| Network links | `decsim/links/links.py` `Link`, `LinkModel`; cards in `link_profiles.py` | FIFO serializer + propagation links.py:436-474; 10-path vocabulary `LinkPath` links.py:239 |
| Buffers/queues | `decsim/syndrome_buffer/syndrome_buffer.py` (Buffer 0, refcounted holds, capacity admission), `syndrome_buffer_1.py` (SB1 room-side store, `ready_tick` gap refusal :120-130) | |
| Window readiness | `decsim/windows/window_manager.py` | stamps window_manager.py:665-674, 751-755, 1004 |
| Decoder scheduling | `decsim/decoders/decoder_manager.py` `DecoderManager` | pools + two-slot units; service claim `_begin_service` :654 through `_on_decode_done` :863 |
| Decoder processing | `decsim/decoders/decoder_engine.py` `DecoderEngine` (staged); latency models `decoders.py:101-160`; real algorithms in `mwpm/ union_find/ bposd/ relay_bp/ belief_matching/ tesseract/` | declared vs measured wall clock decoder_engine.py:153-165 |
| Strong tier / switching | `decsim/decoders/strong_escalation.py`, `weak_strong_switching.py`; confidence in `decsim/confidence/` | |
| Pauli frame | `decsim/pauli_frame/pauli_frame.py` | priced commit; zero cost demands a written justification pauli_frame.py:41-56 |
| Feedback / conditional ops | `decsim/pauli_frame/conditional_release.py`, `decsim/controller/` | OC then CQ dispatch conditional_release.py:33-43 |
| Runtime/planner | `decsim/frontends/execution_runtime.py`, `planner.py`; assembly `run_spec.py` (register :335, plan load :336, streams :339, program :342, run :345) | |
| Observability | `decsim/observe/metrics.py` (utilization, backlog, `WindowLatencyBreakdown` :214-253, `ConditionalReactionTime` :519), `run_views.py`, `links/link_traffic_report.py`, engine log | |

## 2. Current timing model

decsim already decomposes the reaction path into the requested legs; none
of them is collapsed into one decoder number (FACT):

- T_measurement: **inside the QEC cycle** (MODEL CHOICE). Readout is
  emitted at the cycle boundary; measurement duration is not a separate
  event. The QC leg then prices QPU-to-controller transport, and
  "controller-binary-availability" prices controller processing
  (syndrome_packing.py:159-179).
- T_packetization: fragment reassembly plus `t_pack`
  (syndrome_packing.py:181-198).
- T_qpu_to_decoder_transport: CWB (weak path) and CSB (strong side store),
  each an explicit `Link.reserve` with queue wait, serialization, and
  propagation (links.py:448-466). WBD/WSD/SBD price buffer-to-decoder
  input transfers.
- T_queue: DecoderManager pool queues plus the two-slot input/compute
  join; queue wait is stamped per window (`WindowLatencyBreakdown`
  rows: buffer_fill, dep_block, queue_wait, service).
- T_decoder_processing: DecoderEngine stage walk plus algorithm time,
  either declared (`latency(job)`) or measured host wall clock for real
  software decoders (decoder_engine.py:157-165).
- T_decoder_to_controller_transport: WDO/DO transfers, then the priced
  Pauli-frame commit.
- T_controller_reaction: OC then CQ per released decision
  (conditional_release.py:33-43).

Latency vs throughput: links separate serialization (bandwidth) from
propagation (latency) and expose queue_wait explicitly (links.py:448-466).
Rounds pipeline on CWB rather than stop-and-wait
(tests/12_links/test_cwb_q062.py:233). The one conflation is inside a
decoder unit; see section 7, risk R1.

Verified end-to-end trace (declared ticks, 1.0 us rounds, serial
switching, d=3; asserted exactly by tests/21_stabilization): round r
published at r+9 (qc 2 + binary 3 + cwb 4); window data complete 15; weak
done 30 (wbd 5 + weak 10); escalation: strong start 36 = max(30 + sbd 6,
30 + wsd 3), strong done 66, DO lands 70, frame committed 71, next window
released 71.5. Removing any leg fails a named test (mutation matrix
M11, M17-M20, M22).

## 3. Current QEC-cycle behavior

- Every live patch emits one round every cycle, operating or idle
  (cycle_clock.py docstring :1-10, `_emit_idle_rounds` :95-98). FACT,
  tested: tests/16_qpu `test_an_idle_patch_keeps_extracting_every_cycle...`.
- Operations start and end on cycle boundaries and must match the QPU
  cadence (cycle_clock.py:51-52, 65-69). QPU-003 holds.
- The discrete-event engine can jump, but a boundary event is re-armed
  for every cycle while anything is running, idle, or pending
  (cycle_clock.py:92-93), so no live cycle is skipped by time-jumping.
  QPU-004 holds by construction; a global count-conservation property
  test is still missing (section 9, T2).
- Idle extraction stops only at explicit `finish()`
  (cycle_clock.py:59-61, 90-91), a deliberate end-of-program semantics
  with its own test. Idle rounds can also travel the feedback-memory
  route (cycle_clock.py:165-170) under the configured round policy.

## 4. Current decoder timing model

- Timing-only models: `PresetLatencyDecoder` (constant), `PerRoundDecoder`
  (n_rounds * tau), `FunctionLatencyDecoder` (arbitrary job -> us)
  (decoders.py:101-160). Deterministic single values; no latency
  distribution or jitter (MISSING for the validation goal).
- Staged model: `DecoderTiming` prices named stages before and after the
  algorithm stage (decoder_engine.py:125-134).
- Real algorithms (MWPM, union-find, BP+OSD, relay-BP, belief matching,
  Tesseract) run in measured mode: the unit is held for the measured host
  wall time (decoder_engine.py:157-163). Nondeterministic by design; the
  frozen suite compares those points on a semantic projection only.
- Scheduling: pool credits (`pool_free`), FIFO scheduler, two-slot units
  with a decoupled input-DMA/compute join (`_start_job` :585,
  `_offer_compute` :427), service gate parking, idempotent cancellation.
  Results exist only when service time ends (decoder_manager.py:687,
  decoder_engine.py:172-175): DEC-005 holds.
- Throughput: capacity comes from unit count only; one unit is occupied
  for its full service time. See risk R1.

## 5. Current communication model

Ten named paths (QC, CWB, CSB, WBD, WSD, SBD, WDO, DO, OC, CQ;
links.py:239-252). Each wired path is a FIFO serializer: send order is
enforced (out-of-order send ticks raise, links.py:445-446), a payload
waits for the serializer, is serialized at configured bits/us, then
propagates (links.py:448-466). Finite bandwidth therefore yields
queueing, not instantaneous transmission (LINK-003). Unbounded links
price propagation only, an explicit configuration. Every number on a
link card carries a `source` string that travels into the topology
(link_profiles.py:10). Traffic is attributed and countered per path
(`TrafficCounters`, `link_traffic_report.py`).

## 6. Existing verification evidence

- 790 tests pass at HEAD (752 pre-existing + 38 stabilization).
- Engine invariants: tests/00_engine (past-event rejection, same-tick
  priority-then-insertion order, unique sequences, metric ordering).
- QPU clock: tests/16_qpu (boundary starts, idle continuity, finish).
- Links: tests/12_links, including exact FIFO timing equations
  (test_links.py:283) and CWB pipelining/capacity/drop tests.
- End-to-end determinism: tests/21_stabilization, 38 tests whose
  assertions are exact declared-tick arithmetic (the trace in section 2).
- Frozen behavioral gate: validation/responsibility_audit_2026_08_30/
  (9 frozen points, golden sha-pinned, independent verifier with its own
  comparison code).
- Mutation matrix: 23 timing mutations (removed legs, moved timestamps,
  callback reordering, counter semantics), 22 detected, 1 proven inert by
  the refcount design; two first-sweep misses exposed real suite gaps
  that were closed (reports/DECSIM_MUTATION_MATRIX.md). This is exactly
  the commissioned mutation-testing step, already executed.
- Differential verification of windowing semantics: sliding windows
  bit-identical to qLDPC, cuda-qx, and the Tan reference loop except
  documented MWPM ties (2026-08-22 harness, sandbox tmp/reference-decoders;
  not yet a repo pytest, see T10).
- External reference reproduction: SWIPER, PECOS, Helios, QubiC
  distributed_processor reproduced on this host; pins and licenses in
  validation/.../reference_manifest.yaml.

**Validation status, stated plainly: decsim is substantially verified and
essentially unvalidated.** No end-to-end timing distribution has been
compared against a measured hardware trace. Parameters are sourced
(link cards, GateRounds citations) but none is tagged
measured/calibrated; all are literature or spec derived. UNVALIDATED
PARAMETER applies to every hardware number in the cards.

## 7. Suspected correctness risks

- **R1 (P1). Decoder unit latency/throughput conflation.** A unit's
  compute is claimed from `_begin_service` (decoder_manager.py:654) to
  `_on_decode_done` (:863); effective per-unit throughput is 1/latency.
  A pipelined hardware decoder (accepts one round per cycle while a long
  result latency is in flight, e.g. Helios, AFS) cannot be represented
  by one unit; adding units misstates memory and area. FACT about the
  code; the limitation matters only for hardware-decoder studies.
- **R2 (P2). No latency jitter.** Declared models are single-valued;
  only measured mode produces variation, and that variation is host
  noise, not a modeled distribution. DEC-002 is met, but validation
  against a latency distribution is impossible until a seeded
  distribution model exists.
- **R3 (P3). Far-boundary deferral in a backlog regime refuses with a
  bare KeyError** (strong_escalation.py:716) instead of the contract
  message. Correct refusal, wrong error text. Already flagged.
- **R4 (documented constraint, not a bug).** Parallel run-both-at-once
  requires the CSB margin; refused loudly at strong-job build. Pinned by
  test_parallel_requires_the_csb_margin.
- **R5 (P2). Round accounting is reconstructable but not ledgered.**
  Publication ticks, window stamps, decoder stage records
  (decoder_engine.py:166-167), link counters, and log lines exist, but
  no single per-round record proves conservation from emission to
  terminal state (delivered / dropped / expired via
  `_expire_reassembly`, syndrome_packing.py:427). LINK-004 and QPU-001
  hold locally but lack one global check.
- **R6 (validation gap).** No hardware profile bundles a full named
  system with per-parameter provenance; parameters live in link cards,
  experiment YAMLs, and round policies separately.
- ASSUMPTION worth recording: measurement and reset are sub-cycle and
  never separately rate-limiting (they are folded into `cycle_ticks`).
  Acceptable for the intended use (reaction-time co-design at round
  granularity), but it should be written into the conceptual model.

## 8. V&V matrix

Statuses: PASS needs cited evidence; PARTIAL means held by construction
or locally but missing a direct test; MISSING means no evidence.

| ID | Requirement | Implementation | Evidence | Status | Missing work |
|---|---|---|---|---|---|
| SIM-001 | time never moves backward | engine.py:42-45, 89-92 | tests/00_engine past-event rejection | PASS | property test over random schedules (T1) |
| SIM-002 | deterministic tie order | (time, priority, seq) engine.py:14-22 | test_same_tick_events_use_priority_then_insertion_order | PASS | |
| SIM-003 | no event disappears | heap drained in run(); cancel = inert flag, event still fires | run loop engine.py:87-94 | PASS | |
| SIM-004 | no early execution | heap pop + monotone clock | tests/00_engine | PASS | |
| QPU-001 | every live patch, every cycle | _emit_idle_rounds / _emit_operation_rounds | 16_qpu idle-continuity test; 21_stabilization arithmetic | PARTIAL | global round-conservation property (T2) |
| QPU-002 | idle does not stop extraction | idle map until finish() | 16_qpu finish test | PASS | |
| QPU-003 | cycle-boundary alignment | next_boundary; cadence check :51 | 16_qpu boundary test | PASS | |
| QPU-004 | no cycle skipped by DES jump | _arm chain :92-93 | mutation matrix; construction | PARTIAL | same as T2 |
| LINK-001/002 | no arrival before generation + serialization + propagation | reserve equations links.py:448-466 | test_finite_fifo_reservations_obey_exact_timing_equations | PASS | |
| LINK-003 | finite bandwidth queues | serializer FIFO, queue_wait | CWB pipelining and capacity tests | PASS | |
| LINK-004 | every packet accounted | contexts expire; check_work_settled :446; buffer admission records | settledness checks | PARTIAL | terminal-state accounting test (T4) |
| DEC-001 | input precedes decode | input-DMA join _start_job; SB1 ready_tick | M11 detected; test_sb1_gap_cannot_be_served | PASS | |
| DEC-002 | configured service behavior | DecoderEngine stage walk | 21_stabilization exact arithmetic | PASS | distributions (R2) |
| DEC-003 | throughput independent of latency | units only; no initiation interval | none | PARTIAL | decoder-unit v2 (C1) |
| DEC-004 | overload queues/drops by policy | pool queues; buffer capacity + DROP_ROUND | backlog metrics; capacity tests | PASS | sustained-overload test (T6) |
| DEC-005 | no result before completion | decode at service end (decoder_manager.py:687) | M19/M20 detected | PASS | |
| DEC-006 | capacity never exceeded | pool_free credits | DecoderUtilization; dispatch guards | PASS | |
| FBK-001 | return + controller latency | OC then CQ priced | test_release_travels_oc_then_cq_with_exact_cost; M22 | PASS | |
| FBK-002 | conditionals wait for the decode | ConditionalRelease on final results only | test_successor_cannot_start_before_its_release; M23 | PASS | |
| E2E-001 | monotone causal chain | stamps at every stage | exact pinned timelines (specific configs) | PARTIAL | universal property over random configs (T5) |
| VAL-HW | agreement with measured hardware | none | none | MISSING | hardware profiles + trace comparison (C3) |

## 9. Top 10 missing tests

1. **T1** Hypothesis property, engine: random (delay, priority) event
   schedules; assert observed execution times are nondecreasing and
   same-tick order matches (priority, seq). Pins SIM-001/002 universally.
2. **T2** Round conservation: for random programs, rounds emitted by the
   QPU == rounds admitted + refused + dropped at Buffer 0, and per-op
   round counts equal the policy's declaration. Pins QPU-001/004 globally.
3. **T3** Link contention property: N same-tick sends on one finite
   link; assert serializer intervals are disjoint, order preserved, and
   the last delivery equals sum of serializations plus propagation.
4. **T4** Packet terminal-state accounting: every packed round ends in
   exactly one of delivered / dropped / expired, using the packing
   snapshot and buffer admission records. Pins LINK-004.
5. **T5** E2E causal chain property: for every decoded window in a
   randomized-config run, assert publication <= data complete <= queue
   <= service start <= done <= frame accept <= release, from the
   existing stamps. Pins E2E-001 beyond the pinned configs.
6. **T6** Sustained overload: offered round rate above pool capacity for
   many windows; assert backlog growth matches the deficit and drains at
   the theoretical rate after the burst.
7. **T7** Throughput-vs-latency xfail: a unit should accept window i+1
   while window i's long-latency result is in flight; marked xfail today
   to document DEC-003 until C1 lands.
8. **T8** Idle continuity over random gaps: place two operations G
   cycles apart (Hypothesis over G); assert exactly G idle rounds per
   live patch in between.
9. **T9** Shutdown drain: finish() with strong work in flight; assert
   check_decode_work_settled and packing check_work_settled both pass
   and no event remains.
10. **T10** Differential semantics in-repo: one window stream through
    the MWPM window decoder vs PyMatching directly on the same window
    DEM; assert identical corrections (imports the 2026-08-22 harness
    result as a permanent pytest).

## 10. Top 5 highest-priority changes

1. **C1 (P1)** Decoder-unit service model with an initiation interval:
   separate result latency from unit occupancy so pipelined hardware
   decoders are representable (design already drafted in the 2026-08-28
   UF-ASIC study). Test first: T7.
2. **C2 (P2)** Cycle/event ledger as an observe/ metric: one record per
   (patch, round) assembled from existing stamps (emission, publication,
   queue, service, frame, release), exportable per run. No core change;
   it also powers T2, T4, T5.
3. **C3 (P1 for validation)** `hardware_profiles/<system>.yaml`: bundle
   QPU cadence, link cards, decoder timing, controller costs per named
   system with a provenance tag per parameter (measured / spec / paper /
   calibrated / estimated / assumed), extending the link cards' existing
   source-string discipline.
4. **C4 (P2)** `credibility/` directory per NASA-HDBK-7009B: intended
   use, conceptual model (including the sub-cycle measurement
   assumption), traceability.csv mapping the invariant IDs above to
   tests, uncertainty and limitations. Mostly indexing evidence that
   already exists in validation/ and reports/.
5. **C5 (P3)** Replace the far-boundary bare KeyError with the contract
   message (strong_escalation.py:716); the reproducing configuration is
   already recorded in reports/DECSIM_BUFFER_READINESS_AUDIT.md.

## Smallest next implementation step

Create `credibility/traceability.csv` plus `intended_use.md` (C4, first
slice): zero code risk, converts the existing evidence base into the
NASA structure, and makes every PARTIAL and MISSING row above a visible,
assignable gap. The first code-adjacent step after that is T2 (round
conservation), which needs no new instrumentation.
