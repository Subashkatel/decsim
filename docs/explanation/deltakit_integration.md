[decsim docs](../README.md) › [Explanation](README.md)

# Deltakit integration: boundary, evidence and limits

Deltakit is an optional input producer. It supplies ordinary Stim circuits,
explicit measurement schedules or repeated-circuit fragments. DECSIM owns
physical execution, simulated waiting, readout delivery, stores, windows,
decoder scheduling, frame commits and program release. Downstream components
use the same contracts for direct Stim and other producers.

The [run guide](../how-to/run_deltakit_workloads.md) covers finite memory,
live protection, physical idle noise, compiler experiments and SDK-free replay.
The original integration's implementation, independent review and functional
checks are complete. Full suites plus verified targeted repairs cover that
accepted snapshot; the original failure logs remain preserved. The owner accepted
the reviewed
branch baseline for regression acceptance. The maintained gate retains its
separate configuration incompatibility, as explained below. The follow-on
strong-only correction has clean full-suite results recorded separately below.

## Inputs and ownership

The optional `decsim.frontends.deltakit` boundary exports finite circuits
and reusable fragments:

- `memory_circuit` exports finite surface or repetition memory and an absolute
  measurement-index to one-based round map. Public CSS measurement stages
  determine chronology. Terminal data measurements join the last packet.
- `memory_rounds` exports `RepeatedStimCircuit`, four ordinary Stim fragments
  for first, repeated, final and single-round execution. One common qubit map
  preserves physical identity. The final fragment replaces a normal round.
- `css_memory_rounds` exports a supplied public CSS code, retaining every
  logical observable, including the BB example in the run guide.
- `bell_memory_rounds` exports two surface blocks with a transversal CNOT and
  explicit padded cadence, through the same canonical fragment record.

The compiler boundary, `decsim.frontends.deltakit_compiler.compile_experiment`,
exports the same finite circuit/map pair. All optional SDK calls end at these
boundaries. Neither a TICK nor a detector coordinate is interpreted as a round
or physical duration. Unsupported selected inputs fail explicitly.

Finite StimDevice preserves its declared horizon. StreamingStimDevice executes
only controller-requested rounds in one retained ordinary-Stim quantum state.
The controller chooses when destructive readout occurs; model lookahead never
samples that future. Before actual finalization there is no final truth to
score. Partial segments retain correction contributions without claiming
independent logical accuracy; the completed owner is scored once.

Physical and decoder-model sources can be separate implementations. The root
declares the physical stream independently, preserves circuits required by
either consumer, and reconciles finite limits before execution. Physical
completion precedes model finalization. Evolving models explicitly invalidate
pending windows, while finite models retain their stateful slicing ownership.

Final models become ready before a zero-delay readout can reach its decoder.
Protected-region release remains a later same-tick phase, after all protected
emissions. This preserves ordering when a waiting operation spans two patches.
Readout still crosses the normal links, buffers and decoder memory path.

The generic changes, their two-provider use cases and mature references are in
[decisions D22-D24](decisions.md#d22-physical-stream-ownership-is-independent-of-decoder-model-selection).
They borrow gem5's ownership and event-completion principles, Stim's retained
state and same-record conversion, and the existing qLDPC window-ownership law.
The generic `streaming_stim` row uses the existing registration mechanism.
Removing Deltakit leaves these independently useful contracts intact.

## Physical noise and actual stopping time

SD6 assigns noise to gate, reset, measurement and idle locations. Changing a
simulated period alone does not recalibrate an SD6 probability. The separate
physical model uses declared relaxation/dephasing times and the requested round
cadence. Microseconds become SDK seconds at the frontend. Its T1/T2 idle channel
is a Pauli approximation, not exact amplitude damping.

The producer uses an explicit native schedule and calibrates duration with the
public QPU timing API. The live source verifies that the declared period matches
DECSIM's resolved integer tick cadence before execution. A longer wait therefore
executes more noisy protection rounds on the same state. Preparation and
finalization remain part of that actual physical circuit.

At a 1.25 microsecond cadence, the saved physical-noise example runs 14 rounds
with 4 microseconds of feedback propagation and 18 rounds with 8 microseconds.
Actual readout moves from 17.5 to 22.5 microseconds. The earlier raw prefix is
preserved. The finite-horizon example instead retains its scheduled final
readout even when the waiting interval changes.

The live example saves reusable fragments, physical parameters, the actual
executed circuit, explicit map, raw measurements, result, command events and
trace. An actual SDK-absent environment replayed the saved fragments with
byte-identical circuit, measurements, result and trace. Replay with another seed
resamples the canonical program; it does not replay a recorded raw shot.

## Verified capability boundaries

VERIFIED describes the bounded evidence in each row. It does not establish an
unrestricted compiler, fault tolerance or a statistical logical error rate.

| Capability | Status | Physical/backend evidence | DECSIM scope |
| --- | --- | --- | --- |
| Surface and repetition memory | VERIFIED | Noisy finite circuits, explicit maps, same-record Stim detector/observable oracle | Functional decoding and SDK-free supplied-input replay |
| Live surface protection | VERIFIED | One retained state, actual final readout, exact shared-record conversion | Functional decoding with feedback-selected stopping time |
| Duration-aware memory noise | VERIFIED | Native schedule and T1/T2 channels checked against independent formulas | Live physical cadence checked before execution |
| Compiler memory | VERIFIED | Public CircuitBuilder, validated terminal records and logical support | Finite supplied-circuit decoding at distances 3 and 5 |
| Compiler terminal Hadamard | VERIFIED | Both logical bases, conjugate readout, clean and known-fault parity checks | Finite noisy supplied-circuit decoding |
| High-level LogAsm Hadamard/rotation | BLOCKED | Pinned observable backpropagation rejects these operations | Unsupported through that frontend |
| Leakage sampling | VERIFIED sampling only | Explicitly allocated Deltakit FlipSimulator preserves heralds and matches whole-circuit execution | Local leakage decoding BLOCKED by unverified model semantics |
| Shared multi-patch streams | VERIFIED | Retained joint state and complete-record Stim oracle, including interleaved acquisitions | Whole-group lifetime, strong and weak primary routes |
| Transversal Bell memory | VERIFIED | Initial cross-patch check parities, both bases, d3/d5, explicit physical wait noise | Live two-patch functional decoding; no lattice-surgery claim |
| qLDPC/bivariate bicycle | VERIFIED bounded example | Public [[30,8,2]] CSS code, all eight outputs, undecomposed noisy model | BP-OSD agreement, live feedback and higher-index logical-failure detection |
| Cloud stability and AC/LC services | NOT ATTEMPTED | Interfaces inspected | No cloud execution attempted |

Compiler Hadamard is transversal H followed immediately by destructive readout
in the conjugate basis. It does not establish continued extraction in the old
patch orientation, arbitrary logical programs or preserved code distance.
The exporter explicitly declares output from public support and validated
terminal records. It never relabels a detector by position or exposes evaluation
truth as syndrome. Compositional public routines avoid the deep-copy recursion
failure encountered by an earlier monolithic probe.

The live source supports a shared group of physical patches with trailing-buffer
feedback. One retained state preserves their correlations; one block can also
carry several logical outputs. Group ownership and acquisition ordering follow
[D26](decisions.md#d26-a-shared-physical-stream-owns-a-group-not-one-patch).
It does not implement arbitrary conditional quantum gates. Its retained executed
circuit and raw record grow with run length; there is no bounded host-memory claim.
Recorded sources remain finite because absent measurements cannot be invented.
The current fault catalog also refuses detectorless logical noise mechanisms;
retaining their probability and assigning them across windows requires a separate
generic extension. The BB checks use functional decoder vectors. They do not
establish multi-logical output sizing for every timing-only decoder card.

The Deltakit decoder-wrapper oracle also uses PyMatching. Agreement checks
conversion and integration, not independence of decoding algorithms. Windowed
decoding has different boundary semantics from whole-shot decoding and need not
return the same prediction on every noisy shot. Model tests separately check
stable fault identities, linked decompositions and strong-prior translation.

Independent live runtime probes cover direct Stim and Deltakit with 12-round
stores and explicit 12-round memory in each decoder pool. Both switching runs
complete seven real strong decodes, preserve the same-record truth oracle and
drain stores and units within their capacities. A five-round unit refuses a
six-round window. These candidate probes are recorded in
`completion/verifier/live-runtime-coverage.md`; the checked-in live tests do not
yet retain this combined resource/escalation case.

The follow-on strong-only work removes the dynamic-stream restriction through
the existing primary-tier contract. Ordinary window holds now follow their
selected store, and arrivals publish before unheld storage is reclaimed.
Strong-only idle rounds also use the strong store and decoder link. Neither
path visits a weak buffer or invokes escalation. The implementation and its
gem5 references are documented in decision D25. Public tests in
`tests/machine/test_machine.py` cover direct Stim and Deltakit,
controller and decoder event formation, actual final-record truth, declared
timing, bounded resources and static idle delivery. These are model-correctness
checks, not hardware timing calibration or a decoding-threshold claim.

## Upstream limitations preserved

The unchanged upstream ToyNoise example completed generation and decoding, then
failed in reporting on a NumPy integer. A separate variant using the documented
analysis inputs completed. Both original and corrected probe evidence remain.

The upstream high-level memory test emits zero observables and a final logical
check as a detector. Disabling generated flows removes that check without
supplying a complete returned logical-output mapping. The integrated lower-level
provider declares output through public measurement handles; it does not claim
to repair the high-level pipeline.

Ordinary Stim rejects the tested leakage instruction set. The pinned Deltakit
TableauSimulator accepts leakage instructions but treats them as no-ops, so it
cannot serve as a leakage backend. FlipSimulator preserves leakage when all
qubits are allocated initially; automatic table growth failed in the probe.
Across 256 shots, incremental and whole-circuit runs agree on measurement flips,
detectors, observables and herald positions. These are flips relative to an ideal
reference, not arbitrary absolute measurement outcomes.

The pinned leakage error analyzer leaves RELAX backpropagation unimplemented.
An exact local leakage-aware decoder model is therefore unverified. Observed
heralds cannot be discarded, and hidden leakage state cannot be supplied to an
ordinary decoder. No cloud service or upstream source modification was used.

## Original integration reproduction, review and regression status

Evidence lives outside the checkout at `../tmp/deltakit-integration/`.
`upstream/pins.json` and `upstream/requirements.lock` identify Explorer 0.9.2,
circuit/core/decode 0.9.1, Deltakit-Stim 0.2.5 and compiler 0.1.0, including source
commits. `completion/upstream/` contains producer, compiler, leakage and live
example reports with commands, artifacts and preserved failed probes.

The `deltakit` and `deltakit-compile` extras are separate. Both preserve core
Python 3.9 support; the tested SDK requires Python 3.10 through 3.14. Actual
Python 3.9 imports the core and runs direct Stim, then clearly refuses optional
provider selection. Enabled runtime checks use Python 3.11; Python 3.10
dependency resolution does not constitute runtime coverage. Installed pins and
active transitive requirements were checked without global environment changes.

Independent reviews in `completion/reviewer/` record exact file hashes. They
cover source/model separation, finalization ordering, model growth, compiler
output semantics and live replay. The complete line-by-line style audit also
checks new documentation. Automated checkers supplement that audit. The first
current whole-tree check found four unused arguments in two model methods;
separate deletion statements fix them without exemptions or behavior changes.

The supplementary immutable comparison against starting commit `013e2ba3`
passes all 26 points and all 97 real decoder-call fingerprints. It compares all
captured fields, including ordered transfers, queue state, ticks and log hashes.
Real decoding runs, while previously recorded measured service times are replayed
as timing inputs. Six negative controls verify comparison sensitivity.
The final candidate outputs are in `completion/verifier/runs/final-runtime-v1/`,
using the immutable `baseline-013e2ba-v2` bundle. This is a controlled regression
comparison, not a performance measurement or replacement of the maintained gate.

The owner explicitly accepted this reviewed baseline as regression acceptance
for the experiment worktree. `completion/owner-baseline-acceptance.md` records
the decision, immutable bundle identity and verified candidate run. This closes
the branch's acceptance requirement without changing the maintained gate or
reclassifying its preserved failures.

The maintained gate fails before simulation: its newer configurations omit
`unit_memory_rounds`, which this checkout requires. Frozen configurations and
goldens remain untouched. The historical natural-clock comparison also retains
its measured-decoder scheduling difference. Neither failure is reclassified as
a pass by the supplementary deterministic comparison.

Both full suites ran under the original container interpreter against this
worktree. Each found only an outdated listener fixture, which still constructed
a plan without physical-stream metadata. The repaired test uses real public
planning and checks declaration order, planned round count and independent
physical limits. Expected behavior was strengthened rather than relaxed.

| Environment | Original full run | Affected tests after repair |
| --- | --- | --- |
| SDK enabled | 2,116 passed, 6 skipped, 1 fixture failure in 616.89 seconds | 13 passed |
| SDK absent | 1,949 passed, 173 skipped, 1 fixture failure in 511.08 seconds | 13 passed |

The original full-run exit codes remain 1. This is full-suite coverage plus
targeted repair verification, not a subsequently executed clean full-suite run.
Of 433 captured executable/package files, 431 stayed unchanged; the other two
are the repaired listener test and the unused-argument style fix. Both affected
test files were rerun in both environments, including the new parameterized case.
See `completion/verifier/final-runtime-suites-v1/` for commands, logs, exact skip
reasons and the final source-stability proof.

Enabled skips are three unavailable sanitizer runtimes, optional ciw and two
dependency-absence assertions. The SDK-absent job skips the same four baseline
cases plus 169 provider-dependent cases. All enabled feature tests execute.
Actual repository style checks and the ten documentation checks pass. The
independent review records manual coverage beyond the checkers.

Five alternating baseline/candidate direct-Stim pairs produce identical full
results, reference fields, traces and narration. Median setup is 101.02 versus
100.09 milliseconds; median execution is 25.02 versus 24.86 milliseconds. These
short shared-host samples do not establish a speedup or a regression. They
measure one existing workload, not new-provider performance or simulated time.
`completion/upstream/runtime-overhead-final/REPORT.md` records the unchanged
benchmark law, raw samples, source hashes and safe reproduction instructions.

Source edits are confined to the experiment worktree. No commit, push,
publication, frozen-input replacement or maintained-gate adoption was performed.

## Strong-only follow-on validation

The existing primary-tier contract now serves finite and live streams, including
timing-only idle traffic. The 28 new public Machine cases cover both input
producers and event-formation placements, actual terminal truth, a nonzero
correction checked against direct PyMatching, declared timing, finite storage,
backpressure retries and idle-slot lifetime. An independent reviewer read every
changed line across the 18-file delta and rechecked the style fixes.

Clean full suites use the original container interpreter and the final reviewed
sources: 2,147 passed and 6 skipped with the SDK; 1,976 passed and 177 skipped
without it. Both processes exit zero, with no captured source changes during
execution. Enabled skips are three unavailable sanitizer runtimes, optional ciw
and two dependency-absence checks. The absent run skips provider-dependent tests
as well as the four unavailable baseline cases. Both runs retain the existing
plot-legend warning. Exact commands, hashes and skip reasons are in
`strong-only/full-suites-v2/`. An earlier interrupted run is preserved as
`full-suites-v1/ABORTED.json`, with no completed-suite claim.

`strong-only/regression-v2/` again matches all 26 accepted workloads and all 97
real decoder-call checks. The immutable accepted baseline remains unchanged.
`strong-only/check-final.log` and `one-action-final.log` record the actual
whole-tree checks, both passing with zero one-action findings. Existing advisory
reports remain visible. The independent review and exact file hashes are under
`strong-only/reviewer/`; `strong-only/verifier-report.md` records the public
acceptance oracles and their limits. These checks validate the configured timing
model and supported contracts, not physical hardware timing.

## Shared-history follow-on validation

The live source now executes a group on one retained quantum state. Ordered
acquisitions, group lifetime and complete-round event formation also serve direct
Stim and recorded workloads. The optional CSS and Bell producers use those same
contracts. [D26](decisions.md#d26-a-shared-physical-stream-owns-a-group-not-one-patch)
records the existing limitations, gem5 references and ownership decisions.

Independent review covered every changed line in the 67-file follow-on. The full
suite then exposed a stale dictionary key in one migrated test helper. Its
three-site repair preserves every assertion and is explicitly recorded as a
missed review finding. The original failed runs and approvals remain preserved.
Fresh full runs on the repaired snapshot pass: 2,247 tests with 6 skips with the
SDK, and 2,002 tests with 251 skips without it. All 583 captured executable and
configuration inputs remain unchanged during those runs. Enabled skips are three
unavailable sanitizer runtimes, optional ciw and two dependency-absence checks;
the absent job additionally skips provider-dependent tests. Both retain the
existing plotting warning. Exact commands and logs are in
`multipatch/full-suites-v2/`; failed predecessors remain in `full-suites-v1/`.

The accepted baseline again matches all 26 workloads and 97 decoder calls in
`multipatch/regression-final-v2/`. Production is unchanged since that comparison;
the only later executable-file edit is the reviewed test-helper rename. The
actual whole-tree style commands pass with zero one-action findings. Independent
probes cover noisy shared correlations, interleaved acquisition, bounded retries,
full BB vectors, higher-index logical failure and exact configured timing.
`multipatch/REPORT.md` records the evidence and limits. These are model checks,
not neutral-atom calibration, arbitrary logical programs or lattice surgery.

## Changed files and size

`completion/reviewer/final-approved-files-v3.json` inventories every changed
file in the original accepted integration snapshot, with its exact hash and
before/after/substantive line counts. The later strong-only delta has separate
review and validation evidence under `strong-only/`. The 67-file shared-history
delta is recorded in `multipatch/reviewer/approved-files-v3.json`, including
the additive final documentation review after full-suite completion. The v2
manifest preserves the executable repair approval.
Substantive counts omit blank lines, comments and Python docstrings. The
accompanying review records every-line coverage and any size advisories.

- Frontend modules, optional extras and examples provide canonical export,
  supported compilation, live execution and reproducible replay.
- The shared circuit record and live Stim source/model modules preserve actual
  state, explicit cadence, stable fault identities and actual terminal models.
- Existing assembly, planning, ports and window files separate physical source
  ownership from model ownership. Detector formation accepts growing recipes.
- Controller, engine and QPU changes convey final-round intent and establish
  final models before delivery while retaining the region-release phase.
- Machine reporting restricts logical accuracy to a completed physical owner.
- Tests exercise public contracts and source-backed oracles. Documentation and
  generated reference maps describe those changed contracts and runnable paths.

Existing decoder algorithms, links, stores, frame logic and escalation policies
remain the consumers of those generic contracts. Longer literal workload graphs
keep related dependencies visible together; size advisories do not justify
arbitrary helper layers.
