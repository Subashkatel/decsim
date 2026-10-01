[decsim docs](../README.md) › [Explanation](README.md)

# Deltakit integration: boundary and limits

Deltakit is an optional input producer. It supplies ordinary Stim circuits,
explicit measurement schedules or repeated-circuit fragments. decsim owns
physical execution, simulated waiting, readout delivery, stores, windows,
decoder scheduling, frame commits and program release. Downstream components
use the same contracts for direct Stim and other producers, so removing
Deltakit leaves every contract below intact.

The [run guide](../how-to/run_deltakit_workloads.md) covers finite memory,
live protection, physical idle noise, compiler experiments and SDK-free replay.

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

The compiler boundary, `decsim.frontends.deltakit_compiler.compile_experiment`,
exports the same finite circuit and map pair. All optional SDK calls end at
these boundaries. Neither a TICK nor a detector coordinate is interpreted as a
round or a physical duration. Unsupported selected inputs fail explicitly.

The finite `StimDevice` preserves its declared horizon. `StreamingStimDevice`
executes only controller-requested rounds in one retained ordinary Stim
quantum state. The controller chooses when destructive readout occurs; model
lookahead never samples that future. Before actual finalization there is no
final truth to score. Partial segments retain correction contributions without
claiming independent logical accuracy; the completed owner is scored once.

Physical and decoder-model sources can be separate implementations. The root
declares the physical stream independently, preserves circuits required by
either consumer, and reconciles finite limits before execution. Physical
completion precedes model finalization. Evolving models explicitly invalidate
pending windows, while finite models retain their stateful slicing ownership.

Final models become ready before a zero-delay readout can reach its decoder.
Protected-region release remains a later same-tick phase, after all protected
emissions. This preserves ordering when a waiting operation spans two patches.
Readout still crosses the normal links, buffers and decoder memory path.

The generic changes, their two-provider use cases and their references are in
[decisions D22 to D26](decisions.md#d22-physical-stream-ownership-is-independent-of-decoder-model-selection).
They borrow gem5's ownership and event-completion principles, Stim's retained
state and same-record conversion, and the existing qLDPC window-ownership law.
The generic `streaming_stim` row uses the existing registration mechanism.

## Physical noise and actual stopping time

SD6 assigns noise to gate, reset, measurement and idle locations. Changing a
simulated period alone does not recalibrate an SD6 probability. The separate
physical model uses declared relaxation and dephasing times and the requested
round cadence. Microseconds become SDK seconds at the frontend. Its T1/T2 idle
channel is a Pauli approximation, not exact amplitude damping.

The producer uses an explicit native schedule and calibrates duration with the
public QPU timing API. The live source verifies that the declared period
matches decsim's resolved integer tick cadence before execution. A longer wait
therefore executes more noisy protection rounds on the same state.
Preparation and finalization remain part of that actual physical circuit.

The finite-horizon example keeps its scheduled final readout when the waiting
interval changes; the live example executes the rounds that feedback selects.

The live example saves reusable fragments, physical parameters, the actual
executed circuit, explicit map, raw measurements, result, command events and
trace. Replaying the saved fragments without the SDK reproduces the circuit,
measurements, result and trace byte for byte under the same seed. Replay with
another seed resamples the canonical program; it does not replay a recorded
raw shot.

## Capability boundaries

Supported means the tests pin it within the stated scope. It does not
establish an unrestricted compiler, fault tolerance or a statistical
logical error rate.

| Capability | Status | What the tests pin | decsim scope |
| --- | --- | --- | --- |
| Surface and repetition memory | supported | Noisy finite circuits, explicit maps, same-record Stim detector/observable oracle | Functional decoding and SDK-free supplied-input replay |
| Live surface protection | supported | One retained state, actual final readout, exact shared-record conversion | Functional decoding with feedback-selected stopping time |
| Duration-aware memory noise | supported | Native schedule and T1/T2 channels checked against independent formulas | Live physical cadence checked before execution |
| Compiler memory | supported | Public CircuitBuilder, validated terminal records and logical support | Finite supplied-circuit decoding at distances 3 and 5 |
| Compiler terminal Hadamard | supported | Both logical bases, conjugate readout, clean and known-fault parity checks | Finite noisy supplied-circuit decoding |
| High-level LogAsm Hadamard/rotation | not supported | Pinned observable backpropagation rejects these operations | Unsupported through that frontend |
| Leakage sampling | supported, sampling only | Explicitly allocated Deltakit FlipSimulator preserves heralds and matches whole-circuit execution | Local leakage decoding not supported: the leakage model semantics are unverified |
| Shared multi-patch streams | supported | Retained joint state and complete-record Stim oracle, including interleaved acquisitions | Whole-group lifetime, strong and weak primary routes |
| qLDPC/bivariate bicycle | supported, one bounded example | Public [[30,8,2]] CSS code, all eight outputs, undecomposed noisy model | BP-OSD agreement, live feedback and higher-index logical-failure detection |

Compiler Hadamard is transversal H followed immediately by destructive readout
in the conjugate basis. It does not establish continued extraction in the old
patch orientation, arbitrary logical programs or preserved code distance.
The exporter explicitly declares output from public support and validated
terminal records. It never relabels a detector by position or exposes
evaluation truth as syndrome.

The live source supports a shared group of physical patches with
trailing-buffer feedback. One retained state preserves their correlations; one
block can also carry several logical outputs. Group ownership and acquisition
ordering follow
[D26](decisions.md#d26-a-shared-physical-stream-owns-a-group-not-one-patch).
It does not implement arbitrary conditional quantum gates. Its retained
executed circuit and raw record grow with run length; there is no bounded
host-memory claim. Recorded sources remain finite because absent measurements
cannot be invented. The current fault catalog refuses detectorless logical
noise mechanisms; retaining their probability and assigning them across
windows needs a separate generic extension. The BB checks use functional
decoder vectors. They do not establish multi-logical output sizing for every
timing-only decoder card.

The Deltakit decoder-wrapper oracle also uses PyMatching. Agreement checks
conversion and integration, not independence of decoding algorithms. Windowed
decoding has different boundary semantics from whole-shot decoding and need not
return the same prediction on every noisy shot. Model tests separately check
stable fault identities, linked decompositions and strong-prior translation.

Strong-only runs serve finite and live streams through the existing
primary-tier contract, timing-only idle traffic included. Ordinary window holds
follow their selected store, and arrivals publish before unheld storage is
reclaimed. Strong-only idle rounds use the strong store and decoder link.
Neither path visits a weak buffer or invokes escalation; decision D25 holds
the references. The public tests in `tests/machine/test_machine.py` cover
direct Stim and Deltakit, controller and decoder event formation, actual
final-record truth, declared timing, bounded resources and static idle
delivery. These are model-correctness checks, not hardware timing calibration
or a decoding-threshold claim.

## Upstream limitations preserved

The unchanged upstream ToyNoise example completes generation and decoding,
then fails in reporting on a NumPy integer. A variant using the documented
analysis inputs completes.

The upstream high-level memory test emits zero observables and a final logical
check as a detector. Disabling generated flows removes that check without
supplying a complete returned logical-output mapping. The integrated
lower-level provider declares output through public measurement handles; it
does not claim to repair the high-level pipeline.

Ordinary Stim rejects the tested leakage instruction set. The pinned Deltakit
TableauSimulator accepts leakage instructions but treats them as no-ops, so it
cannot serve as a leakage backend. FlipSimulator preserves leakage when all
qubits are allocated initially; automatic table growth is unsupported. Across
256 shots, incremental and whole-circuit runs agree on measurement flips,
detectors, observables and herald positions. These are flips relative to an
ideal reference, not arbitrary absolute measurement outcomes.

The pinned leakage error analyzer leaves RELAX backpropagation unimplemented.
An exact local leakage-aware decoder model is therefore unverified. Observed
heralds cannot be discarded, and hidden leakage state cannot be supplied to an
ordinary decoder. No cloud service or upstream source modification is used.

## Versions

The `deltakit` and `deltakit-compile` extras in `pyproject.toml` pin
Explorer 0.9.2, circuit, core and decode 0.9.1, Deltakit-Stim 0.2.5 and
compiler 0.1.0. The pinned SDK requires Python 3.10 through 3.14, and
decsim's own floor is Python 3.10.

## What changed in the tree

- Frontend modules, optional extras and examples provide canonical export,
  supported compilation, live execution and reproducible replay.
- The shared circuit record and live Stim source and model modules preserve
  actual state, explicit cadence, stable fault identities and actual terminal
  models.
- Existing assembly, planning, ports and window files separate physical source
  ownership from model ownership. Detector formation accepts growing recipes.
- Controller, engine and QPU changes convey final-round intent and establish
  final models before delivery while retaining the region-release phase.
- Machine reporting restricts logical accuracy to a completed physical owner.

Existing decoder algorithms, links, stores, frame logic and escalation policies
remain the consumers of those generic contracts.
