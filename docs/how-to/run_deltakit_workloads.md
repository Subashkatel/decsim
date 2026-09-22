[decsim docs](../README.md) › [How-to guides](README.md)

# Run Deltakit memory, protection and compiler workloads

Deltakit supplies a circuit and its measurement schedule. The existing
decsim machine runs them through readout, stores, windows, decoding and
feedback. The finite example is `tools/deltakit_example.py`;
`tools/live_memory_example.py` keeps memory live until decoded feedback
permits final readout. Neither adds a yaml workload row.

## Install in a separate environment

From the checkout root, with Python 3.10 or newer:

```bash
python3.11 -m venv .venv-deltakit
.venv-deltakit/bin/python -m pip install -e '.[run,deltakit]'
.venv-deltakit/bin/python -m pip install 'ldpc==2.4.1' 'beliefmatching==0.2.0'
```

The second install is required: the machine imports the `ldpc` and
`beliefmatching` decoder backends at build time, and neither extra pins
them. The examples do not select them. Keep this environment separate
from another checkout's editable installation.
The Deltakit extra pins component versions in `pyproject.toml`; it does
not require the umbrella SDK or a cloud account. Core Python 3.9 support
is unchanged; the pinned SDK supports Python >=3.10,<3.15. For the compiler
entrypoint below, install its separate extra in the same isolated
environment:

```bash
.venv-deltakit/bin/python -m pip install -e '.[run,deltakit-compile]'
```

Explorer memory users do not need the compiler extra.

## Run a memory experiment

```bash
PYTHONPATH=. .venv-deltakit/bin/python tools/deltakit_example.py \
  --mode memory --family rotated_surface --distance 3 --rounds 24 \
  --basis Z --probability 0.001 --seed 7 \
  --output results/deltakit-memory
```

Use `--family repetition` for repetition memory, with X or Z selecting
both its check basis and logical preparation/readout basis. Repetition
uses a whole-shot window in this example; the protection mode requires
rotated surface memory. `--patch` names the patch the operations occupy;
the default is `memory-patch`.

The output folder contains the circuit, measurement map, argument values,
result, command arrival/start events and Chrome trace. Inspect
`result.json`, `commands.json` and `trace.json` together. The separate
`setup_seconds.json` measures host export/build time, not simulated latency.
Sampling and execution also consume host time during the run; that cost
is not included in this setup measurement.

For a Python caller, the optional boundary is:

```python
import decsim.frontends.deltakit as deltakit

circuit, measurement_rounds = deltakit.memory_circuit(
    code_family="rotated_surface",
    distance=3,
    round_count=24,
    basis="Z",
    physical_error_probability=0.001,
)
```

Pass these outputs through the existing supplied-circuit API, as
`supplied_settings` in the runnable example does. The map assigns every
absolute measurement index to a one-based round. Final data measurements
join the last packet. Preserve this map with the circuit.

## Keep memory live until feedback releases it

```bash
PYTHONPATH=. .venv-deltakit/bin/python tools/live_memory_example.py \
  --distance 3 --basis Z --physical-error-probability 0.001 \
  --round-period-microseconds 1.25 --feedback-microseconds 4 \
  --output results/live-memory

PYTHONPATH=. .venv-deltakit/bin/python tools/live_memory_example.py \
  --input results/live-memory --feedback-microseconds 8 \
  --output results/live-memory-slower
```

The default noise model is SD6. The prefix requests decoding, protection
continues on the same live stream, and decoded release permits resume
and actual final readout. The prefix ends on a window boundary, so
`--prefix-rounds` is a whole number of the distance (the default 3
suits distance 3; use 5 or 10 at distance 5), or the first commit
refuses it. There is no fixed round horizon. Longer feedback
adds physical syndrome rounds before destructive readout. The ordinary
`StreamingStimDevice` retains the quantum state across those rounds;
decsim owns the wait, the QPU cadence and the stopping decision. The provider
only exports the physical fragments.

For duration-aware noise, generate a separate physical input:

```bash
PYTHONPATH=. .venv-deltakit/bin/python tools/live_memory_example.py \
  --noise-model physical --physical-error-probability 0.001 \
  --relaxation-time-microseconds 50 --dephasing-time-microseconds 40 \
  --round-period-microseconds 1.25 --output results/live-physical
```

Physical noise uses the supplied probability at gates, resets and
measurements, plus a Pauli approximation of T1/T2 noise during idle
intervals. The exporter calibrates the native schedule to the declared
cadence. SD6 instead uses its probability at gates, resets, measurements
and idle locations; it does not use relaxation or dephasing times.

Live output saves four reusable ordinary Stim fragments and
`physical_parameters.json`. It also saves the actual `executed.stim`,
`measurement_rounds.json`, `measurements.json`, result, commands, trace and
arguments. The actual circuit and raw measurements can be checked with
Stim's measurement-to-detector converter after the runtime-selected stop.

Live `--input` loads these canonical fragments without selecting Deltakit.
It preserves every physical parameter, including cadence, and rejects
conflicting overrides. Prefix length, feedback latency, decoder service
time, seed and patch remain runtime choices. Use `--prefix-rounds`,
`--decoder-microseconds`, `--seed` and `--patch` to set them explicitly.
Replay executes the fragments again; `measurements.json` separately records
the completed run's raw measurements for independent conversion.

This path supports rotated-surface memory and trailing-buffer feedback.
It does not implement arbitrary result-dependent quantum gates. Canonical
replay of the physical-noise example works without the SDK installed.

## Supply a joint group or a CSS code block

`RepeatedStimCircuit` describes one shared physical history. Assign the complete
resource footprint to the stream owner's `Operation.patches`, and include that
footprint at both protected-region endpoints. All members use the same declared
round cadence. `ProtectedRegion` refers to the owner stream; it has no separate
patch declaration. The source executes each requested round once for the group.

The optional provider can produce two distinct kinds of input:

```python
import deltakit_explorer.codes as codes

import decsim.frontends.deltakit as deltakit
import decsim.qpu.streaming_stim_device as streaming_stim_device

bell = deltakit.bell_memory_rounds(
    3, "Z", 0.001, round_period_microseconds=1.1
)
bell_source = streaming_stim_device.StreamingStimDevice({100: bell})

code = codes.BivariateBicycleCode(3, 5, [1, 1, 4], [0, 1, 2])
block = deltakit.css_memory_rounds(
    code, "Z", 0.001, round_period_microseconds=1.1
)
block_source = streaming_stim_device.StreamingStimDevice({100: block})
```

Use a two-patch owner for the Bell source. It prepares two rotated-surface
blocks with a transversal CNOT and reports their joint parity in X or Z.
This is a physical Bell-memory experiment, not high-level lattice surgery.
Its explicit wait slots make every exported fragment occupy the declared
period and receive the selected idle noise exactly once.

The BB example is one [[30,8,2]] block with eight logical outputs. Use one
physical resource patch and eight logical qubit identities: the owner's
`qubits` are `range(8)` and its `patches` the one block. Its code card is
`code_geometry.BivariateBicycleCodeModel(qubit_count=30,
logical_qubit_count=8, distance=2)` with the window overrides the run
wants, and its decoder a full-model one such as BP-OSD: the noisy model
contains hyperedges that graphlike matching cannot represent exactly.
`_bb_settings` in `tests/machine/test_machine.py` is the runnable
assembly of those pieces. Patch count never sizes these logical outputs.
The distance describes this small code, not the fault distance of its
syndrome schedule.

The end-to-end examples are reproducible through the public Machine tests:

```bash
PYTHONPATH=. .venv-deltakit/bin/python -m pytest tests/machine/test_machine.py \
  -k 'deltakit_bell or bb_block or bb_live or bb_higher or interleaved_joint' -q
```

They exercise live feedback, controller-side and decoder-side formation, full
logical vectors, deliberate higher-index failure, and a recorded source with
interleaved patch acquisitions. Direct Stim uses the same source contract.
The hardware cadence and link/service costs remain configured model inputs;
these checks do not establish calibrated neutral-atom transport or loss physics.

## Compare feedback waits on one finite history

```bash
PYTHONPATH=. .venv-deltakit/bin/python tools/deltakit_example.py \
  --mode protection --rounds 24 --prefix-rounds 3 \
  --period-microseconds 1.1 --feedback-microseconds 4 \
  --decoder-microseconds 0.1 --seed 7 \
  --output results/deltakit-protection

PYTHONPATH=. .venv-deltakit/bin/python tools/deltakit_example.py \
  --input results/deltakit-protection --mode protection --prefix-rounds 3 \
  --period-microseconds 1.1 --feedback-microseconds 8 \
  --decoder-microseconds 0.1 --seed 7 \
  --output results/deltakit-protection-slower-feedback
```

The prefix's decoded decision releases the continuation. The controller
keeps the patch protected while it waits; the QPU starts the continuation
on its eligible cycle boundary. Arrival and start are separate events.
The prefix ends on a window boundary: the rotated surface card commits
`distance` rounds per window, so `--prefix-rounds` is a multiple of the
distance (3 or 6 at distance 3). A prefix that ends inside a window is
refused by the window's commit, because a scored segment cannot share a
window with its continuation.
The declared decoder service time prices actual functional PyMatching.
It is not a measured decoder benchmark or a timing-only substitute.

The physical history has one preparation and one scheduled final readout
at the declared horizon. Longer feedback can change which rounds protect
the waiting patch and when continuation starts without changing the total
physical round count. This is a scheduling dependency on a decoded result;
it does not implement result-dependent quantum gates. A continuation that
needs rounds beyond the finite source fails explicitly.

## Replay or replace a finite producer

`--input` reads ordinary Stim text and the saved measurement map without
calling Deltakit. It inherits only the physical parameters: family,
distance, horizon, basis and noise probability. An explicit conflicting
physical parameter is refused with a `ValueError` naming the parameter.
Runtime choices, including period,
feedback latency, decoder service time, mode, prefix length, patch and
seed, come from the new command and its defaults. State them explicitly
when comparing timing. Replay resamples the exported circuit; it is not
replay of a saved raw measurement record.

A direct Stim generator or another frontend can supply the same circuit
and explicit map through the existing Python settings. No downstream
component needs the producer's name. Remove the optional Deltakit
packages when generation is no longer needed; ordinary Stim remains
necessary for supplied-circuit execution.

## Compile memory or a terminal Hadamard

The separate optional compiler frontend accepts `memory` and `hadamard`.
It uses public rotated-code schedules and CircuitBuilder measurement
handles, returning ordinary Stim and the same explicit round-map contract:

```python
import decsim.frontends.deltakit_compiler as compiler

circuit, measurement_rounds = compiler.compile_experiment(
    experiment="hadamard",
    distance=5,
    round_count=4,
    basis="Z",
    physical_error_probability=0.003,
)
```

Use `experiment="memory"` for compiled memory, or `basis="X"` for the
other prepared logical basis. Feed these outputs to the existing finite
supplied-circuit settings, with `FixedRounds(round_count)` and the caller's
QPU period and decoder timing. The compiler path uses SD6 and makes no
physical-duration calibration claim.

Hadamard acts transversally on the data qubits and immediately measures
the conjugate basis. It does not continue extraction in the original patch
orientation or establish fault-tolerant distance preservation. The compiler
returns terminal measurement handles; the exporter explicitly reduces the
public logical support to an observable. It never relabels a detector as
logical truth. This supported CircuitBuilder path does not repair the
separate high-level LogAsm Hadamard/rotation limitations.

The entrypoint is tested at distances 3 and 5, both bases, several round
counts, noiseless logical action, SD6 noise and whole decsim runs
(`tests/frontends/test_deltakit_compiler.py`).

## Interpret the result within its scope

SD6 noise is per operation and idle location. Changing a simulator period
does not recalibrate its physical noise probability. The finite protection
example keeps its fixed horizon; the live example executes the physical
rounds selected by feedback. Neither a single run nor replay establishes a
logical error rate.

Partial segments report correction contributions with no independent
truth or accuracy verdict. Stream owner 100 holds the protection
experiment's complete logical result. Ownership and supported limits are
described in the
[integration report](../explanation/deltakit_integration.md).

For the existing feedback and trace contracts, see
[feedback workloads](run_a_feedback_workload.md) and
[reading a trace](read_a_trace.md).
