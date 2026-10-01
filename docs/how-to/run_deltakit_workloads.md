[decsim docs](../README.md) › [How-to guides](README.md)

# Run Deltakit memory and protection workloads

Deltakit supplies a circuit and its measurement schedule. The existing
decsim machine runs them through readout, stores, windows, decoding and
feedback. From a yaml, the makers `decsim.producers:deltakit_memory`
and `decsim.producers:deltakit_live_memory` run the finite and the live
memory ([plug in a workload maker](plug_in_a_workload_maker.md)). The
finite example `tools/deltakit_example.py` adds the repetition family
and the protection mode on one finite history;
`tools/live_memory_example.py` keeps memory live until decoded feedback
permits final readout and saves the executed history.

## Install in a separate environment

From the checkout root, with Python 3.10 or newer:

```bash
python -m venv .venv-deltakit
.venv-deltakit/bin/python -m pip install -e '.[run,deltakit]'
```

Keep this environment separate from another checkout's editable
installation.
The Deltakit extra pins component versions in `pyproject.toml`; it does
not require the umbrella SDK or a cloud account. The pinned SDK supports
Python >=3.10,<3.15, the same floor as decsim.

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

The output folder is a run folder. Its `inputs/<id>/` holds the circuit
and its measurement map, beside the argument values, the result, the
command arrival/start events and the Chrome trace, which `decsim run`
writes in the same places. Inspect `result.json`, `commands.json` and
`trace/seed<seed>.trace.json` together. The separate
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

Hand these outputs to decsim as a workload's physical circuit
(`decsim.records.workload.FiniteCircuit`), as `memory_workload` and
`supplied_settings` in the runnable example do. The map assigns every
absolute measurement index to a one-based round. Final data measurements
join the last packet. Preserve this map with the circuit.

## Keep memory live until feedback releases it

```bash
PYTHONPATH=. .venv-deltakit/bin/python tools/live_memory_example.py \
  --distance 3 --basis Z --physical-error-probability 0.001 \
  --round-period-microseconds 1.25 --feedback-microseconds 4 \
  --output results/live-memory

PYTHONPATH=. .venv-deltakit/bin/python tools/live_memory_example.py \
  --input results/live-memory/inputs/*/fragments --feedback-microseconds 8 \
  --output results/live-memory-slower
```

The rerun runs the point the first folder recorded: its distance,
probability and period come from that point's `resolved/<id>.json`, and
its basis, noise model and T1/T2 from the first run's `arguments.json`,
unknown when neither names them. A distance or a period on the command
line replaces the recorded one; a basis, probability, noise model or
T1/T2 is refused, since the loaded circuit already holds it.

The default noise model is SD6. The prefix requests decoding, protection
continues on the same live stream, and decoded release permits resume
and actual final readout. The prefix's last round ends a window, so its
result is its own windows; left unset, `--prefix-rounds` is one window,
the distance's rounds. There is no fixed round horizon. Longer feedback
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

Live output is a run folder whose `inputs/<id>/fragments` holds the four
reusable ordinary Stim fragments and `physical.json`, the form the
`files` workload row reads. It also saves the actual `executed.stim`,
`measurement_rounds.json`, `measurements.json`, result, commands, trace and
arguments. The actual circuit and raw measurements can be checked with
Stim's measurement-to-detector converter after the runtime-selected stop.

Live `--input` loads a fragments folder in that form without selecting
Deltakit. The fragments carry the physics, and `physical.json` their
cadence; a different `--round-period-microseconds` is refused. Prefix length, feedback latency, decoder service
time, seed and patch remain runtime choices. Use `--prefix-rounds`,
`--decoder-microseconds`, `--seed` and `--patch` to set them explicitly.
Replay executes the fragments again; `measurements.json` separately records
the completed run's raw measurements for independent conversion.

This path supports rotated-surface memory and trailing-buffer feedback.
It does not implement arbitrary result-dependent quantum gates. Canonical
replay of the physical-noise example works without the SDK installed.

## Declare separate readout transport channels

This is a Python configuration surface shared by all circuit providers.
Each emission leaves the QPU as one readout of its operation's whole patch
footprint.

For finite `StimDevice` or `RecordedStimDevice`, use distinct emitting
operation ids for a terminal syndrome prefix and its
separate data finalizer. Their declared fragment slots cover the combined round.
The finite source currently infers separate terminal data only for its supported
generated-circuit layout. With an explicit `measurement_rounds` map, keep final
syndrome and data in one round emission; a separately declared terminal-data
boundary is not yet represented by that map.

Set `FabricSettings.readout_routes` to a tuple of
`links.settings.ReadoutRoute(patch_ids, path_settings)`. A route matches the
complete footprint, independent of patch ordering. Choose different channel
names for independent queues or one name for a shared serializer. Use the
existing `ChannelSettings` and `PathSettings` for bandwidth, propagation and
setup. The default readout path handles any unmatched footprint.

The runnable public-interface tests cover a recorded joint stream and separate
terminal emitters at both detector-formation sites:

```bash
PYTHONPATH=. .venv-deltakit/bin/python -m pytest tests/machine/test_machine.py \
  -k 'recorded_joint_stream or separate_terminal_emitters' -q
```

This models channel transport after round completion. It does not establish
intra-round measurement timing or hardware calibration. DECSIM still owns
protection rounds, decoder scheduling and feedback release.

## Supply a joint group or a CSS code block

`RepeatedStimCircuit` describes one shared physical history. Assign the complete
resource footprint to the stream owner's `Operation.patches`, and include that
footprint at both protected-region endpoints. All members use the same declared
round cadence. `ProtectedRegion` refers to the owner stream; it has no separate
patch declaration. The source executes each requested round once for the group.

The optional provider can produce a CSS code block:

```python
import deltakit_explorer.codes as codes

import decsim.frontends.deltakit as deltakit
import decsim.qpu.streaming_stim_device as streaming_stim_device

code = codes.BivariateBicycleCode(3, 5, [1, 1, 4], [0, 1, 2])
block = deltakit.css_memory_rounds(
    code, "Z", 0.001, round_period_microseconds=1.1
)
block_source = streaming_stim_device.StreamingStimDevice({100: block})
```

The BB example is one [[30,8,2]] block with eight logical outputs. Use one
physical resource patch and eight logical qubit identities: the owner's
`qubits` are `range(8)` and its `patches` the one block. Its code card is
`code_geometry.BivariateBicycleCodeModel(settings=..., distance=2)`, its
`Settings(qubit_count=30, logical_qubit_count=8)`, with the window
overrides the run wants, and its decoder a full-model one such as BP-OSD: the noisy model
contains hyperedges that graphlike matching cannot represent exactly.
`_bb_settings` in `tests/machine/test_machine.py` is the runnable
assembly of those pieces. Patch count never sizes these logical outputs.
The distance describes this small code, not the fault distance of its
syndrome schedule.

The end-to-end examples are reproducible through the public Machine tests:

```bash
PYTHONPATH=. .venv-deltakit/bin/python -m pytest tests/machine/test_machine.py \
  -k 'bb_block or bb_live or bb_higher or recorded_joint_stream' -q
```

They exercise live feedback, controller-side and decoder-side formation, full
logical vectors, deliberate higher-index failure, and a recorded joint source. Direct Stim uses the same source contract.
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
A segment's first round starts a window and its last round ends one,
so a scored segment never shares a window with its continuation: a
`--prefix-rounds` of 4 at distance 3 commits rounds 1 to 3, then round 4
alone. Left unset it is one window, `distance` rounds.
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
