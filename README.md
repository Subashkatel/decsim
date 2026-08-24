# decsim

decsim is a discrete-event simulator for the classical control and decoding
path of a quantum error corrected computer. You describe one system
configuration, run a workload through it, and measure reaction time and
logical error rate.

The simulated path is the full loop: QPU rounds, controller readout, links,
syndrome buffer, window creation, decoder memory, decoder units, and the
Pauli frame. Every hop charges its configured latency and bandwidth, so you
can ask where time goes and which component limits the reaction time.
Decoding is real: windows of a Stim circuit are decoded by PyMatching,
BP-OSD, belief matching, union find, Relay-BP, or Tesseract. Timing-only
runs skip the data path and charge modeled latencies instead.

## Requirements

- Python 3.9 or newer. The core package imports no third-party libraries.
- Runs on real syndrome data need the `experiments` extra:

```bash
python -m pip install -e ".[experiments]"
```

## Quickstart

A run is one `RunSpec` handed to `simulate`. Every field has a default;
you set only what you study. This timing-only run needs no dependencies:

```python
from decsim import RunSpec, simulate, PerRoundDecoder, cnot_plus_two_t_circuit, fmt

completed = simulate(RunSpec(
    ops=cnot_plus_two_t_circuit(),
    decoder=PerRoundDecoder(tau_us=1.0)))
print("workload done at", fmt(completed.result.fully_done_ticks))
```

```
workload done at  79.850 us
```

To decode real data, give the operation a Stim circuit and pick a device
and a decoder:

```python
import stim

from decsim import PresetLatencyDecoder, RunSpec, simulate
from decsim.decoders.mwpm.decoder import PyMatchingDecoder
from decsim.message import Operation
from decsim.qpu.round_policies import FixedRounds
from decsim.qpu.stim_device import StimDevice

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=3, rounds=9,
    after_clifford_depolarization=0.003,
    before_round_data_depolarization=0.003,
    before_measure_flip_probability=0.003,
    after_reset_flip_probability=0.003)
operation = Operation(id=1, name="memory", qubits=(0,), patches=(0,),
                      circuit=circuit)

completed = simulate(RunSpec(
    ops=[operation], d=3, rounds_policy=FixedRounds(9),
    device=StimDevice(), decoder=PyMatchingDecoder(PresetLatencyDecoder(1.0)),
    seed=0))

outcome = completed.result.operation_results[0]
print("prediction:", outcome.logical_observables)
print("truth:     ", outcome.observable_truth)
print("failure:   ", outcome.logical_failure)
```

The device samples the circuit, streams raw measurements round by round,
and the run decodes them in sliding windows. `outcome.logical_failure`
compares the prediction against the sampled truth; count failures over
seeds to estimate a logical error rate. To charge each decode's measured
wall clock instead of a fixed latency, wrap `PyMatchingDecoder()` in a
`DecoderEngine`.

`completed.result` is the immutable outcome (timing, per-operation logical
results, link traffic, metric values). `completed` also carries the runtime
owners (`window_manager`, `decoder_manager`, `controller`, `qpu`, ...) for
inspection after the run.

## Configure a run

Each `RunSpec` field selects one component. A field left `None` takes the
default. `resolve_run_configuration` in `decsim/run_configuration.py`
applies every default and validates the combination in one place.

| To change | Pass | Options in |
| --- | --- | --- |
| Code and distance | `d=` or `code=` or `layout=` (at most one; default surface code, d=3) | `decsim/qpu/code_geometry.py` |
| Windowing scheme | `scheme=` (default sliding) | `decsim/windows/windowing_schemes.py` |
| Decoder | `decoder=`, per-code `decoders=`, or `router=` | `decsim/decoders/` |
| Decoder count and memory | `num_units=`, `decoder_memory=` | `decsim/decoders/decoder_memory.py` |
| Rounds per operation | `rounds_policy=` (default gate rounds) | `decsim/qpu/round_policies.py` |
| Link latency and bandwidth | `links=` (default `logical_reference_profile()`) | `decsim/links/link_profiles.py` |
| Round period and controller costs | `timing=TimingConfig(...)` | `decsim/config.py` |
| Syndrome source | `device=` (timing-only, syndrome bits, or Stim sampling) | `decsim/qpu/` |
| Pauli frame commit cost | `pauli_frame=PauliFrameConfig(...)` | `decsim/pauli_frame/pauli_frame.py` |
| Reproducibility | `seed=` (one root seed drives every component) | `decsim/seeding.py` |

## Package layout

- `run_spec.py`, `run_configuration.py`: composition root; `simulate(RunSpec(...))` is the one entry point.
- `engine.py`: the discrete-event core (integer ticks, 1 tick = 1e-6 us).
- `message.py`, `protocols.py`: the typed objects that flow between components, and the Protocol seams a custom component implements.
- `qpu/`: codes and layouts, round policies, cycle clock, Stim devices, magic-state factories.
- `controller/`: readout handling, detector formation at ingress, feedback streams.
- `links/`: link cards with latency, bandwidth, and traffic accounting per path.
- `syndrome_buffer/`: upstream round retention with explicit backpressure.
- `windows/`: window manager and the windowing schemes (sliding, parallel, sandwich, naive).
- `detector_error_model/`: slices the whole-circuit detector error model into per-window models.
- `decoders/`: manager, engine stages (fetch, algorithm, release), and the six backends.
- `pauli_frame/`, `observe/`: frame commit, conditional release, and run metrics.
- `frontends/`: workload builders that produce `Operation` lists.

To add your own decoder, scheme, or policy, implement the matching Protocol
in `decsim/protocols.py` and pass the instance to `RunSpec`.

## Run the tests

```bash
python -m pytest tests
```
