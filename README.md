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
- Runs on real syndrome data need the `run` extra:

```bash
python -m pip install -e ".[run]"
```

## Quickstart

A run is one `Machine` built from a `MachineSettings` and run once. The
settings come from a yaml file, which is decsim's config script, or
from Python directly.

### From a yaml

```python
from decsim.front.experiment import load_experiment
from decsim.machine import Machine

experiment = load_experiment("configs/reference.yaml")
settings = experiment.point_settings(
    physical_error_probability=0.001, distance=3, round_period_us=1.0)
result = Machine.build(settings, seed=0).run()
```

A yaml names a sweep, so a point of it names one machine;
`configs/reference.yaml` documents every key the yaml layer reads.

### From the command line

One command, one subcommand per word, as sinter's is:

```bash
decsim run configs/reference.yaml --seed 0 --trace   # one seeded shot
decsim collect configs/weak_ler.yaml --processes 8   # the whole sweep
decsim collect configs/weak_ler.yaml --shard 0/4     # one Slurm array task
decsim collect configs/weak_ler.yaml --shots-per-unit 1000   # finer work units
decsim combine results/<a> results/<b>               # the shards' rows
decsim show configs/reference.yaml                   # what the yaml resolves to
decsim plot results/<run> --figure timeline          # a figure from its files
decsim trace follow results/<run>/trace/<shot>.trace.json --round 1:1
```

`python -m decsim <verb>` is the same command. `collect` writes a run
folder under `results/`, which is output and is not tracked: one row per
shot in `shots.csv`, one per point in `sweep.csv`, one per link in
`links.csv`, the config it ran, and the figures. `shot_links.csv` holds
one row per shot per link, that shot's own ledger counters, and
`links.csv` is their mean. `window_samples.csv` holds one row per
distinct microsecond value of each latency point with how many windows
carried it, which is where a point's median and p99 come from. The
timeline figure is one of them whenever the run recorded a trace, since
it is drawn from that file and not from the machine:
`configs/reference.yaml` records one for shot 0, so its `collect` writes
`timeline.png`.

`--shard i/n` gives one array task its share of the work units and
`--shots-per-unit` sets how many shots a unit is: a smaller unit trades
the per-task model cache, since each unit builds the window models
itself, for shards small enough to fit a time limit.

### From Python alone

Every settings field has a default; you set only what you study. This
timing-only run needs no dependencies:

```python
from decsim.config import format_ticks
from decsim.decoders.decoders import PerRoundDecoder
from decsim.decoders.settings import DecoderSettings
from decsim.frontends.circuit_frontend import cnot_plus_two_t_circuit
from decsim.frontends.settings import WorkloadSettings
from decsim.machine import Machine, MachineSettings

settings = MachineSettings(
    workload=WorkloadSettings(operations=cnot_plus_two_t_circuit()),
    weak_decoder=DecoderSettings(decoder=PerRoundDecoder(tau_us=1.0)))
machine = Machine.build(settings)
result = machine.run()
print("workload done at", format_ticks(result.fully_done_ticks))
```

```
workload done at  79.850 us
```

To decode real data, give the operation a Stim circuit and pick a device
and a decoder:

```python
import stim

from decsim.decoders.decoders import PresetLatencyDecoder
from decsim.decoders.minimum_weight_perfect_matching.decoder import PyMatchingDecoder
from decsim.decoders.settings import DecoderSettings
from decsim.frontends.settings import WorkloadSettings
from decsim.machine import Machine, MachineSettings
from decsim.records.program import Operation
from decsim.qpu.round_policies import FixedRounds
from decsim.qpu.settings import QpuSettings
from decsim.qpu.stim_device import StimDevice

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=3, rounds=9,
    after_clifford_depolarization=0.003,
    before_round_data_depolarization=0.003,
    before_measure_flip_probability=0.003,
    after_reset_flip_probability=0.003)
operation = Operation(id=1, name="memory", qubits=(0,), patches=(0,),
                      circuit=circuit)

settings = MachineSettings(
    workload=WorkloadSettings(operations=[operation],
                              rounds_policy=FixedRounds(9)),
    qpu=QpuSettings(distance=3, device=StimDevice()),
    weak_decoder=DecoderSettings(
        decoder=PyMatchingDecoder(PresetLatencyDecoder(1.0))))
machine = Machine.build(settings, seed=0)
result = machine.run()

outcome = result.operation_results[0]
print("prediction:", outcome.logical_observables)
print("truth:     ", outcome.observable_truth)
print("failure:   ", outcome.logical_failure)
```

The device samples the circuit, streams raw measurements round by round,
and the run decodes them in sliding windows. `outcome.logical_failure`
compares the prediction against the sampled truth; count failures over
seeds to estimate a logical error rate. To charge each decode's measured
wall clock instead of a fixed latency, name the decoder by its `kind`
(`DecoderSettings(kind="pymatching")`) so the root stages it in a
`DecoderEngine`.

`result` is the immutable outcome (timing, per-operation logical results,
link traffic, metric values). `machine` carries the components
(`window_manager`, `decoder_manager`, `controller`, `qpu`, ...) for
inspection after the run.

## Configure a run

`MachineSettings` has one settings record per yaml section; each record
is owned by the package it configures, and a pluggable part is named by
its `kind`, a row of the tables at the top of `decsim/machine.py`.

| To change | Set | Options in |
| --- | --- | --- |
| Code and distance | `qpu=QpuSettings(distance=)`, or `code=` or `layout=` (at most one; default surface code, d=3) | `decsim/qpu/settings.py`, `decsim/qpu/code_geometry.py` |
| Syndrome source | `qpu=QpuSettings(kind=)` (stim_device, timing_only, syndrome_bits, recorded_stim) or `device=` | `decsim/qpu/` |
| Round period | `qpu=QpuSettings(round_period_microseconds=)` | `decsim/qpu/settings.py` |
| Controller costs | `controller=ControllerSettings(...)` | `decsim/controller/settings.py` |
| Windowing scheme | `windows=WindowSettings(kind=)` (sliding, parallel, sandwich, naive_online) | `decsim/windows/settings.py` |
| Decoder | `weak_decoder=DecoderSettings(kind=)` (pymatching, belief_matching, or a latency in microseconds) or `decoder=` | `decsim/decoders/settings.py` |
| Decoder count and memory | `DecoderSettings(units=, unit_memory_rounds=)` | `decsim/decoders/settings.py` |
| Escalation | `escalation=EscalationSettings(kind=)` (weak_baseline, strong_only, switching) | `decsim/decoders/settings.py` |
| Rounds per operation | `workload=WorkloadSettings(rounds_policy=)` (default gate rounds) | `decsim/qpu/round_policies.py` |
| Link latency and bandwidth | `links=` (default `logical_reference_profile()`) | `decsim/links/link_profiles.py` |
| Round stores | `round_store=`, `strong_round_store=RoundStoreSettings(rounds=)` | `decsim/syndrome_buffer/settings.py` |
| Pauli frame commit cost | `pauli_frame=PauliFrameConfig(...)` | `decsim/pauli_frame/pauli_frame.py` |
| Reproducibility | `Machine.build(settings, seed)` (one root seed drives every component) | `decsim/seeding.py` |

## Package layout

- `machine.py`: the root; `Machine.build(MachineSettings(...), seed)` builds every component and wires it, `Machine.run()` returns the `RunResult`.
- `engine.py`: the discrete-event core (integer ticks, 1 tick = 1e-6 us).
- `records/`, `ports.py`: the frozen values that flow between components, one module per record family, and the ports (one Protocol per handoff) a component implements.
- `qpu/`: codes and layouts, round policies, cycle clock, Stim devices, magic-state factories.
- `controller/`: readout handling, detector formation at ingress, feedback streams.
- `links/`: link cards with latency, bandwidth, and traffic accounting per path.
- `syndrome_buffer/`: upstream round retention with explicit backpressure.
- `windows/`: window manager and the windowing schemes (sliding, parallel, sandwich, naive).
- `detector_error_model/`: slices the whole-circuit detector error model into per-window models.
- `decoders/`: manager, engine stages (fetch, algorithm, release), and the six backends.
- `pauli_frame/`, `observe/`: frame commit, conditional release, and run metrics.
- `frontends/`: workload builders that produce `Operation` lists.
- `front/`: the way in. The `decsim` command set, the yaml experiment and
  its sweep, one shot to one row, the csv report, the figures, the run
  folder, and the flow view over a trace file.
- `collect.py`: tasks and the shots collected from them, sinter's shape:
  a process pool over tasks, shards, and one window model build per task.
- `configs/`: the yaml experiments, gem5's `configs/`. `results/`: their
  output, gem5's `m5out/`, not tracked.

To add your own decoder, scheme, or policy, implement the matching port
in `decsim/ports.py` and add one row to its table in `decsim/machine.py`,
or pass the instance through its settings record.

## Run the tests

```bash
python -m pytest tests
```

The rules every line of this package is written to are in `REWRITE.md`.
