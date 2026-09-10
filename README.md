# decsim

decsim is a discrete-event simulator for the classical control and decoding
path of a quantum error corrected computer. You describe one system, run a
workload through it, and measure the reaction time and the logical error
rate the system achieves.

The simulated path is the whole loop: QPU rounds, controller readout,
links, the syndrome buffers, window creation, decoder memory, decoder
units, the Pauli frame, and the instruction back to the QPU. Every hop
charges its configured latency and bandwidth, so a run says where the time
went and which component set the reaction time. Decoding is real: windows
of a Stim circuit are decoded by PyMatching, BP-OSD, belief matching, union
find, Relay-BP or Tesseract. A timing-only run skips the data path and
charges modeled latencies instead.

## Install

Python 3.9 or newer, and the `run` extra: the root imports Stim, and a
run on real syndrome data needs PyMatching, numpy, scipy and matplotlib
as well.

```bash
python -m pip install -e ".[run]"
```

The `bb-decoders` extra adds the three optional backends (Relay-BP,
Tesseract, BP-OSD through quits); each one is one row of a table and
nothing else needs it.

## Run one shot

A run is one `Machine`, built from one `MachineSettings`, run once. The
settings come from a yaml file, which is decsim's config script, or from
Python directly.

```bash
decsim run configs/reference.yaml --seed 0 --trace
```

The same thing from Python is a build and a run:

```python
from decsim.front.experiment import load_experiment
from decsim.machine import Machine

experiment = load_experiment("configs/reference.yaml")
settings = experiment.point_settings(
    physical_error_probability=0.001, distance=3, round_period_us=1.0
)
result = Machine.build(settings, seed=0).run()
```

`result` is the immutable outcome: how the run ended and when, one logical
result per operation, the link traffic per path and, when the observation
section asked for it, the data movement. The `Machine` keeps its components
(`qpu`, `controller`, `window_manager`, `decoder_manager`, `pauli_frame`) for
inspection after the run.

A yaml describes a sweep, so one point of it names one machine.
`configs/reference.yaml` documents every key the yaml layer reads, section
by section, and `decsim show configs/reference.yaml` prints what a yaml
resolves to without running anything.

To build a machine with no yaml at all, set only the fields you study;
every settings record has a default. Each package's settings module
holds one table per pluggable part it owns, and the `kind` in a settings
record is a row of it.

## Run one sweep

```bash
decsim collect configs/weak_ler.yaml --processes 8
decsim collect configs/weak_ler.yaml --shard 0/4
decsim combine results/<first> results/<second>
decsim plot results/<run> --figure timeline
decsim plot results/<run> --figure ler_vs_d --probability 0.001
decsim trace follow results/<run>/trace/<shot>.trace.json --round 1:1
```

`configs/weak_ler.yaml` is the real thing: 35 points and 10.4 million
shots, sized for a Slurm array, and its own header says how to launch
one. For a smoke run take `configs/reference.yaml`, whose sweep is two
shots at one point.

`collect` runs every point of the sweep and writes a run folder.
`--processes` gives each worker one task, `--shard i/n` gives one Slurm
array task its share of the work units, and `--shots-per-unit` sets how
many shots a unit is: a smaller unit trades the per-task window-model
cache for shards that fit a time limit. `combine` folds the shards of one
sweep into one folder's rows. `plot` draws one figure from a run
folder; `ler_vs_d` reads one probability out of the sweep, so it asks
for `--probability`.

## Where the output lands

Under `results/`, one folder per collect, which is gem5's `m5out/`: it is
output and is not tracked. A run folder holds the facts that add up, and
every summary is derived from them when it is read.

| File | One row per |
| --- | --- |
| `shots.csv` | shot: its scalars, and each latency point's mean and max over that shot's windows |
| `shot_links.csv` | shot per link: that shot's own ledger counters |
| `shot_data_movement.csv` | shot per path: that shot's own copy and move counters and the memory class the path crosses, written only when `observation.data_movement` is on |
| `window_samples.csv` | point, latency point and distinct microsecond value: how many windows carried it, which is where a median and a p99 come from |
| `latency_samples.csv` | window of a timing run: the sample a wall-clock decoder measured |
| `sweep.csv` | sweep point, summarized from the rows above |
| `links.csv` | sweep point per link, averaged over the point's shots |
| `data_movement.csv` | sweep point per path, then per memory class, averaged over the point's shots |
| `manifest.json` | the run: the resolved config, the git state, the library versions, how it was invoked |
| `trace/<shot>.trace.json` | traced shot: its Chrome trace |
| `*.png` | figure `decsim plot` drew (`timeline`, `stage_breakdown`, `latency`, `ler_vs_d`) |

## Where to look next

- `docs/architecture.md`: the components in pipeline order, and the port
  each hands the next through.
- `docs/plug_in_a_component.md`: how to add your own decoder, link, store
  or window scheme.
- `docs/glossary.md`: what the papers call the names this code uses.
- `docs/reading_a_trace.md`: how to open a trace and what is in it.
- `STYLE.md`: the rules every line of this package is written to.

## Run the tests

```bash
python -m pytest tests
```
