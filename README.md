# decsim

A discrete-event simulator for quantum error correction.

## Requirements

- Python 3.9 or newer. The core has **no required dependencies**.
- Optional, only for the real-decoder adapters in `decsim/adapters/`:
  `stim` and `pymatching` (`pip install stim pymatching`).

## Quickstart

Compose a `RunSpec` and hand it to `simulate` — both live in
[`decsim/run_spec.py`](decsim/run_spec.py):

```python
from decsim.run_spec import RunSpec, simulate
from decsim.decoders import PerRoundDecoder
from decsim.frontends.circuit import cnot_plus_two_t_circuit

completed_run = simulate(RunSpec(
    ops=cnot_plus_two_t_circuit(),
    decoder=PerRoundDecoder(tau_us=1.0),
))
print(
    completed_run.result.execution_done_ticks,
    completed_run.result.fully_done_ticks,
)
```

To build your own workload, create `Operation`s (or use a frontend in
[`decsim/frontends/`](decsim/frontends/)) and pass them as `ops=`.

Planning inputs have exclusive ownership. Supply at most one of `d=`,
`code=`, or `layout=`; omitting all three selects distance 3. Select window
layout and operation-round behavior directly with `scheme=` and
`rounds_policy=`. Custom magic-state factories use
`make_factory(engine, decoder_manager)` so correction jobs share the run's
decoder service. A `CompletedRun` exposes the scientific result and the
runtime owners used by experiments: `window_manager`, `decoder_manager`,
`execution_runtime`, `controller`, `qpu`, `orchestrator`, `factory`,
`syndrome_buffer`, and `syndrome_ingress`.

Custom stochastic runtime parts participate in run-level seeding through
`RunSeedConsumer` in
[`decsim/protocols.py`](decsim/protocols.py). Reservation prepares all work
that can fail without changing the active random state. Once reservation
succeeds, commit is a total, failure-free installation of that prepared state:
it must not allocate, draw randomness, or invoke callbacks. Providers and
planning hooks run before binding and therefore must not draw randomness;
stochastic state belongs in the returned runtime component. A numeric seed
replays Stim sampling only for the same Stim version, machine SIMD width, and
sampler call shape; `seed=None` is intentionally non-reproducible.

## Module map

The one entry point is `simulate(RunSpec(...))`: `RunSpec` (in `run_spec.py`)
is the typed run configuration, and its atomic `build()` wires and drives one
run until no events remain before returning `CompletedRun`. Internally the
simulator is a stable core — the runtime mechanics — plus swappable **parts**
that `RunSpec` wires in. Parts make decisions through narrow protocols while
the core retains lifecycle and event-ordering guarantees. See
[`EXTENDING_DECSIM.md`](EXTENDING_DECSIM.md) for the supported contracts,
validation rules, and a custom window-interaction example.

- `run_spec.py` — the composition root and driver (`RunSpec`, `build()`, `simulate`).
- `message.py` — the typed objects that flow between components.
- `protocols.py` — the numbered port catalog: the seams parts plug into.
- `window_manager.py` — the windowing runtime hub; `dynamic_windows.py`
  owns runtime window layout for unknown-length streams, while
  `syndrome_buffer.py` owns upstream syndrome retention.
- `window_interactions.py` — replaceable boundary, replay-scope, strong-region,
  seam, and ownership decisions.
- `execution_runtime.py` — operation-DAG admission, resource claims, and execution timestamps.
- `controller.py` — command/feedback sequencing, QPU readout conversion, and its optional fixed cost.
- `qpu.py` — physical round cadence and typed QPU readout production.
- `syndrome_ingress.py` — controller-side QC receipt, fragment reassembly, and route arbitration.
- `decoder_input_transfer.py` / `decoder_input_store.py` — decoder-input transport delay, and decoder-side storage admission, round budgets, and stored-input lifetime.
- `planner.py` — compile-time window layout and rounds policies.
- Directly replaceable parts include decoders, schedulers, schemes, policies,
  switching strategies, factories, syndrome ingress, and decoder-input transfer.

A plain-language guide to the Phase A core modules, their state ownership,
and where to make common changes is available at
`tmp/validation/core_evidence/MODULE_GUIDE.md`. Research grounding is
kept out of Python comments and docstrings; the matching evidence map is
`tmp/validation/core_evidence/README.md`.

The decoder resource model is deliberately small: each configured pool has
identical non-preemptive service units and one FIFO ready queue. It models
queueing and service occupancy, not a particular CPU, GPU, FPGA, or ASIC
microarchitecture. Published EDF, elastic-decoder, or Triage policies require
real task deadlines, service estimates, and dependency/conflict metadata and
are not approximated by synthetic priority scores.

Finite retained syndrome storage uses explicit downstream backpressure and does
not silently overwrite packets. The transient ingress reassembly buffer is a
separate configurable concern: capacity is unbounded by default; when a user
selects finite capacity, the default overflow policy fails closed rather than
pretending to model an unverified hardware policy. Ready route heads use a
simple rotating order between the two route kinds; this is a replaceable
least-claim arbitration choice, not a hardware scheduling claim.

The default decoder-I/O composition is the placement-neutral
``logical_reference`` baseline. ``SyndromeBuffer`` owns one upstream allocation
per live round as it moves from assembly through immutable retained readiness;
that state transition does not fabricate a staging-to-retention byte copy.
``WindowManager`` submits only ready window requirements. One CWD transfer per
weak window carries the request, and the manager-owned
``DecoderInputStoreStager`` materializes a distinct immutable ``DecoderInput``
in a ``DecoderInputStore`` on arrival, before the job enters
``DecoderManager``'s FIFO ready queue. The upstream hold is released when
storage admits the request, and overlapping windows retain shared rounds until
their last required transfer completes. Decoder-side storage is optionally
finite in syndrome rounds per decoder unit pool
(``RunSpec.decoder_input_store``); unset, one shared unbounded store keeps the
path unchanged.
Strong input uses CSD once. The generic seam makes no claim about cryogenic or
room-temperature placement, DMA, MMIO, rings, pointers, or streaming; these are
named replaceable research profiles. SB0/SB1 and ``PayloadStore`` are removed
because they were decsim ledgers, not literature entities. See
``docs/architecture/evidence_catalog.md`` and ``syndrome_data_path.md``.

Real-decoder adapters (needing `stim`/`pymatching`) live in
`decsim/adapters/`, `decsim/mwpm_decoder/`, `decsim/bposd_decoder/`, and
`decsim/belief_matching_decoder/`; `detector_error_model.py` slices the
whole-circuit DEM into per-window models for them.

## Running the tests

```bash
python -m pytest tests/
```

## Experiments

The experiment harness runs from a repository checkout and stays outside the
installed simulator package. Install its optional dependencies, then invoke a
runner with an explicit configuration and output directory:

```bash
python -m pip install -e ".[experiments]"
python -m experiments.offline.run_surface --config run.json --output experiment-results
```
