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
    completed_run.result.chip_done_ticks,
    completed_run.result.fully_done_ticks,
)
```

To build your own workload, create `Operation`s (or use a frontend in
[`decsim/frontends/`](decsim/frontends/)) and pass them as `ops=`.

Planning inputs have exclusive ownership. Supply at most one of `d=`,
`code=`, or `layout=`; omitting all three selects distance 3. Select window
layout and operation-round behavior directly with `scheme=` and
`rounds_policy=`. Custom magic-state factories use
`make_factory(engine, cluster)` so the returned factory is bound to the
event engine the run drives. A `CompletedRun` exposes the exact selections as
`completed_run.planning.code`, `completed_run.planning.layout`,
`completed_run.planning.scheme`, and
`completed_run.planning.rounds_policy`.

Custom stochastic runtime parts participate in run-level seeding through
`RunSeedConsumer` in
[`decsim/protocols.py`](decsim/protocols.py). Reservation prepares all work
that can fail without changing the active random state. Once reservation
succeeds, commit is a total, failure-free installation of that prepared state:
it must not allocate, draw randomness, or invoke callbacks.

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
  (runtime window layout for unknown-length streams) and `payload_store.py`
  (syndrome retention) own adjacent runtime state.
- `window_interactions.py` — replaceable boundary, replay-scope, strong-region,
  seam, and ownership decisions.
- `chip.py` — the reaction gate; `devices.py` — the syndrome round clock.
- `planner.py` — compile-time window layout and rounds policies.
- Pluggable parts: `decoders.py`, `schedulers.py`, `schemes.py`,
  `policies.py`, `switching.py`, `factories.py`, `controllers.py`.

Real-decoder adapters (needing `stim`/`pymatching`) live in
`decsim/adapters/`, `decsim/mwpm_decoder/`, `decsim/bposd_decoder/`, and
`decsim/belief_matching_decoder/`; `detector_error_model.py` slices the
whole-circuit DEM into per-window models for them.

## Running the tests

```bash
python -m pytest tests/
```
