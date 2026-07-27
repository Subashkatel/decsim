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

result = simulate(RunSpec(
    ops=cnot_plus_two_t_circuit(),
    decoder=PerRoundDecoder(tau_us=1.0),
))
print(result["chip_done"], result["fully_done"])
```

To build your own workload, create `Operation`s (or use a frontend in
[`decsim/frontends/`](decsim/frontends/)) and pass them as `ops=`.

Planning inputs have exclusive ownership. Supply at most one of `d=`,
`code=`, or `layout=`; omitting all three selects distance 3. A custom
`planner=` supplies its own layout, scheme, and rounds policy, so those
sibling fields are omitted. Custom magic-state factories use
`make_factory(engine, cluster)` so the returned factory is bound to the
event engine the run drives. A built world's exact code, layout, scheme,
rounds policy, and planner are inspectable through `world.planning`.

## Module map

The one entry point is `simulate(RunSpec(...))`: `RunSpec` (in `run_spec.py`)
is the typed run configuration, its `build()` wires the world in a fixed
order, and `simulate` drives it until no events remain. Internally the
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
