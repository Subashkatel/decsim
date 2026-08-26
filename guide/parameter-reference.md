# Parameter reference

Every knob of one simulation run, in one place. A run is one `RunSpec`
(`decsim/run_spec.py`); the yaml experiment layer reads a documented subset
of it. This page is kept honest by
`experiments/tests/test_parameter_reference.py`, which fails when this
table drifts from the `RunSpec` fields or when the reference config drifts
from the loader. When you add or remove a knob, that test tells you to
update this page and `experiments/configs/reference.yaml` in the same
change.

## The yaml surface

`experiments/configs/reference.yaml` is the canonical, runnable listing of
every yaml key with its meaning and defaults; the loader is
`experiments/experiment_config.py` and is the only yaml reader. Run any
config with `python -m experiments.run experiments/configs/<name>.yaml`.
Top-level keys: `mode`, `code_task`, `distance`, `rounds_per_shot`,
`windowing`, `sweep`, `controller`, `links`, `decoder`,
`decoder_memory_rounds`, `pauli_frame`, `trace` (plus `extends`, which
starts from another config in the same folder and overrides the keys it
names). `trace: true` prints the engine narrator live and writes each
shot's full line record to `results/<name>/trace/`; the same lines are
always kept in memory on `completed.engine.log_lines`.

## Every RunSpec field

"yaml" says whether the experiment layer reaches it; Python-only knobs are
set by building a `RunSpec` directly (see `tests/` for working examples of
each).

| field | yaml | what it sets |
|---|---|---|
| `ops` | partly | the workload: `Operation` list with circuits, qubits, patches, dependencies (yaml builds one memory op) |
| `frontend` | no | alternative workload frontend that emits the operation program |
| `decode_ops` | no | decode-only operations appended to the workload |
| `dynamic_streams` | no | live streams whose operations arrive while the run executes |
| `protected_regions` | no | round regions the buffers must retain regardless of consumption |
| `code` | partly | the code model object (yaml: `SurfaceCodeModel` from `distance` + `windowing` overrides) |
| `layout` | no | multi-patch layout object |
| `d` | yes | code distance shorthand when `code` is not given (`distance`) |
| `decoder` | partly | the decoder: engine with staged timing (yaml: PyMatching + fetch/release cycles + algorithm card) |
| `decoders` | no | named per-tier decoder map |
| `router` | no | routes jobs to decoders (e.g. `SwitchingRouter` weak/strong) |
| `escalation_policy` | partly | who decodes a window and when it is final (yaml: default weak Baseline or `StrongOnly`; Python: `Switching` family) |
| `scheduler` | no | ready-queue order policy (default FIFO) |
| `lane_policy` | no | maps jobs to unit pools |
| `unit_pools` | no | named pools with unit counts (e.g. `{"default": 1, "strong": 1}`) |
| `num_units` | yes | unit count of the default pool (`decoder.units`) |
| `scheme` | yes | windowing scheme (`windowing.scheme`) |
| `rounds_policy` | partly | rounds per operation (yaml: `FixedRounds(rounds_per_shot)`) |
| `boundary_policy` | no | commit/release policy at window boundaries (e.g. speculative) |
| `window_interaction` | no | boundary targeting, merging and invalidation between windows |
| `idle_policy` | no | what idle rounds decode |
| `feedback_boundary_mode` | no | feedback stream boundary handling (`trailing_buffer`) |
| `timing` | yes | `TimingConfig`: `round_us`, `t_binary_availability_us`, `t_pack_us` (`controller:` block) |
| `round_us` | yes | round period shorthand when `timing` is not given (`round_period_us` axis) |
| `links` | yes | the link fabric card: latency, capacity, transfer overhead per path (`links:` block) |
| `device` | partly | syndrome source (yaml: `StimDevice`; Python: timing-only or syndrome-bit devices) |
| `error_model_provider` | no | per-window decoder model source when not derived from the circuit |
| `memory_model` | no | Buffer 0 memory technology model |
| `syndrome_buffering` | no | `SyndromeBufferingConfig`: Buffer 0 slots, syndrome buffer 1 slots, packing assembly slots |
| `decoder_memory` | yes | per-unit input SRAM in rounds (`decoder_memory_rounds`); a unit overlaps transfer with compute only when two windows fit |
| `pauli_frame` | yes | `PauliFrameConfig`: frame commit cost (`pauli_frame.commit_us`) |
| `syndrome_packing_policy` | no | packing overflow and queue admission policy |
| `make_syndrome_packing` | no | factory hook replacing the packing stage |
| `make_decoder_memory_transfer` | no | factory hook replacing the input transport |
| `make_factory` | no | magic-state factory hook |
| `make_metrics` | no | metrics observer hook |
| `record_switching_windows` | no | capture per-request/service records for switching studies |
| `make_conditional_release` | no | conditional-release hook (release decisions from final results) |
| `seed` | yes | root seed; sweeps use seeds `0..shots-1` per point |

Not configuration: `ops`-derived plans, and anything the run measures.
Removed by owner ruling 2026-08-26 (no wrong versions in core):
`boundary_application` and `input_staging_depth`; decoder-side boundary
application and per-unit two-slot input staging are the machine.
