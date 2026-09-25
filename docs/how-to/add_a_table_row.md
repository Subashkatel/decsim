[decsim docs](../README.md) › [How-to guides](README.md)

# How to add a row to a table

A pluggable part of decsim is a **table**: a dictionary whose keys are
the names a yaml file may write and whose values are the classes the
machine builds. There are eighteen of them, listed with every row in
[The plug-in tables](../reference/tables.md). This is the recipe for adding a row to any
of them.

Adding a row touches your class, the table, `configs/reference.yaml`,
and the generated page [The plug-in tables](../reference/tables.md),
which `python tools/docs_map.py` rewrites from the tables. If it takes
more, something is wrong with the port rather than with your class.

A row with yaml keys of its own declares them on a nested frozen
dataclass named `Settings`, which fills the `RowSettings` port in
`decsim/ports.py`: its fields are the keys, its classmethod
`from_yaml(section)` reads and checks the ones the yaml wrote, and your
constructor takes the record as `settings`. The section keeps the keys
every row of its table shares and hands your row the rest
(`decsim/tables.py`, `row_settings`); a key that neither declares is
refused by name when the yaml loads. A row with no keys of its own
declares no `Settings`. A decoder row's `from_yaml` is also handed the
run's clocks, since a decoder's own timing names a clock domain, and its
tier's section name (`weak_decoder` or `strong_decoder`), which a
refusal leads with, since both tiers take the same rows:

```python
class MyDecoder(decoder_module.WindowDecoderBase):
    @dataclasses.dataclass(frozen=True)
    class Settings:
        step_count: int = 1

        @classmethod
        def from_yaml(cls, section, clocks, section_name):
            step_count = section.get("step_count", 1)
            if step_count < 1:
                raise ValueError(f"{section_name}.step_count is at least 1")
            return cls(step_count=step_count)

    def __init__(self, settings=None):
        decoder_module.WindowDecoderBase.__init__(self)
        ...
```

```yaml
weak_decoder:
  kind: my_decoder
  step_count: 4          # MyDecoder's own key, beside the tier's keys
```

## 1. Find the port and fill it

Open `decsim/ports.py` and find the port your part fills.
[The ports](../reference/ports.md) is the same file as a page, with every method
and what it does. The ports are in pipeline order, so the part you want
is near the component that uses it.

Write a class with those methods. The ports are `typing.Protocol`
classes and they are structural: a class fits a port by having the right
methods, so you do not inherit anything, and there
is no registration step for the type system. The records each method
takes and returns live in `decsim/records/`.

Some ports declare **members** as well as methods: facts a caller needs
about your row that are answered by the row rather than read off its
class. `Decoder.fault_model_requirement` says which of the four fault
model contracts your decoder wants
(`decsim/detector_error_model/fault_model_contracts.py`), and
`Decoder.stage_recorded` says which trace source it fires, `SILENT` for
a decoder with no internal stages. The members are listed per port in
[The ports](../reference/ports.md).

The port says what your class answers; the build says what its
constructor is handed, and that is not on the port. For the tables a
study most often extends:

| Table | The root builds your row as | Where |
| --- | --- | --- |
| `DECODERS` | `row()`, or `row(settings=...)` for a row with a `Settings` | `decsim/build/decoders.py`, `_algorithm` |
| `WINDOWING_SCHEMES` | `row(card)`, a `WindowingSchemeCard`, or `row(card, settings=...)` for a row with a `Settings` | `decsim/build/plan.py`, `_chosen_scheme` |
| `SYNDROME_SOURCES` | `row()`, with `code=card` when `takes_code_card` and `settings=...` for a row with a `Settings` | `decsim/build/plan.py`, `_syndrome_source` |
| `CODE_CARDS` (the `CodeModel` port) | `row(commit_rounds_override=..., buffer_rounds_override=...)`, the windows section's sizes, with `distance=` when the sweep sets one and `settings=...` for a row with a `Settings` | `decsim/qpu/settings.py`, `QpuSettings._named_card` |
| `WORKLOADS` | not built (the `WorkloadRow` port): the workload section calls `row.workload(settings.row_settings, sweep_values)` once per sweep point for the records.workload `Workload` it lowers | `decsim/frontends/settings.py`, `WorkloadSettings.at_point` |
| `SYNDROME_BUFFERS` | `row(settings)`, the section's record, whose `row_settings` holds the row's own `Settings` | `decsim/build/stores.py` |
| `IDLE_POLICIES` | `row()`, or `row(settings=...)` for a row with a `Settings` | `decsim/build/plan.py`, `_idle_policy` |
| `BOUNDARY_POLICIES`, `BOUNDARY_PAYLOADS` | `row()` | `decsim/build/plan.py` |
| `LINK_FABRICS` | not built: the yaml load calls `row.base_card()` for the numbers the section's per-path cards override, and the root calls `row.build(card, engine)` for the `Link` the run sends on, which also carries `trace.transfer_delivered` for the traffic ledger; a row that keeps the fabric and changes how a wire times its bits hands `LinkFabric` its own `Channel` class instead | `decsim/links/link_profiles.py`, `from_yaml`; `decsim/build/stores.py`, `build_links` |

A syndrome source also says where the run's window models come from,
through its `window_model_source` method, unless Python names another
provider (`QpuSettings.error_model_provider`). A source with a circuit
returns itself and answers `WindowModelSource` too, as `StimDevice`
does; a source with no circuit returns
`syndrome_devices.NO_WINDOW_MODELS`, which answers every model question
with nothing, as `TimingOnlyDevice` does.

A workload row's operations that share a qubit need program-order
edges (`Operation.predecessors`) or the run refuses them; a frontend
fills them (`decsim/frontends/circuit_frontend.py`, `_wire_circuit`).

## 2. Add the row

The table lives beside the classes it lists, in the settings module of
the package that owns the part. Add one entry:

```python
DECODERS = {
    ...
    "my_decoder": my_module.MyDecoder,
}
```

[The plug-in tables](../reference/tables.md) says which file each table is in and which
yaml key names it.

## 3. Name it in the yaml, and in the reference

In your config, write the row's key in the section that owns your part:

```yaml
weak_decoder:
  kind: my_decoder
```

Then add it to `configs/reference.yaml` in the same commit, beside the
rows that section's comment lists: that file is the documentation of
the yaml surface, and no test notices a row it leaves out. Last, run
`python tools/docs_map.py` from the checkout to rewrite the tables page;
`tests/test_docs.py` fails until the page matches the tables.

## What a typo gets

A `kind` that is not a row is refused when the config is loaded, by
name, with the rows printed:

```
weak_decoder.kind 'my_decodr' is not a row of its table; the rows are
['belief_matching', 'bposd', 'dispatch_steps', 'measured_table',
'pymatching', 'relay_bp', 'tesseract', 'union_find',
'unweighted_pymatching']
```

One function does that for every table (`decsim/tables.py`, `row`), so
the refusal reads the same wherever it comes from. It is pinned by
`tests/machine/test_machine.py::test_a_decoder_kind_off_the_table_is_refused_naming_the_rows`.

A key that neither the section nor your row declares is refused the
same way, with the keys the section reads and your row's own listed:

```
weak_syndrome_buffer does not know ['banks']; its keys are ['kind',
'bits', 'clock', 'bank_count']
```

A yaml section that no package owns is refused the same way, with the
sections listed:
`tests/machine/test_machine.py::test_a_yaml_section_nobody_owns_is_refused_naming_the_sections`.

## The worked examples

Each of these is a test that plugs a class in from outside decsim and
runs it. Read the one closest to what you are writing; it is shorter
than any description of it.

| You are writing | Read |
| --- | --- |
| a decoder | `tests/machine/test_machine.py::test_a_new_decoder_is_one_class_and_one_table_row` |
| a decoder, through a whole run of a shipped config | `tests/machine/test_machine.py::test_a_second_table_row_runs_gate_point_one` |
| a syndrome buffer | `tests/machine/test_machine.py::test_a_new_syndrome_buffer_is_one_class_and_one_table_row` |
| a syndrome buffer with a key of its own | `tests/syndrome_buffer/test_settings.py::test_a_buffer_rows_own_key_reaches_its_settings` |
| a syndrome source with a key of its own | `tests/experiments/test_yaml_surface.py::test_a_source_rows_own_key_reaches_the_built_source` |
| a workload maker | `tests/frontends/test_settings.py::test_a_maker_written_outside_decsim_runs_from_a_yaml` |
| a code card | `tests/machine/test_machine.py::test_a_code_card_written_outside_decsim_runs_with_no_registration` |
| a code card with keys of its own, named in the yaml | `tests/experiments/test_yaml_surface.py::test_the_code_card_row_named_in_the_yaml_is_built_with_its_own_keys` |
| a layout | `tests/qpu/test_layouts.py::test_a_layout_written_outside_decsim_hears_every_hook_of_a_run` |
| a boundary or idle policy | `tests/machine/test_machine.py::test_a_policy_written_outside_decsim_is_used_on_its_own_axis` |
| a windowing scheme | `tests/windows/test_window_planner.py`, and the `WindowingScheme` port |
| a windowing scheme with a key of its own | `tests/windows/test_settings.py::test_a_scheme_rows_own_key_reaches_its_settings` |
| a link fabric | `tests/links/test_link_profiles.py::test_a_fabric_row_written_outside_decsim_runs_from_a_yaml`, and the `Link` port |
| an escalation policy | `tests/escalation/test_policies.py`, and the `EscalationPolicy` port |

## Two rules your class has to keep

Your class runs inside the machine, so it keeps the machine's contract.

**Charge your own time through the engine.** A component never sleeps
and never polls. The engine reaches your class the way its port says: a
decoder is handed one by `start`, and a magic state factory reads it off
the collaborators record its constructor takes. Call `engine.schedule`
with the delay you just charged.

**Derive your randomness from the run's seed.** A stochastic component
reports its own seed source and derives its generator from the root seed
and its path, so one seed reproduces one shot exactly. See
`decsim/seeding.py` and
`tests/machine/test_machine.py::test_a_component_gets_the_seed_derived_from_the_runs_seed_and_its_path`.

## What will tell you what you forgot

- `tests/observe/test_wiring.py`: if your component fires a trace source,
  the wiring census fails until something hears it, and fails again if
  two listeners of one class hear it twice.
- `tests/test_import_surface.py`: every module is imported, so a
  component that only imports under one setting is caught.
- `tools/check_row_recognition.py`, run by `tools/check.sh`: it fails on
  any module that tests against a row's class, because a fact a caller
  needs about a row is declared on the port and answered by every row.

## Read next

- [How to plug a component in without a table row](plug_in_without_a_table_row.md): skip step 2 while the
  class is still changing.
- [How to add a decoder backend](add_a_decoder_backend.md): the decoder case in full.
- [The plug-in tables](../reference/tables.md): all eighteen tables and their rows.
