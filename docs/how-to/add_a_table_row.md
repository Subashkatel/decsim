[decsim docs](../README.md) › [How-to guides](README.md)

# How to add a row to a table

A pluggable part of decsim is a **table**: a dictionary whose keys are
the names a yaml file may write and whose values are the classes the
machine builds. There are seventeen of them, listed with every row in
[The plug-in tables](../reference/tables.md). This is the recipe for adding a row to any
of them.

Adding a row must touch exactly three things: your class, the table, and
`configs/reference.yaml`. If it takes more, something is wrong with the
port rather than with your class.

## 1. Find the port and fill it

Open `decsim/ports.py` and find the port your part fills.
[The ports](../reference/ports.md) is the same file as a page, with every method
and what it does. The ports are in pipeline order, so the part you want
is near the component that uses it.

Write a class with those methods. The ports are `typing.Protocol`
classes and they are structural: you do not inherit anything, and there
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

Then add it to `configs/reference.yaml` in the same commit. That file is
the documentation of the yaml surface, and
`tests/front/test_yaml_surface.py` fails when the file and the readers
drift apart.

## What a typo gets

A `kind` that is not a row is refused when the config is loaded, by
name, with the rows printed:

```
weak_decoder.kind 'my_decodr' is not a row of its table; the rows are
['belief_matching', 'bposd', 'pymatching', 'relay_bp', 'tesseract',
'union_find', 'unweighted_pymatching']
```

One function does that for every table (`decsim/tables.py`, `row`), so
the refusal reads the same wherever it comes from. It is pinned by
`tests/machine/test_machine.py::test_a_decoder_kind_off_the_table_is_refused_naming_the_rows`.

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
| a decoder, through a whole gate point | `tests/machine/test_machine.py::test_a_second_table_row_runs_gate_point_one` |
| a round store | `tests/machine/test_machine.py::test_a_new_round_store_is_one_class_and_one_table_row` |
| a code card | `tests/machine/test_machine.py::test_a_code_card_written_outside_decsim_runs_with_no_registration` |
| a layout | `tests/qpu/test_layouts.py::test_a_layout_written_outside_decsim_hears_every_hook_of_a_run` |
| a boundary or idle policy | `tests/machine/test_machine.py::test_a_policy_written_outside_decsim_is_used_on_its_own_axis` |
| a windowing scheme | `tests/windows/test_window_planner.py`, and the `WindowingScheme` port |
| a link card | `tests/links/test_link_profiles.py`, and the `Link` port |
| an escalation policy | `tests/escalation/test_policies.py`, and the `EscalationPolicy` port |

## Two rules your class has to keep

Your class runs inside the machine, so it keeps the machine's contract.

**Charge your own time through the engine.** A component never sleeps
and never polls. Take the `Engine` in your constructor and call
`engine.schedule` with the delay you just charged.

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
- [The plug-in tables](../reference/tables.md): all seventeen tables and their rows.
