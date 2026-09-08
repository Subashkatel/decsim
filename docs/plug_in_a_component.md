# Plug in a component

You want to try your own decoder, link card, round store, windowing
scheme, code, layout or policy. The rule is the same for all of them: a
component depends on a port, so yours is one class that fills the port,
and nothing else in the machine changes.

There are two ways in, and which one you want depends on whether a yaml
has to be able to name your class.

**Pass the instance.** Every settings record takes the object itself, so
a class you wrote in your own file runs with no registration at all:
`DecoderSettings(decoder=MyDecoder())`, `WindowSettings(scheme=MyScheme())`,
`QpuSettings(code=MyCard())` or `QpuSettings(layout=MyLayout())`. Use this
while you are still changing the class.

**Add a table row.** A `kind` string in a settings record is a row of a
table in that package's own settings module, which is what lets a yaml
name your class and what makes it appear in a sweep. One row, nothing
more.

## The three steps

1. **Fill the port.** Find the port in `decsim/ports.py` (they are in
   pipeline order, one method per handoff) and write a class with those
   methods. The Protocols are structural: you do not inherit anything.
   The record each method takes and returns lives in `decsim/records/`.
   A port with members that are not methods says where their values
   come from: the `Decoder`'s `fault_model_requirement` is one of the
   four in `decsim/detector_error_model/fault_model_contracts.py`, and
   its `stage_recorded` is a `decsim/observe/trace_source.py` source,
   `SILENT` for a decoder with no internal stages. Inheriting
   `decsim/decoders/decoder.py`'s `DecoderBase` gives you both, and
   `start`, `cancel`, `occupancy` and `pipeline_depth` besides, so a
   decoder can be `decode` and `latency` alone.
2. **Add the row**, if a yaml needs to name it: one entry in the table
   your part belongs to, which lives beside the classes it lists.

   | Table | Where it lives |
   | --- | --- |
   | `DECODERS` | `decsim/decoders/settings.py` |
   | `SYNDROME_SOURCES`, `MAGIC_STATE_FACTORIES` | `decsim/qpu/settings.py` |
   | `ROUND_STORES` | `decsim/syndrome_buffer/round_store.py` |
   | `ESCALATIONS`, `STRONG_WINDOW_SHAPES` | `decsim/escalation/settings.py` |
   | `WINDOWING_SCHEMES`, `BOUNDARY_PAYLOADS` | `decsim/windows/settings.py` |
   | `IDLE_POLICIES` | `decsim/controller/settings.py` |
   | `WORKLOADS` | `decsim/frontends/settings.py` |
   | `CONFIDENCE_SIGNALS` | `decsim/confidence/signals.py` |
   | `WINDOW_CHECKS` | `decsim/observe/settings.py` |
3. **Name it in the yaml** by the row's key, in the section that owns
   your part: `weak_decoder: {kind: my_decoder}`. A kind that is not a
   row is refused at load with the rows listed, so a typo never runs.

## The worked examples, as tests

Each of these is a real test that plugs a class in from outside decsim
and runs it. Read the one closest to what you are writing; it is shorter
than any description of it.

| You are writing | Read |
| --- | --- |
| a decoder | `tests/machine/test_machine.py::test_a_new_decoder_is_one_class_and_one_table_row` |
| a decoder, run through a whole gate point | `tests/machine/test_machine.py::test_a_second_table_row_runs_gate_point_one` |
| a round store | `tests/machine/test_machine.py::test_a_new_round_store_is_one_class_and_one_table_row` |
| a code card | `tests/machine/test_machine.py::test_a_code_card_written_outside_decsim_runs_with_no_registration` |
| a layout | `tests/qpu/test_layouts.py::test_a_layout_written_outside_decsim_hears_every_hook_of_a_run` |
| a boundary or idle policy | `tests/machine/test_machine.py::test_a_policy_written_outside_decsim_is_used_on_its_own_axis` |
| a windowing scheme | `tests/windows/test_window_planner.py`, and the `WindowingScheme` port |
| a link card | `tests/links/test_link_profiles.py`, and the `Link` port |
| an escalation policy | `tests/escalation/test_policies.py`, and the `EscalationPolicy` port |

## What tells you what you forgot

- A `kind` off the table is refused by name, and the refusal lists the
  rows: `tests/machine/test_machine.py::test_a_decoder_kind_off_the_table_is_refused_naming_the_rows`.
- A yaml section nobody owns is refused with the sections listed:
  `tests/machine/test_machine.py::test_a_yaml_section_nobody_owns_is_refused_naming_the_sections`.
- If your component fires a trace source, the wiring census fails until
  something hears it, and fails again if two listeners of one class hear
  it twice: `tests/observe/test_wiring.py`.
- `tests/test_import_surface.py` imports every module, so a component
  that only imports under one setting is caught.

## Two rules that will bite you

Your class runs inside the machine, so it keeps the machine's contract:

- **Charge your own time through the engine.** A component never sleeps
  and never polls: it schedules the action that follows the cost it
  charged. Take the `Engine` in your constructor and use
  `engine.schedule`.
- **Derive your randomness from the run's seed.** A stochastic component
  reports its own seed source and derives its generator from the root
  seed and its path, so one seed reproduces one shot exactly. See
  `decsim/seeding.py` and
  `tests/machine/test_machine.py::test_a_component_gets_the_seed_derived_from_the_runs_seed_and_its_path`.

## Read next

- `decsim/ports.py`: the port you are filling.
- the settings module of the package you are plugging into: its table.
- `configs/reference.yaml`: the section your `kind` goes in.
