[decsim docs](../README.md) › [How-to guides](README.md)

# How to plug a component in without a table row

While a class is still changing, you do not want to edit decsim at all.
You do not have to: every settings record takes the object itself, so a
class in your own file runs with no registration step.

## Pass the instance

```python
import dataclasses

import decsim.decoders.settings as decoder_settings
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.qpu.settings as qpu_settings
import decsim.records.program as program_records
import decsim.settings as machine_settings
import decsim.windows.settings as window_settings

@dataclasses.dataclass(frozen=True)
class MyCardSettings:
    """My card's record: the qpu hands its build the distance and sizes."""

    def build(self, distance, commit_rounds_override, buffer_rounds_override):
        return MyCard()


memory = program_records.Operation(id=1, name="memory", qubits=(0,))
workload = workload_settings.WorkloadSettings(operations=(memory,))
settings = machine_settings.MachineSettings(
    workload=workload,
    weak_decoder=decoder_settings.DecoderSettings(decoder=MyDecoder()),
    windows=window_settings.WindowSettings(scheme=MyScheme()),
    qpu=qpu_settings.QpuSettings(code_card=MyCardSettings()),
)
machine = machine_module.Machine.build(settings, 0)
result = machine.run()
```

The default workload is empty, so a run with no operations completes
at tick zero without calling any of your classes; name one.

Every field of `MachineSettings` has a default, so you name only what
differs from the default machine. The qpu's slots take a record whose
`build` makes the component, and the build calls whatever record the
slot holds: `source.build(code, circuit_arguments)`,
`code_card.build(distance, commit_rounds_override,
buffer_rounds_override)` and `layout.build(code)`. A table row's record
is built the same way, so yours needs no table.

## What you give up

Only the yaml, and with it what a sweep point sets for a code card:
your card's record is handed the point's `qpu.distance` and the
windows section's `commit_rounds` and `buffer_rounds`, and a card that
ignores them keeps its own.

Without a row your class cannot be named from a config file, which
means it cannot appear in a sweep run by `decsim run`, and a run
folder records it by its class and its attributes, which a later run
cannot rebuild it from. Everything else
works: the ports, the engine, the trace, the metrics.

## When to add the row

When the class stops changing, or the moment you want to sweep it.
[How to add a row to a table](add_a_table_row.md) is the recipe; a code
card's table is `CODE_CARDS`, named by `qpu.code_card`.

## The worked examples

Each of these builds a class outside decsim and runs it:

- a code card:
  `tests/machine/test_machine.py::test_a_code_card_written_outside_decsim_runs_with_no_registration`
- a layout:
  `tests/qpu/test_layouts.py::test_a_layout_written_outside_decsim_hears_every_hook_of_a_run`
- a policy:
  `tests/machine/test_machine.py::test_a_policy_written_outside_decsim_is_used_on_its_own_axis`

## Read next

- [How to add a row to a table](add_a_table_row.md): the three steps when you want the
  yaml to name it.
- [The ports](../reference/ports.md): the port your class fills.
