# How to plug a component in without a table row

While a class is still changing, you do not want to edit decsim at all.
You do not have to: every settings record takes the object itself, so a
class in your own file runs with no registration step.

## Pass the instance

```python
from decsim.decoders.settings import DecoderSettings
from decsim.windows.settings import WindowSettings
from decsim.qpu.settings import QpuSettings
from decsim.machine import Machine
from decsim.settings import MachineSettings

settings = MachineSettings(
    weak_decoder=DecoderSettings(decoder=MyDecoder()),
    windows=WindowSettings(scheme=MyScheme()),
    qpu=QpuSettings(code=MyCard()),
)
result = Machine.build(settings, seed=0).run()
```

Every field of `MachineSettings` has a default, so you name only what
differs from the default machine. The build site prefers your object
over the table: `decsim/build/plan.py` and `decsim/build/decoders.py`
check the settings record's object field first and fall back to the
`kind` only when it is `None`. That order is sinter's, which resolves a
caller's own object before its table
(`decsim/tables.py`'s own docstring says so).

## What you give up

Only the yaml. Without a row your class cannot be named from a config
file, which means it cannot appear in a sweep run by `decsim collect`
and cannot be recorded in a run folder's manifest as a name. Everything
else works: the ports, the engine, the trace, the metrics.

## When to add the row

When the class stops changing, or the moment you want to sweep it.
`docs/how-to/add_a_table_row.md` is one entry in a dictionary and one
key in `configs/reference.yaml`.

## The worked examples

Each of these builds a class outside decsim and runs it:

- a code card:
  `tests/machine/test_machine.py::test_a_code_card_written_outside_decsim_runs_with_no_registration`
- a layout:
  `tests/qpu/test_layouts.py::test_a_layout_written_outside_decsim_hears_every_hook_of_a_run`
- a policy:
  `tests/machine/test_machine.py::test_a_policy_written_outside_decsim_is_used_on_its_own_axis`

## Read next

- `docs/how-to/add_a_table_row.md`: the three steps when you want the
  yaml to name it.
- `docs/reference/ports.md`: the port your class fills.
