[decsim docs](../README.md) › [Tutorials](README.md)

# Build a machine step by step

`decsim run` and `Machine.build` build the whole machine in one call.
This lesson does the same build by hand, one part at a time, so you can
see what each part holds, which wires it makes itself, and which wires
the machine makes between parts. It takes about ten minutes.

It assumes you have done [Two tiers](two_tiers.md), because it builds
that lesson's machine.

## The six parts

A machine is six parts. Each is a small record in `decsim/build/` of the
components one stage of the loop holds:

| Part | What it holds |
| --- | --- |
| `Qpu` | the device, its syndrome source, the magic state factory |
| `Control` | the program's execution and the Pauli frame |
| `Readout` | the controller, the packing stage, the syndrome buffers |
| `Windows` | the planner, the window manager, the verdict |
| `Decoders` | each tier's decoder units and their managers |
| links | the link fabric every hop rides |

A part's `build` makes its components and wires them to one another.
`Machine.assemble` wires the parts to each other.

## Step 1. Read the settings

Run the lines of every step in one Python session, from the top of the
repository.

```python
import decsim.build.control as control_part
import decsim.build.decoders as decoders_part
import decsim.build.escalation as escalation_build
import decsim.build.plan as plan_build
import decsim.build.qpu as qpu_part
import decsim.build.readout as readout_part
import decsim.build.windows as windows_part
import decsim.engine as engine_module
import decsim.experiments.experiment as experiment
import decsim.machine as machine_module

config = experiment.load_experiment("configs/examples/two_tiers.yaml")
point = config.first_point_task()
settings = point.shot_settings()
```

`settings` is a `MachineSettings`: one record per section of the yaml.
You could also write it in Python, as
[How to plug a component in without a table row](../how-to/plug_in_without_a_table_row.md)
does.

## Step 2. Compile what the parts are built for

Some things are read from the settings before any part exists, because
more than one part is built for them: the escalation policy, the plan
(the windows, the circuit and the round clock), the burst detector, the
detection event formation, the decoder units, and the store slots: which
syndrome buffers a decoder reads, and whether it reads them in place.

```python
engine = engine_module.Engine()
escalation_policy = escalation_build.build_escalation_policy(
    settings.escalation, settings.weak_decoder
)
plan = plan_build.build_plan(settings, escalation_policy)
burst_detector = escalation_build.build_burst_detector(
    settings, engine, plan, escalation_policy
)
window_tier = escalation_policy.primary_tier
escalates = escalation_policy.requires_strong_context
detection_events = readout_part.build_detection_events(
    settings.detection_events,
    settings.clock,
    plan.device,
    window_tier,
    escalates,
    burst_detector,
)
pool = decoders_part.build_decoder_pool(
    settings, plan, escalation_policy, detection_events
)
weak_store_slot, strong_store_slot = machine_module.store_slots(
    settings, window_tier, pool
)
print(type(escalation_policy).__name__, plan.round_ticks)
```

It prints:

```text
Switching 1000000
```

The policy is the `switching` row the yaml names, and one round is a
million ticks, one microsecond.

## Step 3. Build each part

One line per part:

```python
links = machine_module.build_links(settings, engine)
qpu = qpu_part.Qpu.build(settings.magic_state_factory, engine, plan)
control = control_part.Control.build(
    settings.controller,
    settings.pauli_frame,
    settings.clock,
    engine,
    plan,
    links,
)
readout = readout_part.Readout.build(
    settings.controller,
    settings.links,
    weak_store_slot,
    strong_store_slot,
    settings.clock,
    engine,
    detection_events,
    links,
)
windows = windows_part.Windows.build(
    settings, engine, plan, escalation_policy, burst_detector, links
)
decoders = decoders_part.Decoders.build(
    settings.decoder_manager, engine, pool, escalation_policy
)
```

Each part is now whole on the inside. The controller already hands its
rounds to the readout part's own packing stage:

```python
print(readout.controller.assembler is readout.assembler)
```

```text
True
```

The wires to other parts are not made yet. The syndrome round sender
tells the window manager of each stored round, and the window manager
is in another part, so its port is still empty:

```python
readout.syndrome_round_sender.windows
```

```text
RuntimeError: SyndromeRoundSender.windows was read before it was bound
```

A component a run does not need is `None`. This run escalates, so its
readout part has a strong syndrome buffer; a weak-only run's would be
`None`.

## Step 4. Assemble

```python
machine = machine_module.Machine.assemble(
    settings, engine, plan, links, qpu, control, readout, windows, decoders
)
print(readout.syndrome_round_sender.windows is windows.window_manager)
print(windows.window_manager.requester.decode_queue is decoders.decoder_manager)
```

```text
True
True
```

`assemble` makes every wire that crosses from one part to another, in
four `connect` calls you can read in `decsim/machine.py`. It then starts
the parts, gives each random component its seed, connects the observers
and loads the program.

## Step 5. Run

```python
result = machine.run()
print(result.terminal_status, result.fully_done_ticks)
print(result.operation_results[0].logical_failure)
```

```text
complete 279888000
False
```

The run took 279.888 microseconds of machine time, and its one memory
operation came out right. `Machine.build(settings, 0)` builds the same
machine, and gives the same result.

`tests/machine/test_machine.py::test_a_machine_built_part_by_part_runs_as_the_one_call_does`
runs the steps of this page and checks every output on it.

## What you did

You built the machine the way `Machine.build` does: the shared pieces
first, then one part per line, then `assemble`. To change one part,
build your own in that part's line and keep the rest.

## Read next

- [How to add a component to a part](../how-to/add_a_component_to_a_part.md):
  a new kind of component, one no table lists.
- [How to add a row to a table](../how-to/add_a_table_row.md): a new
  row of a component decsim already has, such as a decoder.
- [Architecture](../explanation/architecture.md): the parts and the
  components, and why they are shaped this way.
