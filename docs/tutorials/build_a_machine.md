[decsim docs](../README.md) › [Tutorials](README.md)

# Build a machine step by step

`decsim run` and `Machine.build` build the whole machine in one call.
This lesson does the same build by hand, one part at a time, so you can
see what each part holds, which wires it makes itself, and which wires
the machine makes between parts. It takes about ten minutes.

It assumes you have done [Two tiers](two_tiers.md), because it builds
that lesson's machine.

## The parts

A machine is six parts, and a switching machine has a seventh. Each is
a small record in `decsim/build/` of the components one stage of the
loop holds:

| Part | What it holds |
| --- | --- |
| `Qpu` | the device, its syndrome source, the magic state factory |
| `Control` | the program's execution and the Pauli frame |
| `Readout` | the controller, the packing stage, the syndrome buffers |
| `Windows` | the planner, the window manager, the verdict |
| `Decoders` | each tier's decoder units and their managers |
| links | the link fabric every hop rides |
| `Switching` | the confidence signal, the decision and the strong window side |

A part's `build` makes its components and wires them to one another.
`Machine.assemble` wires the parts to each other.

## Step 1. Read the graph

Run the blocks of every step in one Python session, from the top of the
repository.

```python
import decsim.build.control as control_part
import decsim.build.decoders as decoders_part
import decsim.build.escalation as escalation_build
import decsim.build.qpu as qpu_part
import decsim.build.readout as readout_part
import decsim.build.windows as windows_part
import decsim.engine as engine_module
import decsim.links.fabric as fabric
import decsim.machine as machine_module
import decsim.windows.built_window_models as built_window_models
import examples.two_tiers as two_tiers

task = two_tiers.tasks[0]
graph = task.machine
layout = task.layout
clock = graph.clock
```

`graph` is the run file's first task at distance 3, its parts by name
and the wires between them (`decsim/topology/graph.py`). `layout` is
the graph laid out: each pool, the store it reads, whether it reads it
in place, each escalate edge, and the card that prices every hop
(`decsim/topology/graph_layout.py`). A part the machine holds once is
read with `graph.held`, its record or its kind's default.

## Step 2. Compile what the parts are built for

Some things are built before any part exists, because more than one
part is built for them: a switching part per edge, the plan (the
windows, the circuit and the round clock), the detection event
formation, and the decoder units.

```python
engine = engine_module.Engine()
switching_parts = escalation_build.build_switching(
    layout, engine, machine_clock=clock
)
plan = machine_module.plan_of(graph, layout, switching_parts)
formation_paths = layout.formation_paths()
detection_events = readout_part.build_detection_events(
    graph.held("detection_events"),
    clock,
    plan.device,
    formation_paths,
)
signal_by_pool = escalation_build.signal_by_pool(switching_parts)
pools = decoders_part.build_pool_units(
    layout, clock, detection_events, signal_by_pool
)
entry_switching = switching_parts[0]
policy_type = type(entry_switching.policy)
print(policy_type.__name__, plan.round_ticks)
```

It prints:

```text
Switching 1000000
```

The policy is the switching part's, and one round is a million ticks,
one microsecond.

## Step 3. Build each part

One call per part:

```python
links = fabric.LinkFabric(layout.card, engine)
factory = graph.held("magic_state_factory")
qpu = qpu_part.Qpu.build(factory, engine, plan)
controller = graph.held("controller")
control = control_part.Control.build(
    controller,
    graph.held("pauli_frame"),
    clock,
    engine,
    plan,
    links,
)
readout = readout_part.Readout.build(
    controller,
    layout.card,
    layout,
    clock,
    engine,
    detection_events,
    links,
)
window_models = built_window_models.BuiltWindowModels()
windows = windows_part.Windows.build(
    graph.held("windows"),
    graph.held("workload"),
    layout,
    clock,
    engine,
    plan,
    links,
    window_models,
)
target_by_source = layout.target_by_source()
decoders = decoders_part.Decoders.build(
    graph.held("decoder_manager"),
    clock,
    engine,
    pools,
    target_by_source,
)
```

Each part is now whole on the inside. The controller already hands its
rounds to the readout part's own packing stage:

```python
is_shared_assembler = readout.controller.assembler is readout.assembler
print(is_shared_assembler)
```

```text
True
```

The wires to other parts are not made yet. The window manager's
requester asks for each window's decode, and the decoder manager that
takes the request is in another part, so its port is still empty:

```python
windows.window_manager.requester.decode_queue
```

```text
RuntimeError: DecodeRequester.decode_queue was read before it was bound
```

The readout part builds one store for each store the layout holds.
This run escalates, so it holds two, the weak pool's and the strong
pool's; a weak-only run's holds one.

## Step 4. Assemble

```python
machine = machine_module.Machine.assemble(
    graph,
    engine,
    plan,
    links,
    qpu,
    control,
    readout,
    windows,
    decoders,
    layout,
    switching_parts,
)
window_manager = windows.window_manager
decode_queue = window_manager.requester.decode_queue
is_requester_wired = decode_queue is decoders.decoder_manager
print(is_requester_wired)
```

```text
True
```

`assemble` makes every wire that crosses from one part to another, in
the `connect` calls you can read in `decsim/machine.py`. It then starts
the parts, gives each random component its seed, connects the observers
and loads the program.

## Step 5. Run

```python
result = machine.run()
print(result.terminal_status, result.fully_done_ticks)
print(result.operation_results[0].logical_failure)
```

```text
complete 289904000
False
```

The run took 289.904 microseconds of machine time, and its one memory
operation came out right. `Machine.build(graph, 0)` builds the same
machine, and gives the same result.

## What you did

You built the machine the way `Machine.build` does: the shared pieces
first, then one part per call, then `assemble`. To change one part,
build your own in that part's call and keep the rest.

## Read next

- [How to add a component to a part](../how-to/add_a_component_to_a_part.md):
  a new kind of component.
- [How to add a decoder backend](../how-to/add_a_decoder_backend.md): a
  new decoder.
- [The parts](../reference/parts.md): every settings record a machine is
  built from.
