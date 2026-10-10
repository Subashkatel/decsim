[decsim docs](../README.md) › [How-to guides](README.md)

# How to write a machine as a graph

A machine can be written as named parts and the wires between them,
in Python or in YAML. Each part is a decsim settings record. Its
parameters sit at its name (`uf.unit_count`), so `--set`, `decsim params`
and `decsim describe` all reach it. Today the graph runs as the
`MachineSettings` it resolves to, and a task keeps that record's id.

## 1. Name the parts and wire them

```python
import decsim
import decsim.settings as machine_settings
import decsim.topology.graph as graph

FRIDGE = machine_settings.FRIDGE_CLOCK

machine = graph.Machine(
    clock=FRIDGE,
    parts=(
        graph.Part("qpu", qpu),
        graph.Part("controller", controller),
        graph.Part("chip_buffer", buffer, placed="fridge"),
        graph.Part("uf", union_find_pool, placed="fridge"),
        graph.Part("frame", frame),
        graph.Part("workload", workload),
    ),
    wires=(
        graph.Wire("qpu.readout", "controller.readout"),
        graph.Wire("controller.rounds", "chip_buffer.rounds"),
        graph.Wire("chip_buffer.reads", "uf.rounds"),
        graph.Wire("uf.corrections", "frame.corrections"),
        graph.Wire("frame.decisions", "controller.decisions"),
        graph.Wire("controller.pulses", "qpu.pulses"),
    ),
)
task = decsim.Task("uf_d5", machine, collection=collection)
```

A part's kind comes from the `MachineSettings` field that takes its
record, so your own record joins a kind by meeting that field's type.
The kinds and their ports are in `decsim/topology/part_kinds.py`. A
wire runs from an output to an input that carries the same signal. A
wrong wire is refused when the machine is made. A required port left
unwired is refused when the task is built.

For two decoders, add a second store and pool, wire the first pool's
`escalate` to the second pool's `escalated`, and add a
`SwitchingSettings` part. A pool alone placed on the host is the strong
decoder.

## 2. Price the links

A wire may carry its own link:

```python
import decsim.topology.fabrics as fabrics

uplink = fabrics.ClockedLink(latency_cycles=40, bits_per_cycle=38.79, lane_count=4)
graph.Wire("controller.rounds", "chip_buffer.rounds", link=uplink)
```

Every other hop takes the machine's `fabric`. The fabric prices a hop by
the sites of its two ends:

```python
leg = fabrics.RoundTripLeg(2.305, 128, 100_000)
host_poll = fabrics.TimedLink(0.0)
machine = graph.Machine(..., fabric=fabrics.FridgeHostFabric(cable=leg, host_local=host_poll))
```

`ReferenceFabric` is the default card, and `PathLinks` gives a link
to a path by its name.

## 3. Or write it in YAML

```yaml
clock: 250MHz
parts:
  qpu: {type: QpuSettings, distance: 5, round_period: 1us}
  uf:
    type: DecoderPoolSettings
    placed: fridge
    unit_count: ${union_find.units}
    algorithm: {type: PyMatchingDecoder, preset_latency: 28ns}
  workload: python
wires:
  - qpu.readout -> controller.readout
  - wire: uf.corrections -> frame.corrections
    link: {type: ClockedLink, latency: "${fiber.down}"}
```

```python
import decsim.topology.machine_file as machine_file

machine = machine_file.load(
    "machine.yaml", "constants.yaml", python_parts={"workload": workload}
)
```

A part writes only the fields it changes from its record's defaults. A
part that holds a Python object, such as the workload's Stim circuit, is
written `python` and given in `python_parts`. `${name}` reads a
constant from the constants file, where a comment after a value is its
source. In a flow mapping, `{...}`, quote it. A constant that nothing
reads is refused. `machine_file.dumps(machine)` writes any graph back as
YAML.

## 4. Set, list and describe

```
decsim params run.py --only uf_d5
decsim run run.py --set uf.unit_count=2 --set uf.placed=host
decsim describe run.py --only uf_d5
```

`describe` prints the task's machine as YAML, then every path's
latency and rate as the run prices them. A task built from
`MachineSettings` is printed as the graph it resolves from.

A graph that today's builder cannot build is refused with "not
implemented yet". That covers three pools, two pools with no escalation
between them, or a store feeding a store. Stage 4 builds the graph
itself.
