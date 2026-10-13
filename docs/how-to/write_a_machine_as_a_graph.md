[decsim docs](../README.md) › [How-to guides](README.md)

# How to write a machine as a graph

A machine is named parts and the wires between them, written in
Python. Each part is a decsim settings record. Its
parameters sit at its name (`uf.unit_count`), so `--set`, `decsim params`
and `decsim describe` all reach it. A task is identified by its graph.

## 1. Name the parts and wire them

```python
import decsim
import decsim.presets.clocks as clocks
import decsim.topology.graph as graph

machine = graph.Machine(
    clock=clocks.FRIDGE_CLOCK,
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

A part's kind is the one whose record type takes its record, so your
own record joins a kind by meeting that type. The kinds, their ports
and the defaults of the kinds a machine holds once are in
`decsim/topology/part_kinds.py`; a kind held once that the graph leaves
out runs on its default. A
wire runs from an output to an input that carries the same signal. A
wrong wire is refused when the machine is made. A required port left
unwired is refused when the task is built.

For two decoders, add a second store and pool and a `SwitchingSettings`
part, the escalate edge between them: wire the first pool's `escalate`
to the switching part's `results`, and its `redecode` to the second
pool's `escalated`. A pool alone placed on the host is the strong
decoder. A machine of more pools adds a switching part per edge; every
pool but the first is the target of one edge, and its parts' names name
its hops (`strong_to_deep`), its statistics and its seeds.

The machines decsim ships are graphs already (`decsim/presets/`), with
their parts named as above. Start from one and change it by name:

```python
import decsim.presets.machines as machines

base = machines.weak_baseline(5, 0.001, 1.0)
two_units = base.with_settings(machines.WEAK_POOL, unit_count=2)
task = decsim.Task("weak_d5", two_units, collection=collection)
```

`with_settings` changes fields of one part's record, `with_record`
replaces its record and keeps its site, `with_held` replaces the record
of a kind held once, `with_part` replaces a part of the same name or
adds one, `with_wires` adds wires, and `without_part` drops a part and
its wires. `machines.base` writes the parts held once with their loop
wired, `machines.with_pool` adds a pool that reads a store of its own,
and `machines.with_strong_tier` adds a strong decoder on the host that
the weak one escalates to.

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

`ReferenceFabric` is the default card, and `RoleLinks` gives a link to
every hop of a role, such as `answer_fridge` or `write_landed`
(`decsim/records/routes.py`).

## 3. Set, list and describe

```
decsim params run.py --only uf_d5
decsim run run.py --set uf.unit_count=2 --set uf.placed=host
decsim describe run.py --only uf_d5
decsim describe run.py --only uf_d5 --json
```

`describe` prints the task's machine as YAML, then every path's
latency and rate as the run prices them. `--json`
writes the same graph and paths as one json object, for a script to
read. A record decsim does not define is written by its import path,
the name `--set` takes for it.

[How to build a chain of decoders](build_a_chain_of_decoders.md)
grows a graph of more pools. A pool no edge
reaches, or a store feeding a store, is refused.
