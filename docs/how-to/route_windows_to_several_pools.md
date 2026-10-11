[decsim docs](../README.md) › [How-to guides](README.md)

# How to route windows to several pools

A router sits between the entry store and several decoder pools. It
sends each window to one of them, and each of those pools decodes the
windows it is sent as the entry pool does
(`decsim/windows/routers.py`). The sources route by a schedule fixed
before the run. VQD gives each logical qubit its own decoder as its
baseline (Maurya and Tannu 2406.17995v2 Sec. 4.1, Fig. 5), and its
Arbiter routes each qubit's syndromes to it (Sec. 5).

## 1. Put a router in front of the entry pool

```python
import decsim.presets.machines as machines
import decsim.windows.routers as routers

rule = routers.ByPatch.Settings(pool_by_patch=((0, "weak"), (1, "fpga_b")))
weak = base.part("weak").settings
routed = machines.with_router(base, rule, (("fpga_b", weak),))
```

`with_router` rewires `weak_buffer.reads` to the router and the
router's `windows` to `weak` and to each new pool, which answers the
frame. In a machine file, the router is a part of type `router` with a
`rule`, and its wires are `weak_buffer.reads -> router.rounds` and
`router.windows -> <pool>.rounds`, the first pool wired first.

## 2. Choose the rule

`by_patch` sends each patch's windows to the pool its table names. A
window over several patches, a merge, goes to the pool of its
operation's first patch; VQD does not say how a decode that spans two
qubits is split, so this is decsim's reading. An operation on a patch
the table does not name is refused when the machine is built.

A rule that reads the run's state, such as the least loaded pool or a
backlog, has no paper behind it and is not built. The `rule` slot takes
a record of your own with `name`, `pool_for(operation)` and
`check_operations(operations, pool_names)`
([add a record of your own](add_a_store_card_signal_policy_or_scheme.md)).
Choosing which waiting window one shared pool serves next (VQD's MFD,
RR and MLS) is that pool's scheduler, not a router.

## 3. What it costs and what it reports

Every routed pool reads the entry store over a hop of its own,
`weak_buffer_to_<pool>`. A boundary rides from the pool that committed
it to the pool of the window that reads it. Each frame record's `tier`
is the pool that answered, and each pool's statistics sit at its name.
With the same decoder on every pool, routing changes no correction,
only the timing (`tests/windows/test_routers.py`).

## What a router does not do yet

An escalate edge may leave only the first entry pool, and an absorbing
strong window (`double_window`) is refused beside a router; both say
"not implemented yet (stage 5)".
