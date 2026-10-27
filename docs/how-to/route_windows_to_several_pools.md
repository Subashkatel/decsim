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
frame. `describe` prints the router as a part of type `router` with a
`rule`, and its wires as `weak_buffer.reads -> router.rounds` and
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
only the timing (`tests/windows/test_routers.py`; with a shared strong
pool and a store per pool, `tests/escalation/test_shared_target.py`
and `tests/presets/test_machines.py`).

`decsim run` reads the pools two ways:

- **By tier**, in `sweep.csv`: the weak tier is every entry pool, the
  strong tier every pool an entry pool escalates to. A tier's
  `*_busy_fraction` and `*_offered_load` are its busy unit time over
  all its units, its `*_queue_max` its largest pool's queue, and
  `escalated_windows` the windows any strong-tier pool committed. The
  `algorithm` column names every entry pool's decoder kind, joined by
  `+` when they differ.
- **By pool**, in `shot_pools.csv` (a row per shot and pool) and
  `pool_summary.csv` (a row per task and pool): each pool's queue
  peak, busy fraction, offered load, the windows it committed and the
  windows it escalated. This is where one patch's pool falling behind
  shows while the tier as a whole keeps up.

`decsim stream` runs the same machines over a long computation: each
memory patch is its own stream, and its record holds the same tier
numbers in `pools` and each pool's row in `pool_rows`
([How to run a memory as a stream](run_a_memory_as_a_stream.md)).

## 4. Escalate from any routed pool, to a pool they share

Any entry pool may escalate. Several may escalate to one target, as
Delfosse's lazy decoders share decoding units (2001.11427 Sec. V) and
Toshio's weak decoders share one strong decoder (2510.25222 Sec. III C):

```python
grown = machines.with_escalation(routed, "weak", "strong", strong, switching)
shared = machines.with_escalation_to(grown, "fpga_b", "strong", switching)
```

A window climbs the chain its own entry pool starts; each pool leaves
by at most one edge. An answer that goes back through a chip
(`strong_answer_route=through_weak_chip`) goes to the chip that asked.
Every edge out of an entry pool must agree on `strong_window` and
`confidence`, since the planner lays one plan for all of them, and a
task's learned threshold is refused over more than one entry edge:
give each a fixed one. `tests/escalation/test_shared_target.py` runs
both answer routes and a third level behind the shared pool.

## 5. Name the pool that decodes the factory

Idle rounds go to the pool of their operation's patch. No paper
dedicates a decoder to a magic-state factory (DecoNet 2504.11805
Sec. VIII-F and SWIPER 2412.05115 Fig. 10 decode distillation as
ordinary patches), so with more than one entry pool the
router names one: `with_router(..., factory_pool="fpga_b")`, else the
build refuses.

## 6. Put pools on one device

A device is a site whose links are shared. Place the pools at one site
name, any name, and give the fabric one shared link per site pair:

```python
import decsim.topology.fabrics as fabrics

cable = fabrics.TimedLink(1.0, 100_000.0)
shared = fabrics.SiteLink((fabrics.FRIDGE, "gpu0"), cable, shared=True)
fabric = fabrics.SiteLinks((shared,), base_fabric)
```

Every hop from `fridge` into `gpu0` then queues on one channel,
`fridge_to_gpu0`, and every hop back on `gpu0_to_fridge`.

Links are all an FPGA's or an ASIC's pools share: each decoder instance
there is logic of its own (DecoNet 2504.11805 p7-p10), so each pool
keeps its own units. Pools on a GPU or a CPU host also share its
compute. CUDA-Q QEC runs N decoder replicas as N streams on one GPU,
"an interchangeable pool" (cudaqx host_side_dispatcher_design.md line
27), and Google decodes in real time only "when
provided exclusive access to a subset of CPUs" (2408.13687 p15). Give
the site a device with that many execution slots:

```python
import decsim.decoders.execution_slots as execution_slots
import decsim.topology.graph as graph

slots = execution_slots.ExecutionSlots.Settings(2)
device = graph.Part("gpu0", slots, "gpu0")
shared = machine.with_part(device)
```

A decode takes a slot when it starts and gives it back when its unit
frees, as a GPU stream is busy until its result is sent. One with no free slot waits with its input landed, and the oldest
waiting decode on the device starts when a slot frees, whichever pool
it is from. A pool's `unit_count` still caps the pool, so a device with
a slot for each unit of its pools changes nothing. The device keeps a
`slot_wait` distribution at its name. A site holds at most one device,
and a device at a site no pool sits at is refused. No slowdown with
load is modelled; one measured curve exists, for batches, not for
concurrent streams (`specs/decoder-resources/data-landing.md`).

## 7. Give each pool its own store

Put the router before the stores instead of after the entry store:

```python
stored = machines.with_store_per_pool(base, rule, (("fpga_b", weak),))
```

The controller's rounds reach the router, which feeds `weak_buffer` and
a store per added pool, `fpga_b_buffer`, each read by its pool. Each
round is written to the store of the pool that decodes its stream, as
DecoNet's router routes "measurement data to the appropriate decoder
instances" (2504.11805 Sec. VI-B.2). A round another pool's window
reads, a merge's partner region or a successor's first rounds, is also
copied to that pool's store as it is written; the copy rides the
controller's hop into that store, `controller_to_fpga_b_buffer`, and
its transfer row says `{"copy_into": "fpga_b_buffer"}`. Committed
boundaries still go pool to pool, one message per pair. No source has
a decoder fetch a neighbour's raw syndromes when it decodes, so decsim
does not either (`specs/decoder-resources/data-landing.md`).

A round leaves only when every store it goes to has room, so a full
store holds back the rounds bound for it. A window's read of another
pool's round must be known before the round is written: a merge
couples its patches as its body starts, before its rounds are
measured, and a read asked later stops the run with the round, the
store and the stream named. `tests/controller/test_patch_streams.py`
runs the T gate this way and checks the copied rounds.

## What a router does not do yet

An absorbing strong window (`double_window`) is refused beside a
router, "not implemented yet" ([the refusals](build_each_switching_pattern.md#what-is-refused-for-now)).
