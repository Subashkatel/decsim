"""The decoder bases and the strong tier, as the graphs they are.

The referent is the layout each graph lays out to
(topology/graph_layout.py), the one a task runs on: every port of a base is
bound, each pool sits where its decoder runs, the weak base's hops are
one 250 MHz fridge cycle (4 ns, 2605.04892 line 1063), and a strong tier
adds the strong store and pool on the host, the escalation wire and the
strong window's shape (window_settings.switching_windows).

A router, or a router before the stores, that names only the entry pool
runs shot for shot as the machine without it: the referent is the
regression suite's whole-shot digest (tests/regression), recorded
before a router could sit in front of a pool. With a pool and a store
per patch, every window still reads the rounds it read from the one
store, so it answers as it did (tests/patch_pools.py).
"""

import dataclasses
import functools

import pytest

import decsim.frontends.lattice_surgery as lattice_surgery
import decsim.presets.clocks as clocks
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines
import decsim.records.windows as window_records
import decsim.topology.fabrics as fabrics
import decsim.topology.graph_layout as graph_layout
import decsim.windows.routers as routers
import decsim.windows.settings as window_settings
import tests.controller.test_patch_streams as patch_stream_tests
import tests.declared_run as declared_run
import tests.escalation.test_strong_window_shapes as shape_tests
import tests.frontends.test_patch_lives as patch_lives_tests
import tests.patch_pools as patch_pools
import tests.regression.test_shot_results as regression
import tests.topology.test_graph_layout as layout_tests

# one fridge cycle at 250 MHz, in ticks
ONE_FRIDGE_CYCLE_TICKS = 4000


def _layout(machine):
    at_task = machine.at_task()
    return graph_layout.layout_of(at_task)


def _wire_ends(machine) -> list:
    return [(wire.source, wire.target) for wire in machine.wires]


def _weak_base_switching(fabric=None):
    """The weak base escalating to the strong base's pool."""
    weak_base = machines.weak_baseline(3, 0.001, 1.0)
    switching = shape_tests.switching_slot(declared_run.DOUBLE_WINDOW)
    return machines.with_strong_tier(
        weak_base, machines.STRONG_BASELINE_POOL, switching, fabric
    )


def test_every_port_of_the_weak_base_is_bound():
    weak_base = machines.weak_baseline(3, 0.001, 1.0)

    layout = _layout(weak_base)

    pools = [plan.names.seat for plan in layout.pools]
    assert weak_base.unbound_ports() == []
    assert pools == [machines.WEAK_POOL]
    assert layout.entry_pool.settings == machines.WEAK_BASELINE_POOL
    assert layout.entry_switching() is None


def test_every_port_of_the_strong_base_is_bound():
    strong_base = machines.strong_baseline(3, 0.001, 1.0)

    layout = _layout(strong_base)

    pools = [plan.names.seat for plan in layout.pools]
    assert strong_base.unbound_ports() == []
    assert pools == [machines.STRONG_POOL]
    assert layout.entry_pool.settings == machines.STRONG_BASELINE_POOL
    assert layout.entry_switching() is None


def test_the_strong_base_holds_its_own_store_alone():
    strong_base = machines.strong_baseline(3, 0.001, 1.0)

    stores = strong_base.parts_of("syndrome_buffer")

    names = [store.name for store in stores]
    assert names == [machines.STRONG_STORE]


def test_the_weak_pool_sits_in_the_fridge_and_the_strong_pool_on_the_host():
    weak_base = machines.weak_baseline(3, 0.001, 1.0)
    strong_base = machines.strong_baseline(3, 0.001, 1.0)

    weak_pool = weak_base.part(machines.WEAK_POOL)
    weak_store = weak_base.part(machines.WEAK_STORE)
    strong_pool = strong_base.part(machines.STRONG_POOL)
    strong_store = strong_base.part(machines.STRONG_STORE)

    assert weak_pool.placed == fabrics.FRIDGE
    assert weak_store.placed == fabrics.FRIDGE
    assert strong_pool.placed == fabrics.HOST
    assert strong_store.placed == fabrics.HOST


def test_a_bases_shot_is_ten_rounds_a_unit_of_distance_on_the_fridge_clock():
    weak_base = machines.weak_baseline(5, 0.002, 1.0)
    fifty_rounds = machines.memory_workload(5, 0.002, 50)

    at_task = weak_base.at_task()

    qpu = at_task.held(machines.QPU)
    assert at_task.held("workload") == fifty_rounds
    assert at_task.clock == clocks.FRIDGE_CLOCK
    assert qpu.distance == 5


def test_the_weak_bases_chain_is_one_fridge_cycle_a_hop_on_unbounded_wires():
    weak_base = machines.weak_baseline(3, 0.001, 1.0)

    layout = _layout(weak_base)

    card = layout.card
    chain = [path.settings for path in card.paths]
    latencies = [path.channel.propagation_latency_ticks for path in chain]
    capacities = [path.channel.capacity for path in chain]
    names = card.path_names()
    assert set(names) == {
        "qpu_to_controller",
        "controller_to_weak_buffer",
        "weak_buffer_to_weak",
        "weak_to_weak",
        "weak_to_frame",
        "frame_to_controller",
        "controller_to_qpu",
    }
    assert latencies == [ONE_FRIDGE_CYCLE_TICKS] * 7
    assert capacities == [None] * 7


def test_a_strong_tier_adds_the_strong_side_and_the_escalation_wire():
    machine = _weak_base_switching()

    strong_pool = machine.part(machines.STRONG_POOL)
    strong_store = machine.part(machines.STRONG_STORE)
    layout = _layout(machine)

    strong_id = window_records.PoolId(machines.STRONG_POOL)
    strong_plan = layout.pool(strong_id)
    switching = layout.entry_switching()
    ends = _wire_ends(machine)
    assert machine.unbound_ports() == []
    assert ("weak.escalate", "switching.results") in ends
    assert ("switching.redecode", "strong.escalated") in ends
    assert strong_pool.placed == fabrics.HOST
    assert strong_store.placed == fabrics.HOST
    assert layout.entry_pool.settings == machines.WEAK_BASELINE_POOL
    assert strong_plan.settings == machines.STRONG_BASELINE_POOL
    assert switching.strong_window == declared_run.DOUBLE_WINDOW


def test_a_strong_tier_gives_the_windows_the_strong_windows_shape():
    weak_base = machines.weak_baseline(3, 0.001, 1.0)
    plain_windows = weak_base.part(machines.WINDOWS)
    switching_windows = window_settings.switching_windows(
        plain_windows.settings, declared_run.DOUBLE_WINDOW
    )

    machine = _weak_base_switching()

    windows = machine.part(machines.WINDOWS)
    assert windows.settings == switching_windows
    assert windows.settings.terminal_policy == "lookahead"


def test_a_strong_tier_keeps_the_machines_fabric_unless_it_is_given_one():
    weak_base = machines.weak_baseline(3, 0.001, 1.0)
    strong_side = link_cards.strong_side(weak_base.fabric)

    kept = _weak_base_switching()
    replaced = _weak_base_switching(strong_side)

    assert kept.fabric == link_cards.WEAK_BASELINE_FABRIC
    assert replaced.fabric == strong_side


def test_a_strong_tier_on_the_strong_side_reaches_the_host_in_a_room_cycle():
    weak_base = machines.weak_baseline(3, 0.001, 1.0)
    strong_side = link_cards.strong_side(weak_base.fabric)
    machine = _weak_base_switching(strong_side)

    layout = _layout(machine)

    escalation_path = layout.card.path_settings("weak_to_strong")
    escalation = escalation_path.channel
    room_cycle_ticks = clocks.ROOM_CLOCK.period_ticks
    assert escalation.propagation_latency_ticks == room_cycle_ticks
    assert escalation.capacity is None


def test_a_part_changed_by_name_reaches_the_layout():
    weak_base = machines.weak_baseline(3, 0.001, 1.0)

    two_units = weak_base.with_settings(machines.WEAK_POOL, unit_count=2)

    layout = _layout(two_units)
    expected = dataclasses.replace(machines.WEAK_BASELINE_POOL, unit_count=2)
    assert layout.entry_pool.settings == expected


def test_an_escalation_grows_a_chain_one_pool_and_edge_at_a_time():
    """The strong pool escalates on to a deep pool, a third, on the host."""
    two_pools = _weak_base_switching()
    deep_switching = layout_tests.DEEP_SWITCHING_SLOT

    chain = machines.with_escalation(
        two_pools,
        machines.STRONG_POOL,
        "deep",
        machines.STRONG_BASELINE_POOL,
        deep_switching,
    )

    layout = _layout(chain)
    edges = [(edge.name, edge.carried_from) for edge in layout.edges]
    deep = chain.part("deep")
    assert chain.unbound_ports() == []
    assert deep.placed == fabrics.HOST
    assert edges == [
        ("switching", "weak_buffer"),
        ("deep_switching", "strong_buffer"),
    ]


ONE_POOL = routers.OnePool.Settings(machines.WEAK_POOL)
SWITCHING_SHAPES = ("redo_window_switching", "cluster_gap_switching")


@pytest.mark.parametrize("name", SWITCHING_SHAPES)
def test_a_router_naming_only_the_entry_pool_runs_shot_for_shot_as_none(name):
    shape = regression.SHAPES[name]
    routed = functools.partial(_routed, shape)

    routed_digest = regression.shot_digest(routed)
    plain_digest = regression.shot_digest(shape)

    assert routed_digest == plain_digest


@pytest.mark.parametrize("name", SWITCHING_SHAPES)
def test_a_store_per_pool_with_only_the_entry_pool_runs_as_one_store(name):
    shape = regression.SHAPES[name]
    stored = functools.partial(_stored, shape)

    stored_digest = regression.shot_digest(stored)
    plain_digest = regression.shot_digest(shape)

    assert stored_digest == plain_digest


def test_a_patch_table_naming_only_the_entry_pool_runs_as_no_router(tmp_path):
    """The T gate's three patch streams, every one routed to weak."""
    plain = _t_gate(tmp_path)
    table = (((0, 0), "weak"), ((0, 1), "weak"), ((1, 0), "weak"))
    rule = routers.ByPatch.Settings(pool_by_patch=table)
    routed = machines.with_router(plain, rule, ())
    stored = machines.with_store_per_pool(plain, rule, ())

    plain_digest = regression.shot_digest(lambda: plain)
    routed_digest = regression.shot_digest(lambda: routed)
    stored_digest = regression.shot_digest(lambda: stored)

    assert routed_digest == plain_digest
    assert stored_digest == plain_digest


def test_a_store_per_pool_sends_the_factorys_decodes_to_the_pool_named():
    plain = machines.weak_baseline(3, 0.001, 1.0)
    rule = routers.OnePool.Settings("weak")
    weak = plain.part(machines.WEAK_POOL)
    fpga_b = weak.settings

    stored = machines.with_store_per_pool(
        plain, rule, (("fpga_b", fpga_b),), factory_pool="fpga_b"
    )

    router = stored.part(machines.ROUTER)
    assert router.settings.factory_pool == "fpga_b"


def test_a_store_per_pool_answers_every_window_as_one_shared_store_does():
    """Three patches escalating to one strong pool, its rounds carried.

    An escalated window's strong decode reads the rounds kept in the
    store of the pool that escalated it, here one of three stores.
    """
    shared_store = patch_pools.shared_strong_per_patch()
    store_per_pool = patch_pools.shared_strong_per_patch(True)
    shared_shot = patch_pools.shot(shared_store)
    per_pool_shot = patch_pools.shot(store_per_pool)

    shared_answers = patch_pools.answers(shared_shot)
    per_pool_answers = patch_pools.answers(per_pool_shot)

    committing = patch_pools.committing_pools(per_pool_shot)
    assert committing == {"weak", "w1", "w2", "strong"}
    assert per_pool_answers == shared_answers


def _routed(shape):
    machine = shape()
    return machines.with_router(machine, ONE_POOL, ())


def _stored(shape):
    machine = shape()
    return machines.with_store_per_pool(machine, ONE_POOL, ())


def _t_gate(tmp_path):
    path = tmp_path / "t_gate.lss"
    path.write_text(patch_lives_tests.T_GATE)
    workload = lattice_surgery.read_schedule(path, patch_stream_tests.DISTANCE)
    decode_microseconds = patch_stream_tests.T_GATE_DECODE_MICROSECONDS
    _settings, plain = patch_stream_tests.patch_stream_graph(
        workload, decode_microseconds
    )
    return plain
