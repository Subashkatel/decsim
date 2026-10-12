"""The decoder bases and the strong tier, as the graphs they are.

The referent is the record each graph resolves to (topology/slotted.py),
the record a task keeps and is identified by: every port of a base is
bound, each pool sits where its decoder runs, the weak base's hops are
one 250 MHz fridge cycle (4 ns, 2605.04892 line 1063), and a strong tier
adds the strong store and pool on the host, the escalation wire and the
strong window's shape (window_settings.switching_windows).
"""

import dataclasses

import decsim.links.link_profiles as link_profiles
import decsim.presets.clocks as clocks
import decsim.presets.link_cards as link_cards
import decsim.presets.machines as machines
import decsim.topology.fabrics as fabrics
import decsim.topology.graph_layout as graph_layout
import decsim.topology.slotted as slotted
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run
import tests.escalation.test_strong_window_shapes as shape_tests
import tests.topology.test_graph_layout as layout_tests

# one fridge cycle at 250 MHz, in ticks
ONE_FRIDGE_CYCLE_TICKS = 4000


def _weak_base_switching(fabric=None):
    """The weak base escalating to the strong base's pool."""
    weak_base = machines.weak_baseline(3, 0.001, 1.0)
    switching = shape_tests.switching_slot(declared_run.DOUBLE_WINDOW)
    return machines.with_strong_tier(
        weak_base, machines.STRONG_BASELINE_POOL, switching, fabric
    )


def test_every_port_of_the_weak_base_is_bound():
    weak_base = machines.weak_baseline(3, 0.001, 1.0)

    settings = slotted.resolved(weak_base)

    assert weak_base.unbound_ports() == []
    assert settings.weak_decoder == machines.WEAK_BASELINE_POOL
    assert settings.strong_decoder is None
    assert settings.switching is None


def test_every_port_of_the_strong_base_is_bound():
    strong_base = machines.strong_baseline(3, 0.001, 1.0)

    settings = slotted.resolved(strong_base)

    assert strong_base.unbound_ports() == []
    assert settings.weak_decoder is None
    assert settings.strong_decoder == machines.STRONG_BASELINE_POOL
    assert settings.switching is None


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

    resolved = slotted.resolved(weak_base)
    settings = resolved.at_task()

    assert settings.workload == fifty_rounds
    assert settings.clock == clocks.FRIDGE_CLOCK
    assert settings.qpu.distance == 5


def test_the_weak_bases_chain_is_one_fridge_cycle_a_hop_on_unbounded_wires():
    weak_base = machines.weak_baseline(3, 0.001, 1.0)
    reference = link_profiles.logical_reference_profile()

    settings = slotted.resolved(weak_base)
    links = settings.links

    chain = (
        links.qpu_to_controller,
        links.controller_to_weak_buffer,
        links.weak_buffer_to_weak_decoder,
        links.weak_decoder_to_frame,
        links.frame_to_controller,
        links.controller_to_qpu,
    )
    latencies = [path.channel.propagation_latency_ticks for path in chain]
    capacities = [path.channel.capacity for path in chain]
    assert latencies == [ONE_FRIDGE_CYCLE_TICKS] * 6
    assert capacities == [None] * 6
    assert links.controller_to_strong_buffer == (
        reference.controller_to_strong_buffer
    )


def test_a_strong_tier_adds_the_strong_side_and_the_escalation_wire():
    machine = _weak_base_switching()

    strong_pool = machine.part(machines.STRONG_POOL)
    strong_store = machine.part(machines.STRONG_STORE)
    settings = slotted.resolved(machine)

    assert machine.unbound_ports() == []
    assert set(machines.ESCALATION_WIRES) <= set(machine.wires)
    assert strong_pool.placed == fabrics.HOST
    assert strong_store.placed == fabrics.HOST
    assert settings.weak_decoder == machines.WEAK_BASELINE_POOL
    assert settings.strong_decoder == machines.STRONG_BASELINE_POOL
    assert settings.switching.strong_window == declared_run.DOUBLE_WINDOW


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

    settings = slotted.resolved(machine)
    links = settings.links

    escalation = links.weak_decoder_to_strong_decoder.channel
    room_cycle_ticks = clocks.ROOM_CLOCK.period_ticks
    assert escalation.propagation_latency_ticks == room_cycle_ticks
    assert escalation.capacity is None


def test_a_part_changed_by_name_reaches_the_record():
    weak_base = machines.weak_baseline(3, 0.001, 1.0)

    two_units = weak_base.with_settings(machines.WEAK_POOL, unit_count=2)

    settings = slotted.resolved(two_units)
    expected = dataclasses.replace(machines.WEAK_BASELINE_POOL, unit_count=2)
    assert settings.weak_decoder == expected


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

    entry = graph_layout.entry_settings(chain)
    settings = entry.at_task()
    facts = settings.task_facts()
    layout = graph_layout.layout_of(chain, facts)
    edges = [(edge.name, edge.carried_from) for edge in layout.edges]
    deep = chain.part("deep")
    assert chain.unbound_ports() == []
    assert not slotted.holds(chain)
    assert deep.placed == fabrics.HOST
    assert edges == [
        ("switching", "weak_buffer"),
        ("deep_switching", "strong_buffer"),
    ]
