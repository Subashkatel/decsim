"""A graph of a new shape, laid out (topology/graph_layout.py).

The referent is a chain written by hand: the weak base escalating to a
union-find pool on the host, whose cluster gap escalates on to the
strong base's pool, a third. Its pools, stores and edges go by their
parts' names, entry first; each hop is named by the parts at its ends
and priced on the card; and the task is identified by the graph itself.
"""

import dataclasses

import pytest

import decsim.confidence.cluster as cluster
import decsim.decoders.execution_slots as execution_slots
import decsim.decoders.union_find.cycle_count as cycle_count
import decsim.decoders.union_find.decoder as union_find
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.collect as collect
import decsim.presets.machines as machines
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.syndrome_buffer.syndrome_round_receiver as receiver_module
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph
import decsim.topology.graph_layout as graph_layout
import decsim.windows.routers as routers
import tests.declared_run as declared_run
import tests.escalation.test_strong_window_shapes as shape_tests

DEEP_POOL = "deep"
DEEP_STORE = "deep_buffer"
DEEP_SWITCHING = "deep_switching"
# the middle pool: union-find on the host, charged the host's time, whose
# one solve carries the cluster gap's evidence
_HOST_TIME = cycle_count.HostMeasuredTime()
_UNION_FIND = union_find.UnionFindDecoder.Settings(timing=_HOST_TIME)
MIDDLE_POOL = dataclasses.replace(
    machines.STRONG_BASELINE_POOL, algorithm=_UNION_FIND
)
# the deep edge keeps a cluster gap of 20 dB or more from the middle pool
_CLUSTER_GAP = cluster.ClusterGap.Settings()
_TWENTY_DECIBELS = threshold_sources.FixedThreshold.Settings(
    threshold_decibels=20.0
)
DEEP_SWITCHING_SLOT = escalation_settings.SwitchingSettings(
    confidence=_CLUSTER_GAP, threshold=_TWENTY_DECIBELS
)


def chain_machine(
    deep_switching: escalation_settings.SwitchingSettings = (
        DEEP_SWITCHING_SLOT
    ),
    entry_window: escalation_settings.StrongWindowSettings = (
        declared_run.DOUBLE_WINDOW
    ),
) -> graph.Machine:
    """The strong pool escalating on to a deep pool on the host.

    The entry edge lays its windows out by entry_window.
    """
    two_pools = _two_pools(entry_window)
    return machines.with_escalation(
        two_pools,
        machines.STRONG_POOL,
        DEEP_POOL,
        machines.STRONG_BASELINE_POOL,
        deep_switching,
    )


def laid_out(machine: graph.Machine):
    """The machine's layout, at its task."""
    at_task = machine.at_task()
    return graph_layout.layout_of(at_task)


def _two_pools(
    entry_window: escalation_settings.StrongWindowSettings = (
        declared_run.DOUBLE_WINDOW
    ),
) -> graph.Machine:
    weak_base = machines.weak_baseline(3, 0.001, 1.0)
    switching = shape_tests.switching_slot(entry_window)
    return machines.with_strong_tier(weak_base, MIDDLE_POOL, switching)


def test_a_chains_pools_go_by_their_parts_entry_first():
    chain = chain_machine()
    layout = laid_out(chain)

    pools = [plan.pool for plan in layout.pools]
    stores = [plan.name for plan in layout.stores]
    edges = [plan.name for plan in layout.edges]

    assert pools == [
        window_records.PoolId("weak"),
        window_records.PoolId("strong"),
        window_records.PoolId(DEEP_POOL),
    ]
    assert stores == ["weak_buffer", "strong_buffer", DEEP_STORE]
    assert edges == ["switching", DEEP_SWITCHING]


def test_a_chain_seeds_its_entry_edge_as_the_two_pool_machine_does():
    """A pool added past the entry moves no other part's seed."""
    chain_graph = chain_machine()
    chain = laid_out(chain_graph)
    two_pool_machine = _two_pools()
    two_pools = laid_out(two_pool_machine)

    chain_pools = _pool_seeds(chain)
    chain_edges = _policy_seeds(chain)
    two_pool_pools = _pool_seeds(two_pools)
    two_pool_edges = _policy_seeds(two_pools)

    assert chain_pools == [*two_pool_pools, (DEEP_POOL, "deep_manager")]
    assert chain_edges == [*two_pool_edges, f"{DEEP_SWITCHING}_policy"]


def test_each_pool_reads_its_input_off_its_own_store_and_keeps_its_name():
    chain = chain_machine()
    layout = laid_out(chain)

    input_hops = layout.input_hop_by_pool()
    paths = layout.statistics_path_by_pool()

    deep = window_records.PoolId(DEEP_POOL)
    assert input_hops[deep] == f"{DEEP_STORE}_to_{DEEP_POOL}"
    assert paths[deep] == DEEP_POOL
    assert len(input_hops) == 3


def test_each_hop_is_named_by_the_parts_at_its_ends():
    chain = chain_machine()
    layout = laid_out(chain)

    deep_id = window_records.PoolId(DEEP_POOL)
    deep = layout.pool(deep_id)
    deep_edge = layout.edges[1]
    deep_store = layout.store(DEEP_STORE)

    assert deep.hops.answer == "deep_to_frame"
    assert deep.hops.same_pool == "deep_to_deep"
    assert deep_edge.hops.forward == "strong_to_deep"
    assert deep_edge.hops.reverse == "deep_to_strong"
    assert deep_store.read_hop == "deep_buffer_to_deep"


def test_the_entry_store_is_written_and_a_targets_store_landed():
    """An escalated window's rounds come from its source pool's store."""
    chain = chain_machine()
    layout = laid_out(chain)

    entry_store = layout.stores[0]
    deep_store = layout.store(DEEP_STORE)
    deep_edge = layout.edges[1]

    assert entry_store.intake.shape == receiver_module.WRITTEN
    assert deep_store.intake.shape == receiver_module.LANDED
    assert deep_edge.hops.forward == "strong_to_deep"
    assert deep_edge.carried_from == "strong_buffer"


def test_a_round_reaches_each_pool_through_one_path_of_seats():
    chain = chain_machine()
    layout = laid_out(chain)

    paths = layout.formation_paths()

    assert paths == (
        ("controller", "weak_buffer", "weak"),
        ("controller", "weak_buffer", "strong_buffer", "strong"),
        ("controller", "weak_buffer", "strong_buffer", DEEP_STORE, DEEP_POOL),
    )


def test_the_card_prices_the_graphs_own_hops_after_the_shared_three():
    chain = chain_machine()
    layout = laid_out(chain)

    card = layout.card
    names = card.path_names()

    assert names[:3] == (
        transfer_records.QPU_TO_CONTROLLER,
        transfer_records.CONTROLLER_TO_QPU,
        transfer_records.FRAME_TO_CONTROLLER,
    )
    assert "strong_to_deep" in names
    assert "deep_to_frame" in names
    assert card.path_settings("deep_to_deep") is not None


def test_the_layout_holds_the_entry_edge():
    chain = chain_machine()

    layout = laid_out(chain)

    strong_id = window_records.PoolId(machines.STRONG_POOL)
    strong = layout.pool(strong_id)
    assert layout.entry_pool.settings == machines.WEAK_BASELINE_POOL
    assert strong.settings == MIDDLE_POOL
    switching = chain.part("switching")
    assert layout.entry_switching() == switching.settings


def test_pools_no_edge_joins_are_refused():
    chain = chain_machine()
    cut = chain.without_part(DEEP_SWITCHING)

    with pytest.raises(graph.GraphError, match="reached by no escalate edge"):
        laid_out(cut)


def test_a_chain_that_comes_back_to_a_pool_it_passed_is_refused():
    """The chain weak, b, c, d escalates back into b, a pool it passed."""
    graph_with_circle = declared_run.weak_only_graph(rounds=3)
    switching = declared_run.declared_switching()
    weak = graph_with_circle.part("weak")
    pool = weak.settings
    for source, target in (("weak", "b"), ("b", "c"), ("c", "d")):
        graph_with_circle = machines.with_escalation(
            graph_with_circle, source, target, pool, switching
        )
    graph_with_circle = machines.with_escalation_to(
        graph_with_circle, "d", "b", switching, switching_name="d_to_b"
    )

    with pytest.raises(graph.GraphError) as refusal:
        laid_out(graph_with_circle)

    message = str(refusal.value)
    assert "['b', 'c', 'd'] escalate in a circle" in message
    assert "from weak comes back to b" in message


def test_a_router_makes_each_pool_it_feeds_an_entry_pool():
    routed = _routed_chain()

    layout = laid_out(routed)
    entry_names = [plan.names.seat for plan in layout.entry_pools]

    assert entry_names == [machines.WEAK_POOL, _ROUTED_POOL]
    assert layout.router.rule == _ROUTED_BY_PATCH


def test_two_entry_pools_escalate_to_one_shared_strong_pool():
    """Toshio's N_window weak decoders and a single strong decoder.

    2510.25222 Sec. III C: fpga_b escalates to the strong pool the
    weak pool escalates to, which is laid out once.
    """
    routed = _routed_chain()
    switching = graph.Part("routed_switching", DEEP_SWITCHING_SLOT)
    results = graph.Wire(f"{_ROUTED_POOL}.escalate", "routed_switching.results")
    redecode = graph.Wire(
        "routed_switching.redecode", f"{machines.STRONG_POOL}.escalated"
    )
    with_edge = routed.with_part(switching)
    sharing = with_edge.with_wires(results, redecode)

    layout = laid_out(sharing)
    pool_names = [plan.names.seat for plan in layout.pools]
    routed_id = window_records.PoolId(_ROUTED_POOL)
    routed_edge = layout.edge_leaving(routed_id)

    assert pool_names == [
        machines.WEAK_POOL,
        _ROUTED_POOL,
        machines.STRONG_POOL,
        DEEP_POOL,
    ]
    assert routed_edge.hops.target.word == machines.STRONG_POOL


def test_a_pool_reached_by_no_edge_and_no_router_is_refused():
    routed = _routed_chain()
    cut = routed.without_part(DEEP_SWITCHING)

    with pytest.raises(graph.GraphError, match="no escalate edge and no"):
        laid_out(cut)


def test_a_chains_task_is_identified_by_its_graph():
    """A pool past the entry edge is part of the id."""
    chain = chain_machine()
    task = collect.Task("chain", chain)
    two_pool_machine = _two_pools()
    two_pools = collect.Task("two_pools", two_pool_machine)

    assert task.layout is not None
    assert task.strong_id() != two_pools.strong_id()


_ROUTED_POOL = "fpga_b"
_ROUTED_BY_PATCH = routers.ByPatch.Settings(
    pool_by_patch=((0, machines.WEAK_POOL),)
)


def _routed_chain() -> graph.Machine:
    """The chain whose weak store feeds a router to weak and fpga_b."""
    chain = chain_machine()
    weak = chain.part(machines.WEAK_POOL)
    return machines.with_router(
        chain, _ROUTED_BY_PATCH, ((_ROUTED_POOL, weak.settings),)
    )


def _pool_seeds(layout) -> list:
    return [
        (plan.names.unit_seed, plan.names.manager_seed) for plan in layout.pools
    ]


def _policy_seeds(layout) -> list:
    return [plan.policy_seed for plan in layout.edges]


def _store_per_pool_chain() -> graph.Machine:
    """The chain whose router feeds weak's store and fpga_b's own."""
    chain = chain_machine()
    weak = chain.part(machines.WEAK_POOL)
    return machines.with_store_per_pool(
        chain, _ROUTED_BY_PATCH, ((_ROUTED_POOL, weak.settings),)
    )


def test_a_router_before_the_stores_gives_each_entry_pool_a_written_store():
    stored = _store_per_pool_chain()
    layout = laid_out(stored)
    routed_id = window_records.PoolId(_ROUTED_POOL)
    routed = layout.pool(routed_id)
    routed_store = layout.store(routed.store)
    entry_stores = layout.entry_stores()

    assert entry_stores == (machines.WEAK_STORE, "fpga_b_buffer")
    assert routed_store.intake.shape == receiver_module.WRITTEN
    assert routed_store.other_readers == ()
    assert routed_store.intake.controller_write.path_name == (
        "controller_to_fpga_b_buffer"
    )


def test_a_router_that_feeds_both_a_store_and_a_pool_is_refused():
    stored = _store_per_pool_chain()
    weak = stored.part(machines.WEAK_POOL)
    loose = graph.Part("loose", weak.settings)
    grown = stored.with_part(loose)
    wire = graph.Wire(f"{machines.ROUTER}.windows", "loose.rounds")
    mixed = grown.with_wires(wire)

    with pytest.raises(graph.GraphError, match="both stores and pools"):
        laid_out(mixed)


def test_a_router_that_feeds_a_store_no_pool_reads_is_refused():
    stored = _store_per_pool_chain()
    unread = stored.without_part(_ROUTED_POOL)

    with pytest.raises(graph.GraphError) as refusal:
        laid_out(unread)

    assert str(refusal.value) == (
        "the router feeds the store fpga_b_buffer, which no pool reads: a "
        "router before the stores feeds one store per entry pool"
    )


def test_a_device_shared_by_a_sites_pools_is_laid_out_at_that_site():
    chain = chain_machine()
    slots = execution_slots.ExecutionSlots.Settings(2)
    device = graph.Part("gpu", slots, fabrics.HOST)
    shared = chain.with_part(device)

    layout = laid_out(shared)

    (device,) = layout.devices
    assert (device.name, device.site) == ("gpu", fabrics.HOST)
    assert device.settings.slot_count == 2


def test_a_second_device_at_one_site_is_refused():
    chain = chain_machine()
    slots = execution_slots.ExecutionSlots.Settings(2)
    gpu = graph.Part("gpu", slots, fabrics.HOST)
    cpu = graph.Part("cpu", slots, fabrics.HOST)
    shared = chain.with_part(gpu)
    doubled = shared.with_part(cpu)

    with pytest.raises(graph.GraphError, match="another device already sits"):
        laid_out(doubled)


def test_a_device_at_a_site_with_no_pool_is_refused():
    chain = chain_machine()
    slots = execution_slots.ExecutionSlots.Settings(2)
    device = graph.Part("gpu", slots, "rack")
    lone = chain.with_part(device)

    with pytest.raises(graph.GraphError, match="where no pool sits"):
        laid_out(lone)


def test_a_pool_the_router_feeds_and_an_edge_reaches_is_refused():
    routed = _routed_chain()
    both = machines.with_escalation_to(
        routed,
        DEEP_POOL,
        _ROUTED_POOL,
        DEEP_SWITCHING_SLOT,
        switching_name="deep_to_fpga_b",
    )

    with pytest.raises(graph.GraphError, match="fed by the router and reached"):
        laid_out(both)


def test_the_tier_measures_over_two_entry_pools_are_refused():
    routed = _routed_chain()
    layout = laid_out(routed)

    with pytest.raises(NotImplementedError, match="over 2 entry pools"):
        layout.tier_pools()
