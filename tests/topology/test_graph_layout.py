"""A graph of a new shape, laid out (topology/graph_layout.py).

The referent is a chain written by hand: the weak base escalating to a
union-find pool on the host, whose cluster gap escalates on to the
strong base's pool, a third. Its pools, stores and edges go by their
parts' names, entry first; each hop is named by the parts at its ends
and priced on the card; the task is identified by the graph itself and
reads back from YAML as the same graph, with the same id.
"""

import dataclasses

import pytest

import decsim.build.layout as layout_module
import decsim.confidence.cluster as cluster
import decsim.decoders.union_find.cycle_count as cycle_count
import decsim.decoders.union_find.decoder as union_find
import decsim.escalation.settings as escalation_settings
import decsim.escalation.threshold_sources as threshold_sources
import decsim.experiments.collect as collect
import decsim.presets.machines as machines
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer
import decsim.syndrome_buffer.syndrome_round_receiver as receiver_module
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph
import decsim.topology.graph_layout as graph_layout
import decsim.topology.legacy as legacy
import decsim.topology.machine_file as machine_file
import decsim.windows.routers as routers
import tests.declared_run as declared_run
import tests.escalation.test_strong_window_shapes as shape_tests

DEEP_POOL = "deep"
DEEP_STORE = "deep_buffer"
DEEP_SWITCHING = "deep_switching"
WORKLOAD = "workload"
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
    store = syndrome_buffer.SyndromeBufferSettings()
    parts = (
        graph.Part(DEEP_STORE, store, fabrics.HOST),
        graph.Part(DEEP_POOL, machines.STRONG_BASELINE_POOL, fabrics.HOST),
        graph.Part(DEEP_SWITCHING, deep_switching),
    )
    wires = (
        graph.Wire("controller.rounds", f"{DEEP_STORE}.rounds"),
        graph.Wire(f"{DEEP_STORE}.reads", f"{DEEP_POOL}.rounds"),
        graph.Wire(f"{DEEP_POOL}.corrections", "frame.corrections"),
        graph.Wire("strong.escalate", f"{DEEP_SWITCHING}.results"),
        graph.Wire(f"{DEEP_SWITCHING}.redecode", f"{DEEP_POOL}.escalated"),
    )
    grown = two_pools
    for part in parts:
        grown = grown.with_part(part)
    return grown.with_wires(*wires)


def laid_out(machine: graph.Machine):
    """The machine's layout, from its entry edge's facts."""
    entry = graph_layout.entry_settings(machine)
    settings = entry.at_task()
    facts = settings.task_facts()
    return graph_layout.layout_of(machine, facts)


def _two_pools(
    entry_window: escalation_settings.StrongWindowSettings = (
        declared_run.DOUBLE_WINDOW
    ),
) -> graph.Machine:
    weak_base = machines.weak_baseline(3, 0.001, 1.0)
    switching = shape_tests.switching_slot(entry_window)
    return machines.with_strong_tier(weak_base, MIDDLE_POOL, switching)


def test_a_chain_is_a_new_shape():
    chain = chain_machine()

    two_pools = _two_pools()

    assert not legacy.holds(chain)
    assert legacy.holds(two_pools)


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
    two_pool_settings = legacy.resolved(two_pool_machine)
    two_pools = layout_module.legacy_layout(two_pool_settings)

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
    assert deep_store.intake.escalation.path_name == "strong_to_deep"
    assert deep_store.intake.escalation.source_name == "strong_buffer"
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


def test_the_card_prices_the_graphs_own_hops_beside_the_legacy_twelve():
    chain = chain_machine()
    settings = graph_layout.entry_settings(chain)

    card = settings.links
    names = card.path_names()

    assert names[:12] == transfer_records.LEGACY_PATHS
    assert "strong_to_deep" in names
    assert "deep_to_frame" in names
    assert card.path_settings("deep_to_deep") is not None


def test_the_entry_settings_hold_the_entry_edge():
    chain = chain_machine()

    settings = graph_layout.entry_settings(chain)

    assert settings.weak_decoder == machines.WEAK_BASELINE_POOL
    assert settings.strong_decoder == MIDDLE_POOL
    switching = chain.part("switching")
    assert settings.switching == switching.settings


def test_pools_no_edge_joins_are_refused():
    chain = chain_machine()
    cut = chain.without_part(DEEP_SWITCHING)

    with pytest.raises(graph.GraphError, match="reached by no escalate edge"):
        graph_layout.entry_settings(cut)


def test_a_router_makes_each_pool_it_feeds_an_entry_pool():
    routed = _routed_chain()

    layout = laid_out(routed)
    entry_names = [plan.names.seat for plan in layout.entry_pools]

    assert entry_names == [machines.WEAK_POOL, _ROUTED_POOL]
    assert layout.router.rule == _ROUTED_BY_PATCH


def test_an_edge_from_an_entry_pool_past_the_first_is_refused():
    """Only the first entry pool escalates yet (stage 5)."""
    routed = _routed_chain()
    switching = graph.Part("routed_switching", DEEP_SWITCHING_SLOT)
    without_deep_edge = routed.without_part(DEEP_SWITCHING)
    results = graph.Wire(f"{_ROUTED_POOL}.escalate", "routed_switching.results")
    redecode = graph.Wire("routed_switching.redecode", f"{DEEP_POOL}.escalated")
    with_edge = without_deep_edge.with_part(switching)
    escalating = with_edge.with_wires(results, redecode)

    with pytest.raises(graph.GraphError, match="not implemented yet"):
        laid_out(escalating)


def test_a_pool_reached_by_no_edge_and_no_router_is_refused():
    routed = _routed_chain()
    cut = routed.without_part(DEEP_SWITCHING)

    with pytest.raises(graph.GraphError, match="no escalate edge and no"):
        graph_layout.entry_settings(cut)


def test_a_chains_task_is_identified_by_its_graph():
    """Its entry edge's record is a legacy one, whose id it never takes."""
    chain = chain_machine()
    task = collect.Task("chain", chain)
    two_pools = collect.Task("two_pools", task.machine)

    assert task.layout is not None
    assert task.strong_id() != two_pools.strong_id()


def test_a_chain_read_back_from_its_file_keeps_its_id():
    chain = chain_machine()
    workload = chain.part(WORKLOAD)
    text = machine_file.dumps(chain)

    read = machine_file.loads(text, python_parts={WORKLOAD: workload})

    first = collect.Task("chain", chain)
    again = collect.Task("chain", read)
    assert read == chain
    assert again.strong_id() == first.strong_id()


def test_a_routed_machine_read_back_from_its_file_keeps_its_id():
    routed = _routed_chain()
    workload = routed.part(WORKLOAD)
    text = machine_file.dumps(routed)

    read = machine_file.loads(text, python_parts={WORKLOAD: workload})

    first = collect.Task("routed", routed)
    again = collect.Task("routed", read)
    assert read == routed
    assert again.strong_id() == first.strong_id()


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
