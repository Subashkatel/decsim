"""A graph of a new shape, laid out (topology/graph_layout.py).

The referent is a chain written by hand: the weak base escalating to
the strong base's pool, which escalates on to a third pool on the
host. Its pools, stores and edges go by their parts' names, entry first;
each hop is named by the parts at its ends and priced on the card; the
task is identified by the graph itself and reads back from YAML as the
same graph, with the same id.
"""

import pytest

import decsim.experiments.collect as collect
import decsim.machine as machine_module
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
import tests.declared_run as declared_run
import tests.escalation.test_strong_window_shapes as shape_tests

DEEP_POOL = "deep"
DEEP_STORE = "deep_buffer"
DEEP_SWITCHING = "deep_switching"
WORKLOAD = "workload"


def _two_pools() -> graph.Machine:
    weak_base = machines.weak_baseline(3, 0.001, 1.0)
    switching = shape_tests.switching_slot(declared_run.DOUBLE_WINDOW)
    return machines.with_strong_tier(
        weak_base, machines.STRONG_BASELINE_POOL, switching
    )


def _chain() -> graph.Machine:
    """The strong pool escalating on to a deep pool on the host."""
    two_pools = _two_pools()
    deep_switching = shape_tests.switching_slot(declared_run.REDO_WINDOW)
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


def _layout(machine: graph.Machine):
    settings = graph_layout.entry_settings(machine)
    facts = settings.task_facts()
    return graph_layout.layout_of(machine, facts)


def test_a_chain_is_a_new_shape():
    chain = _chain()

    two_pools = _two_pools()

    assert not legacy.holds(chain)
    assert legacy.holds(two_pools)


def test_a_chains_pools_go_by_their_parts_entry_first():
    chain = _chain()
    layout = _layout(chain)

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


def test_each_hop_is_named_by_the_parts_at_its_ends():
    chain = _chain()
    layout = _layout(chain)

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
    """An escalated window's rounds come from the entry store (stage 4)."""
    chain = _chain()
    layout = _layout(chain)

    entry_store = layout.stores[0]
    deep_store = layout.store(DEEP_STORE)
    deep_edge = layout.edges[1]

    assert entry_store.intake.shape == receiver_module.WRITTEN
    assert deep_store.intake.shape == receiver_module.LANDED
    assert deep_store.intake.escalation.path_name == "strong_to_deep"
    assert deep_edge.carried_from == "weak_buffer"


def test_a_round_reaches_each_pool_through_one_path_of_seats():
    chain = _chain()
    layout = _layout(chain)

    paths = layout.formation_paths()

    assert paths == (
        ("controller", "weak_buffer", "weak"),
        ("controller", "weak_buffer", "strong_buffer", "strong"),
        ("controller", "weak_buffer", DEEP_STORE, DEEP_POOL),
    )


def test_the_card_prices_the_graphs_own_hops_beside_the_legacy_twelve():
    chain = _chain()
    settings = graph_layout.entry_settings(chain)

    card = settings.links
    names = card.path_names()

    assert names[:12] == transfer_records.LEGACY_PATHS
    assert "strong_to_deep" in names
    assert "deep_to_frame" in names
    assert card.path_settings("deep_to_deep") is not None


def test_the_entry_settings_hold_the_entry_edge():
    chain = _chain()

    settings = graph_layout.entry_settings(chain)

    assert settings.weak_decoder == machines.WEAK_BASELINE_POOL
    assert settings.strong_decoder == machines.STRONG_BASELINE_POOL
    switching = chain.part("switching")
    assert settings.switching == switching.settings


def test_pools_no_edge_joins_are_refused():
    chain = _chain()
    cut = chain.without_part(DEEP_SWITCHING)

    with pytest.raises(graph.GraphError, match="reached by no escalate edge"):
        graph_layout.entry_settings(cut)


def test_a_chains_task_is_identified_by_its_graph():
    """Its entry edge's record is a legacy one, whose id it never takes."""
    chain = _chain()
    task = collect.Task("chain", chain)
    two_pools = collect.Task("two_pools", task.machine)

    assert task.layout is not None
    assert task.strong_id() != two_pools.strong_id()


def test_a_chain_read_back_from_its_file_keeps_its_id():
    chain = _chain()
    workload = chain.part(WORKLOAD)
    text = machine_file.dumps(chain)

    read = machine_file.loads(text, python_parts={WORKLOAD: workload})

    first = collect.Task("chain", chain)
    again = collect.Task("chain", read)
    assert read == chain
    assert again.strong_id() == first.strong_id()


def test_a_chain_waits_for_stage_4g_to_run():
    chain = _chain()
    task = collect.Task("chain", chain)

    with pytest.raises(NotImplementedError, match=r"stage 4g"):
        machine_module.Machine.build(task.machine, 0, layout=task.layout)
