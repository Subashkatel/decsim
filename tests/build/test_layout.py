"""A machine's layout: the paths its rounds take and the factory's pool.

A router's entry pools share one store, each reading it over a hop of
its own, as VQD's decoders take their syndromes from the Arbiter (Maurya
and Tannu 2406.17995v2 Sec. 5). No paper sets decoders aside for the
magic-state factory (DecoNet 2504.11805 Sec. VIII-F; VQD Sec. 5.3), so a
run of several entry pools names the one that serves it.
"""

import pytest

import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.windows.routers as routers
import tests.declared_run as declared_run
import tests.topology.test_graph_layout as layout_tests

_WEAK_ONLY_ROUTED = routers.ByPatch.Settings(
    pool_by_patch=((0, machines.WEAK_POOL),)
)


def _routed_chain(factory_pool=None):
    """The weak and fpga_b pools on weak's store; weak climbs to deep."""
    chain = layout_tests.chain_machine(entry_window=declared_run.REDO_WINDOW)
    weak = chain.part(machines.WEAK_POOL)
    return machines.with_router(
        chain,
        _WEAK_ONLY_ROUTED,
        (("fpga_b", weak.settings),),
        factory_pool=factory_pool,
    )


def test_two_entry_pools_on_one_store_reach_their_shared_target_once():
    """fpga_b climbs weak's chain from the same store: no path twice."""
    routed = _routed_chain()
    switching = routed.part(machines.SWITCHING)
    both = machines.with_escalation_to(
        routed, "fpga_b", machines.STRONG_POOL, switching.settings
    )
    layout = layout_tests.laid_out(both)

    paths = layout.formation_paths()

    assert paths == (
        ("controller", "weak_buffer", "weak"),
        ("controller", "weak_buffer", "strong_buffer", "strong"),
        ("controller", "weak_buffer", "strong_buffer", "deep_buffer", "deep"),
        ("controller", "weak_buffer", "fpga_b"),
    )


def test_a_factory_pool_the_machine_does_not_hold_is_refused():
    routed = _routed_chain(factory_pool="nowhere")

    with pytest.raises(ValueError) as refusal:
        machine_module.Machine.build(routed, 0)

    assert str(refusal.value) == (
        "router.factory_pool names 'nowhere', which is no pool of this "
        "machine; its pools are ['weak', 'fpga_b', 'strong', 'deep']"
    )
