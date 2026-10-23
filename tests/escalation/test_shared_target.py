"""Several pools escalate to one shared target pool.

A weak decoder per patch with one strong decoder they all share is
Delfosse's lazy decoders over shared decoding units (2001.11427 Sec. V)
and Toshio's N_window weak decoders over a single strong one
(2510.25222 Sec. III C). Here patch 1 enters at weak and patch 2 at
fpga_b, both on the chip, and both escalate every window to strong on
the host. Each answer that goes back through the weak chip goes to the
chip that asked.
"""

import dataclasses

import pytest

import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.records.windows as window_records
import decsim.windows.routers as routers
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run
import tests.escalation.test_switching_mode as switching_mode


def _shared_strong(answer_route: str, factory_pool="weak"):
    """Two chip pools, by patch, each escalating every window to strong.

    The magic-state factory's decodes go to factory_pool.
    """
    weak, strong = declared_run.switching_decoders(True)
    engine = declared_run.DECLARED_ENGINE
    weak_pool = decoder_settings.DecoderPoolSettings(
        algorithm=weak, engine=engine
    )
    strong_pool = decoder_settings.DecoderPoolSettings(
        algorithm=strong, engine=engine
    )
    operations = [
        declared_run.memory_operation(1),
        declared_run.memory_operation(2),
    ]
    workload = declared_run.declared_workload(operations, 6)
    windows = window_settings.switching_windows(
        window_settings.WindowSettings(), declared_run.REDO_WINDOW
    )
    switching = declared_run.declared_switching(
        strong_window=declared_run.REDO_WINDOW
    )
    switching = dataclasses.replace(
        switching, strong_answer_route=answer_route, clock=config.Clock(4)
    )
    machine = machines.base(
        workload,
        declared_run.declared_qpu(),
        windows,
        declared_run.declared_controller(),
        declared_run.declared_frame(),
        fabric=declared_run.declared_fabric(),
    )
    machine = machines.with_pool(machine, weak_pool)
    machine = machines.with_strong_tier(machine, strong_pool, switching)
    by_patch = routers.ByPatch.Settings(
        pool_by_patch=((1, "weak"), (2, "fpga_b"))
    )
    machine = machines.with_router(
        machine, by_patch, (("fpga_b", weak_pool),), factory_pool=factory_pool
    )
    return machines.with_escalation_to(machine, "fpga_b", "strong", switching)


def _run(answer_route: str):
    machine = machine_module.Machine.build(_shared_strong(answer_route), 0)
    result = machine.run()
    return machine, result


def test_both_pools_escalate_every_window_to_the_one_strong_pool():
    machine, result = _run("direct")
    tiers = dict(declared_run.frame_tiers(machine))
    transfers = switching_mode.transfers_by_path(result)
    assert tiers == {
        (1, 0): "strong",
        (1, 1): "strong",
        (2, 0): "strong",
        (2, 1): "strong",
    }
    assert transfers["strong_to_frame"] == 4
    assert transfers["weak_to_strong"] >= 2
    assert transfers["fpga_b_to_strong"] >= 2


def test_through_the_weak_chip_each_answer_goes_down_to_the_chip_that_asked():
    _machine, result = _run("through_weak_chip")
    transfers = switching_mode.transfers_by_path(result)
    answers = {
        path: transfers.get(path, 0)
        for path in (
            "strong_to_weak",
            "strong_to_fpga_b",
            "weak_to_frame",
            "fpga_b_to_frame",
            "strong_to_frame",
        )
    }
    assert answers == {
        "strong_to_weak": 2,
        "strong_to_fpga_b": 2,
        "weak_to_frame": 2,
        "fpga_b_to_frame": 2,
        "strong_to_frame": 0,
    }


def test_one_learned_threshold_over_two_entry_edges_is_refused():
    """The task's calibrator would decide on weak's edge and not fpga_b's."""
    # the refusal reads only that the task learns a threshold
    calibrator = object()

    with pytest.raises(ValueError, match="2 edges leave entry pools"):
        machine_module.Machine.build(
            _shared_strong("direct"), 0, online_threshold=calibrator
        )


def test_the_factorys_decodes_go_to_the_pool_the_router_names():
    machine = machine_module.Machine.build(
        _shared_strong("direct", factory_pool="fpga_b"), 0
    )

    fpga_b = machine.decoders.manager_of(window_records.PoolId("fpga_b"))
    assert machine.qpu.factory.decode_queue is fpga_b


def test_two_entry_pools_with_no_factory_pool_are_refused():
    """No paper sets a default; the run names which pool serves them."""
    graph = _shared_strong("direct")
    router_part = graph.part(machines.ROUTER)
    unnamed = dataclasses.replace(router_part.settings, factory_pool=None)
    unnamed_graph = graph.with_record(machines.ROUTER, unnamed)

    with pytest.raises(ValueError, match="router.factory_pool names no pool"):
        machine_module.Machine.build(unnamed_graph, 0)
