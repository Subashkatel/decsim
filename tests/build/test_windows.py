"""Building the window side: the gap a run with no switching records.

The referent is the same machine switching at 0 dB, where no gap is
below the threshold and nothing escalates (Toshio 2510.25222 Sec. III
A): it decodes the same windows on the same weak tier and reads each
window's cluster gap, so a run alone that records the signal records
those gaps beside the same predictions.
"""

import dataclasses

import pytest

import decsim.build.windows as windows_part
import decsim.confidence.cluster as cluster
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.records.program as program_records
import decsim.windows.routers as routers
import tests.declared_run as declared_run
import tests.escalation.test_switching_mode as switching_tests
import tests.topology.test_graph_layout as layout_tests

CLUSTER_GAP = cluster.ClusterGap.Settings(walk_microseconds=0.0)


def test_a_run_alone_records_the_gaps_its_switching_twin_reads():
    switching = switching_tests.never_escalating_graph()
    without_switching = switching.without_part(machines.SWITCHING)
    without_strong = without_switching.without_part(machines.STRONG_POOL)
    weak_only = without_strong.without_part(machines.STRONG_STORE)
    recording = _with_recorded_confidence(weak_only, CLUSTER_GAP)

    twin, twin_result = switching_tests.run_shot(switching, 0)
    machine, result = switching_tests.run_shot(recording, 0)

    twin_gaps = twin.observation.confidence.windows()
    recorded_gaps = machine.observation.confidence.windows()
    assert len(recorded_gaps) == len(twin.observation.windows.windows)
    assert recorded_gaps == twin_gaps
    predictions = switching_tests.predictions(result)
    assert predictions == switching_tests.predictions(twin_result)


def test_a_recorded_confidence_on_a_switching_run_is_refused():
    switching = switching_tests.never_escalating_graph()
    recording = _with_recorded_confidence(switching, CLUSTER_GAP)
    layout = layout_tests.laid_out(recording)
    window_settings = recording.held("windows")
    with pytest.raises(
        ValueError,
        match="records a confidence on a run with no switching; a "
        "switching run records its edge's own",
    ):
        windows_part.recorded_signal(window_settings, layout)


def test_a_recorded_confidence_on_a_routed_run_is_refused():
    weak_only = declared_run.weak_only_graph()
    weak = weak_only.part(machines.WEAK_POOL)
    rule = routers.ByPatch.Settings(pool_by_patch=((0, machines.WEAK_POOL),))
    routed = machines.with_router(weak_only, rule, (("fpga_b", weak.settings),))
    recording = _with_recorded_confidence(routed, CLUSTER_GAP)
    layout = layout_tests.laid_out(recording)
    window_settings = recording.held("windows")
    with pytest.raises(
        ValueError,
        match="records the entry pool's windows, and this run routes its "
        "windows to 2 entry pools",
    ):
        windows_part.recorded_signal(window_settings, layout)


def _with_recorded_confidence(graph, confidence):
    windows = graph.held("windows")
    recorded = dataclasses.replace(windows, recorded_confidence=confidence)
    return graph.with_held("windows", recorded)


def test_a_router_leaves_an_operation_with_no_patch_and_no_rounds_alone():
    """A classical step after the memory holds no patch and sends nothing.

    The router routes rounds and idle decodes, so it has nothing of this
    operation's to route, and its table need not name it.
    """
    classical = program_records.Operation(
        2, "classical", (), predecessors=(1,), emits_detector_data=False
    )
    operations = [declared_run.memory_operation(1), classical]
    graph = declared_run.weak_only_graph(rounds=6, operations=operations)
    rule = routers.ByPatch.Settings(pool_by_patch=((1, "weak"),))
    routed = machines.with_router(graph, rule, ())
    machine = machine_module.Machine.build(routed, 0)

    result = machine.run()

    stamps = machine.observation.runtime_stamps
    assert result.terminal_status == "complete"
    assert set(stamps.body_done) == {1, 2}
