"""The switching slot's record and the decoder slots it switches between.

SwitchingSettings (decsim/escalation/settings.py) is Toshio et al.
2510.25222 Sec. III A's switch, weak first and strong on low
confidence, a switching part joining two pools of a machine's graph
(decsim/topology/graph_layout.py). The laws here are the record's own
refusals and the build's stop when the graph lacks a pool it needs.
"""

import dataclasses

import pytest

import decsim.confidence.complementary as complementary
import decsim.escalation.deadlines as deadlines
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.topology.graph as graph
import tests.escalation.test_strong_window_shapes as shape_tests


def _switching(**changes) -> escalation_settings.SwitchingSettings:
    confidence = complementary.ComplementaryGap.Settings()
    threshold = threshold_sources.FixedThreshold.Settings(
        threshold_decibels=20.0
    )
    return escalation_settings.SwitchingSettings(
        confidence=confidence, threshold=threshold, **changes
    )


def test_a_switching_record_refuses_a_cost_by_its_own_field():
    with pytest.raises(ValueError, match="threshold_cycles"):
        _switching(threshold_cycles=-1)


def test_a_switching_record_refuses_a_strong_answer_route_it_lacks():
    with pytest.raises(ValueError) as refusal:
        _switching(strong_answer_route="by_courier")
    assert str(refusal.value) == (
        "switching.strong_answer_route must be one of "
        "('direct', 'through_weak_chip'), got 'by_courier'"
    )


def test_a_deadline_beside_both_decoders_started_at_once_is_refused():
    """A rescue starts the target on its trigger; Step 1 starts it always."""
    deadline = deadlines.Deadline.Settings(slack_microseconds=10.0)

    with pytest.raises(ValueError, match="run_both_at_once"):
        _switching(deadline=deadline, run_both_at_once=True)


def test_a_deadline_beside_an_absorbing_window_is_refused():
    """A rescue starts the target early; a double window starts it late."""
    deadline = deadlines.Deadline.Settings(slack_microseconds=10.0)
    double_window = strong_window_shapes.DoubleWindow.Settings()

    with pytest.raises(ValueError, match="double_window"):
        _switching(deadline=deadline, strong_window=double_window)


def test_a_route_through_the_weak_chip_with_no_clock_is_refused():
    """The chip's commit is one cycle, and a run with no clock has none."""
    gate = shape_tests.gate_switching()
    switching_part = gate.part(machines.SWITCHING)
    switching = dataclasses.replace(
        switching_part.settings, strong_answer_route="through_weak_chip"
    )
    through_chip = gate.with_record(machines.SWITCHING, switching)
    clockless = dataclasses.replace(through_chip, clock=None)

    with pytest.raises(ValueError) as refusal:
        machine_module.Machine.build(clockless, 0)
    assert str(refusal.value) == (
        "switching.strong_answer_route 'through_weak_chip' commits on the "
        "source chip in clock cycles, so it needs switching.clock or the "
        "machine's clock"
    )


def test_two_pools_without_switching_are_refused():
    """With no switching every window decodes on one pool.

    The strong pool would be built and never handed a window, so the
    run would go on as if it were not there.
    """
    gate = shape_tests.gate_switching()
    two_pools = gate.without_part(machines.SWITCHING)

    with pytest.raises(graph.GraphError, match="reached by no escalate edge"):
        machine_module.Machine.build(two_pools, 0)


@pytest.mark.parametrize(
    "gone, sentence",
    [
        (machines.WEAK_POOL, "switching.results hears no pool"),
        (machines.STRONG_POOL, "switching.redecode reaches no pool"),
    ],
)
def test_switching_with_a_pool_gone_is_refused_before_the_run(gone, sentence):
    """The switching part's wire to the missing pool went with it."""
    gate = shape_tests.gate_switching()
    one_pool = gate.without_part(gone)

    with pytest.raises(graph.GraphError, match=sentence):
        machine_module.Machine.build(one_pool, 0)
