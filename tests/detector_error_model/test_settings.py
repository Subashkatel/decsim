"""The detection_events section: its seats, its clock and its cost law.

The cost is a pipelined stage's, Yang et al. 2605.04892 lines 1273-1275:
a fixed latency once, then one rate for every round after the first.
"""

import pytest

import decsim.config as config
import decsim.detector_error_model.settings as event_settings
import decsim.topology.graph_layout as graph_layout
import tests.escalation.declared_fabric as declared_fabric

FRIDGE_CLOCK = config.Clock.from_megahertz(250.0)


def _switching_layout():
    """The declared weak-primary switching graph, laid out at its task."""
    graph = declared_fabric.switching_graph(rounds=6, escalated_windows=())
    at_task = graph.at_task()
    return graph_layout.layout_of(at_task)


def test_n_rounds_formed_together_cost_the_latency_once_then_the_rate():
    settings = event_settings.DetectionEventSettings(
        formed_at=("weak",),
        clock=FRIDGE_CLOCK,
        latency_cycles=5,
        cycles_per_round=1,
    )

    assert settings.cycles_for(0) == 0
    assert settings.cycles_for(1) == 5
    assert settings.cycles_for(7) == 11


def test_a_seat_named_twice_is_refused():
    formed_at = ("controller", "controller")

    with pytest.raises(ValueError, match="names a seat twice"):
        event_settings.DetectionEventSettings(formed_at=formed_at)


@pytest.mark.parametrize(
    "formed_at, seat",
    [
        (("controller",), None),
        (("weak_buffer",), None),
        (("weak", "strong"), "strong"),
        (("weak", "strong_buffer"), "strong_buffer"),
    ],
)
def test_the_seat_past_the_weak_buffer_that_forms_is_named(formed_at, seat):
    layout = _switching_layout()
    landed_seats = layout.landed_seats()
    settings = event_settings.DetectionEventSettings(formed_at=formed_at)

    assert settings.landed_seat(landed_seats) == seat


@pytest.mark.parametrize("key", ["latency_cycles", "cycles_per_round"])
def test_a_negative_formation_cycle_count_is_refused_by_its_name(key):
    """A negative count would form a round's events before it arrived."""
    sentence = f"detection_events.{key} must not be negative"

    with pytest.raises(ValueError, match=sentence):
        event_settings.DetectionEventSettings(**{key: -1})
