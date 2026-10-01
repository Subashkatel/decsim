"""The detection_events section: its seats, its clock and its cost law.

The cost is a pipelined stage's, Yang et al. 2605.04892 lines 1273-1275:
a fixed latency once, then one rate for every round after the first.
"""

import pytest

import decsim.config as config
import decsim.detector_error_model.settings as event_settings

CLOCKS = config.ClockSettings({"fridge": 250.0, "room": 500.0})


def test_the_section_left_out_forms_at_the_controller_for_nothing():
    fridge = CLOCKS.clock("fridge")

    settings = event_settings.DetectionEventSettings.from_yaml(
        {}, CLOCKS, fridge
    )

    assert settings.formed_at == ("controller",)
    assert settings.clock == fridge
    assert settings.cycles_for(3) == 0


def test_a_named_clock_replaces_the_controllers():
    fridge = CLOCKS.clock("fridge")
    section = {"formed_at": ["weak_decoder"], "clock": "room"}

    settings = event_settings.DetectionEventSettings.from_yaml(
        section, CLOCKS, fridge
    )

    assert settings.clock == CLOCKS.clock("room")


def test_n_rounds_formed_together_cost_the_latency_once_then_the_rate():
    fridge = CLOCKS.clock("fridge")
    settings = event_settings.DetectionEventSettings(
        formed_at=("weak_decoder",),
        clock=fridge,
        latency_cycles=5,
        cycles_per_round=1,
    )

    assert settings.cycles_for(0) == 0
    assert settings.cycles_for(1) == 5
    assert settings.cycles_for(7) == 11


def test_a_seat_off_the_path_is_refused_with_the_seats():
    section = {"formed_at": ["workstation"]}

    with pytest.raises(ValueError) as refusal:
        event_settings.DetectionEventSettings.from_yaml(section, CLOCKS, None)

    sentence = str(refusal.value)
    assert "detection_events.formed_at names 'workstation'" in sentence
    assert "strong_syndrome_buffer" in sentence


def test_a_seat_named_twice_is_refused():
    section = {"formed_at": ["controller", "controller"]}

    with pytest.raises(ValueError, match="names a seat twice"):
        event_settings.DetectionEventSettings.from_yaml(section, CLOCKS, None)


def test_one_seat_written_bare_is_refused_as_a_list():
    section = {"formed_at": "controller"}

    with pytest.raises(ValueError, match=r"as in \[controller\]"):
        event_settings.DetectionEventSettings.from_yaml(section, CLOCKS, None)


def test_a_key_the_section_does_not_have_is_refused():
    section = {"cycles_per_job": 1}

    with pytest.raises(ValueError, match="detection_events does not know"):
        event_settings.DetectionEventSettings.from_yaml(section, CLOCKS, None)


def test_a_charged_cost_with_no_clock_is_refused():
    section = {"latency_cycles": 5}

    with pytest.raises(ValueError, match="needs the clock"):
        event_settings.DetectionEventSettings.from_yaml(section, CLOCKS, None)


@pytest.mark.parametrize(
    "formed_at, seat",
    [
        (("controller",), None),
        (("weak_syndrome_buffer",), None),
        (("weak_decoder", "strong_decoder"), "strong_decoder"),
        (("weak_decoder", "strong_syndrome_buffer"), "strong_syndrome_buffer"),
    ],
)
def test_the_seat_past_the_weak_buffer_that_forms_is_named(formed_at, seat):
    settings = event_settings.DetectionEventSettings(formed_at=formed_at)

    assert settings.strong_side_seat() == seat
