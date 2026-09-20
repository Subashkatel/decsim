"""The weak syndrome buffer's access cycles, on gem5's clock and cycle law."""

import pytest

import decsim.config as config
import decsim.syndrome_buffer.settings as syndrome_buffer_settings


def test_a_charged_store_cost_needs_its_clock():
    with pytest.raises(
        ValueError, match="charged weak_syndrome_buffer costs need a clock"
    ):
        syndrome_buffer_settings.SyndromeBufferSettings(read_cycles=1)


def test_the_chips_formation_charge_is_read_from_the_section():
    clocks = config.ClockSettings.from_yaml({"fridge": 250.0})
    section = {"clock": "fridge", "detection_event_cycles_per_round": 5}

    settings = syndrome_buffer_settings.SyndromeBufferSettings.from_yaml(
        section, clocks
    )

    assert settings.detection_event_cycles_per_round == 5


@pytest.mark.parametrize(
    "cost",
    ["write_cycles", "read_cycles", "detection_event_cycles_per_round"],
)
def test_a_cost_on_the_strong_syndrome_buffer_is_refused(cost):
    section = {cost: 3}

    with pytest.raises(
        ValueError,
        match=f"strong_syndrome_buffer.{cost} is a cost of the weak",
    ):
        syndrome_buffer_settings.check_strong_section_charges_nothing(section)


def test_a_strong_syndrome_buffer_with_no_cost_is_accepted():
    section = {"kind": "syndrome_buffer", "rounds": 40}

    syndrome_buffer_settings.check_strong_section_charges_nothing(section)
