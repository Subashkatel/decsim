"""A store section: its capacity in bits, and its access cycles.

The capacity is the unit gem5 declares a packet store in, bytes on the
store itself (`rx_fifo_size = Param.MemorySize("384KiB", ...)`,
src/dev/net/Ethernet.py); the access cycles follow gem5's clock and
cycle law.
"""

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
        section, "weak_syndrome_buffer", clocks
    )

    assert settings.detection_event_cycles_per_round == 5


@pytest.mark.parametrize("capacity", [0, -8, True, 8.0])
def test_a_store_capacity_that_is_not_whole_bits_is_refused_by_name(capacity):
    clocks = config.ClockSettings.from_yaml({"fridge": 250.0})
    section = {"bits": capacity}

    with pytest.raises(
        ValueError, match="weak_syndrome_buffer.bits must be at least one bit"
    ):
        syndrome_buffer_settings.SyndromeBufferSettings.from_yaml(
            section, "weak_syndrome_buffer", clocks
        )


def test_an_unknown_key_under_a_store_section_is_refused_by_name():
    clocks = config.ClockSettings.from_yaml({"fridge": 250.0})
    section = {"rounds": 8}

    with pytest.raises(
        ValueError,
        match=r"strong_syndrome_buffer does not know \['rounds'\]",
    ):
        syndrome_buffer_settings.SyndromeBufferSettings.from_yaml(
            section, "strong_syndrome_buffer", clocks
        )


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
    section = {"kind": "syndrome_buffer", "bits": 40}

    syndrome_buffer_settings.check_strong_section_charges_nothing(section)
