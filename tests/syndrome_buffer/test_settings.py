"""A store's settings: its capacity in bits, and its access cycles.

The capacity is the unit gem5 declares a packet store in, bytes on the
store itself (`rx_fifo_size = Param.MemorySize("384KiB", ...)`,
src/dev/net/Ethernet.py); the access cycles follow gem5's clock and
cycle law.
"""

import pytest

import decsim.config as config
import decsim.engine as engine_module
import decsim.syndrome_buffer.settings as syndrome_buffer_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module


def test_a_charged_store_cost_needs_its_clock():
    settings = syndrome_buffer_module.SyndromeBufferSettings(read_cycles=1)
    engine = engine_module.Engine()

    with pytest.raises(
        ValueError, match="charged weak_syndrome_buffer costs need a clock"
    ):
        syndrome_buffer_module.SyndromeBuffer(settings, engine)


def test_a_store_sections_keys_are_its_records_fields():
    clocks = config.ClockSettings.from_yaml({"fridge": 250.0})
    section = {"clock": "fridge", "write_cycles": 2, "read_cycles": 3}

    settings = syndrome_buffer_settings.from_yaml(
        section, "weak_syndrome_buffer", clocks
    )

    fridge = config.Clock(4000)
    expected = syndrome_buffer_module.SyndromeBufferSettings(
        clock=fridge, write_cycles=2, read_cycles=3
    )
    assert settings == expected


@pytest.mark.parametrize("capacity", [0, -8, True, 8.0])
def test_a_store_capacity_that_is_not_whole_bits_is_refused_by_name(capacity):
    with pytest.raises(
        ValueError, match="syndrome_buffer.bits must be at least one bit"
    ):
        syndrome_buffer_module.SyndromeBufferSettings(bits=capacity)


def test_a_store_kind_that_names_no_record_is_refused():
    clocks = config.ClockSettings.from_yaml({"fridge": 250.0})
    section = {"kind": "tape"}

    with pytest.raises(
        ValueError, match="weak_syndrome_buffer.kind 'tape' is not a row"
    ):
        syndrome_buffer_settings.from_yaml(
            section, "weak_syndrome_buffer", clocks
        )


def test_an_unknown_key_under_a_store_section_is_refused_by_name():
    clocks = config.ClockSettings.from_yaml({"fridge": 250.0})
    section = {"rounds": 8}

    with pytest.raises(
        ValueError,
        match=r"strong_syndrome_buffer does not know \['rounds'\]",
    ):
        syndrome_buffer_settings.from_yaml(
            section, "strong_syndrome_buffer", clocks
        )
