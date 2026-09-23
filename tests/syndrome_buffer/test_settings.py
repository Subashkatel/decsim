"""A store section: its capacity in bits, and its access cycles.

The capacity is the unit gem5 declares a packet store in, bytes on the
store itself (`rx_fifo_size = Param.MemorySize("384KiB", ...)`,
src/dev/net/Ethernet.py); the access cycles follow gem5's clock and
cycle law.
"""

import dataclasses

import pytest

import decsim.config as config
import decsim.syndrome_buffer.settings as syndrome_buffer_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module

BUFFER_ROWS = syndrome_buffer_module.SYNDROME_BUFFERS


def test_a_charged_store_cost_needs_its_clock():
    with pytest.raises(
        ValueError, match="charged weak_syndrome_buffer costs need a clock"
    ):
        syndrome_buffer_settings.SyndromeBufferSettings(read_cycles=1)


def test_the_chips_formation_charge_is_read_from_the_section():
    clocks = config.ClockSettings.from_yaml({"fridge": 250.0})
    section = {"clock": "fridge", "detection_event_cycles_per_round": 5}

    settings = syndrome_buffer_settings.SyndromeBufferSettings.from_yaml(
        section, "weak_syndrome_buffer", clocks, BUFFER_ROWS
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
            section, "weak_syndrome_buffer", clocks, BUFFER_ROWS
        )


class _BankedBuffer(syndrome_buffer_module.SyndromeBuffer):
    """A store with a key of its own, for the section's split."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        bank_count: int = 1

        @classmethod
        def from_yaml(cls, section):
            return cls(**section)


def test_a_buffer_rows_own_key_reaches_its_settings():
    """The section keeps its shared keys and hands the row the rest."""
    clocks = config.ClockSettings.from_yaml({"fridge": 250.0})
    rows = {"banked": _BankedBuffer}
    section = {"kind": "banked", "bits": 400, "bank_count": 4}

    settings = syndrome_buffer_settings.SyndromeBufferSettings.from_yaml(
        section, "weak_syndrome_buffer", clocks, rows
    )

    assert settings.bits == 400
    assert settings.row_settings == _BankedBuffer.Settings(bank_count=4)


def test_an_unknown_key_under_a_store_section_is_refused_by_name():
    clocks = config.ClockSettings.from_yaml({"fridge": 250.0})
    section = {"rounds": 8}

    with pytest.raises(
        ValueError,
        match=r"strong_syndrome_buffer does not know \['rounds'\]",
    ):
        syndrome_buffer_settings.SyndromeBufferSettings.from_yaml(
            section, "strong_syndrome_buffer", clocks, BUFFER_ROWS
        )


@pytest.mark.parametrize(
    "key",
    [
        "clock",
        "write_cycles",
        "read_cycles",
        "detection_event_cycles_per_round",
    ],
)
@pytest.mark.parametrize("value", [3, 0, "3x"])
def test_a_cost_or_its_clock_on_the_strong_syndrome_buffer_is_refused(
    key, value
):
    """Whatever the value: the strong store reads none of these keys."""
    section = {key: value}

    with pytest.raises(
        ValueError,
        match=f"strong_syndrome_buffer.{key} belongs to the weak",
    ):
        syndrome_buffer_settings.check_strong_section_charges_nothing(section)


def test_a_strong_syndrome_buffer_with_no_cost_is_accepted():
    section = {"kind": "syndrome_buffer", "bits": 40}

    syndrome_buffer_settings.check_strong_section_charges_nothing(section)
