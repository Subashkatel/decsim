"""Buffer 0 access cycles, on gem5's named clock and integer-cycle law."""

import pytest

import decsim.syndrome_buffer.settings as round_store_settings


def test_a_charged_store_cost_needs_its_clock():
    with pytest.raises(
        ValueError, match="charged round_store costs need a clock"
    ):
        round_store_settings.RoundStoreSettings(read_cycles=1)
