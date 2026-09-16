"""The weak syndrome buffer's access cycles, on gem5's clock and cycle law."""

import pytest

import decsim.syndrome_buffer.settings as syndrome_buffer_settings


def test_a_charged_store_cost_needs_its_clock():
    with pytest.raises(
        ValueError, match="charged weak_syndrome_buffer costs need a clock"
    ):
        syndrome_buffer_settings.SyndromeBufferSettings(read_cycles=1)
