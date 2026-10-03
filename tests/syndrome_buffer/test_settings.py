"""A store's settings: its capacity in bits, and its access cycles.

The capacity is the unit gem5 declares a packet store in, bytes on the
store itself (`rx_fifo_size = Param.MemorySize("384KiB", ...)`,
src/dev/net/Ethernet.py); the access cycles follow gem5's clock and
cycle law.
"""

import pytest

import decsim.engine as engine_module
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module


def test_a_charged_store_cost_needs_its_clock():
    settings = syndrome_buffer_module.SyndromeBufferSettings(read_cycles=1)
    engine = engine_module.Engine()

    with pytest.raises(
        ValueError, match="charged weak_syndrome_buffer costs need a clock"
    ):
        syndrome_buffer_module.SyndromeBuffer(settings, engine)


@pytest.mark.parametrize("capacity", [0, -8, True, 8.0])
def test_a_store_capacity_that_is_not_whole_bits_is_refused_by_name(capacity):
    with pytest.raises(
        ValueError, match="syndrome_buffer.bits must be at least one bit"
    ):
        syndrome_buffer_module.SyndromeBufferSettings(bits=capacity)
