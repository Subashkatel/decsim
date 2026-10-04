"""A store's settings: its capacity in bits.

The capacity is the unit gem5 declares a packet store in, bytes on the
store itself (`rx_fifo_size = Param.MemorySize("384KiB", ...)`,
src/dev/net/Ethernet.py).
"""

import pytest

import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module


@pytest.mark.parametrize("capacity", [0, -8, True, 8.0])
def test_a_store_capacity_that_is_not_whole_bits_is_refused_by_name(capacity):
    with pytest.raises(
        ValueError, match="syndrome_buffer.bits must be at least one bit"
    ):
        syndrome_buffer_module.SyndromeBufferSettings(bits=capacity)
