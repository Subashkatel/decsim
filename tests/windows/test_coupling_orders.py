"""The coupling orders: which block of a coupling reads the other.

The coordinate order is SWIPER's (2412.05115; swiper/window_manager.py
SlidingWindowManager.process_round: a merge face is read by the patch
whose coordinates are both at least the other's).
"""

import decsim.windows.coupling_orders as coupling_orders


def test_the_block_whose_coordinates_are_both_at_least_reads_the_other():
    order = coupling_orders.CoordinateOrder()

    beside = order.orient(((0, 0), (0, 1)))
    below = order.orient(((1, 0), (0, 0)))

    assert beside == ((0, 1), (0, 0))
    assert below == ((1, 0), (0, 0))


def test_blocks_neither_of_which_is_at_least_the_other_read_nothing():
    order = coupling_orders.CoordinateOrder()

    oriented = order.orient(((0, 1), (1, 0)))

    assert oriented is None
