"""The burst flag record: the first round fired on from a given round."""

import decsim.observe.burst_flags as burst_flags


def test_the_first_flag_is_the_first_round_at_or_after_the_start():
    flags = _flags_on(3, 9, 10)

    assert flags.first_flag_from(1) == 3
    assert flags.first_flag_from(4) == 9
    assert flags.first_flag_from(9) == 9


def test_no_round_flagged_from_the_start_on_is_round_zero():
    flags = _flags_on(3)

    assert flags.first_flag_from(4) == 0


def _flags_on(*rounds):
    flags = burst_flags.BurstFlags()
    for round_index in rounds:
        flags.round_flagged(1, round_index)
    return flags
