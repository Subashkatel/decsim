"""The hold bookkeeping of a round store: a round lives while a hold names it.

The laws are the store's own (decsim/syndrome_buffer/round_holds.py): a
round loses its last holder exactly once, a hold moves without freeing
its rounds, and a token is registered once in its life.
"""

import pytest

import decsim.syndrome_buffer.round_holds as round_holds


def record(*round_keys) -> round_holds.HoldRecord:
    operation_ids = frozenset(round_key[0] for round_key in round_keys)
    return round_holds.HoldRecord(tuple(round_keys), operation_ids)


def test_a_round_stays_held_until_its_last_holder_releases():
    holds = round_holds.RoundHolds()
    one_round = record((1, 1))
    holds.register("first", one_round)
    holds.register("second", one_round)

    orphaned_by_first = holds.release("first")
    still_held = holds.is_held((1, 1))
    orphaned_by_second = holds.release("second")

    assert orphaned_by_first == []
    assert still_held is True
    assert orphaned_by_second == [(1, 1)]
    assert holds.is_held((1, 1)) is False


def test_replacing_a_hold_reports_the_rounds_it_no_longer_names():
    holds = round_holds.RoundHolds()
    first_three = record((1, 1), (1, 2), (1, 3))
    holds.register("window", first_three)

    next_three = record((1, 2), (1, 3), (1, 4))
    orphaned = holds.replace("window", next_three)

    assert orphaned == [(1, 1)]
    assert holds.round_keys_of("window") == ((1, 2), (1, 3), (1, 4))
    assert holds.is_held((1, 4)) is True


def test_a_transferred_hold_keeps_its_rounds_alive():
    holds = round_holds.RoundHolds()
    two_rounds = record((1, 1), (1, 2))
    holds.register("window", two_rounds)

    holds.transfer("window", "request")

    assert holds.is_live("window") is False
    assert holds.is_live("request") is True
    assert holds.is_held((1, 1)) is True
    assert holds.round_keys_of("request") == ((1, 1), (1, 2))


def test_a_token_registered_before_is_refused():
    holds = round_holds.RoundHolds()
    first_round = record((1, 1))
    holds.register("window", first_round)
    holds.release("window")

    second_round = record((1, 2))
    with pytest.raises(RuntimeError, match="'window' was registered before"):
        holds.register("window", second_round)


def test_a_released_token_is_forgotten_once_its_operations_close():
    holds = round_holds.RoundHolds()
    first_round = record((1, 1))
    holds.register("window", first_round)
    holds.release("window")

    holds.forget_released_outside({2})

    assert "window" not in holds.released_holders
