"""When a round copied to several stores leaves the seats.

A round goes to its own store and to each store whose windows read it
(controller/round_destinations.py); each store lets it go when its own
readers are done. The seats may form its events until the last store
lets it go, so they hear it once, then.
"""

import pytest

import decsim.controller.round_destinations as round_destinations
import tests.controller.test_syndrome_round_sender as sender_tests


class Seats:
    """The rounds the seats were told left for good, in order."""

    def __init__(self) -> None:
        self.retired: list = []

    def retire_round(self, round_key: tuple) -> None:
        self.retired.append(round_key)


def _side(name: str) -> round_destinations.StoreSide:
    return round_destinations.StoreSide(name, None, None, f"to_{name}", False)


def _two_stores() -> tuple:
    """Destinations over a store per entry pool, telling recorded seats."""
    side_by_pool = {"weak": _side("weak_buffer"), "fpga_b": _side("b_buffer")}
    destinations = round_destinations.RoundDestinations(side_by_pool)
    seats = Seats()
    destinations.detection_events = seats
    return destinations, seats


def test_a_round_copied_to_two_stores_leaves_the_seats_when_the_last_lets_go():
    destinations, seats = _two_stores()
    copied = sender_tests.packed(3)
    destinations.note_written(copied, 2)

    destinations.retire_round(copied.round_key)
    told_after_the_first = list(seats.retired)
    destinations.retire_round(copied.round_key)

    assert told_after_the_first == []
    assert seats.retired == [copied.round_key]


def test_a_round_written_to_its_own_store_alone_leaves_the_seats_with_it():
    destinations, seats = _two_stores()
    own = sender_tests.packed(4)
    destinations.note_written(own, 1)

    destinations.retire_round(own.round_key)

    assert seats.retired == [own.round_key]


def test_a_round_leaving_a_store_it_was_never_written_to_is_a_contract_break():
    destinations, _seats = _two_stores()

    with pytest.raises(RuntimeError, match="never written to"):
        destinations.retire_round((1, 5))
