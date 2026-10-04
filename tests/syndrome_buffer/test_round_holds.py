"""A syndrome buffer's hold bookkeeping: a round lives while a hold names it.

The laws are the store's own (decsim/syndrome_buffer/round_holds.py): a
round loses its last holder exactly once, and a hold moves without freeing
its rounds.
"""

import decsim.syndrome_buffer.round_holds as round_holds


def record(*round_keys) -> round_holds.HoldRecord:
    operation_ids = frozenset(round_key[0] for round_key in round_keys)
    return round_holds.HoldRecord(tuple(round_keys), operation_ids)


def test_a_released_token_is_forgotten_once_its_operations_close():
    holds = round_holds.RoundHolds()
    first_round = record((1, 1))
    holds.register("window", first_round)
    holds.release("window")

    holds.forget_released_outside({2})

    assert "window" not in holds.released_holders
