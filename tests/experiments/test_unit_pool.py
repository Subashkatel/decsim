"""The pool of decsim/experiments/unit_pool.py.

Its referent is sinter's worker loop (sinter/_collection/
_collection_manager.py, process_message): a worker lives for the whole
collection and is handed its next work when it reports back, so what
it compiled stays with it.
"""

import os

import decsim.experiments.unit_pool as unit_pool


def unit_and_process(unit, measure) -> tuple:
    """The unit, its measure, and the process that ran it."""
    process_id = os.getpid()
    return unit, measure, process_id


def test_one_process_runs_each_unit_here_in_the_order_handed_in():
    with unit_pool.UnitPool(unit_and_process, "m", 1) as pool:
        pool.submit("first", 10)
        first = pool.next_ended()
        pool.submit("second", 20)
        second = pool.next_ended()

    here = os.getpid()
    assert first == ("first", 10, (10, "m", here))
    assert second == ("second", 20, (20, "m", here))


def test_a_pool_has_room_for_as_many_units_as_its_processes():
    with unit_pool.UnitPool(unit_and_process, "m", 2) as pool:
        pool.submit("a", 1)
        has_room_for_one = pool.has_room()
        pool.submit("b", 2)
        has_room_for_none = pool.has_room()
        pool.next_ended()
        pool.next_ended()

    assert has_room_for_one
    assert not has_room_for_none


def test_a_pools_workers_live_from_one_unit_to_the_next():
    """Six units one after another run on at most the pool's two workers."""
    process_ids = set()
    with unit_pool.UnitPool(unit_and_process, "m", 2) as pool:
        for unit in range(6):
            pool.submit(unit, unit)
            _tag, _unit, outcome = pool.next_ended()
            process_ids.add(outcome[2])

    assert len(process_ids) <= 2
    assert os.getpid() not in process_ids
