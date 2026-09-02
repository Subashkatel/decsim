"""The engine runs actions in the order a discrete-event simulator must.

The law, shared with SimPy (simpy.core.Environment.schedule: events are
keyed by time, then priority, then a rising sequence number): an action
runs at its due tick; two actions due at the same tick run lower priority
number first; two with the same priority run in the order they were
scheduled. Validated exact against SimPy on 200 random programs in the
component matrix (row X1); this test pins the same law with a sorted-list
oracle so the suite needs no SimPy install.
"""

import random

from decsim.engine import Engine


def scheduled_order_oracle(requests):
    """The order a correct engine must run `requests` in."""
    keyed = [
        (due_time, priority, arrival_index)
        for arrival_index, (due_time, priority) in enumerate(requests)
    ]
    return [arrival_index for _, _, arrival_index in sorted(keyed)]


def test_random_programs_run_in_time_priority_arrival_order():
    rng = random.Random(7)
    for _ in range(200):
        engine = Engine(verbose=False)
        request_count = rng.randint(1, 30)
        requests = [
            (rng.randint(0, 10), rng.randint(0, 3)) for _ in range(request_count)
        ]
        ran = []
        for arrival_index, (due_time, priority) in enumerate(requests):
            engine.schedule(
                due_time,
                lambda index=arrival_index: ran.append(index),
                priority=priority)
        engine.run()
        assert ran == scheduled_order_oracle(requests)


def test_an_action_sees_its_own_due_tick_as_now():
    engine = Engine(verbose=False)
    seen = []
    engine.schedule(5, lambda: seen.append(engine.now))
    engine.schedule(2, lambda: seen.append(engine.now))
    engine.run()
    assert seen == [2, 5]
    assert engine.now == 5


def test_actions_scheduled_while_running_join_the_same_order():
    engine = Engine(verbose=False)
    ran = []

    def schedule_two_more():
        ran.append("first")
        engine.schedule(0, lambda: ran.append("same tick, later arrival"))
        engine.schedule(3, lambda: ran.append("three ticks later"))

    engine.schedule(1, schedule_two_more)
    engine.schedule(1, lambda: ran.append("same tick, earlier arrival"),
                    priority=1)
    engine.run()
    assert ran == [
        "first",
        "same tick, later arrival",
        "same tick, earlier arrival",
        "three ticks later",
    ]


def test_a_negative_delay_is_refused():
    engine = Engine(verbose=False)
    try:
        engine.schedule(-1, lambda: None)
    except ValueError as error:
        assert "past" in str(error)
    else:
        raise AssertionError("a negative delay was accepted")
