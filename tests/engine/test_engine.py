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


def record_arrival(ran, arrival_index):
    """An action that notes which request ran."""

    def action():
        ran.append(arrival_index)

    return action


def test_random_programs_run_in_time_priority_arrival_order():
    rng = random.Random(7)
    for _ in range(200):
        engine = Engine()
        request_count = rng.randint(1, 30)
        requests = []
        for _ in range(request_count):
            due_time = rng.randint(0, 10)
            priority = rng.randint(0, 3)
            requests.append((due_time, priority))
        ran = []
        for arrival_index, (due_time, priority) in enumerate(requests):
            action = record_arrival(ran, arrival_index)
            engine.schedule(due_time, action, priority=priority)
        engine.run()
        assert ran == scheduled_order_oracle(requests)


def test_an_action_sees_its_own_due_tick_as_now():
    engine = Engine()
    seen = []
    engine.schedule(5, lambda: seen.append(engine.now))
    engine.schedule(2, lambda: seen.append(engine.now))
    engine.run()
    assert seen == [2, 5]
    assert engine.now == 5


def test_actions_scheduled_while_running_join_the_same_order():
    engine = Engine()
    ran = []

    def schedule_two_more():
        ran.append("first")
        engine.schedule(0, lambda: ran.append("same tick, later arrival"))
        engine.schedule(3, lambda: ran.append("three ticks later"))

    engine.schedule(1, schedule_two_more)
    engine.schedule(
        1, lambda: ran.append("same tick, earlier arrival"), priority=1
    )
    engine.run()
    assert ran == [
        "first",
        "same tick, later arrival",
        "same tick, earlier arrival",
        "three ticks later",
    ]


def test_a_negative_delay_is_refused():
    engine = Engine()
    try:
        engine.schedule(-1, lambda: None)
    except ValueError as error:
        assert "past" in str(error)
    else:
        raise AssertionError("a negative delay was accepted")


def test_action_done_carries_the_tick_after_every_action():
    engine = Engine()
    heard = []
    engine.action_done.connect(heard.append)
    engine.schedule(2, lambda: None)
    engine.schedule(3, lambda: None)
    engine.run()
    assert heard == [2, 3]


def test_the_engine_runs_the_same_ticks_with_no_listener_at_all():
    def program(engine, seen):
        engine.schedule(5, lambda: seen.append(engine.now))
        engine.schedule(2, lambda: seen.append(engine.now))
        engine.schedule(2, lambda: engine.log("worker", "ready"))

    bare = Engine()
    bare_seen = []
    program(bare, bare_seen)
    bare.run()
    heard = Engine()
    heard_seen = []
    heard_lines = []
    heard_ticks = []
    heard.line.connect(heard_lines.append)
    heard.action_done.connect(heard_ticks.append)
    program(heard, heard_seen)
    heard.run()
    assert bare_seen == [2, 5]
    assert heard_seen == [2, 5]
    assert heard_ticks == [2, 2, 5]
    assert heard_lines == ["[  0.000 us] worker: ready"]
    assert bare.now == heard.now


def test_an_action_that_raises_stops_the_run_and_leaves_the_rest_queued():
    """An action's exception reaches the caller unchanged.

    The engine wraps no action, so a component's bug arrives as its own
    traceback (STYLE.md rule 4: a wrong caller is a bug and a loud
    stop is better than a wrong number; gem5's panic, src/base/logging.hh).
    The failing action is already off the queue, so a second run resumes
    with what is left.
    """
    engine = Engine()
    failure = RuntimeError("the component is wrong")
    ran = []

    def failing_action():
        ran.append("failing")
        raise failure

    engine.schedule(2, failing_action)
    engine.schedule(3, lambda: ran.append("later"))
    try:
        engine.run()
    except RuntimeError as error:
        assert error is failure
    else:
        raise AssertionError("the exception did not reach the caller")
    assert ran == ["failing"]
    assert engine.now == 2
    engine.run()
    assert ran == ["failing", "later"]
