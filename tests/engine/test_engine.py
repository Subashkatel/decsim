"""The engine runs actions in the order a discrete-event simulator must.

The law, shared with SimPy (simpy.core.Environment.schedule: events are
keyed by time, then priority, then a rising sequence number): an action
runs at its due tick; two actions due at the same tick run lower priority
number first; two with the same priority run in the order they were
scheduled. Validated exact against SimPy on 200 random programs in the
component matrix (row X1); this test pins the same law with a sorted-list
oracle so the suite needs no SimPy install.

The priority a component passes is one of the engine's named ones, the
way gem5's events name theirs (gem5 src/sim/eventq.hh
lines 138-244), which the last test below reads the package to check.
"""

import ast
import pathlib
import random

import pytest

import decsim.engine as engine_module


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


def test_random_programs_run_in_time_priority_arrival_order_property():
    rng = random.Random(7)
    for _ in range(200):
        engine = engine_module.Engine()
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


def test_the_engine_runs_the_same_ticks_with_no_listener_at_all():
    def program(engine, seen):
        engine.schedule(5, lambda: seen.append(engine.now))
        engine.schedule(2, lambda: seen.append(engine.now))
        engine.schedule(2, lambda: engine.log("worker", "ready"))

    bare = engine_module.Engine()
    bare_seen = []
    program(bare, bare_seen)
    bare.run()
    heard = engine_module.Engine()
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


def test_a_negative_delay_is_refused():
    """An action due in the past would run out of order, so it stops."""
    engine = engine_module.Engine()
    with pytest.raises(RuntimeError, match="cannot schedule an action"):
        engine.schedule(-1, lambda: None)


def test_an_action_that_raises_stops_the_run_and_leaves_the_rest_queued():
    """An action's exception reaches the caller unchanged.

    The engine wraps no action, so a component's bug arrives as its own
    traceback (STYLE.md rule 4: a wrong caller is a bug and a loud
    stop is better than a wrong number; gem5's panic, src/base/logging.hh).
    The failing action is already off the queue, so a second run resumes
    with what is left.
    """
    engine = engine_module.Engine()
    failure = RuntimeError("the component is wrong")
    ran = []

    def failing_action():
        ran.append("failing")
        raise failure

    engine.schedule(2, failing_action)
    engine.schedule(3, lambda: ran.append("later"))
    with pytest.raises(RuntimeError) as raised:
        engine.run()
    assert raised.value is failure
    assert ran == ["failing"]
    assert engine.now == 2
    engine.run()
    assert ran == ["failing", "later"]


def test_no_scheduled_action_in_the_package_names_a_bare_priority_number():
    """A call site says which event it is scheduling, not which number.

    gem5 gives every priority a name beside the reason for it
    (gem5 src/sim/eventq.hh lines 138-244); a bare integer
    at the call says nothing about what must run before what, and two
    call sites that share a number look unrelated.
    """
    engine_path = pathlib.Path(engine_module.__file__)
    package = engine_path.parent
    bare = bare_priority_numbers(package)
    assert bare == []


def bare_priority_numbers(package):
    """Every schedule call in the package that passes a literal priority."""
    bare = []
    modules = package.rglob("*.py")
    for path in sorted(modules):
        source = path.read_text()
        tree = ast.parse(source)
        found = literal_priorities(tree, path)
        bare.extend(found)
    return bare


def literal_priorities(tree, path):
    """The `path:line` of every schedule call with a literal priority."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None)
        if name != "schedule":
            continue
        literal = literal_keywords(node, path)
        found.extend(literal)
    return found


def literal_keywords(call, path):
    """The `path:line` of the call when its priority is a literal."""
    for keyword in call.keywords:
        if keyword.arg != "priority":
            continue
        if isinstance(keyword.value, ast.Constant):
            return [f"{path}:{call.lineno}"]
    return []


def test_a_descheduled_action_never_runs_and_never_sets_the_time():
    """gem5 EventQueue::deschedule (src/sim/eventq.hh:790)."""
    engine = engine_module.Engine()
    ran = []
    first = record_arrival(ran, 0)
    second = record_arrival(ran, 1)
    engine.schedule(5, first)
    stopped = engine.schedule(50, second)
    engine.deschedule(stopped)

    engine.run()

    assert ran == [0]
    assert engine.now == 5
