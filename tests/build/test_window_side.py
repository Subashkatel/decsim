"""The windows facade and everything behind it, bound by assignment.

STYLE.md rule 7: a component declares each neighbour as a port, its
constructor takes settings only, and the root binds every wire by
assignment once every component exists. This file pins the three binds
the window side could only make late: the courier names the facade it
tells of a landing, and the committer and the verdict name the strong
redecode built after them, which a run that never escalates leaves
unbound.
"""

import tests.declared_run as declared_run
import tests.escalation.declared_fabric as fabric


def test_the_courier_names_the_facade_it_tells_a_landing():
    machine = declared_run.weak_only_run(rounds=6)
    courier = machine.window_manager.courier

    assert courier.windows is machine.window_manager


def test_the_committer_and_the_verdict_name_the_strong_redecode():
    machine = fabric.switching_machine(rounds=9, escalated_windows={1})
    redecode = machine.window_manager.strong_redecode
    verdict = machine.window_manager.requester.verdict

    assert verdict.strong_redecode is redecode
    assert verdict.committer.strong_redecode is redecode


def test_a_run_that_never_escalates_leaves_the_redecode_unbound():
    """An optional port nobody binds reads as None; nothing binds None."""
    machine = declared_run.weak_only_run(rounds=6)
    verdict = machine.window_manager.requester.verdict

    assert machine.window_manager.strong_redecode is None
    assert verdict.strong_redecode is None
    assert verdict.committer.strong_redecode is None
