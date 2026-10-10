"""The assignments in force while a run file builds (in_force.py).

ns-3's Config::SetDefault: every object built while a value is in force
takes it (manual, Attributes). Outside a load nothing is in force, and
an assignment no task reaches is refused at the end.
"""

import pytest

import decsim.experiments.collect as collect
import decsim.parameters.fields as fields
import decsim.parameters.in_force as in_force
import decsim.parameters.tree as tree
import tests.experiments.run_files as run_files

ORIGIN = "--set"


def _assignment(pattern: str, written) -> tree.Assignment:
    return tree.Assignment(pattern, written, ORIGIN)


def test_nothing_is_in_force_outside_a_load():
    machine = run_files.minimal_machine()

    applied = in_force.applied(machine)

    assert applied is machine


def test_every_task_built_inside_takes_the_assignments():
    machine = run_files.minimal_machine()
    unit_count = _assignment("weak_decoder.unit_count", 2)

    with in_force.in_force([unit_count]):
        first = collect.Task("first", machine)
        second = collect.Task("second", machine)
    after = collect.Task("after", machine)

    assert first.machine.weak_decoder.unit_count == 2
    assert second.machine.weak_decoder.unit_count == 2
    assert after.machine.weak_decoder.unit_count == 1


def test_an_assignment_no_task_reaches_is_refused_with_the_near_path():
    machine = run_files.minimal_machine()
    misspelt = _assignment("weak_decoder.unit_cont", 2)

    with pytest.raises(fields.ParameterError) as refused:
        with in_force.in_force([misspelt]):
            collect.Task("first", machine)

    message = str(refused.value)
    assert "--set weak_decoder.unit_cont=2 reaches no parameter" in message
    assert "Did you mean weak_decoder.unit_count" in message


def test_an_assignment_one_task_reaches_is_not_refused_for_the_others():
    minimal = run_files.minimal_machine()
    switching = run_files.switching_machine()
    strong = _assignment("strong_decoder.unit_count", 2)

    with in_force.in_force([strong]):
        collect.Task("weak_only", minimal)
        switched = collect.Task("switching", switching)

    assert switched.machine.strong_decoder.unit_count == 2
