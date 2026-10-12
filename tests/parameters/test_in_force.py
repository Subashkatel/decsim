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
import decsim.presets.machines as machines
import decsim.topology.constants as constants_module
import tests.experiments.run_files as run_files

ORIGIN = "--set"


def _assignment(pattern: str, written) -> tree.Assignment:
    return tree.Assignment(pattern, written, ORIGIN)


def _unit_count(task: collect.Task, pool_name: str) -> int:
    pool = task.machine.part(pool_name)
    return pool.settings.unit_count


def test_nothing_is_in_force_outside_a_load():
    machine = run_files.minimal_machine()

    applied = in_force.applied(machine)

    assert applied is machine


def test_every_task_built_inside_takes_the_assignments():
    machine = run_files.minimal_machine()
    unit_count = _assignment("weak.unit_count", 2)

    with in_force.in_force([unit_count]):
        first = collect.Task("first", machine)
        second = collect.Task("second", machine)
    after = collect.Task("after", machine)

    assert _unit_count(first, machines.WEAK_POOL) == 2
    assert _unit_count(second, machines.WEAK_POOL) == 2
    assert _unit_count(after, machines.WEAK_POOL) == 1


def test_an_assignment_no_task_reaches_is_refused_with_the_near_path():
    machine = run_files.minimal_machine()
    misspelt = _assignment("weak.unit_cont", 2)

    with pytest.raises(fields.ParameterError) as refused:
        with in_force.in_force([misspelt]):
            collect.Task("first", machine)

    message = str(refused.value)
    assert "--set weak.unit_cont=2 reaches no parameter" in message
    assert "Did you mean weak.unit_count" in message


def test_an_assignment_one_task_reaches_is_not_refused_for_the_others():
    minimal = run_files.minimal_machine()
    switching = run_files.switching_machine()
    strong = _assignment("strong.unit_count", 2)

    with in_force.in_force([strong]):
        collect.Task("weak_only", minimal)
        switched = collect.Task("switching", switching)

    assert _unit_count(switched, machines.STRONG_POOL) == 2


def test_a_misspelt_constant_is_refused_with_the_near_constant(tmp_path):
    path = tmp_path / "constants.yaml"
    path.write_text("memory:\n  probability: 0.002\n")
    misspelt = _assignment("constants.memory.probabilty", 0.004)

    with pytest.raises(fields.ParameterError) as refused:
        with in_force.in_force([misspelt]):
            constants_module.read(path)

    message = str(refused.value)
    assert "Did you mean constants.memory.probability" in message


def test_the_assignments_in_force_are_none_outside_a_load():
    machine = run_files.minimal_machine()
    unit_count = _assignment("weak.unit_count", 2)

    with in_force.in_force([unit_count]):
        inside = in_force.current()
        collect.Task("first", machine)
    outside = in_force.current()

    assert inside == (unit_count,)
    assert outside == ()
