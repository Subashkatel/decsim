"""The assignments in force while a run file builds its tasks.

ns-3's command line sets attribute defaults that every object built
after it takes (manual, Attributes: Config::SetDefault). Here `--set`
and DECSIM_SET are in force while `decsim run` executes a run file
(experiments/experiment.py load), and every Task built then takes them
on its machine before its threshold table is read
(collect.Task.__post_init__), so a row is read at the machine the run
actually runs. Outside a load nothing is in force and a Task keeps the
machine it is given.

An assignment that reaches no path of any task is refused when the run
file ends, with the paths it was near.
"""

import contextlib
import contextvars
from collections.abc import Iterator, Sequence
from typing import Any, Optional

import decsim.parameters.fields as fields
import decsim.parameters.tree as tree
import decsim.parameters.units as units

_IN_FORCE = contextvars.ContextVar("decsim_assignments_in_force", default=None)


@contextlib.contextmanager
def in_force(assignments: Sequence[tree.Assignment]) -> Iterator[None]:
    """The assignments in force inside; the unmatched refused after."""
    use = _Use(assignments)
    token = _IN_FORCE.set(use)
    try:
        yield
    finally:
        _IN_FORCE.reset(token)
    use.refuse_the_unmatched()


def applied(machine: Any) -> Any:
    """The machine with the assignments in force applied, if any are."""
    use: Optional[_Use] = _IN_FORCE.get()
    if use is None or not use.assignments:
        return machine
    return use.applied(machine)


class _Use:
    """One load's assignments, the ones some task reached, a machine seen."""

    def __init__(self, assignments: Sequence[tree.Assignment]) -> None:
        self.assignments = tuple(assignments)
        self.matched = set()
        self.first_machine = None

    def applied(self, machine: Any) -> Any:
        """The machine with the assignments applied; the hits recorded."""
        if self.first_machine is None:
            self.first_machine = machine
        applied = tree.apply(machine, self.assignments)
        self.matched.update(applied.matched)
        return applied.settings

    def refuse_the_unmatched(self) -> None:
        """Refuse the first assignment that reached no path of any task."""
        for position, assignment in enumerate(self.assignments):
            if position in self.matched:
                continue
            self._refuse_an_unmatched(assignment)

    def _refuse_an_unmatched(self, assignment: tree.Assignment) -> None:
        paths = []
        if self.first_machine is not None:
            settings = tree.settings_of(self.first_machine)
            paths = [setting.path for setting in settings]
        near = units.did_you_mean(assignment.pattern, paths)
        raise fields.ParameterError(
            f"{assignment.origin} {assignment.text} reaches no parameter of "
            f"any task.{near} `decsim params <run file>` lists them"
        )
