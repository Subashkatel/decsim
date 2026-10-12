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
from collections.abc import Iterator, Mapping, Sequence
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


def constant_overrides(values: Mapping, prefix: str) -> dict:
    """The assignments in force to prefix.<name>, by the names values has.

    Each is recorded as reaching a parameter: a constants file read
    while a run file loads takes it (topology/constants.py).
    """
    use: Optional[_Use] = _IN_FORCE.get()
    if use is None:
        return {}
    return use.constant_overrides(values, prefix)


def current() -> tuple:
    """The assignments in force; none outside a load."""
    use: Optional[_Use] = _IN_FORCE.get()
    if use is None:
        return ()
    return use.assignments


def reaching(paths: Sequence[str]) -> list:
    """The assignments in force whose pattern reaches one of the paths."""
    use: Optional[_Use] = _IN_FORCE.get()
    if use is None:
        return []
    return use.reaching(paths)


class _Use:
    """One load's assignments, those reached, the machine and constants seen."""

    def __init__(self, assignments: Sequence[tree.Assignment]) -> None:
        self.assignments = tuple(assignments)
        self.matched = set()
        self.first_machine = None
        self.constant_names = set()

    def applied(self, machine: Any) -> Any:
        """The machine with the assignments applied; the hits recorded."""
        if self.first_machine is None:
            self.first_machine = machine
        applied = tree.apply(machine, self.assignments)
        self.matched.update(applied.matched)
        return applied.settings

    def constant_overrides(self, values: Mapping, prefix: str) -> dict:
        """Each assignment to prefix.<name> values has, recorded as a hit."""
        for name in values:
            self.constant_names.add(f"{prefix}.{name}")
        found = {}
        for position, assignment in enumerate(self.assignments):
            head, _, name = assignment.pattern.partition(".")
            if head != prefix or name not in values:
                continue
            self.matched.add(position)
            found[name] = assignment
        return found

    def reaching(self, paths: Sequence[str]) -> list:
        """The assignments whose pattern reaches one of the paths."""
        return [
            assignment
            for assignment in self.assignments
            if _reaches(assignment.pattern, paths)
        ]

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
        names = sorted(self.constant_names)
        near = units.did_you_mean(assignment.pattern, [*paths, *names])
        raise fields.ParameterError(
            f"{assignment.origin} {assignment.text} reaches no parameter of "
            f"any task.{near} `decsim params <run file>` lists them"
        )


def _reaches(pattern: str, paths: Sequence[str]) -> bool:
    """Whether the pattern matches one of the dotted paths."""
    for path in paths:
        segments = path.split(".")
        if tree.matches(pattern, segments):
            return True
    return False
