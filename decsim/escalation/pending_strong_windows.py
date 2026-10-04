"""The strong windows held until the conditions their row declared fire.

A shape row that cannot build its job at the escalation declares what
releases it: the weak windows whose commits it waits on, or the
operation whose stored rounds release it when it has no later window.
Toshio et al. 2510.25222 Sec. III C starts the strong decoder "after the
boundary conditions at both ends have been determined by the weak
decoder" (lines 1248-1250); which boundaries those are is the row's
geometry, so the condition is a declaration rather than a hook per row.
"""

import dataclasses
from typing import Any, Optional

import decsim.records.identity as identity_records


@dataclasses.dataclass(frozen=True)
class ReleaseConditions:
    """What must happen before a held strong job may be built.

    committed_windows are the weak windows whose commits the row waits
    on; stored_data_of_operation is the operation whose stored rounds
    release it. name labels the wait in the run's view,
    released_description in the log.
    """

    committed_windows: tuple = ()
    stored_data_of_operation: Optional[int] = None
    name: str = "nothing"
    released_description: str = ""


@dataclasses.dataclass(frozen=True)
class PendingStrongWindow:
    """One held strong window: what was assigned, and what releases it."""

    key: tuple
    assignment: Any
    conditions: ReleaseConditions
    selection_arrival_ticks: int


class PendingStrongWindows:
    """The held strong windows, under every condition that can release one.

    An entry is indexed under each window it waits on and under the
    operation whose data releases it, and a release takes it out of
    every index at once.
    """

    def __init__(self) -> None:
        self.by_key: dict = {}
        self.keys_by_committed_window: dict = {}
        self.keys_by_stored_operation: dict = {}
        self.uncommitted_windows: dict = {}

    def register(self, held: PendingStrongWindow) -> None:
        """Hold the assignment under the conditions its row declared."""
        conditions = held.conditions
        waits_on_data = conditions.stored_data_of_operation is not None
        has_condition = bool(conditions.committed_windows) or waits_on_data
        if not has_condition:
            raise RuntimeError(
                f"the strong window for {held.key} is held with no release "
                f"condition: a row that holds its job names what releases it"
            )
        assert held.key not in self.by_key, held.key
        self.by_key[held.key] = held
        self.uncommitted_windows[held.key] = set(conditions.committed_windows)
        for window_key in conditions.committed_windows:
            _index(self.keys_by_committed_window, window_key, held.key)
        if waits_on_data:
            _index(
                self.keys_by_stored_operation,
                conditions.stored_data_of_operation,
                held.key,
            )

    def released_by_commit(self, window_key: tuple) -> tuple:
        """The entries this weak commit satisfies, in registration order."""
        waiting = _waiting_on(self.keys_by_committed_window, window_key)
        released = []
        for held_key in waiting:
            self.uncommitted_windows[held_key].discard(window_key)
            if self.uncommitted_windows[held_key]:
                continue
            released.append(self.by_key[held_key])
        return tuple(released)

    def released_by_stored_data(
        self,
        operation_id: Any,  # an opaque identity
    ) -> tuple:
        """The entries of this operation whose weak commits have landed."""
        waiting = _waiting_on(self.keys_by_stored_operation, operation_id)
        released = []
        for held_key in waiting:
            if self.uncommitted_windows[held_key]:
                continue
            released.append(self.by_key[held_key])
        return tuple(released)

    def take(self, held: PendingStrongWindow) -> None:
        """The row built the job: the entry leaves every index."""
        assert self.by_key.get(held.key) is held, held.key
        del self.by_key[held.key]
        del self.uncommitted_windows[held.key]
        for window_key in held.conditions.committed_windows:
            _unindex(self.keys_by_committed_window, window_key, held.key)
        operation_id = held.conditions.stored_data_of_operation
        if operation_id is not None:
            _unindex(self.keys_by_stored_operation, operation_id, held.key)

    def held_for(self, window_key: tuple) -> Optional[PendingStrongWindow]:
        """The entry held for one escalated window, or None."""
        return self.by_key.get(window_key)

    def any_held(self) -> bool:
        """Whether a strong window is still waiting for its conditions."""
        return bool(self.by_key)

    def work(self) -> tuple:
        """The held windows as (key, wait name, rounds), in stable order."""
        records = []
        for key, held in self.by_key.items():
            phase_name = f"waiting_{held.conditions.name}"
            record = (key, phase_name, held.assignment.round_count)
            records.append(record)
        ordered = sorted(records, key=_work_record_order)
        return tuple(ordered)


def waiting_text(conditions: ReleaseConditions) -> str:
    """What a held window waits on, named for the run's views."""
    parts = []
    for window_key in conditions.committed_windows:
        parts.append(f"commit of window {window_key}")
    operation_id = conditions.stored_data_of_operation
    if operation_id is not None:
        parts.append(f"stored rounds of operation {operation_id}")
    return ", ".join(parts)


def _index(index: dict, condition_key, held_key: tuple) -> None:
    """Record that the held window waits on this condition."""
    waiting = index.setdefault(condition_key, [])
    assert held_key not in waiting, held_key
    waiting.append(held_key)


def _unindex(index: dict, condition_key, held_key: tuple) -> None:
    """Drop the held window from a condition it no longer waits on."""
    waiting = index[condition_key]
    waiting.remove(held_key)
    if not waiting:
        del index[condition_key]


def _waiting_on(index: dict, condition_key) -> list:
    """The held windows under one condition, as a list the caller may edit."""
    waiting = index.get(condition_key, ())
    return list(waiting)


def _work_record_order(record: tuple) -> bytes:
    return identity_records.stable_identity_bytes(record[0])
