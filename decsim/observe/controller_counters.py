"""The controller's counters: how many idle rounds it emitted."""

from typing import Any


class ControllerCounters:
    """Every idle round emitted so far, over all patches."""

    def __init__(self) -> None:
        self.idle_rounds = 0

    def idle_round_emitted(
        self,
        operation_id: int,
        patch: Any,  # an opaque identity
        round_index: int,
    ) -> None:
        """One more idle round left the QPU."""
        del operation_id, patch, round_index
        self.idle_rounds += 1
