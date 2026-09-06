"""The controller's counters: how many idle rounds it emitted.

A listener on the idle accounting's idle_round_emitted(operation_id,
patch, round_index); the count is what the gate pins as
controller_idle_rounds.
"""


class ControllerCounters:
    """Every idle round emitted so far, over all patches."""

    def __init__(self) -> None:
        self.idle_rounds = 0

    def idle_round_emitted(self, operation_id, patch, round_index: int) -> None:
        """One more idle round left the QPU."""
        del operation_id, patch, round_index
        self.idle_rounds += 1
