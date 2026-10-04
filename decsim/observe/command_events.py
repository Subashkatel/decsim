"""When each command arrived at the QPU and when it started."""

import decsim.qpu.cycle_clock as cycle_clock


class CommandEvents:
    """The QPU's command events, in the order they happened."""

    def __init__(self) -> None:
        self.events: list = []

    def command_event(self, event: cycle_clock.QPUCommandEvent) -> None:
        """One more arrival or start."""
        self.events.append(event)
