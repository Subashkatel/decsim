"""When each command arrived at the QPU and when it started.

A listener on the QPU's command_event(event); the flight recorder reads
the events to close every command's chain at the QPU.
"""


class CommandEvents:
    """The QPU's command events, in the order they happened."""

    def __init__(self) -> None:
        self.events: list = []

    def command_event(self, event) -> None:
        """One more arrival or start."""
        self.events.append(event)
