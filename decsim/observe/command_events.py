"""When each command arrived at the QPU and when it started.

A listener on the QPU's command_event(event); the run command reads the
events, and the run ledger the tests check closes every command's chain
at the QPU with them (tests/observe/run_ledger.py).
"""


class CommandEvents:
    """The QPU's command events, in the order they happened."""

    def __init__(self) -> None:
        self.events: list = []

    def command_event(self, event) -> None:
        """One more arrival or start."""
        self.events.append(event)
