"""The two listeners of the engine's narrator: the record and the console.

gem5's split between trace text and its sink (src/base/trace.hh).
"""


class LogWriter:
    """Every line the engine fired, in order."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, line: str) -> None:
        """Keep one line."""
        self.lines.append(line)


class ConsolePrinter:
    """Every line the engine fired, printed as it happens."""

    def write(self, line: str) -> None:
        """Print one line."""
        print(line)
