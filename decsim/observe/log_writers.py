"""The two listeners of the engine's narrator: the record and the console.

The engine builds each line's text and fires it on its line source (and
on its io_line source for a component's I/O line); LogWriter keeps the
lines in order, ConsolePrinter prints each as it is fired. That is gem5's
split between the trace text and its sink (src/base/trace.hh
DPRINTF builds curTick(), name() and the text; the logger writes it).
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
