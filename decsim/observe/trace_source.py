"""A trace source: one named event a component fires and listeners hear.

A component owns one TraceSource per event it reports, fires it where
the event happens, and knows nothing of who listens; a listener
connects at build. That is ns-3's TracedCallback (the device declares
m_macTxTrace at point-to-point-net-device.h:309 and fires it at
transmit, .cc:530; with no sink connected it fires into an empty list)
and SimPy's per-event callback list (simpy/core.py step()). A source
with no listener costs one empty loop per fire.

SilentSource is the same surface for an event one row of a table never
has, so every row of that table carries every source the port declares
and a listener connects to all of them by name.
"""

from typing import Callable


class TraceSource:
    """The listeners of one event, called in connection order."""

    def __init__(self) -> None:
        self._listeners: list[Callable] = []

    def connect(self, listener: Callable) -> None:
        """Hear every fire from now on, after the listeners already there."""
        self._listeners.append(listener)

    def fire(self, *values) -> None:
        """Tell every listener, in connection order."""
        for listener in self._listeners:
            listener(*values)

    @property
    def has_listeners(self) -> bool:
        """Whether anyone hears this source; asked before costly work."""
        return bool(self._listeners)


class SilentSource:
    """A declared event that this implementation never has.

    A port declares the sources every row carries; a row with nothing to
    report exposes one of these instead of leaving the name off, so a
    listener connects by name without asking what the row is. It never
    fires, so it keeps no listener and holds no state, and one instance
    serves every row that is silent.
    """

    def connect(self, listener: Callable) -> None:
        """There is nothing to hear, so the listener is not kept."""
        del listener

    @property
    def has_listeners(self) -> bool:
        """No one hears an event that never happens."""
        return False


# the one silent source every row that reports nothing exposes
SILENT = SilentSource()
