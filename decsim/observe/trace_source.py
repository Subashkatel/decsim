"""A trace source: one named event a component fires and listeners hear.

A component owns one TraceSource per event it reports, fires it where
the event happens, and knows nothing of who listens; a listener
connects at build. That is ns-3's TracedCallback (the device declares
m_macTxTrace at point-to-point-net-device.h:309 and fires it at
transmit, .cc:530; with no sink connected it fires into an empty list)
and SimPy's per-event callback list (simpy/core.py step()). A source
with no listener costs one empty loop per fire.
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
