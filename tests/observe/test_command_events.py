"""The QPU's command events, kept in the order they happened.

A listener on the QPU's command_event source. The flight recorder reads
them to close every command's chain at the QPU, so their order is the
one thing this listener promises.
"""

import decsim.observe.command_events as command_events


def test_a_run_with_no_command_holds_no_event():
    events = command_events.CommandEvents()

    assert events.events == []


def test_every_event_is_kept_in_the_order_it_was_heard():
    events = command_events.CommandEvents()

    events.command_event("arrived")
    events.command_event("started")

    assert events.events == ["arrived", "started"]


def test_the_listener_reads_nothing_inside_an_event():
    """It closes a chain by identity, so an event is opaque to it."""
    events = command_events.CommandEvents()
    opaque = object()

    events.command_event(opaque)

    assert events.events[0] is opaque
