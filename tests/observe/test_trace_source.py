"""A trace source fires to its listeners in connection order, or to none.

The shape is ns-3's TracedCallback (point-to-point-net-device.h:309,
.cc:530, on disk under tmp/resources/l5_buffers): a source with no sink
fires into an empty list.
"""

import decsim.observe.trace_source as trace_source


def test_a_source_with_no_listener_fires_and_nothing_happens():
    source = trace_source.TraceSource()
    source.fire(1, "round")
    assert not source.has_listeners


def test_every_listener_hears_every_fire_once_in_connection_order():
    source = trace_source.TraceSource()
    heard = []

    def first(tick):
        heard.append(("first", tick))

    def second(tick):
        heard.append(("second", tick))

    source.connect(first)
    source.connect(second)
    source.fire(3)
    source.fire(7)
    assert heard == [("first", 3), ("second", 3), ("first", 7), ("second", 7)]
    assert source.has_listeners
