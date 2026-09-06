"""The flight recorder keeps every round event it is told, in order.

The rows are the run ledger's source (decsim/observe/run_views.py); the
recorder is a listener on the components' round_event, output_event and
round_stored sources and never schedules or decides.
"""

import functools

import decsim.engine as engine_module
import decsim.message as message
import decsim.observe.round_events as round_events


def test_an_event_is_kept_with_the_tick_it_carries():
    engine = engine_module.Engine()
    recorder = round_events.RoundEventRecorder(engine)
    packed = message.RoundEvent.of(
        "PACKED", 40, 1, 3, message.WINDOW_INPUT_ROUTE
    )
    published = message.RoundEvent.of(
        "PUBLISHED", 25, 1, 3, message.WINDOW_INPUT_ROUTE
    )

    recorder.record(packed)
    recorder.record(published)

    rows = [(event.kind, event.tick, event.route) for event in recorder.events]
    assert rows == [
        ("PACKED", 40, "WINDOW_INPUT"),
        ("PUBLISHED", 25, "WINDOW_INPUT"),
    ]


def test_a_dropped_round_is_counted_and_kept_as_dropped():
    engine = engine_module.Engine()
    recorder = round_events.RoundEventRecorder(engine)
    dropped = message.RoundEvent.of(
        "DROPPED", 0, 1, 2, message.WINDOW_INPUT_ROUTE, 0
    )

    recorder.record(dropped)

    assert recorder.packing_drops == 1
    (event,) = recorder.events
    assert (event.kind, event.round_index, event.patch_id) == ("DROPPED", 2, 0)


def test_an_output_event_carries_the_payload_itself():
    engine = engine_module.Engine()
    recorder = round_events.RoundEventRecorder(engine)
    decision = message.Decision(9, releases_operation=False)
    event = message.ControllerOutputEvent("DECISION_AVAILABLE", 0, 9, decision)

    recorder.output(event)

    (kept,) = recorder.output_events
    assert kept.payload is decision
    assert (kept.kind, kept.tick, kept.operation_id) == (
        "DECISION_AVAILABLE",
        0,
        9,
    )


def test_a_strong_store_landing_is_kept_with_its_tick():
    engine = engine_module.Engine()
    recorder = round_events.RoundEventRecorder(engine)
    landing = functools.partial(recorder.round_stored, (1, 4), None)
    engine.schedule(12, landing)

    engine.run()

    assert recorder.stored_rounds == [(12, 1, 4)]
