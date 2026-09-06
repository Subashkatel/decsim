"""The flight recorder keeps every round event at the tick it was told.

The rows are the run ledger's source (decsim/observe/run_views.py); the
recorder never schedules or decides, and the Buffer 0 trace line it
narrates appears only when the engine's I/O trace is on.
"""

import decsim.engine as engine_module
import decsim.message as message
import decsim.observe.log_writers as log_writers
import decsim.observe.round_events as round_events
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings


def packet(round_index):
    fragment = message.RetainedSyndromeFragment(
        operation_id=1,
        patch_id=0,
        round_index=round_index,
        bits=(1, 0),
        size_bits=2,
        fragment_index=0,
    )
    return message.SyndromeRoundPacket(1, round_index, (fragment,))


def test_an_event_carries_the_engine_tick_unless_one_is_given():
    engine = engine_module.Engine()
    recorder = round_events.RoundEventRecorder(engine)
    engine.schedule(
        40,
        lambda: recorder.record("PACKED", 1, 3, message.WINDOW_INPUT_ROUTE),
    )

    engine.run()
    recorder.record("PUBLISHED", 1, 3, message.WINDOW_INPUT_ROUTE, tick=25)

    rows = [(event.kind, event.tick, event.route) for event in recorder.events]
    assert rows == [
        ("PACKED", 40, "WINDOW_INPUT"),
        ("PUBLISHED", 25, "WINDOW_INPUT"),
    ]


def test_a_dropped_round_is_counted_and_recorded_as_dropped():
    engine = engine_module.Engine()
    recorder = round_events.RoundEventRecorder(engine)

    recorder.round_dropped(1, 2, message.WINDOW_INPUT_ROUTE, patch_id=0)

    assert recorder.packing_drops == 1
    (event,) = recorder.events
    assert (event.kind, event.round_index, event.patch_id) == ("DROPPED", 2, 0)


def test_an_output_event_carries_the_payload_itself():
    engine = engine_module.Engine()
    recorder = round_events.RoundEventRecorder(engine)
    decision = message.Decision(9, releases_operation=False)

    recorder.output("DECISION_AVAILABLE", 9, decision)

    (event,) = recorder.output_events
    assert event.payload is decision
    assert (event.kind, event.tick, event.operation_id) == (
        "DECISION_AVAILABLE",
        0,
        9,
    )


def test_a_strong_store_landing_is_kept_with_its_tick():
    engine = engine_module.Engine()
    recorder = round_events.RoundEventRecorder(engine)
    engine.schedule(12, lambda: recorder.round_stored(1, 4))

    engine.run()

    assert recorder.stored_rounds == [(12, 1, 4)]


def test_the_buffer_0_line_is_narrated_only_on_the_io_trace():
    silent_engine = engine_module.Engine()
    silent_log = log_writers.LogWriter()
    silent_engine.line.connect(silent_log.write)
    silent = round_events.RoundEventRecorder(silent_engine)
    narrating_engine = engine_module.Engine()
    narrating_log = log_writers.LogWriter()
    narrating_engine.io_line.connect(narrating_log.write)
    narrating = round_events.RoundEventRecorder(narrating_engine)
    settings = round_store_settings.RoundStoreSettings()
    store = round_store_module.RoundStore(settings)
    first = packet(1)
    store.accept_packed_round(first, publication_tick=0)

    silent.weak_store_received(first, store)
    narrating.weak_store_received(first, store)

    assert silent_log.lines == []
    (line,) = narrating_log.lines
    assert line.endswith(
        "Buffer 0: received round 1 of op 1 from packing; "
        "defects {0}; holds op 1 rounds 1 (1)"
    )
