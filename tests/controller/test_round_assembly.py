"""The assembler: fragments become one packed round after the packing time.

The packing time is charged once per complete round, however many
fragments it arrived in (Caune et al. 2410.05202 measure 250 to 370 FPGA
cycles for packetization, bus transfer, result return and the
conditional together, an upper bound); a two-fragment round is the QLX
frontend's shape (decsim/frontends/qlx_frontend.py, the terminal data
readout as its own fragment). The bound counts every round in flight
through the stage, in assembly, held for store room or on its route
(controller.packing_rounds_in_flight); a full stage stops the run with a
sentence naming the setting (design note, slice 4, ruling 7), or drops
the new round under the drop knob.
"""

import functools
import types

import pytest

import decsim.config as config
import decsim.controller.round_assembly as round_assembly
import decsim.controller.settings as controller_settings
import decsim.engine as engine_module
import decsim.observe.round_events as round_events
import decsim.records.rounds as round_records

PACKING_TICKS = config.microseconds_to_ticks(1.0)
DROP = controller_settings.PackingOverflowPolicy.DROP_ROUND


def fragment(round_index, fragment_index=0, bits=(1, 0)):
    return round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_id=0,
        round_index=round_index,
        bits=bits,
        size_bits=len(bits),
        fragment_index=fragment_index,
    )


def rounds_in_flight(capacity, held=0, on_route=0):
    """The bound over `held` held rounds and `on_route` on their route."""
    held_rounds = types.SimpleNamespace(count=held)
    transmitter = types.SimpleNamespace(in_flight=on_route)
    return round_assembly.RoundsInFlight(capacity, held_rounds, transmitter)


def assembler_with(engine, packed, recorder, **settings_fields):
    settings = controller_settings.ControllerSettings(**settings_fields)
    bound = rounds_in_flight(settings.packing_rounds_in_flight)
    assembler = round_assembly.RoundAssembler(
        engine,
        settings,
        form_round=None,
        on_packed=packed.append,
        rounds_in_flight=bound,
    )
    if recorder is not None:
        assembler.trace.round_event.connect(recorder.record)
    return assembler


def test_a_two_fragment_round_is_packed_once_after_the_packing_time():
    engine = engine_module.Engine()
    packed = []
    recorder = round_events.RoundEventRecorder(engine)
    assembler = assembler_with(
        engine, packed, recorder, packing_microseconds_per_round=1.0
    )
    first = fragment(1, fragment_index=0, bits=(1, 0))
    second = fragment(1, fragment_index=1, bits=(1, 1))
    add_first = functools.partial(
        assembler.add, first, 2, round_records.WINDOW_INPUT_ROUTE
    )
    add_second = functools.partial(
        assembler.add, second, 2, round_records.WINDOW_INPUT_ROUTE
    )
    engine.schedule(0, add_first)
    engine.schedule(5, add_second)

    engine.run()

    (round,) = packed
    (merged,) = round.packet.fragments
    assert merged.bits == (1, 0, 1, 1)
    assert merged.size_bits == 4
    assert round.wire_bits == 4
    assert round.route is round_records.WINDOW_INPUT_ROUTE
    packed_events = [
        event for event in recorder.events if event.kind == "PACKED"
    ]
    assert [event.tick for event in packed_events] == [5 + PACKING_TICKS]


def test_a_full_workspace_stops_the_run_naming_the_setting():
    engine = engine_module.Engine()
    packed = []
    recorder = None
    assembler = assembler_with(
        engine, packed, recorder, packing_rounds_in_flight=1
    )
    first = fragment(1, fragment_index=0)
    assembler.add(first, 2, round_records.WINDOW_INPUT_ROUTE)

    second_round = fragment(2)
    with pytest.raises(
        RuntimeError, match="controller.packing_rounds_in_flight is 1"
    ):
        assembler.add(second_round, 1, round_records.WINDOW_INPUT_ROUTE)

    last = fragment(1, fragment_index=1)
    assembler.add(last, 2, round_records.WINDOW_INPUT_ROUTE)
    assert len(packed) == 1


def test_the_bound_counts_rounds_held_and_on_their_route():
    """Two rounds past assembly fill a bound of two; three admits a fragment.

    One round held for store room and one on its route count against
    the bound before any round is in assembly.
    """
    engine = engine_module.Engine()
    settings = controller_settings.ControllerSettings()
    full = rounds_in_flight(2, held=1, on_route=1)
    packed = []
    assembler = round_assembly.RoundAssembler(
        engine,
        settings,
        form_round=None,
        on_packed=packed.append,
        rounds_in_flight=full,
    )
    first = fragment(1)

    with pytest.raises(RuntimeError, match="held for store room: 1, on"):
        assembler.add(first, 1, round_records.WINDOW_INPUT_ROUTE)
    assert full.count(0) == 2

    room_for_one = rounds_in_flight(3, held=1, on_route=1)
    assembler.workspace.rounds_in_flight = room_for_one
    assembler.add(first, 1, round_records.WINDOW_INPUT_ROUTE)
    assert [round.packet.round_index for round in packed] == [1]


def test_the_drop_knob_drops_only_the_round_that_found_no_context():
    engine = engine_module.Engine()
    packed = []
    recorder = round_events.RoundEventRecorder(engine)
    assembler = assembler_with(
        engine,
        packed,
        recorder,
        packing_rounds_in_flight=1,
        packing_overflow=DROP,
    )
    first = fragment(1, fragment_index=0)
    assembler.add(first, 2, round_records.WINDOW_INPUT_ROUTE)
    second_round = fragment(2)
    assembler.add(second_round, 1, round_records.WINDOW_INPUT_ROUTE)
    last = fragment(1, fragment_index=1)
    assembler.add(last, 2, round_records.WINDOW_INPUT_ROUTE)

    assert recorder.packing_drops == 1
    dropped = [
        event.round_index
        for event in recorder.events
        if event.kind == "DROPPED"
    ]
    assert dropped == [2]
    assert [round.packet.round_index for round in packed] == [1]


def test_detection_events_are_formed_once_from_the_merged_bits():
    engine = engine_module.Engine()
    packed = []
    formed_from = []

    def form_round(operation_id, round_index, bits):
        formed_from.append((operation_id, round_index, bits))
        return (0, 1, 1)

    settings = controller_settings.ControllerSettings()
    unbounded = rounds_in_flight(None)
    assembler = round_assembly.RoundAssembler(
        engine,
        settings,
        form_round=form_round,
        on_packed=packed.append,
        rounds_in_flight=unbounded,
    )
    first = fragment(1, fragment_index=0, bits=(1, 0))
    second = fragment(1, fragment_index=1, bits=(0, 1))

    assembler.add(first, 2, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(second, 2, round_records.WINDOW_INPUT_ROUTE)

    assert formed_from == [(1, 1, (1, 0, 0, 1))]
    (round,) = packed
    (formed,) = round.packet.fragments
    assert formed.bits == (0, 1, 1)
    assert formed.size_bits == 3
    assert round.wire_bits == 4


def test_settlement_reports_a_round_still_in_assembly():
    engine = engine_module.Engine()
    recorder = None
    assembler = assembler_with(engine, [], recorder)
    first = fragment(1, fragment_index=0)
    assembler.add(first, 2, round_records.WINDOW_INPUT_ROUTE)

    with pytest.raises(RuntimeError, match="incomplete syndrome packing"):
        assembler.check_settled()
