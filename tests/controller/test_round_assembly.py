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
from decsim.detector_error_model import detection_event_formation

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


def formation(former, departure_ticks=0):
    """The controller row: this assembler forms the round before it leaves."""
    return detection_event_formation.ControllerSideFormation(
        former, departure_ticks
    )


def assembler_with(engine, packed, recorder, **settings_fields):
    settings = controller_settings.ControllerSettings(**settings_fields)
    bound = rounds_in_flight(settings.packing_rounds_in_flight)
    events = formation(None)
    assembler = round_assembly.RoundAssembler(
        engine,
        settings,
        detection_events=events,
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
    events = formation(None)
    assembler = round_assembly.RoundAssembler(
        engine,
        settings,
        detection_events=events,
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
    former = _Former((0, 1, 1))

    settings = controller_settings.ControllerSettings()
    unbounded = rounds_in_flight(None)
    events = formation(former)
    assembler = round_assembly.RoundAssembler(
        engine,
        settings,
        detection_events=events,
        on_packed=packed.append,
        rounds_in_flight=unbounded,
    )
    first = fragment(1, fragment_index=0, bits=(1, 0))
    second = fragment(1, fragment_index=1, bits=(0, 1))

    assembler.add(first, 2, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(second, 2, round_records.WINDOW_INPUT_ROUTE)

    assert former.asked == [(1, 1, (1, 0, 0, 1))]
    (round,) = packed
    (formed,) = round.packet.fragments
    assert formed.bits == (0, 1, 1)
    assert formed.size_bits == 3
    # the row forms events here, so the round leaves three bits wide,
    # not the four measurement outcomes it was merged from
    assert round.wire_bits == 3


def test_events_formed_at_the_decoder_keep_the_raw_measurement_width():
    """The store and the input link carry the outcomes, not the events."""
    engine = engine_module.Engine()
    packed = []
    former = _Former((0, 1, 1))

    settings = controller_settings.ControllerSettings()
    events = detection_event_formation.DecoderSideFormation(former, 0)
    unbounded = rounds_in_flight(None)
    assembler = round_assembly.RoundAssembler(
        engine,
        settings,
        detection_events=events,
        on_packed=packed.append,
        rounds_in_flight=unbounded,
    )
    first = fragment(1, fragment_index=0, bits=(1, 0))
    second = fragment(1, fragment_index=1, bits=(0, 1))

    assembler.add(first, 2, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(second, 2, round_records.WINDOW_INPUT_ROUTE)

    (round,) = packed
    (left,) = round.packet.fragments
    assert left.bits == (1, 0, 0, 1)
    assert left.size_bits == 4
    assert round.wire_bits == 4
    assert former.asked == []


def test_the_controller_row_delays_the_round_by_its_formation_time():
    """The charge moves the departure and nothing else."""
    engine = engine_module.Engine()
    departures = _Departures(engine)
    formation_ticks = config.microseconds_to_ticks(0.02)
    settings = controller_settings.ControllerSettings()
    former = _Former((0, 1, 1))
    events = formation(former, departure_ticks=formation_ticks)
    unbounded = rounds_in_flight(None)
    assembler = round_assembly.RoundAssembler(
        engine,
        settings,
        detection_events=events,
        on_packed=departures.record,
        rounds_in_flight=unbounded,
    )

    only_fragment = fragment(1, bits=(1, 0))

    assembler.add(only_fragment, 1, round_records.WINDOW_INPUT_ROUTE)
    engine.run()

    departure_ticks, round = departures.records[0]
    assert departure_ticks == formation_ticks
    assert round.wire_bits == 3


def test_an_uncharged_controller_row_hands_the_round_on_at_once():
    engine = engine_module.Engine()
    departures = _Departures(engine)
    settings = controller_settings.ControllerSettings()
    former = _Former((0, 1, 1))
    events = formation(former)
    unbounded = rounds_in_flight(None)
    assembler = round_assembly.RoundAssembler(
        engine,
        settings,
        detection_events=events,
        on_packed=departures.record,
        rounds_in_flight=unbounded,
    )
    only_fragment = fragment(1, bits=(1, 0))

    assembler.add(only_fragment, 1, round_records.WINDOW_INPUT_ROUTE)

    departure_ticks, _round = departures.records[0]
    assert departure_ticks == 0


class _Departures:
    """The tick each packed round left the assembler at, and the round."""

    def __init__(self, engine):
        self.engine = engine
        self.records = []

    def record(self, packed_round):
        """One round handed on."""
        self.records.append((self.engine.now, packed_round))


class _Former:
    """A formation table answering one round's events, and what it was asked."""

    def __init__(self, events):
        self.events = events
        self.asked = []

    def form_round(self, operation_id, round_index, bits):
        """One round's detection events."""
        self.asked.append((operation_id, round_index, bits))
        return self.events


def test_settlement_reports_a_round_still_in_assembly():
    engine = engine_module.Engine()
    recorder = None
    assembler = assembler_with(engine, [], recorder)
    first = fragment(1, fragment_index=0)
    assembler.add(first, 2, round_records.WINDOW_INPUT_ROUTE)

    with pytest.raises(RuntimeError, match="incomplete syndrome packing"):
        assembler.check_settled()
