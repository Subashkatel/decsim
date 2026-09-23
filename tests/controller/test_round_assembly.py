"""The assembler: fragments become one packed round after the packing time.

The packing time is charged once per complete round, however many
fragments it arrived in (Caune et al. 2410.05202 measure 250 to 370 FPGA
cycles for packetization, bus transfer, result return and the
conditional together, an upper bound); a round arrives in several
fragments when its source reads it out in more than one acquisition
group (RepeatedStimCircuit.readout_partitions in decsim/records/
circuits.py). The bound counts every round in flight
through the stage, in assembly, held for store room or on its route
(controller.packing_rounds_in_flight); a full stage stops the run with a
sentence naming the setting, or drops
the new round under the drop knob.
"""

import dataclasses
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
# a 1 MHz controller, so one packing cycle is the packing time above
PACKING_CLOCK = config.Clock(PACKING_TICKS)
DROP = controller_settings.PackingOverflowPolicy.DROP_ROUND


def fragment(round_index, fragment_index=0, bits=(1, 0)):
    return round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=round_index,
        bits=bits,
        size_bits=len(bits),
        fragment_index=fragment_index,
    )


def rounds_in_flight(capacity, held=0, on_route=0):
    """The bound over `held` held rounds and `on_route` on their route."""
    bound = round_assembly.RoundsInFlight(capacity)
    bound.held_rounds = types.SimpleNamespace(count=held)
    bound.transmitter = types.SimpleNamespace(in_flight=on_route)
    return bound


def formation(former, detection_event_formation_cycles=0):
    """The controller row: this assembler forms the round before it leaves."""
    return detection_event_formation.ControllerSideFormation(
        former, detection_event_formation_cycles
    )


def assembler_with(engine, packed, recorder, **settings_fields):
    settings = controller_settings.ControllerSettings(**settings_fields)
    bound = rounds_in_flight(settings.packing_rounds_in_flight)
    events = formation(None)
    assembler = round_assembly.RoundAssembler(engine, settings)
    assembler.detection_events = events
    assembler.rounds_in_flight = bound
    assembler.syndrome_round_sender = types.SimpleNamespace(admit=packed.append)
    if recorder is not None:
        assembler.trace.round_event.connect(recorder.record)
    return assembler


def test_later_completed_round_waits_for_the_prior_emitted_round() -> None:
    engine = engine_module.Engine()
    packed = []
    assembler = assembler_with(engine, packed, None)
    first = fragment(1, bits=(1, 0))
    second = fragment(2, bits=(0, 1))
    assembler.expect_round(first, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(second, round_records.WINDOW_INPUT_ROUTE)

    assembler.add(second, 1, round_records.WINDOW_INPUT_ROUTE)
    assert packed == []
    assembler.add(first, 1, round_records.WINDOW_INPUT_ROUTE)

    round_indices = [round.packet.round_index for round in packed]
    assert round_indices == [1, 2]
    assert packed[0].packet.fragments[0].bits == (1, 0)
    assert packed[1].packet.fragments[0].bits == (0, 1)
    assembler.check_settled()


def test_waiting_for_another_channel_does_not_block_an_independent_stream() -> (
    None
):
    engine = engine_module.Engine()
    packed = []
    assembler = assembler_with(engine, packed, None)
    first = fragment(1)
    second = fragment(2)
    independent = dataclasses.replace(first, operation_id="independent")
    assembler.expect_round(first, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(second, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(independent, round_records.WINDOW_INPUT_ROUTE)

    assembler.add(second, 1, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(independent, 1, round_records.WINDOW_INPUT_ROUTE)

    operation_ids = [round.packet.operation_id for round in packed]
    assert operation_ids == ["independent"]
    assembler.add(first, 1, round_records.WINDOW_INPUT_ROUTE)
    round_indices = [round.packet.round_index for round in packed]
    assert round_indices == [1, 1, 2]


def test_a_completed_round_waiting_for_its_predecessor_consumes_capacity() -> (
    None
):
    engine = engine_module.Engine()
    packed = []
    assembler = assembler_with(engine, packed, None, packing_rounds_in_flight=1)
    first = fragment(1)
    second = fragment(2)
    assembler.expect_round(first, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(second, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(second, 1, round_records.WINDOW_INPUT_ROUTE)

    with pytest.raises(RuntimeError, match="packing workspace is full"):
        assembler.add(first, 1, round_records.WINDOW_INPUT_ROUTE)


def test_formation_retains_capacity_until_the_round_leaves() -> None:
    engine = engine_module.Engine()
    departures = _Departures(engine)
    settings = controller_settings.ControllerSettings(
        clock=PACKING_CLOCK, packing_rounds_in_flight=1
    )
    assembler = round_assembly.RoundAssembler(engine, settings)
    assembler.detection_events = formation(None, 10)
    assembler.rounds_in_flight = rounds_in_flight(1)
    assembler.syndrome_round_sender = types.SimpleNamespace(
        admit=departures.record
    )
    first = fragment(1)
    second = fragment(2)
    assembler.expect_round(first, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(second, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, 1, round_records.WINDOW_INPUT_ROUTE)

    assert departures.records == []
    with pytest.raises(RuntimeError, match="packing workspace is full"):
        assembler.add(second, 1, round_records.WINDOW_INPUT_ROUTE)
    engine.run()
    (departure_ticks, packed) = departures.records[0]
    expected_ticks = 10 * PACKING_TICKS
    assert departure_ticks == expected_ticks
    assert packed.packet.round_index == 1


def test_dropping_an_earlier_round_releases_a_ready_later_round() -> None:
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
    first = fragment(1)
    second = fragment(2)
    assembler.expect_round(first, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(second, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(second, 1, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, 1, round_records.WINDOW_INPUT_ROUTE)

    round_indices = [round.packet.round_index for round in packed]
    assert round_indices == [2]
    dropped = [
        event.round_index
        for event in recorder.events
        if event.kind == "DROPPED"
    ]
    assert dropped == [1]
    assembler.check_settled()


def test_the_assembler_hands_a_packed_round_to_its_round_sender() -> None:
    engine = engine_module.Engine()
    settings = controller_settings.ControllerSettings()
    events = formation(None)
    unbounded = rounds_in_flight(None)
    assembler = round_assembly.RoundAssembler(engine, settings)
    assembler.detection_events = events
    assembler.rounds_in_flight = unbounded
    admitted = []
    assembler.syndrome_round_sender = types.SimpleNamespace(
        admit=admitted.append
    )

    only_fragment = fragment(1)

    assembler.expect_round(only_fragment, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(only_fragment, 1, round_records.WINDOW_INPUT_ROUTE)

    assert [round.packet.round_index for round in admitted] == [1]


def test_packing_starts_at_the_edge_after_the_last_fragment_arrives() -> None:
    engine = engine_module.Engine()
    packed = []
    recorder = round_events.RoundEventRecorder(engine)
    assembler = assembler_with(
        engine,
        packed,
        recorder,
        clock=PACKING_CLOCK,
        packing_cycles_per_round=3,
    )
    first = fragment(1, fragment_index=0, bits=(1, 0))
    second = fragment(1, fragment_index=1, bits=(1, 1))
    assembler.expect_round(first, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(second, round_records.WINDOW_INPUT_ROUTE)
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
    expected_tick = 4 * PACKING_TICKS
    assert [event.tick for event in packed_events] == [expected_tick]


def test_a_full_workspace_stops_the_run_naming_the_setting() -> None:
    engine = engine_module.Engine()
    packed = []
    recorder = None
    assembler = assembler_with(
        engine, packed, recorder, packing_rounds_in_flight=1
    )
    first = fragment(1, fragment_index=0)
    assembler.expect_round(first, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, 2, round_records.WINDOW_INPUT_ROUTE)

    second_round = fragment(2)
    with pytest.raises(
        RuntimeError, match="controller.packing_rounds_in_flight is 1"
    ):
        assembler.expect_round(second_round, round_records.WINDOW_INPUT_ROUTE)
        assembler.add(second_round, 1, round_records.WINDOW_INPUT_ROUTE)

    last = fragment(1, fragment_index=1)
    assembler.expect_round(last, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(last, 2, round_records.WINDOW_INPUT_ROUTE)
    assert len(packed) == 1


def test_the_bound_counts_rounds_held_and_on_their_route() -> None:
    """Two rounds past assembly fill a bound of two; three admits a fragment.

    One round held for store room and one on its route count against
    the bound before any round is in assembly.
    """
    engine = engine_module.Engine()
    settings = controller_settings.ControllerSettings()
    full = rounds_in_flight(2, held=1, on_route=1)
    packed = []
    events = formation(None)
    assembler = round_assembly.RoundAssembler(engine, settings)
    assembler.detection_events = events
    assembler.rounds_in_flight = full
    assembler.syndrome_round_sender = types.SimpleNamespace(admit=packed.append)
    first = fragment(1)

    with pytest.raises(RuntimeError, match="held for store room: 1, on"):
        assembler.expect_round(first, round_records.WINDOW_INPUT_ROUTE)
        assembler.add(first, 1, round_records.WINDOW_INPUT_ROUTE)
    assert full.count(0) == 2

    full.capacity = 3
    assembler.expect_round(first, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, 1, round_records.WINDOW_INPUT_ROUTE)
    assert [round.packet.round_index for round in packed] == [1]


def test_the_drop_knob_drops_only_the_round_that_found_no_context() -> None:
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
    assembler.expect_round(first, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, 2, round_records.WINDOW_INPUT_ROUTE)
    second_round = fragment(2)
    assembler.expect_round(second_round, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(second_round, 1, round_records.WINDOW_INPUT_ROUTE)
    last = fragment(1, fragment_index=1)
    assembler.expect_round(last, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(last, 2, round_records.WINDOW_INPUT_ROUTE)

    assert recorder.packing_drops == 1
    dropped = [
        event.round_index
        for event in recorder.events
        if event.kind == "DROPPED"
    ]
    assert dropped == [2]
    assert [round.packet.round_index for round in packed] == [1]


def test_detection_events_are_formed_once_from_the_merged_bits() -> None:
    engine = engine_module.Engine()
    packed = []
    former = _Former((0, 1, 1))

    settings = controller_settings.ControllerSettings()
    unbounded = rounds_in_flight(None)
    events = formation(former)
    assembler = round_assembly.RoundAssembler(engine, settings)
    assembler.detection_events = events
    assembler.rounds_in_flight = unbounded
    assembler.syndrome_round_sender = types.SimpleNamespace(admit=packed.append)
    first = fragment(1, fragment_index=0, bits=(1, 0))
    second = fragment(1, fragment_index=1, bits=(0, 1))

    assembler.expect_round(first, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, 2, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(second, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(second, 2, round_records.WINDOW_INPUT_ROUTE)

    assert former.asked == [(1, 1, (1, 0, 0, 1))]
    (round,) = packed
    (formed,) = round.packet.fragments
    assert formed.bits == (0, 1, 1)
    assert formed.size_bits == 3
    # the row forms events here, so the round leaves three bits wide,
    # not the four measurement outcomes it was merged from
    assert round.wire_bits == 3


def test_events_formed_at_the_decoder_keep_the_raw_measurement_width() -> None:
    """The store and the input link carry the outcomes, not the events."""
    engine = engine_module.Engine()
    packed = []
    former = _Former((0, 1, 1))

    settings = controller_settings.ControllerSettings()
    events = detection_event_formation.DecoderSideFormation(former, 0)
    unbounded = rounds_in_flight(None)
    assembler = round_assembly.RoundAssembler(engine, settings)
    assembler.detection_events = events
    assembler.rounds_in_flight = unbounded
    assembler.syndrome_round_sender = types.SimpleNamespace(admit=packed.append)
    first = fragment(1, fragment_index=0, bits=(1, 0))
    second = fragment(1, fragment_index=1, bits=(0, 1))

    assembler.expect_round(first, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, 2, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(second, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(second, 2, round_records.WINDOW_INPUT_ROUTE)

    (round,) = packed
    (left,) = round.packet.fragments
    assert left.bits == (1, 0, 0, 1)
    assert left.size_bits == 4
    assert round.wire_bits == 4
    assert former.asked == []


def test_the_controller_row_delays_the_round_by_its_formation_time() -> None:
    """The charge moves the departure and nothing else."""
    engine = engine_module.Engine()
    departures = _Departures(engine)
    formation_ticks = config.microseconds_to_ticks(0.02)
    formation_clock = config.Clock(formation_ticks)
    settings = controller_settings.ControllerSettings(clock=formation_clock)
    former = _Former((0, 1, 1))
    events = formation(former, detection_event_formation_cycles=1)
    unbounded = rounds_in_flight(None)
    assembler = round_assembly.RoundAssembler(engine, settings)
    assembler.detection_events = events
    assembler.rounds_in_flight = unbounded
    assembler.syndrome_round_sender = types.SimpleNamespace(
        admit=departures.record
    )

    only_fragment = fragment(1, bits=(1, 0))

    assembler.expect_round(only_fragment, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(only_fragment, 1, round_records.WINDOW_INPUT_ROUTE)
    engine.run()

    departure_ticks, round = departures.records[0]
    assert departure_ticks == formation_ticks
    assert round.wire_bits == 3


def test_an_uncharged_controller_row_hands_the_round_on_at_once() -> None:
    engine = engine_module.Engine()
    departures = _Departures(engine)
    settings = controller_settings.ControllerSettings()
    former = _Former((0, 1, 1))
    events = formation(former)
    unbounded = rounds_in_flight(None)
    assembler = round_assembly.RoundAssembler(engine, settings)
    assembler.detection_events = events
    assembler.rounds_in_flight = unbounded
    assembler.syndrome_round_sender = types.SimpleNamespace(
        admit=departures.record
    )
    only_fragment = fragment(1, bits=(1, 0))

    assembler.expect_round(only_fragment, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(only_fragment, 1, round_records.WINDOW_INPUT_ROUTE)

    departure_ticks, _round = departures.records[0]
    assert departure_ticks == 0


def test_interleaved_patch_fragments_keep_measurement_order() -> None:
    engine = engine_module.Engine()
    packed = []
    assembler = assembler_with(engine, packed, None)
    first = fragment(1, fragment_index=0, bits=(1,))
    second = fragment(1, fragment_index=1, bits=(0,))
    second = dataclasses.replace(second, patch_ids=("other",))
    third = fragment(1, fragment_index=2, bits=(1,))

    assembler.expect_round(third, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(third, 3, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(first, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, 3, round_records.WINDOW_INPUT_ROUTE)
    assert packed == []
    assembler.expect_round(second, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(second, 3, round_records.WINDOW_INPUT_ROUTE)
    engine.run()

    (round,) = packed
    assert round.packet.fragments == (first, second, third)
    assert round.wire_bits == 3


def test_settlement_reports_a_round_still_in_assembly() -> None:
    engine = engine_module.Engine()
    recorder = None
    assembler = assembler_with(engine, [], recorder)
    first = fragment(1, fragment_index=0)
    assembler.expect_round(first, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, 2, round_records.WINDOW_INPUT_ROUTE)

    with pytest.raises(RuntimeError, match="incomplete syndrome packing"):
        assembler.check_settled()


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
