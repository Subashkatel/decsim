"""The assembler: fragments become one packed round after the packing time.

The packing time is charged once per complete round, however many
fragments it arrived in (Caune et al. 2410.05202 measure 250 to 370 FPGA
cycles for packetization, bus transfer, result return and the
conditional together, an upper bound); a round arrives in several
fragments when its last checks and its data readout leave as separate
emissions (finalize_stream_round in decsim/ports.py). The bound counts
every round in flight
through the stage from its emission, on the readout link, in assembly,
held for store room or on its route (controller.packing_rounds_in_flight).
Places are taken in emission order, and a round that finds the stage full
waits in front of it and enters on the retry, never lost
(gem5 src/mem/port.hh:244-262).
"""

import dataclasses
import functools
import operator
import types

import pytest

import decsim.config as config
import decsim.controller.round_assembly as round_assembly
import decsim.controller.settings as controller_settings
import decsim.controller.syndrome_round_sender as syndrome_round_sender
import decsim.engine as engine_module
import decsim.observe.round_events as round_events
import decsim.records.rounds as round_records

PACKING_TICKS = config.microseconds_to_ticks(1.0)
# a 1 MHz controller, so one packing cycle is the packing time above
PACKING_CLOCK = config.Clock(PACKING_TICKS)


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
    bound.syndrome_round_sender = types.SimpleNamespace(strong_crossing_count=0)
    return bound


def formation(former, cycles=0, clock=PACKING_CLOCK):
    """The former seated at the controller, charging cycles on clock."""
    return _Placement(former, "controller", cycles, clock)


def assembler_with(engine, packed, recorder, **settings_fields):
    settings = controller_settings.ControllerSettings(**settings_fields)
    bound = rounds_in_flight(settings.packing_rounds_in_flight)
    events = formation(None)
    assembler = round_assembly.RoundAssembler(engine, settings)
    assembler.detection_events = events
    assembler.rounds_in_flight = bound
    assembler.syndrome_round_sender = types.SimpleNamespace(admit=packed.append)
    assembler.packing_line = syndrome_round_sender.HeldRounds(engine)
    if recorder is not None:
        assembler.trace.round_event.connect(recorder.record)
    return assembler


def test_later_completed_round_waits_for_the_prior_emitted_round() -> None:
    engine = engine_module.Engine()
    packed = []
    assembler = assembler_with(engine, packed, None)
    first = fragment(1, bits=(1, 0))
    second = fragment(2, bits=(0, 1))
    assembler.expect_round(first, 1, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(second, 1, round_records.WINDOW_INPUT_ROUTE)

    assembler.add(second, round_records.WINDOW_INPUT_ROUTE)
    assert packed == []
    assembler.add(first, round_records.WINDOW_INPUT_ROUTE)

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
    assembler.expect_round(first, 1, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(second, 1, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(independent, 1, round_records.WINDOW_INPUT_ROUTE)

    assembler.add(second, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(independent, round_records.WINDOW_INPUT_ROUTE)

    operation_ids = [round.packet.operation_id for round in packed]
    assert operation_ids == ["independent"]
    assembler.add(first, round_records.WINDOW_INPUT_ROUTE)
    round_indices = [round.packet.round_index for round in packed]
    assert round_indices == [1, 1, 2]


def test_formation_retains_capacity_until_the_round_leaves() -> None:
    engine = engine_module.Engine()
    departures = _Departures(engine)
    settings = controller_settings.ControllerSettings(
        clock=PACKING_CLOCK, packing_rounds_in_flight=1
    )
    assembler = round_assembly.RoundAssembler(engine, settings)
    assembler.detection_events = formation(None, cycles=10)
    assembler.rounds_in_flight = rounds_in_flight(1)
    assembler.syndrome_round_sender = types.SimpleNamespace(
        admit=departures.record
    )
    assembler.packing_line = syndrome_round_sender.HeldRounds(engine)
    first = fragment(1)
    second = fragment(2)
    assembler.expect_round(first, 1, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(second, 1, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, round_records.WINDOW_INPUT_ROUTE)

    assert departures.records == []
    assembler.add(second, round_records.WINDOW_INPUT_ROUTE)
    assert assembler.packing_line.count == 1
    engine.run()
    (departure_ticks, packed) = departures.records[0]
    expected_ticks = 10 * PACKING_TICKS
    assert departure_ticks == expected_ticks
    assert packed.packet.round_index == 1


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
    assembler.expect_round(first, 2, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(second, 2, round_records.WINDOW_INPUT_ROUTE)
    add_first = functools.partial(
        assembler.add, first, round_records.WINDOW_INPUT_ROUTE
    )
    add_second = functools.partial(
        assembler.add, second, round_records.WINDOW_INPUT_ROUTE
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


def test_a_round_that_finds_the_stage_full_waits_until_a_round_leaves() -> None:
    """The refused round is held and re-offered on the retry, never lost.

    It packs at the retry's own tick. gem5's refused requester "must
    wait for a recvReqRetry" (src/mem/port.hh:244-255). The QPU keeps
    measuring meanwhile.
    """
    engine = engine_module.Engine()
    packed = []
    recorder = round_events.RoundEventRecorder(engine)
    assembler = assembler_with(
        engine, packed, recorder, packing_rounds_in_flight=1
    )
    first = fragment(1, fragment_index=0)
    last = fragment(1, fragment_index=1)
    second_round = fragment(2)
    assembler.expect_round(first, 2, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(second_round, 1, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(second_round, round_records.WINDOW_INPUT_ROUTE)
    waiting_count = assembler.packing_line.count
    assembler.add(last, round_records.WINDOW_INPUT_ROUTE)
    packed_before_the_retry = len(packed)

    assembler.packing_line.retry()
    engine.run()

    round_indices = [round.packet.round_index for round in packed]
    assert waiting_count == 1
    assert packed_before_the_retry == 1
    assert round_indices == [1, 2]
    assert engine.now == 0
    assembler.check_settled()


def test_places_are_taken_in_emission_order_so_a_late_round_one_packs():
    """Round 2 arrives first and round 1 still gets the one place.

    A round takes its place in the order the QPU emitted it, as gem5's
    O3 rename stalls in program order when the reorder buffer has no free
    entry (src/cpu/o3/rename.cc:556-584) and commit inserts in that order
    (src/cpu/o3/commit.cc:1295-1316). A round that arrived early cannot
    take the place its predecessor needs to retire it.
    """
    engine = engine_module.Engine()
    packed = []
    assembler = assembler_with(engine, packed, None, packing_rounds_in_flight=1)
    first = fragment(1)
    second = fragment(2)
    assembler.expect_round(first, 1, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(second, 1, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(second, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, round_records.WINDOW_INPUT_ROUTE)
    packed_before_the_retry = len(packed)

    assembler.packing_line.retry()
    engine.run()

    round_indices = [round.packet.round_index for round in packed]
    assert packed_before_the_retry == 1
    assert round_indices == [1, 2]
    assembler.check_settled()


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
    assembler.packing_line = syndrome_round_sender.HeldRounds(engine)
    first = fragment(1)
    assembler.expect_round(first, 1, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, round_records.WINDOW_INPUT_ROUTE)
    waiting_count = assembler.packing_line.count

    full.capacity = 3
    assembler.packing_line.retry()
    engine.run()

    assert waiting_count == 1
    assert full.count(0) == 2
    assert [round.packet.round_index for round in packed] == [1]


def test_a_round_joined_from_two_fragments_states_both_event_widths() -> None:
    """A valueless round's checks and data readout form 8 and 4 events."""
    engine = engine_module.Engine()
    packed = []
    assembler = assembler_with(engine, packed, None)
    checks = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=1,
        bits=None,
        size_bits=8,
        fragment_index=0,
        event_bits=8,
    )
    readout = dataclasses.replace(
        checks, size_bits=9, fragment_index=1, event_bits=4
    )
    assembler.expect_round(checks, 2, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(checks, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(readout, round_records.WINDOW_INPUT_ROUTE)

    engine.run()

    (round,) = packed
    (joined,) = round.packet.fragments
    assert joined.size_bits == 17
    assert joined.event_bits == 12


def test_detection_events_are_formed_once_from_the_merged_bits() -> None:
    engine = engine_module.Engine()
    packed = []
    former = _Former((0, 1, 1))

    settings = controller_settings.ControllerSettings()
    unbounded = rounds_in_flight(None)
    events = formation(former)
    assembler = round_assembly.RoundAssembler(engine, settings)
    assembler.packing_line = syndrome_round_sender.HeldRounds(engine)
    assembler.detection_events = events
    assembler.rounds_in_flight = unbounded
    assembler.syndrome_round_sender = types.SimpleNamespace(admit=packed.append)
    first = fragment(1, fragment_index=0, bits=(1, 0))
    second = fragment(1, fragment_index=1, bits=(0, 1))

    assembler.expect_round(first, 2, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(second, 2, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(second, round_records.WINDOW_INPUT_ROUTE)

    assert former.asked == [(1, 1, (1, 0, 0, 1))]
    (round,) = packed
    (formed,) = round.packet.fragments
    assert formed.bits == (0, 1, 1)
    assert formed.size_bits == 3
    # the seat forms events here, so the round leaves three bits wide,
    # not the four measurement outcomes it was merged from
    assert round.wire_bits == 3


def test_events_formed_at_the_decoder_keep_the_raw_measurement_width() -> None:
    """The store and the input link carry the outcomes, not the events."""
    engine = engine_module.Engine()
    packed = []
    former = _Former((0, 1, 1))

    settings = controller_settings.ControllerSettings()
    events = _Placement(former, "weak_decoder")
    unbounded = rounds_in_flight(None)
    assembler = round_assembly.RoundAssembler(engine, settings)
    assembler.packing_line = syndrome_round_sender.HeldRounds(engine)
    assembler.detection_events = events
    assembler.rounds_in_flight = unbounded
    assembler.syndrome_round_sender = types.SimpleNamespace(admit=packed.append)
    first = fragment(1, fragment_index=0, bits=(1, 0))
    second = fragment(1, fragment_index=1, bits=(0, 1))

    assembler.expect_round(first, 2, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(second, 2, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(second, round_records.WINDOW_INPUT_ROUTE)

    (round,) = packed
    (left,) = round.packet.fragments
    assert left.bits == (1, 0, 0, 1)
    assert left.size_bits == 4
    assert round.wire_bits == 4
    assert former.asked == []


def test_the_controller_seat_delays_the_round_by_its_formation_time() -> None:
    """The charge moves the departure and nothing else."""
    engine = engine_module.Engine()
    departures = _Departures(engine)
    formation_ticks = config.microseconds_to_ticks(0.02)
    formation_clock = config.Clock(formation_ticks)
    settings = controller_settings.ControllerSettings()
    former = _Former((0, 1, 1))
    events = formation(former, cycles=1, clock=formation_clock)
    unbounded = rounds_in_flight(None)
    assembler = round_assembly.RoundAssembler(engine, settings)
    assembler.packing_line = syndrome_round_sender.HeldRounds(engine)
    assembler.detection_events = events
    assembler.rounds_in_flight = unbounded
    assembler.syndrome_round_sender = types.SimpleNamespace(
        admit=departures.record
    )

    only_fragment = fragment(1, bits=(1, 0))

    assembler.expect_round(only_fragment, 1, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(only_fragment, round_records.WINDOW_INPUT_ROUTE)
    engine.run()

    departure_ticks, round = departures.records[0]
    assert departure_ticks == formation_ticks
    assert round.wire_bits == 3


def test_interleaved_patch_fragments_keep_measurement_order() -> None:
    engine = engine_module.Engine()
    packed = []
    assembler = assembler_with(engine, packed, None)
    first = fragment(1, fragment_index=0, bits=(1,))
    second = fragment(1, fragment_index=1, bits=(0,))
    second = dataclasses.replace(second, patch_ids=("other",))
    third = fragment(1, fragment_index=2, bits=(1,))

    assembler.expect_round(third, 3, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(third, round_records.WINDOW_INPUT_ROUTE)
    assembler.expect_round(first, 3, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, round_records.WINDOW_INPUT_ROUTE)
    assert packed == []
    assembler.expect_round(second, 3, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(second, round_records.WINDOW_INPUT_ROUTE)
    engine.run()

    (round,) = packed
    assert round.packet.fragments == (first, second, third)
    assert round.wire_bits == 3


def test_settlement_reports_a_round_still_in_assembly() -> None:
    engine = engine_module.Engine()
    recorder = None
    assembler = assembler_with(engine, [], recorder)
    first = fragment(1, fragment_index=0)
    assembler.expect_round(first, 2, round_records.WINDOW_INPUT_ROUTE)
    assembler.add(first, round_records.WINDOW_INPUT_ROUTE)

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
    """One round's events whatever its bits, and the bits it was asked."""

    def __init__(self, events):
        self.events = events
        self.asked = []

    def form(self, fragments):
        """The round's fragments as one fragment of the events."""
        first = fragments[0]
        bits = []
        by_fragment_index = operator.attrgetter("fragment_index")
        for fragment in sorted(fragments, key=by_fragment_index):
            bits.extend(fragment.bits)
        self.asked.append((first.operation_id, first.round_index, tuple(bits)))
        size_bits = len(self.events)
        formed = dataclasses.replace(
            first, bits=self.events, size_bits=size_bits
        )
        return (formed,)


class _Placement:
    """A detection event placement seated at one seat, with its own cost."""

    def __init__(self, former, seat, cycles=0, clock=None):
        self.former = former
        self.seat = seat
        self.cycles = cycles
        self.clock = clock

    def forms_at(self, seat):
        """Whether this is the seat."""
        return seat == self.seat

    def form_at(self, seat, fragments):
        """The round formed at the seat, as it came anywhere else."""
        if seat != self.seat or self.former is None:
            return fragments
        return self.former.form(fragments)

    def cycles_at(self, seat, round_count):
        """The cost at the seat, nothing anywhere else."""
        del round_count
        if seat != self.seat:
            return 0
        return self.cycles
