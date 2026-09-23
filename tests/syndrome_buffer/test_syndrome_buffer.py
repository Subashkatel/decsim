"""A syndrome buffer answers room, keeps rounds while held, and frees in order.

Referents: the closed form of a store of K slots with Type I blocking,
round k enters at max(arrival k, the (k minus K)-th smallest exit among
the rounds before it), which is the exit of round k minus K when rounds
leave in order; that is Ciw's blocking law
(Ciw's ciw/node.py: finish_service blocks when
the next node is at node_capacity, release_blocked_individual releases
the longest blocked one when a customer leaves); the same trace runs
through a two-node Ciw network when Ciw imports from the resources
folder, one slot standing for one round's bits. gem5's packet store
answers `avail()` against the packet's own length before it lands (gem5
src/dev/net/pktfifo.hh) and its blocked port retries the requester
(src/mem/cache/base.cc: clearBlocked, processSendRetry); the store
never refuses a write. Over rounds of unequal width, the last the
widest, held by several readers, the occupied bits and the room answer
follow that packet store with a holder set per round.

The two whole-run laws at the end of the file place the store in the
pipeline: its publication tick is what makes a weak window ready, on
the declared card of tests/declared_run.py.
"""

import collections
import functools
import random

import pytest

import decsim.config as config
import decsim.controller.settings as controller_settings
import decsim.controller.syndrome_round_sender as syndrome_round_sender
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.syndrome_buffer.settings as syndrome_buffer_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import tests.declared_run as declared_run

STALL = controller_settings.PackingOverflowPolicy.STALL
# every round this file stores carries one fragment of two bits
BITS_PER_ROUND = 2
# the key the random programs ask room for; no program writes it
PROBE_ROUND = ("probe", 0)


def packet(
    round_index: int, operation_id=1
) -> round_records.SyndromeRoundPacket:
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=operation_id,
        patch_ids=(0,),
        round_index=round_index,
        bits=(1, 0),
        size_bits=2,
        fragment_index=0,
    )
    return round_records.SyndromeRoundPacket(
        operation_id, round_index, (fragment,)
    )


def packed(round_index: int) -> round_records.PackedRound:
    stored = packet(round_index)
    return round_records.PackedRound(
        stored, round_records.WINDOW_INPUT_ROUTE, BITS_PER_ROUND
    )


def unsized_packet(round_index: int) -> round_records.SyndromeRoundPacket:
    """A timing-only round: it carries no bits and states no size."""
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=round_index,
        bits=None,
        size_bits=None,
        fragment_index=0,
    )
    return round_records.SyndromeRoundPacket(1, round_index, (fragment,))


def held_rounds() -> syndrome_round_sender.HeldRounds:
    engine = engine_module.Engine()
    return syndrome_round_sender.HeldRounds(engine, STALL)


def store(bits=None, waiting_line=None, listener=None):
    settings = syndrome_buffer_settings.SyndromeBufferSettings(bits=bits)
    engine = engine_module.Engine()
    the_store = syndrome_buffer_module.SyndromeBuffer(settings, engine)
    if waiting_line is not None:
        the_store.held_rounds = waiting_line
    if listener is not None:
        the_store.trace.round_stored.connect(listener.round_stored)
        the_store.trace.round_released.connect(listener.round_released)
    return the_store


class RecordingWaitingLine:
    """A waiting line that counts the slots the store frees."""

    def __init__(self):
        self.freed_count = 0

    def retry(self):
        self.freed_count += 1


class RecordingListener:
    def __init__(self):
        self.stored = []
        self.released = []

    def round_stored(self, round_key, _packet):
        self.stored.append(round_key)

    def round_released(self, round_key):
        self.released.append(round_key)


def closed_form(arrivals, holds, capacity):
    """Round k enters at max(arrival k, (k - K)-th smallest earlier exit)."""
    enters = []
    exits = []
    for index, arrival in enumerate(arrivals):
        enter = arrival
        if index >= capacity:
            earlier_exits = sorted(exits)
            slot_free = earlier_exits[index - capacity]
            enter = max(arrival, slot_free)
        enters.append(enter)
        exit_tick = enter + holds[index]
        exits.append(exit_tick)
    return enters


def run_trace(arrivals, holds, capacity):
    """The trace through the store: enter ticks, in round order."""
    engine = engine_module.Engine()
    held = held_rounds()
    capacity_bits = capacity * BITS_PER_ROUND
    the_store = store(bits=capacity_bits, waiting_line=held)
    enters = {}

    def admit(round: round_records.PackedRound) -> bool:
        if not the_store.has_room(round.round_key, round.wire_bits, {}):
            return False
        the_store.accept_packed_round(round.packet, publication_tick=engine.now)
        round_index = round.packet.round_index
        enters[round_index] = engine.now
        release_at = holds[round_index - 1]
        release = functools.partial(the_store.release_round, round.round_key)
        engine.schedule(release_at, release)
        return True

    def arrive(round_index) -> None:
        round = packed(round_index)
        admitted = admit(round)
        if not admitted:
            held.refuse(round, admit)

    for round_index, arrival in enumerate(arrivals, start=1):
        engine.schedule(arrival, lambda index=round_index: arrive(index))
    engine.run()
    last = len(arrivals) + 1
    round_indices = range(1, last)
    return [enters[round_index] for round_index in round_indices]


def ciw_trace(arrivals, holds, capacity):
    """The same trace through Ciw: node 1 blocks until node 2 has room."""
    period = arrivals[1] - arrivals[0]
    ciw = pytest.importorskip("ciw")
    arrivals_every_period = ciw.dists.Deterministic(period)
    no_service = ciw.dists.Deterministic(0.0)
    scripted_holds = ciw.dists.Sequential(list(holds))
    network = ciw.create_network(
        arrival_distributions=[arrivals_every_period, None],
        service_distributions=[no_service, scripted_holds],
        number_of_servers=[float("inf"), capacity],
        queue_capacities=[float("inf"), 0],
        routing=[[0.0, 1.0], [0.0, 0.0]],
    )
    simulation = ciw.Simulation(network)
    simulation.simulate_until_max_customers(len(arrivals), method="Finish")
    records = simulation.get_all_records()
    node_two = [record for record in records if record.node == 2]
    node_two.sort(key=lambda record: record.id_number)
    return [record.arrival_date for record in node_two]


def test_a_round_enters_at_its_arrival_or_when_a_slot_frees_over_random_holds():
    for seed in range(20):
        generator = random.Random(seed)
        capacity = generator.choice([1, 2, 3])
        holds = [generator.randint(0, 8) for _ in range(30)]
        arrivals = [2 * index for index in range(1, 31)]
        expected = closed_form(arrivals, holds, capacity)
        assert run_trace(arrivals, holds, capacity) == expected, seed


def test_ciw_blocks_at_the_same_ticks_over_random_holds():
    for seed in range(5):
        generator = random.Random(seed)
        capacity = generator.choice([1, 2, 3])
        holds = [generator.randint(0, 8) for _ in range(30)]
        arrivals = [2 * index for index in range(1, 31)]
        ours = run_trace(arrivals, holds, capacity)
        theirs = ciw_trace(arrivals, holds, capacity)
        assert [float(tick) for tick in ours] == theirs, seed


class PacketStoreWithHoldCounts:
    """The reference: gem5's packet store plus a holder set per round.

    Room is `avail() = _maxsize - _size` against the round's own length
    (src/dev/net/pktfifo.hh:104), a push adds that length (:125-138)
    and a pop takes it back (:143-152). A round is popped at the call
    that drops its last holder, and on arrival when nobody holds it.
    """

    def __init__(self, capacity_bits: int) -> None:
        self.capacity_bits = capacity_bits
        self.occupied_bits = 0
        self.bits_by_round = {}
        self.holders_by_round = collections.defaultdict(set)
        self.rounds_by_holder = {}

    def has_room(self, bits: int) -> bool:
        available = self.capacity_bits - self.occupied_bits
        return available >= bits

    def accept(self, round_key, bits: int) -> None:
        self.bits_by_round[round_key] = bits
        self.occupied_bits += bits
        if not self.holders_by_round[round_key]:
            self.pop(round_key)

    def hold(self, holder, round_keys) -> None:
        self.rounds_by_holder[holder] = round_keys
        for round_key in round_keys:
            self.holders_by_round[round_key].add(holder)

    def release(self, holder) -> None:
        round_keys = self.rounds_by_holder.pop(holder)
        for round_key in round_keys:
            holders = self.holders_by_round[round_key]
            holders.discard(holder)
            if not holders:
                self.pop(round_key)

    def pop(self, round_key) -> None:
        bits = self.bits_by_round.pop(round_key, 0)
        self.occupied_bits -= bits


def sized_packet(round_key, bits: int) -> round_records.SyndromeRoundPacket:
    operation_id, round_index = round_key
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=operation_id,
        patch_ids=(0,),
        round_index=round_index,
        bits=None,
        size_bits=bits,
        fragment_index=0,
    )
    return round_records.SyndromeRoundPacket(
        operation_id, round_index, (fragment,)
    )


def random_hold_program(generator: random.Random) -> list:
    """Holds, releases and in-order writes of rounds of unequal width.

    The last round of a memory is the widest (it closes on the data
    readout), so it is drawn wider than the rest.
    """
    round_count = 12
    past_the_last_round = round_count + 1
    steps = []
    for round_index in range(1, past_the_last_round):
        bits = generator.randint(1, 12)
        if round_index == round_count:
            bits += 8
        steps.append(("write", (1, round_index), bits))
    reader_count = generator.randint(1, 5)
    for reader in range(reader_count):
        holder = decoding_records.WindowReads((1, reader))
        first = generator.randint(1, round_count)
        span = generator.randint(1, 6)
        past_the_span = first + span
        last = min(past_the_span, past_the_last_round)
        round_keys = tuple((1, index) for index in range(first, last))
        register_at = generator.randint(0, len(steps))
        steps.insert(register_at, ("hold", holder, round_keys))
        first_release_place = register_at + 1
        release_at = generator.randint(first_release_place, len(steps))
        steps.insert(release_at, ("release", holder, None))
    return steps


def run_on_both(steps: list, capacity_bits: int) -> tuple:
    """Each side's occupied bits and room answers after every step."""
    the_store = store(bits=capacity_bits)
    reference = PacketStoreWithHoldCounts(capacity_bits)
    ours = []
    theirs = []
    for kind, subject, detail in steps:
        if kind == "hold":
            the_store.register_hold(subject, detail)
            reference.hold(subject, detail)
        if kind == "release":
            the_store.release_hold(subject)
            reference.release(subject)
        if kind == "write":
            write_if_room(the_store, reference, subject, detail)
        our_room = the_store.has_room(PROBE_ROUND, 9, {})
        their_room = reference.has_room(9)
        ours.append((the_store.occupied_bits, our_room))
        theirs.append((reference.occupied_bits, their_room))
    return ours, theirs


def write_if_room(the_store, reference, round_key, bits: int) -> None:
    """Both sides answer room for themselves; a refused round is skipped."""
    if the_store.has_room(round_key, bits, {}):
        written = sized_packet(round_key, bits)
        the_store.accept_packed_round(written, publication_tick=0)
        the_store.release_round_if_unheld(round_key)
    if reference.has_room(bits):
        reference.accept(round_key, bits)


def test_occupancy_and_room_follow_the_packet_store_over_random_holds():
    """A property test: 300 random programs against the reference store."""
    for seed in range(300):
        generator = random.Random(seed)
        capacity_bits = generator.choice([24, 40, 64, 100])
        steps = random_hold_program(generator)
        ours, theirs = run_on_both(steps, capacity_bits)
        assert ours == theirs, seed


def test_a_full_store_answers_no_room_and_is_unchanged():
    two_rounds_bits = 2 * BITS_PER_ROUND
    the_store = store(bits=two_rounds_bits)
    first = packet(1)
    second = packet(2)
    the_store.accept_packed_round(first, publication_tick=10)
    the_store.accept_packed_round(second, publication_tick=11)

    assert the_store.has_room((1, 3), BITS_PER_ROUND, {}) is False
    assert the_store.occupied_bits == two_rounds_bits
    assert the_store.occupancy == 2
    assert the_store.publication_tick((1, 1)) == 10
    assert the_store.publication_tick((1, 2)) == 11


def test_a_held_round_enters_when_a_slot_frees_in_completion_order():
    held = held_rounds()
    the_store = store(bits=BITS_PER_ROUND, waiting_line=held)
    entered = []

    def admit(round: round_records.PackedRound) -> bool:
        if not the_store.has_room(round.round_key, round.wire_bits, {}):
            return False
        the_store.accept_packed_round(round.packet, publication_tick=0)
        entered.append(round.packet.round_index)
        return True

    first = packed(1)
    second = packed(2)
    third = packed(3)
    admit(first)
    held.refuse(second, admit)
    held.refuse(third, admit)
    the_store.release_round((1, 1))
    after_first_release = list(entered)
    the_store.release_round((1, 2))

    assert after_first_release == [1, 2]
    assert entered == [1, 2, 3]
    assert held.count == 0


def test_a_stored_round_is_released_when_its_last_hold_releases():
    waiting_line = RecordingWaitingLine()
    listener = RecordingListener()
    the_store = store(waiting_line=waiting_line, listener=listener)
    weak_reads = decoding_records.WindowReads((1, 0))
    strong_reads = decoding_records.PotentialStrong((1, 0))
    the_store.register_hold(weak_reads, [(1, 1)])
    the_store.register_hold(strong_reads, [(1, 1)])
    first = packet(1)
    the_store.accept_packed_round(first, publication_tick=0)

    the_store.release_hold(weak_reads)
    held_after_first = the_store.retained_fragments((1, 1)) is not None
    the_store.release_hold(strong_reads)

    assert held_after_first is True
    assert the_store.retained_fragments((1, 1)) is None
    assert listener.released == [(1, 1)]
    assert waiting_line.freed_count == 1


def test_the_listener_hears_a_stored_round_once():
    listener = RecordingListener()
    the_store = store(listener=listener)
    reads = decoding_records.WindowReads((1, 0))
    the_store.register_hold(reads, [(1, 1)])

    first = packet(1)
    the_store.accept_packed_round(first, publication_tick=0)

    assert listener.stored == [(1, 1)]


def test_a_round_is_readable_at_the_tick_it_is_stored_and_not_before():
    """One call, one tick: the bits and the publication arrive together.

    A round with no publication tick is a timing-only round, which no
    window reads (syndrome_buffer/weak_syndrome_round_receiver.py,
    send_memory_round).
    """
    the_store = store()
    reads = decoding_records.WindowReads((1, 0))
    the_store.register_hold(reads, [(1, 1)])
    first = packet(1)
    unstored = the_store.publication_tick((1, 1))

    the_store.accept_packed_round(first, publication_tick=7)

    assert unstored is None
    assert the_store.publication_tick((1, 1)) == 7


def test_an_unheld_round_is_freed_on_arrival_and_a_held_one_is_not():
    the_store = store()
    reads = decoding_records.WindowReads((1, 0))
    the_store.register_hold(reads, [(1, 2)])
    first = packet(1)
    the_store.accept_packed_round(first, publication_tick=0)
    second = packet(2)
    the_store.accept_packed_round(second, publication_tick=0)

    freed_unheld = the_store.release_round_if_unheld((1, 1))
    freed_held = the_store.release_round_if_unheld((1, 2))

    assert freed_unheld is True
    assert freed_held is False
    assert the_store.occupancy == 1


def test_a_closed_operation_identity_never_reopens():
    the_store = store()
    the_store.open_operation(1)
    the_store.close_operation(1)

    with pytest.raises(RuntimeError, match="closed operation identities"):
        the_store.open_operation(1)


def test_settlement_reports_a_hold_on_a_round_never_written():
    the_store = store()
    reads = decoding_records.WindowReads((1, 0))
    the_store.register_hold(reads, [(1, 5)])

    with pytest.raises(RuntimeError, match=r"unresolved holds on \[\(1, 5\)\]"):
        the_store.check_settled()


def test_a_bounded_store_ending_under_a_live_hold_names_its_widest_hold():
    """The hold waits for a round its own stored rounds leave no room for."""
    two_rounds_bits = 2 * BITS_PER_ROUND
    the_store = store(bits=two_rounds_bits)
    wide = decoding_records.WindowReads((1, 0))
    narrow = decoding_records.WindowReads((1, 1))
    the_store.register_hold(wide, [(1, 1), (1, 2), (1, 3)])
    the_store.register_hold(narrow, [(1, 2)])
    first = packet(1)
    second = packet(2)
    the_store.accept_packed_round(first, publication_tick=0)
    the_store.accept_packed_round(second, publication_tick=0)

    with pytest.raises(RuntimeError) as stop:
        the_store.check_settled()

    sentence = str(stop.value)
    assert "4 of its 4 bits" in sentence
    assert "widest live hold WindowReads(window_key=(1, 0))" in sentence
    assert "keeps 4 bits of them and waits for [(1, 3)]" in sentence


def test_the_hold_sources_carry_the_token_and_its_rounds():
    heard = []
    the_store = store()

    def registered(holder, keys):
        heard.append(("registered", holder, keys))

    def transferred(old, new):
        heard.append(("transferred", old, new))

    def released(holder):
        heard.append(("released", holder))

    the_store.trace.hold_registered.connect(registered)
    the_store.trace.hold_transferred.connect(transferred)
    the_store.trace.hold_released.connect(released)
    reads = decoding_records.WindowReads((1, 0))
    potential = decoding_records.PotentialStrong((1, 0))
    the_store.register_hold(reads, [(1, 1), (1, 2)])
    the_store.transfer_hold(reads, potential)
    the_store.release_hold(potential)
    assert heard == [
        ("registered", reads, ((1, 1), (1, 2))),
        ("transferred", reads, potential),
        ("released", potential),
    ]


def test_a_bounded_store_has_room_while_the_bits_fit_beside_what_is_taken():
    """gem5's avail(): the capacity less what is stored and reserved."""
    three_rounds_bits = 3 * BITS_PER_ROUND
    one_round_reserved = BITS_PER_ROUND
    the_store = store(bits=three_rounds_bits)
    first = packet(1)
    the_store.accept_packed_round(first, publication_tick=0)

    fits_beside_the_stored = the_store.has_room((1, 2), BITS_PER_ROUND, {})
    fits_beside_the_reserved = the_store.has_room(
        (1, 3), BITS_PER_ROUND, {(1, 2): one_round_reserved}
    )
    exceeds_the_capacity = the_store.has_room((1, 2), three_rounds_bits, {})

    assert fits_beside_the_stored is True
    assert fits_beside_the_reserved is True
    assert exceeds_the_capacity is False
    assert the_store.occupied_bits == BITS_PER_ROUND


def test_a_bounded_store_refuses_a_round_that_states_no_size():
    """A bound is measured against a size, so the round must state one."""
    the_store = store(bits=BITS_PER_ROUND)

    with pytest.raises(
        RuntimeError, match="a bounded syndrome buffer needs sized rounds"
    ):
        the_store.has_room((1, 1), None, {})


def test_a_bounded_store_refuses_a_round_wider_than_itself():
    """No free makes room for it (gem5 Network.cc:64-65 refuses the same)."""
    the_store = store(bits=BITS_PER_ROUND)
    wide_bits = BITS_PER_ROUND + 1

    with pytest.raises(RuntimeError) as refusal:
        the_store.has_room((1, 4), wide_bits, {})

    sentence = str(refusal.value)
    assert f"holds {BITS_PER_ROUND} bits and round (1, 4) states 3" in sentence


def test_an_unbounded_store_takes_a_round_that_states_no_size():
    the_store = store()
    timing_only = unsized_packet(1)

    the_store.accept_packed_round(timing_only, publication_tick=None)

    assert the_store.occupancy == 1
    assert the_store.occupied_bits == 0


# ---- the publication tick in the whole pipeline


def test_the_weak_primary_pipeline_runs_on_the_declared_ticks():
    """The store's publication is the window's readiness, hop by hop.

    Round r becomes public once the readout has crossed
    qpu_to_controller, been classified into bits, and crossed
    controller_to_weak_buffer; the window that round completes is
    queued and dispatched on that same tick, and the weak decode is
    charged the weak_buffer_to_weak_decoder transfer before its own
    latency. Every latency is the declared card's
    (tests/declared_run.py), so each stamp is exact arithmetic.
    """
    machine = declared_run.weak_only_run(rounds=6)
    windows = machine.observation.windows.windows
    window = windows[(1, 0)]
    snapshot = machine.pauli_frame.snapshot()
    (record,) = snapshot.records
    expected_first_round = config.microseconds_to_ticks(10.0)
    expected_data_complete = config.microseconds_to_ticks(15.0)
    expected_done = config.microseconds_to_ticks(30.0)
    expected_accepted = config.microseconds_to_ticks(32.0)
    expected_committed = config.microseconds_to_ticks(33.0)

    assert window.t_first_round == expected_first_round
    assert window.t_data_complete == expected_data_complete
    assert window.t_queued == expected_data_complete
    assert window.t_dispatch == expected_data_complete
    assert window.t_done == expected_done
    assert record.tier == "weak"
    assert record.accepted_ticks == expected_accepted
    assert record.committed_ticks == expected_committed


def test_a_weak_window_is_ready_on_this_store_and_nothing_lands_room_side():
    """Readiness listens to the store the weak decoder actually reads.

    Toshio arXiv:2510.25222 Sec. III A: the weak tier reads the
    fridge-side store, and the strong syndrome buffer only holds what an
    escalation carried up. A switching run whose windows are all kept
    lands nothing on the room side, so nothing there could delay a weak
    window: round 6 is published here at 15 us, the declared card's
    readout, controller and store hops after its 10 us emission.
    """
    machine = declared_run.switching_run(rounds=9, io_trace=True)
    windows = machine.observation.windows.windows
    first_window = windows[(1, 0)]
    log_lines = machine.observation.log.lines
    published = declared_run.log_tick(log_lines, "round 6 of mem1 arrived")
    landings = [line for line in log_lines if "received round" in line]
    room_side_landings = [line for line in landings if "strong" in line]
    expected_data_complete = config.microseconds_to_ticks(15.0)

    assert first_window.t_data_complete == expected_data_complete
    assert published == expected_data_complete
    assert room_side_landings == []
    assert machine.strong_syndrome_buffer.occupancy == 0


def test_a_write_completes_its_write_cycles_after_the_edge_at_or_after_now():
    """gem5's clockEdge then the latency (src/mem/simple_mem.cc:174).

    From tick 1 on a 10-tick clock the edge is 10, and 3 cycles end at 40.
    """
    engine = engine_module.Engine()
    engine.now = 1
    clock = config.Clock(10)
    settings = syndrome_buffer_settings.SyndromeBufferSettings(
        clock=clock, write_cycles=3
    )
    the_store = syndrome_buffer_module.SyndromeBuffer(settings, engine)

    assert the_store.book_write((1, 1), BITS_PER_ROUND) == 40
