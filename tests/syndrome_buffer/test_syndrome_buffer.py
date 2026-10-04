"""A syndrome buffer answers room, keeps rounds while held, and frees in order.

Referents: the closed form of a store of K slots with Type I blocking,
round k enters at max(arrival k, the (k minus K)-th smallest exit among
the rounds before it), which is the exit of round k minus K when rounds
leave in order; that is Ciw's blocking law
(Ciw's ciw/node.py: finish_service blocks when
the next node is at node_capacity, release_blocked_individual releases
the longest blocked one when a customer leaves); the same trace runs
through a two-node Ciw network, Ciw 3.2.7 from the test extra, one slot
standing for one round's bits. gem5's packet store
answers `avail()` against the packet's own length before it lands (gem5
src/dev/net/pktfifo.hh) and its blocked port retries the requester
(src/mem/cache/base.cc: clearBlocked, processSendRetry); the store
never refuses a write. Over rounds of unequal width, the last the
widest, held by several readers, the occupied bits and the room answer
follow that packet store with a holder set per round.
"""

import collections
import dataclasses
import functools
import random

import pytest

import decsim.config as config
import decsim.controller.syndrome_round_sender as syndrome_round_sender
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.windows.schemes.parallel as parallel_scheme
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run

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
    return syndrome_round_sender.HeldRounds(engine)


def store(bits=None, waiting_line=None, listener=None):
    settings = syndrome_buffer_module.SyndromeBufferSettings(bits=bits)
    engine = engine_module.Engine()
    the_store = syndrome_buffer_module.SyndromeBuffer(settings, engine)
    if waiting_line is not None:
        the_store.held_rounds = waiting_line
    if listener is not None:
        the_store.trace.round_stored.connect(listener.round_stored)
        the_store.trace.round_released.connect(listener.round_released)
    return the_store


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


def test_a_round_enters_at_arrival_or_when_a_slot_frees_property():
    for seed in range(20):
        generator = random.Random(seed)
        capacity = generator.choice([1, 2, 3])
        holds = [generator.randint(0, 8) for _ in range(30)]
        arrivals = [2 * index for index in range(1, 31)]
        expected = closed_form(arrivals, holds, capacity)
        assert run_trace(arrivals, holds, capacity) == expected, seed


def test_ciw_blocks_at_the_same_ticks_over_random_holds_property():
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
    readout), so it is drawn wider than the rest. A hold here may end
    before its rounds are all stored, as a potential strong read does;
    a window's read never does, and the store stops on one it cannot
    hold.
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
        holder = decoding_records.PotentialStrong((1, reader))
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


def test_occupancy_and_room_follow_the_packet_store_property():
    """A property test: 300 random programs against the reference store."""
    for seed in range(300):
        generator = random.Random(seed)
        capacity_bits = generator.choice([24, 40, 64, 100])
        steps = random_hold_program(generator)
        ours, theirs = run_on_both(steps, capacity_bits)
        assert ours == theirs, seed


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


def test_settlement_reports_a_hold_on_a_round_never_written():
    the_store = store()
    reads = decoding_records.WindowReads((1, 0))
    the_store.register_hold(reads, [(1, 5)])

    with pytest.raises(RuntimeError):
        the_store.check_settled()


def blocked_successor() -> list:
    """mem1, then mem2 blocked by mem1's feedback.

    mem1's patch keeps measuring while mem2 waits, so a run whose rounds
    wait for room forever never reaches its end-of-run checks.
    """
    first = declared_run.memory_operation(1)
    second = declared_run.memory_operation(2, predecessors=(1,), blocked_by=1)
    return [first, second]


NARROWER_THAN_A_ROUND = syndrome_buffer_module.SyndromeBufferSettings(bits=1)


def test_a_weak_store_narrower_than_a_round_stops_the_run():
    """No free makes room for the round (gem5 Network.cc:64-65)."""
    operations = blocked_successor()

    with pytest.raises(RuntimeError, match="no round leaving it makes room"):
        declared_run.weak_only_run(
            operations=operations, weak_syndrome_buffer=NARROWER_THAN_A_ROUND
        )


def test_a_weak_store_too_small_for_a_window_stops_the_run():
    """Window (1, 0) reads mem1's six rounds; 8 bits hold round 1 alone.

    Round 2's 8 bits must sit beside the 4 of round 1, which the window
    keeps until it reads both, so no round leaving makes its room.
    """
    operations = blocked_successor()
    store_of_8_bits = syndrome_buffer_module.SyndromeBufferSettings(bits=8)

    with pytest.raises(RuntimeError, match=r"round \(1, 2\) needs 12 bits"):
        declared_run.weak_only_run(
            operations=operations, weak_syndrome_buffer=store_of_8_bits
        )


def test_interleaved_successors_waiting_for_room_stop_the_run():
    """mem2 and mem3 follow mem1, their rounds interleave; mem4 waits on mem2.

    The first window of each keeps rounds 1 to 5 until round 6 is
    stored, and mem3's round 6 waits behind mem2's, so round (2, 6)'s 12
    bits must sit beside 72, past an 80-bit store.
    """
    memory = declared_run.memory_operation
    operations = [
        memory(1),
        memory(2, predecessors=(1,)),
        memory(3, predecessors=(1,)),
        memory(4, predecessors=(2,), blocked_by=2),
    ]
    store_of_80_bits = syndrome_buffer_module.SyndromeBufferSettings(bits=80)

    with pytest.raises(RuntimeError, match=r"round \(2, 6\) needs 84 bits"):
        declared_run.weak_only_run(
            operations=operations, weak_syndrome_buffer=store_of_80_bits
        )


def test_a_window_waiting_for_a_later_boundary_stops_the_run():
    """Parallel windows of one round: window 1 waits for window 2's boundary.

    Window 1's input takes no slot before window 2 decodes, and window 2
    reads round 6, so round 6's 12 bits must sit beside the 24 of rounds
    3 to 5, past a 32-bit store.
    """
    plain_windows = window_settings.WindowSettings()
    windows = declared_run.windows_on(
        plain_windows,
        parallel_scheme.ParallelWindowScheme,
        commit_rounds=1,
        buffer_rounds=1,
    )
    store_of_32_bits = syndrome_buffer_module.SyndromeBufferSettings(bits=32)
    operations = blocked_successor()

    with pytest.raises(RuntimeError, match=r"round \(1, 6\) needs 36 bits"):
        declared_run.weak_only_run(
            operations=operations,
            windows=windows,
            weak_syndrome_buffer=store_of_32_bits,
        )


def store_of(monkeypatch, field: str, store_settings) -> None:
    """Every declared run from here on sizes that store so."""
    run_machine = declared_run.run_machine

    def run_with_the_store(settings, seed=0, probes=()):
        sized = dataclasses.replace(settings, **{field: store_settings})
        return run_machine(sized, seed, probes)

    monkeypatch.setattr(declared_run, "run_machine", run_with_the_store)


def test_a_strong_store_narrower_than_a_round_stops_the_run(monkeypatch):
    """The strong-primary run writes every round into the strong store."""
    store_of(monkeypatch, "strong_syndrome_buffer", NARROWER_THAN_A_ROUND)
    operations = blocked_successor()

    with pytest.raises(RuntimeError, match="no round leaving it makes room"):
        declared_run.strong_only_run(operations=operations)


def test_a_strong_store_too_small_for_a_region_stops_the_run(monkeypatch):
    """An escalated region cannot wait for room, so it must fit at once.

    Each 6-round region is 48 bits against a store of 8.
    """
    store_of_8_bits = syndrome_buffer_module.SyndromeBufferSettings(bits=8)
    store_of(monkeypatch, "strong_syndrome_buffer", store_of_8_bits)
    operations = blocked_successor()

    with pytest.raises(RuntimeError, match="strong_syndrome_buffer.bits"):
        declared_run.switching_run(escalates=True, operations=operations)


def test_a_region_held_for_its_restart_window_stops_the_run(monkeypatch):
    """The double window keeps mem1's region until its restart commits.

    Every window escalates. The region, rounds 1 to 9, waits for the
    restart window, which reads rounds 7 to 12, so round 12's 12 bits
    must sit beside the 84 of rounds 1 to 11, past an 88-bit store.
    """
    store_of_88_bits = syndrome_buffer_module.SyndromeBufferSettings(bits=88)
    store_of(monkeypatch, "weak_syndrome_buffer", store_of_88_bits)
    operations = blocked_successor()

    with pytest.raises(RuntimeError, match=r"round \(1, 12\) needs 96 bits"):
        declared_run.switching_run(
            rounds=12,
            escalates=True,
            operations=operations,
            strong_window=declared_run.DOUBLE_WINDOW,
        )


def test_an_unbounded_store_takes_a_round_that_states_no_size():
    the_store = store()
    timing_only = unsized_packet(1)

    the_store.accept_packed_round(timing_only, publication_tick=None)

    assert the_store.occupancy == 1
    assert the_store.occupied_bits == 0


def test_a_write_completes_its_write_cycles_after_the_edge_at_or_after_now():
    """gem5's clockEdge then the latency (src/mem/simple_mem.cc:174).

    From tick 1 on a 10-tick clock the edge is 10, and 3 cycles end at 40.
    """
    engine = engine_module.Engine()
    engine.now = 1
    clock = config.Clock(10)
    settings = syndrome_buffer_module.SyndromeBufferSettings(
        clock=clock, write_cycles=3
    )
    the_store = syndrome_buffer_module.SyndromeBuffer(settings, engine)

    assert the_store.book_write((1, 1), BITS_PER_ROUND) == 40
