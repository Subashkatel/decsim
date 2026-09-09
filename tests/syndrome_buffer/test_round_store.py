"""A round store answers room, keeps rounds while held, and frees in order.

Referents: the closed form of a store of K slots with Type I blocking,
round k enters at max(arrival k, the (k minus K)-th smallest exit among
the rounds before it), which is the exit of round k minus K when rounds
leave in order; that is Ciw's blocking law
(Ciw's ciw/node.py: finish_service blocks when
the next node is at node_capacity, release_blocked_individual releases
the longest blocked one when a customer leaves); the same trace runs
through a two-node Ciw network when Ciw imports from the resources
folder. gem5's queue answers isFull
before allocate (gem5 src/mem/cache/queue.hh:150-153)
and its blocked port retries the requester (src/mem/cache/base.cc:
clearBlocked, processSendRetry); the store never refuses a write.

The two whole-run laws at the end of the file place the store in the
pipeline: its publication tick is what makes a weak window ready, on
the declared card of tests/declared_run.py.
"""

import functools
import random

import pytest

import decsim.config as config
import decsim.controller.round_writes as round_writes
import decsim.controller.settings as controller_settings
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings
import tests.declared_run as declared_run

STALL = controller_settings.PackingOverflowPolicy.STALL


def packet(
    round_index: int, operation_id=1
) -> round_records.SyndromeRoundPacket:
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=operation_id,
        patch_id=0,
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
        stored, round_records.WINDOW_INPUT_ROUTE, 2
    )


def held_rounds() -> round_writes.HeldRounds:
    engine = engine_module.Engine()
    return round_writes.HeldRounds(engine, STALL)


def store(rounds=None, on_slot_freed=None, listener=None):
    settings = round_store_settings.RoundStoreSettings(rounds=rounds)
    the_store = round_store_module.RoundStore(
        settings, on_slot_freed=on_slot_freed
    )
    if listener is not None:
        the_store.trace.round_stored.connect(listener.round_stored)
        the_store.trace.round_released.connect(listener.round_released)
    return the_store


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
    the_store = store(rounds=capacity, on_slot_freed=held.retry)
    enters = {}

    def admit(round: round_records.PackedRound) -> bool:
        if not the_store.has_room():
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


def test_a_full_store_answers_no_room_and_is_unchanged():
    the_store = store(rounds=2)
    first = packet(1)
    second = packet(2)
    the_store.accept_packed_round(first, publication_tick=10)
    the_store.accept_packed_round(second, publication_tick=11)

    assert the_store.has_room() is False
    assert the_store.occupancy == 2
    assert the_store.publication_tick((1, 1)) == 10
    assert the_store.publication_tick((1, 2)) == 11


def test_a_held_round_enters_when_a_slot_frees_in_completion_order():
    held = held_rounds()
    the_store = store(rounds=1, on_slot_freed=held.retry)
    entered = []

    def admit(round: round_records.PackedRound) -> bool:
        if not the_store.has_room():
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
    freed = []
    listener = RecordingListener()
    the_store = store(
        on_slot_freed=lambda: freed.append(True), listener=listener
    )
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
    assert freed == [True]


def test_the_listener_hears_a_stored_round_once():
    listener = RecordingListener()
    the_store = store(listener=listener)
    reads = decoding_records.WindowReads((1, 0))
    the_store.register_hold(reads, [(1, 1)])

    first = packet(1)
    the_store.accept_packed_round(first, publication_tick=0)

    assert listener.stored == [(1, 1)]


def test_publication_is_stamped_at_delivery_for_a_priced_hop():
    the_store = store()
    reads = decoding_records.WindowReads((1, 0))
    the_store.register_hold(reads, [(1, 1)])
    first = packet(1)
    the_store.accept_packed_round(first, publication_tick=None)

    unpublished = the_store.publication_tick((1, 1))
    the_store.mark_publication_tick((1, 1), 7)

    assert unpublished is None
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


def test_a_weak_window_is_ready_on_this_store_not_the_room_side_landing():
    """Readiness listens to the store the weak decoder actually reads.

    Toshio arXiv:2510.25222 Sec. III A: the weak tier reads the
    fridge-side store, and syndrome buffer 1 only holds the context an
    escalation would need. On the declared card the room-side hop is
    7 us against this store's 4 us, so the same round lands in syndrome
    buffer 1 three microseconds after this store published it; a
    readiness rule that waited for the slower path would delay every
    weak window by that lag.
    """
    machine = declared_run.switching_run(rounds=9, io_trace=True)
    windows = machine.observation.windows.windows
    first_window = windows[(1, 0)]
    log_lines = machine.observation.log.lines
    published = declared_run.log_tick(log_lines, "round 6 of mem1 arrived")
    landing_line = "received round 6 of op 1 from controller_to_strong_buffer"
    landed = declared_run.log_tick(log_lines, landing_line)
    expected_data_complete = config.microseconds_to_ticks(15.0)
    expected_landing = config.microseconds_to_ticks(18.0)

    assert first_window.t_data_complete == expected_data_complete
    assert published == expected_data_complete
    assert landed == expected_landing
    assert published < landed
