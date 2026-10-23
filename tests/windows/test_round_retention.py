"""The round retention's laws on a real syndrome buffer.

A window holds [start_round, buffer_hi] plus the successor overflow
(Skoric et al. 2209.08552: the buffer region is re-read by the next
window); the hold moves to the request at admission and is released
once the input lands; an unheld round is freed on arrival. Under the
double window a window's potential restart read (PotentialRestart,
placed by the planner) keeps the rounds its restart decode would read
past the landing, follows a re-slice, and ends when the retention is
told no earlier escalation can re-slice the window (Toshio 2510.25222
Sec. III C). A read whose reader forms the detection events also holds
the raw rounds its first round's recipes read, the rounds Stim's
converter reads for that round's detectors
(stim.Circuit.compile_m2d_converter). A chain keeps a window's potential
read on each escalate edge, in the store the edge carries from and the
one it lands in, until the window is final.
"""

import types

import pytest
import stim

import decsim.detector_error_model.detection_event_formation as formation
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.settings as event_settings
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.windows.round_retention as round_retention
import tests.one_entry as one_entry


def _store() -> syndrome_buffer_module.SyndromeBuffer:
    settings = syndrome_buffer_module.SyndromeBufferSettings()
    engine = engine_module.Engine()
    return syndrome_buffer_module.SyndromeBuffer(settings, engine)


def _retention(store, round_counts: dict, successors: dict):
    planner = types.SimpleNamespace(successors_by_stream=successors)

    def effective_round_count_for_window(stream_id, _window):
        return round_counts[stream_id]

    tracker = types.SimpleNamespace(
        effective_round_count_for_window=effective_round_count_for_window,
        rounds_arrived=lambda _stream_id: 0,
        strong_rounds_arrived=lambda _stream_id, _store: 0,
    )
    retention = round_retention.RoundRetention(
        is_strong_context_retained=False,
        primary_tier=window_records.WEAK_POOL_ID,
    )
    retention.router = one_entry.stores_router(
        {window_records.WEAK_POOL_ID: store}, _all_to_weak
    )
    retention.planner = planner
    retention.tracker = tracker
    return retention


def _publish_rounds(store, stream_id: int, round_indices) -> None:
    """Each round stored at the tick of its own index."""
    for round_index in round_indices:
        packet = _packet(stream_id, round_index)
        store.accept_packed_round(packet, publication_tick=round_index)


def _packet(stream_id, round_index) -> round_records.SyndromeRoundPacket:
    fragment = round_records.RetainedSyndromeFragment(
        stream_id=stream_id,
        patch_ids=(0,),
        round_index=round_index,
        bits=None,
        size_bits=None,
        fragment_index=0,
    )
    return round_records.SyndromeRoundPacket(
        stream_id, round_index, (fragment,)
    )


def _all_to_weak(_stream_id) -> window_records.PoolId:
    return window_records.WEAK_POOL_ID


def _strong_retention(
    strong_store, rounds_arrived: int, entry_of=_all_to_weak, other_entries=()
):
    """A retention whose strong syndrome buffer is the one under test.

    entry_of names each stream's entry pool, the router's answer;
    other_entries are (pool, store) pairs of entry pools past weak.
    """
    planner = types.SimpleNamespace(successors_by_stream={1: []})
    tracker = types.SimpleNamespace(
        effective_round_count_for_window=lambda _stream_id, _window: 9,
        rounds_arrived=lambda _stream_id: rounds_arrived,
        strong_rounds_arrived=lambda _stream_id, _store: 0,
    )
    weak_store = _store()
    retention = round_retention.RoundRetention(
        is_strong_context_retained=True,
        primary_tier=window_records.WEAK_POOL_ID,
    )
    store_by_pool = {window_records.WEAK_POOL_ID: weak_store}
    store_by_pool.update(other_entries)
    retention.planner = planner
    retention.tracker = tracker
    retention.router = one_entry.stores_router(store_by_pool, entry_of)
    retention.add_edge(
        window_records.WEAK_POOL_ID,
        window_records.STRONG_POOL_ID,
        weak_store,
        strong_store,
    )
    return retention


def _weak_store_of(retention):
    """The store the weak entry pool reads, as the router's side names it."""
    side = retention.router.side(window_records.WEAK_POOL_ID)
    return side.store


class _OneTable:
    """A source whose recipes are one circuit's, for every operation."""

    def __init__(self, table) -> None:
        self.table = table

    def formation_table(self, stream_id):
        """The one table."""
        del stream_id
        return self.table


def _placement(table):
    """The run's real placement, forming at the strong decoder."""
    settings = event_settings.DetectionEventSettings(
        formed_at=("strong_decoder",)
    )
    source = _OneTable(table)
    return formation.SeatedFormation(source, settings)


def _surface_code_table():
    """Four rounds of a d=3 memory: each bulk round reads the one before."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=4, distance=3
    )
    return detector_formation.build_formation_table(circuit, 4)


def _lookback_table():
    """Ten rounds of one qubit; round r's detector is rec[-1] ^ rec[-3]."""
    circuit = stim.Circuit(
        "R 0\nM 0\nDETECTOR rec[-1]\nM 0\nDETECTOR rec[-1]\n"
        "REPEAT 8 {\nM 0\nDETECTOR rec[-1] rec[-3]\n}\n"
    )
    measurement_rounds = {index: index + 1 for index in range(10)}
    return detector_formation.build_formation_table(
        circuit, 10, measurement_rounds=measurement_rounds
    )


def _reach_growing_table():
    """Ten rounds of rec[-1], then rec[-1] ^ rec[-4] from round 4 on."""
    circuit = stim.Circuit(
        "R 0\nREPEAT 3 {\nM 0\nDETECTOR rec[-1]\n}\n"
        "REPEAT 7 {\nM 0\nDETECTOR rec[-1] rec[-4]\n}\n"
    )
    measurement_rounds = {index: index + 1 for index in range(10)}
    return detector_formation.build_formation_table(
        circuit, 10, measurement_rounds=measurement_rounds
    )


def _forming_retention(table, **forms):
    retention = round_retention.RoundRetention(
        is_strong_context_retained=True,
        primary_tier=window_records.WEAK_POOL_ID,
        **forms,
    )
    retention.detection_events = _placement(table)
    return retention


def test_a_strong_side_that_forms_reads_the_raw_round_before_a_redo():
    """A surface-code bulk detector reads the round before its own."""
    table = _surface_code_table()
    retention = _forming_retention(table, strong_side_seat="strong_decoder")

    assert retention.strong_rounds_before(1, 3, 3) == [(1, 2)]


def test_a_strong_read_carries_the_round_its_first_rounds_recipes_read():
    """Round 5's detector reads rounds 5 and 3, so round 3 alone goes up."""
    table = _lookback_table()
    retention = _forming_retention(table, strong_side_seat="strong_decoder")

    assert retention.strong_rounds_before(1, 5, 5) == [(1, 3)]


def test_a_strong_read_carries_the_round_a_later_round_of_it_reads():
    """Its first round 3 reads only itself; its round 4 reads round 1."""
    table = _reach_growing_table()
    retention = _forming_retention(table, strong_side_seat="strong_decoder")

    assert retention.strong_rounds_before(1, 3, 4) == [(1, 1)]


def _raw_fragment(round_index: int) -> round_records.RetainedSyndromeFragment:
    """One raw bit of a one-qubit round, as the QPU emits it."""
    return round_records.RetainedSyndromeFragment(
        stream_id=1,
        patch_ids=(0,),
        round_index=round_index,
        bits=(0,),
        size_bits=1,
        fragment_index=0,
    )


def test_a_strong_read_carries_no_round_its_seat_formed():
    """An earlier region landed rounds 3 and 4 there; round 6 still reads 4.

    Round 5 reads round 3 and round 6 reads round 4, so the seat keeps
    both packets for them and the read from round 5 carries nothing.
    """
    table = _lookback_table()
    retention = _forming_retention(table, strong_side_seat="strong_decoder")
    earlier_region = (_raw_fragment(3), _raw_fragment(4))
    rounds_before = (_raw_fragment(1), _raw_fragment(2))
    retention.detection_events.form_at(
        "strong_decoder", earlier_region, rounds_before
    )

    assert retention.strong_rounds_before(1, 5, 6) == []


def test_an_operations_first_round_has_no_round_before_to_read():
    table = _surface_code_table()
    retention = _forming_retention(table, strong_side_seat="strong_decoder")

    assert retention.strong_rounds_before(1, 1, 1) == []


def test_a_strong_side_that_does_not_form_reads_nothing_before_a_redo():
    table = _lookback_table()
    retention = _forming_retention(table)

    assert retention.strong_rounds_before(1, 5, 5) == []


def test_a_source_with_no_recipes_reads_nothing_before_a_redo():
    settings = event_settings.DetectionEventSettings(
        formed_at=("strong_decoder",)
    )
    retention = round_retention.RoundRetention(
        is_strong_context_retained=True,
        primary_tier=window_records.WEAK_POOL_ID,
        strong_side_seat="strong_decoder",
    )
    retention.detection_events = formation.SeatedFormation(None, settings)

    assert retention.strong_rounds_before(1, 5, 5) == []


def test_a_context_round_still_crossing_is_told_apart_from_one_released():
    """The two states the strong context's readiness check tells apart.

    A round with a live hold and no fragments is on
    controller_to_strong_buffer and the strong window waits for it, the
    way a gem5 port waits for its retry rather than failing
    (src/mem/port.hh:244-255). A round that arrived at the weak syndrome buffer
    with neither fragments nor a hold was released while a reader still
    needs it, which is the mistake the check catches.
    """
    store = _store()
    retention = _strong_retention(store, rounds_arrived=3)
    potential = decoding_records.PotentialStrong((1, 0))
    store.register_hold(potential, [(1, 1), (1, 2)])
    packet = _packet(1, 1)
    store.accept_packed_round(packet, publication_tick=1)
    crossing = retention.context_rounds_in_flight((1, 0), ((1, 1), (1, 2)))
    assert crossing == ((1, 2),)
    with pytest.raises(RuntimeError, match="arrived at the store it is"):
        retention.context_rounds_in_flight((1, 0), ((1, 3),))


def test_a_context_round_the_qpu_has_not_produced_is_not_awaited():
    """A round not yet at the weak syndrome buffer is not late anywhere."""
    store = _store()
    retention = _strong_retention(store, rounds_arrived=2)
    crossing = retention.context_rounds_in_flight((1, 0), ((1, 5),))
    assert crossing == ()


def test_a_window_holds_its_read_range_plus_the_successor_overflow():
    store = _store()
    retention = _retention(store, {1: 4, 2: 9}, {1: [2]})
    window = window_records.Window(
        stream_id=1,
        window_index=0,
        commit_lo=1,
        commit_hi=3,
        buffer_hi=6,
        round_count=6,
    )
    retention.register_window((1, 0), window)
    reads = decoding_records.WindowReads((1, 0))
    held = store.hold_round_identities(reads)
    assert held == ((1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2))


def test_the_hold_moves_to_the_request_and_releases_when_the_input_lands():
    store = _store()
    retention = _retention(store, {1: 6}, {1: []})
    window = window_records.Window(
        stream_id=1,
        window_index=0,
        commit_lo=1,
        commit_hi=3,
        buffer_hi=5,
        round_count=5,
    )
    retention.register_window((1, 0), window)
    _publish_rounds(store, 1, (1, 2, 3, 4, 5))
    request_key = window_records.DecoderRequestKey(
        1, 0, window_records.WEAK_POOL_ID, 0
    )
    job = decoding_records.DecodeJob(
        stream_id=1, window_id=0, round_count=5, request_key=request_key
    )
    reads = decoding_records.WindowReads((1, 0))
    retention.bind_input_hold(job, reads, store)
    assert not store.has_hold(reads)
    input_hold = decoding_records.DecoderInputHold(request_key)
    assert store.has_hold(input_hold)
    assert retention.holds_input(job)
    assert store.occupancy == 5
    job.input_hold()
    assert store.occupancy == 0


def test_an_unheld_round_is_freed_on_arrival():
    store = _store()
    retention = _retention(store, {1: 6}, {1: []})
    packet = _packet(1, 6)
    store.accept_packed_round(packet, publication_tick=6)
    retention.release_round_if_unheld(store, (1, 6))
    assert store.occupancy == 0


def test_a_clipped_tail_keeps_only_its_commit_range():
    store = _store()
    retention = _retention(store, {"stream": 9}, {"stream": []})
    window = window_records.Window(
        stream_id="stream",
        window_index=1,
        commit_lo=4,
        commit_hi=6,
        buffer_hi=8,
        round_count=5,
    )
    retention.register_window(("stream", 1), window)
    window.commit_hi = 5
    window.buffer_hi = 7
    retention.reset_clipped_window_reads(window)
    reads = decoding_records.WindowReads(("stream", 1))
    held = store.hold_round_identities(reads)
    assert held == (("stream", 4), ("stream", 5))


def test_a_clipped_merge_window_keeps_the_region_it_reads_across():
    """The seal clips the window's own stream, not the partner region.

    With a store per pool those rounds are copies only this window
    holds (controller/round_destinations.py).
    """
    store = _store()
    retention = _retention(
        store, {"stream": 9, "partner": 9}, {"stream": [], "partner": []}
    )
    partner = window_records.RoundSpan("partner", 2, 3)
    window = window_records.Window(
        stream_id="stream",
        window_index=1,
        commit_lo=4,
        commit_hi=6,
        buffer_hi=8,
        round_count=7,
        spatial_reads=(partner,),
    )
    retention.register_window(("stream", 1), window)
    window.commit_hi = 5
    window.buffer_hi = 7
    retention.reset_clipped_window_reads(window)
    reads = decoding_records.WindowReads(("stream", 1))
    held = store.hold_round_identities(reads)
    assert held == (
        ("stream", 4),
        ("stream", 5),
        ("partner", 2),
        ("partner", 3),
    )


def test_a_potential_restart_read_outlives_the_landing_and_follows_a_reslice():
    store = _store()
    retention = _retention(store, {1: 12}, {1: []})
    window = window_records.Window(
        stream_id=1,
        window_index=1,
        commit_lo=4,
        commit_hi=6,
        buffer_hi=9,
        round_count=6,
    )
    retention.register_window((1, 1), window)
    claim = decoding_records.PotentialRestart((1, 1))
    claimed = [(1, index) for index in range(1, 10)]
    store.register_hold(claim, claimed)
    _publish_rounds(store, 1, range(1, 10))
    request_key = window_records.DecoderRequestKey(
        1, 1, window_records.WEAK_POOL_ID, 0
    )
    job = decoding_records.DecodeJob(
        stream_id=1, window_id=1, round_count=6, request_key=request_key
    )
    reads = decoding_records.WindowReads((1, 1))
    retention.bind_input_hold(job, reads, store)
    job.input_hold()  # the input landed: the request's hold ends
    assert not store.has_hold(reads)
    assert store.retained_fragments((1, 4)) is not None
    assert store.retained_fragments((1, 9)) is not None
    # the re-slice: the window now reads one buffer into the strong region
    window.buffer_lo = 1
    retention.replace_window_reads((1, 1), window)
    assert store.hold_round_identities(claim) == tuple(claimed)
    retention.release_restart_reads((1, 1))
    assert not store.has_hold(claim)
    assert store.retained_fragments((1, 1)) is None
    assert store.retained_fragments((1, 9)) is None


def _absorbing_retention(absorbed_rounds, request_rounds, restart_rounds):
    """Both stores holding one absorbed window, a request and a restart."""
    store = _store()
    retention = _strong_retention(store, rounds_arrived=0)
    absorbed = decoding_records.PotentialStrong((1, 1))
    request = decoding_records.PendingStrong("request")
    restart = decoding_records.PotentialStrong((1, 2))
    holds = (
        (absorbed, absorbed_rounds),
        (request, request_rounds),
        (restart, restart_rounds),
    )
    for holder, rounds in holds:
        round_keys = _round_keys(rounds)
        store.register_hold(holder, round_keys)
        weak_store = _weak_store_of(retention)
        weak_store.register_hold(holder, round_keys)
    return retention, store, request


def _round_keys(rounds) -> list:
    keys = []
    for round_index in rounds:
        keys.append((1, round_index))
    return keys


def test_an_absorbed_read_behind_the_strong_window_needs_no_new_holder():
    """A buffer wider than the commit reaches behind a pinned near face.

    The absorbed window would have read rounds 4 to 12; the strong
    window pinned at round 7 reads 7 to 9 and the restart window 10 to
    12, so rounds 4 to 6 have no reader left and the absorption lands.
    """
    absorbed_rounds = range(4, 13)
    request_rounds = range(7, 10)
    restart_rounds = range(10, 13)
    retention, store, request = _absorbing_retention(
        absorbed_rounds, request_rounds, restart_rounds
    )
    retention.release_absorbed_strong_hold((1, 1), (1, 2), request)
    absorbed = decoding_records.PotentialStrong((1, 1))
    assert not store.has_hold(absorbed)


def test_an_absorbed_read_past_the_first_round_that_nobody_holds_is_refused():
    absorbed_rounds = range(7, 13)
    request_rounds = range(7, 10)
    restart_rounds = range(11, 13)
    retention, _store_under_test, request = _absorbing_retention(
        absorbed_rounds, request_rounds, restart_rounds
    )
    with pytest.raises(RuntimeError, match="are held by neither the strong"):
        retention.release_absorbed_strong_hold((1, 1), (1, 2), request)


DEEP_POOL = window_records.PoolId("deep")


def _chain_retention():
    """Weak escalating to strong, strong on to deep: three stores."""
    strong = _store()
    deep = _store()
    retention = _strong_retention(strong, rounds_arrived=9)
    weak = _weak_store_of(retention)
    retention.add_edge(window_records.STRONG_POOL_ID, DEEP_POOL, strong, deep)
    return retention, (weak, strong, deep)


def _chain_window() -> window_records.Window:
    return window_records.Window(
        stream_id=1,
        window_index=0,
        commit_lo=1,
        commit_hi=3,
        buffer_hi=5,
        round_count=6,
    )


def test_a_chain_keeps_a_windows_reads_in_each_edges_two_stores():
    retention, stores = _chain_retention()
    weak, strong, deep = stores
    window = _chain_window()

    retention.register_window((1, 0), window)

    to_strong = decoding_records.PotentialStrong((1, 0))
    to_deep = decoding_records.PotentialStrong((1, 0), DEEP_POOL)
    held_by_store = [
        (weak.has_hold(to_strong), weak.has_hold(to_deep)),
        (strong.has_hold(to_strong), strong.has_hold(to_deep)),
        (deep.has_hold(to_strong), deep.has_hold(to_deep)),
    ]
    assert held_by_store == [(True, False), (True, True), (False, True)]


FPGA_POOL = window_records.PoolId("fpga_b")


def _shared_target_retention():
    """Weak and fpga_b each escalate to strong, and strong on to deep.

    Stream 1 enters at weak and stream 2 at fpga_b, whose store is its
    own: Delfosse's lazy decoders sharing one strong unit (2001.11427
    Sec. V).
    """
    strong = _store()
    deep = _store()
    fpga = _store()
    entry_by_stream = {1: window_records.WEAK_POOL_ID, 2: FPGA_POOL}
    retention = _strong_retention(
        strong,
        rounds_arrived=9,
        entry_of=entry_by_stream.get,
        other_entries=((FPGA_POOL, fpga),),
    )
    weak = _weak_store_of(retention)
    retention.add_edge(FPGA_POOL, window_records.STRONG_POOL_ID, fpga, strong)
    retention.add_edge(window_records.STRONG_POOL_ID, DEEP_POOL, strong, deep)
    return retention, (weak, fpga, strong, deep)


def test_a_window_holds_its_rounds_only_on_its_own_entry_pools_chain():
    retention, stores = _shared_target_retention()
    window = window_records.Window(
        stream_id=2,
        window_index=0,
        commit_lo=1,
        commit_hi=3,
        buffer_hi=5,
        round_count=6,
    )

    retention.register_window((2, 0), window)

    to_strong = decoding_records.PotentialStrong((2, 0))
    to_deep = decoding_records.PotentialStrong((2, 0), DEEP_POOL)
    held_by_store = []
    for store in stores:
        held = (store.has_hold(to_strong), store.has_hold(to_deep))
        held_by_store.append(held)
    assert held_by_store == [
        (False, False),
        (True, False),
        (True, True),
        (False, True),
    ]


def test_a_final_window_ends_its_reads_on_every_edge():
    retention, stores = _chain_retention()
    window = _chain_window()
    retention.register_window((1, 0), window)

    retention.release_potential_strong((1, 0))

    to_strong = decoding_records.PotentialStrong((1, 0))
    to_deep = decoding_records.PotentialStrong((1, 0), DEEP_POOL)
    held = []
    for store in stores:
        for holder in (to_strong, to_deep):
            is_held = store.has_hold(holder)
            held.append(is_held)
    assert held == [False] * 6


def test_a_middle_jobs_rounds_stay_held_for_the_edge_after_it():
    """The deep edge redoes the strong revision, rounds 6 and 7 too.

    The strong job reads past the window's own rounds 1-5, as a double
    window does; once its input lands, the strong store still keeps
    every round it read for the deep edge's verdict.
    """
    retention, stores = _chain_retention()
    _weak, strong, _deep = stores
    window = _chain_window()
    retention.register_window((1, 0), window)
    request_key = window_records.DecoderRequestKey(
        1, 0, window_records.STRONG_POOL_ID, 0
    )
    payloads = []
    for round_index in range(1, 8):
        fragment = _raw_fragment(round_index)
        payloads.append(fragment)
    job = decoding_records.DecodeJob(
        stream_id=1,
        window_id=0,
        round_count=7,
        strong_decode_for=(1, 0),
        request_key=request_key,
        payloads=payloads,
    )

    retention.hold_strong_input(job)
    job.input_hold()

    to_deep = decoding_records.PotentialStrong((1, 0), DEEP_POOL)
    held = []
    for round_index in range(1, 8):
        is_held = strong.is_round_held((1, round_index))
        held.append(is_held)
    assert strong.has_hold(to_deep)
    assert held == [True] * 7


def test_a_deep_edge_carries_its_rounds_from_the_middle_store():
    retention, stores = _chain_retention()
    _weak, strong, _deep = stores
    packet = _packet(1, 2)
    strong.accept_packed_round(packet, publication_tick=2)

    packets = retention.escalated_rounds(1, [(1, 2)], DEEP_POOL)

    assert packets == (packet,)


def _two_store_retention(last_written: int):
    """Stream 1 enters at weak, stream 2 at fpga_b, each its own store."""
    weak = _store()
    fpga = _store()
    entry_by_stream = {1: window_records.WEAK_POOL_ID, 2: FPGA_POOL}
    store_by_pool = {window_records.WEAK_POOL_ID: weak, FPGA_POOL: fpga}
    retention = round_retention.RoundRetention(
        is_strong_context_retained=False,
        primary_tier=window_records.WEAK_POOL_ID,
    )
    retention.router = one_entry.stores_router(
        store_by_pool, entry_by_stream.get
    )
    retention.writes = types.SimpleNamespace(
        last_round_written=lambda _stream_id: last_written
    )
    retention.planner = types.SimpleNamespace(
        successors_by_stream={1: [], 2: []}
    )
    retention.tracker = types.SimpleNamespace(
        effective_round_count_for_window=lambda _stream_id, _window: 7
    )
    retention.open_stream_store(1)
    retention.open_stream_store(2)
    return retention, weak, fpga


def test_a_merge_read_of_another_pools_stream_is_held_in_the_readers_store():
    """The hold is what makes the controller copy each round there."""
    retention, weak, fpga = _two_store_retention(last_written=0)
    reader = window_records.RoundSpan(2, 1, 7)
    read = window_records.RoundSpan(1, 1, 7)

    retention.hold_copied_span(reader, read)

    assert fpga.is_round_held((1, 1))
    assert fpga.is_round_held((1, 7))
    assert not weak.is_round_held((1, 1))


def test_a_copy_asked_after_its_round_was_written_is_refused():
    """A round already gone has no copy coming, so no window could read it."""
    retention, _weak, _fpga = _two_store_retention(last_written=3)
    reader = window_records.RoundSpan(2, 1, 7)
    read = window_records.RoundSpan(1, 1, 7)

    with pytest.raises(RuntimeError, match="wrote through round 3"):
        retention.hold_copied_span(reader, read)


def _reader_window(window_index: int, commit_lo: int, commit_hi: int):
    """A window of stream 2 that reads stream 1's rounds 1-7 across."""
    read = window_records.RoundSpan(1, 1, 7)
    return window_records.Window(
        stream_id=2,
        window_index=window_index,
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        buffer_hi=commit_hi,
        round_count=7,
        spatial_reads=(read,),
    )


def test_a_copy_hold_stays_until_the_window_committing_its_last_round():
    """The reader's span is rounds 1-7; its first window commits 1-3.

    The copies stay held for the windows still to come, and the window
    that commits round 7 claims them in its own read, ending the hold.
    """
    retention, _weak, fpga = _two_store_retention(last_written=0)
    reader = window_records.RoundSpan(2, 1, 7)
    read = window_records.RoundSpan(1, 1, 7)
    retention.hold_copied_span(reader, read)
    (copy_hold,) = retention.copy_holds_by_stream[2]
    first = _reader_window(0, 1, 3)
    last = _reader_window(1, 4, 7)

    retention.register_window((2, 0), first)
    held_after_the_first = fpga.has_hold(copy_hold)
    retention.register_window((2, 1), last)

    assert held_after_the_first
    assert not fpga.has_hold(copy_hold)
    assert retention.copy_holds_by_stream[2] == []


def test_a_stream_whose_entry_pool_escalates_nowhere_has_no_strong_hold():
    """Weak escalates to strong; fpga_b, stream 2's pool, escalates nowhere.

    A router may give one region a switching pair and another a lone
    decoder (Tour de gross 2506.03094), so stream 2 has no edge whose
    stores could keep a strong hold, and ending one ends nothing.
    """
    strong = _store()
    fpga = _store()
    entry_by_stream = {1: window_records.WEAK_POOL_ID, 2: FPGA_POOL}
    retention = _strong_retention(
        strong,
        rounds_arrived=9,
        entry_of=entry_by_stream.get,
        other_entries=((FPGA_POOL, fpga),),
    )
    later_reads = decoding_records.LaterStreamReads(2)

    retention.release_strong_hold_if_live(later_reads, 2)

    assert not strong.has_hold(later_reads)
    assert not fpga.has_hold(later_reads)
