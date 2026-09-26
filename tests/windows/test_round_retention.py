"""The round retention's laws on a real syndrome buffer.

A window holds [start_round, buffer_hi] plus the successor overflow
(Skoric et al. 2209.08552: the buffer region is re-read by the next
window); the hold moves to the request at admission and is released
once the input lands; an unheld round is freed on arrival. Under the
forward window a window's potential restart read (PotentialRestart,
placed by the planner) keeps the rounds its restart decode would read
past the landing, follows a re-slice, and ends when the retention is
told no earlier escalation can re-slice the window (Toshio 2510.25222
Sec. III C).
"""

import types

import pytest

import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.syndrome_buffer.settings as syndrome_buffer_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.windows.round_retention as round_retention


def _store() -> syndrome_buffer_module.SyndromeBuffer:
    settings = syndrome_buffer_settings.SyndromeBufferSettings()
    engine = engine_module.Engine()
    return syndrome_buffer_module.SyndromeBuffer(settings, engine)


def _retention(store, round_counts: dict, successors: dict):
    planner = types.SimpleNamespace(successors_by_operation=successors)

    def effective_round_count_for_window(operation_id, _window):
        return round_counts[operation_id]

    tracker = types.SimpleNamespace(
        effective_round_count_for_window=effective_round_count_for_window,
        rounds_arrived=lambda _operation_id: 0,
        strong_rounds_arrived=lambda _operation_id: 0,
    )
    retention = round_retention.RoundRetention(
        is_strong_context_retained=False,
        primary_tier=window_records.DecoderTier.WEAK,
    )
    retention.weak_store = store
    retention.planner = planner
    retention.tracker = tracker
    return retention


def _publish_rounds(store, operation_id: int, round_indices) -> None:
    """Each round stored at the tick of its own index."""
    for round_index in round_indices:
        packet = _packet(operation_id, round_index)
        store.accept_packed_round(packet, publication_tick=round_index)


def _packet(operation_id, round_index) -> round_records.SyndromeRoundPacket:
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=operation_id,
        patch_ids=(0,),
        round_index=round_index,
        bits=None,
        size_bits=None,
        fragment_index=0,
    )
    return round_records.SyndromeRoundPacket(
        operation_id, round_index, (fragment,)
    )


def _strong_retention(strong_store, rounds_arrived: int):
    """A retention whose strong syndrome buffer is the one under test."""
    planner = types.SimpleNamespace(successors_by_operation={1: []})
    tracker = types.SimpleNamespace(
        effective_round_count_for_window=lambda _operation_id, _window: 9,
        rounds_arrived=lambda _operation_id: rounds_arrived,
        strong_rounds_arrived=lambda _operation_id: 0,
    )
    weak_store = _store()
    retention = round_retention.RoundRetention(
        is_strong_context_retained=True,
        primary_tier=window_records.DecoderTier.WEAK,
    )
    retention.weak_store = weak_store
    retention.strong_store = strong_store
    retention.planner = planner
    retention.tracker = tracker
    return retention


def test_a_strong_side_that_forms_reads_the_raw_round_before_a_redo():
    """LILLIPUT 2108.06569 lines 499-510: a detector reads the round before."""
    retention = round_retention.RoundRetention(
        is_strong_context_retained=True,
        primary_tier=window_records.DecoderTier.WEAK,
        strong_side_forms=True,
    )

    assert retention.strong_round_before(1, 4) == [(1, 3)]


def test_an_operations_first_round_has_no_round_before_to_read():
    retention = round_retention.RoundRetention(
        is_strong_context_retained=True,
        primary_tier=window_records.DecoderTier.WEAK,
        strong_side_forms=True,
    )

    assert retention.strong_round_before(1, 1) == []


def test_a_strong_side_that_does_not_form_reads_nothing_before_a_redo():
    retention = round_retention.RoundRetention(
        is_strong_context_retained=True,
        primary_tier=window_records.DecoderTier.WEAK,
    )

    assert retention.strong_round_before(1, 4) == []


def test_a_context_round_still_crossing_is_told_apart_from_one_released():
    """The two states the strong context's readiness check conflated.

    A round with a live hold and no fragments is on
    controller_to_strong_buffer and the strong window waits for it, the
    way a gem5 port waits for its retry rather than failing
    (src/mem/port.hh:244-255). A round that arrived at the weak syndrome buffer
    with neither fragments nor a hold was released while a reader still
    needs it, which is the mistake the check was written to catch.
    """
    store = _store()
    retention = _strong_retention(store, rounds_arrived=3)
    potential = decoding_records.PotentialStrong((1, 0))
    store.register_hold(potential, [(1, 1), (1, 2)])
    packet = _packet(1, 1)
    store.accept_packed_round(packet, publication_tick=1)
    crossing = retention.context_rounds_in_flight((1, 0), ((1, 1), (1, 2)))
    assert crossing == ((1, 2),)
    with pytest.raises(RuntimeError, match="released while a strong window"):
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
        operation_id=1,
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
        operation_id=1,
        window_index=0,
        commit_lo=1,
        commit_hi=3,
        buffer_hi=5,
        round_count=5,
    )
    retention.register_window((1, 0), window)
    _publish_rounds(store, 1, (1, 2, 3, 4, 5))
    request_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.WEAK, 0
    )
    job = decoding_records.DecodeJob(
        operation_id=1, window_id=0, round_count=5, request_key=request_key
    )
    reads = decoding_records.WindowReads((1, 0))
    retention.bind_input_hold(job, reads)
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
    retention.release_round_if_unheld((1, 6))
    assert store.occupancy == 0


def test_a_clipped_tail_keeps_only_its_commit_range():
    store = _store()
    retention = _retention(store, {"stream": 9}, {"stream": []})
    window = window_records.Window(
        operation_id="stream",
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


def test_a_potential_restart_read_outlives_the_landing_and_follows_a_reslice():
    store = _store()
    retention = _retention(store, {1: 12}, {1: []})
    window = window_records.Window(
        operation_id=1,
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
        1, 1, window_records.DecoderTier.WEAK, 0
    )
    job = decoding_records.DecodeJob(
        operation_id=1, window_id=1, round_count=6, request_key=request_key
    )
    reads = decoding_records.WindowReads((1, 1))
    retention.bind_input_hold(job, reads)
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
        retention.weak_store.register_hold(holder, round_keys)
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
    with pytest.raises(RuntimeError, match=r"rounds \[\(1, 10\)\] are held"):
        retention.release_absorbed_strong_hold((1, 1), (1, 2), request)
