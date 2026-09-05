"""The round retention's laws on a real round store.

A window holds [start_round, buffer_hi] plus the successor overflow
(Skoric et al. 2209.08552: the buffer region is re-read by the next
window); the hold moves to the request at admission and is released
once the input lands; an unheld round is freed on arrival.
"""

import types

import decsim.message as message
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings
import decsim.windows.round_retention as round_retention


def _store() -> round_store_module.RoundStore:
    settings = round_store_settings.RoundStoreSettings()
    return round_store_module.RoundStore(settings)


def _retention(store, round_counts: dict, successors: dict):
    planner = types.SimpleNamespace(successors_by_operation=successors)

    def effective_round_count_for_window(operation_id, _window):
        return round_counts[operation_id]

    tracker = types.SimpleNamespace(
        effective_round_count_for_window=effective_round_count_for_window,
        rounds_arrived=lambda _operation_id: 0,
        strong_rounds_arrived=lambda _operation_id: 0,
    )
    return round_retention.RoundRetention(
        store,
        None,
        planner,
        tracker,
        is_strong_context_retained=False,
        primary_tier=message.DecoderTier.WEAK,
    )


def _packet(operation_id, round_index) -> message.SyndromeRoundPacket:
    fragment = message.RetainedSyndromeFragment(
        operation_id=operation_id,
        patch_id=0,
        round_index=round_index,
        bits=None,
        size_bits=None,
        fragment_index=0,
    )
    return message.SyndromeRoundPacket(operation_id, round_index, (fragment,))


def test_a_window_holds_its_read_range_plus_the_successor_overflow():
    store = _store()
    retention = _retention(store, {1: 4, 2: 9}, {1: [2]})
    window = message.Window(
        op_id=1, k=0, commit_lo=1, commit_hi=3, buffer_hi=6, n_rounds=6
    )
    retention.register_window((1, 0), window)
    held = store.hold_round_identities((1, 0))
    assert held == ((1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2))


def test_the_hold_moves_to_the_request_and_releases_when_the_input_lands():
    store = _store()
    retention = _retention(store, {1: 6}, {1: []})
    window = message.Window(
        op_id=1, k=0, commit_lo=1, commit_hi=3, buffer_hi=5, n_rounds=5
    )
    retention.register_window((1, 0), window)
    for round_index in (1, 2, 3, 4, 5):
        packet = _packet(1, round_index)
        store.accept_packed_round(packet, publication_tick=round_index)
    request_key = message.DecoderRequestKey(1, 0, message.DecoderTier.WEAK, 0)
    job = message.DecodeJob(
        op_id=1, window_id=0, n_rounds=5, request_key=request_key
    )
    retention.bind_input_hold(job, (1, 0))
    assert not store.has_hold((1, 0))
    input_hold = message.DecoderInputHold(request_key)
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
    window = message.Window(
        op_id="stream", k=1, commit_lo=4, commit_hi=6, buffer_hi=8, n_rounds=5
    )
    retention.register_window(("stream", 1), window)
    window.commit_hi = 5
    window.buffer_hi = 7
    retention.reset_clipped_window_reads(window)
    held = store.hold_round_identities(("stream", 1))
    assert held == (("stream", 4), ("stream", 5))


def test_strong_context_is_one_buffer_on_each_side_of_the_commit():
    window = message.Window(
        op_id=1, k=2, commit_lo=7, commit_hi=9, buffer_hi=11, n_rounds=5
    )
    bounds = round_retention.strong_context_bounds(window)
    assert bounds == (5, 7, 9, 11)
