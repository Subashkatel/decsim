"""The operation results' laws through the release receiver.

The result is the ledger's XOR over [1, rounds], delivered once after
every window committed and no strong wait; a segment releases when the
committed prefix covers it, its value the stream's from round 1 (Skoric
et al. 2209.08552: the logical correction of a stream is the sum over
its committed windows).
"""

import types

import decsim.observe.result_ledger as result_ledger
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.windows.committed_rounds as committed_rounds
import decsim.windows.operation_results as operation_results
import decsim.windows.release_rules as release_rules


class _Release:
    def __init__(self) -> None:
        self.released = []

    def release_waiters(self, operation) -> None:
        self.released.append(operation.id)


class _Store:
    def __init__(self) -> None:
        self.closed = []

    def has_live_stream_reference(self, _stream_id) -> bool:
        return False

    def has_stream(self, _stream_id) -> bool:
        return True

    def close_stream(self, stream_id) -> None:
        self.closed.append(stream_id)


def _window(stream_id, index, commit_lo, commit_hi) -> window_records.Window:
    round_count = commit_hi - commit_lo + 1
    return window_records.Window(
        stream_id=stream_id,
        window_index=index,
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        buffer_hi=commit_hi,
        round_count=round_count,
    )


class _Factory:
    """The factory the last result of the workload stops."""

    def __init__(self, completions: list) -> None:
        self.completions = completions

    def shutdown(self) -> None:
        """The workload is complete."""
        self.completions.append("done")


class _Fixture:
    """One six-round operation in two windows, committed by hand."""

    def __init__(self) -> None:
        self.operation = program_records.Operation(
            1, "memory", (0,), patches=(0,)
        )
        self.windows = {
            (1, 0): _window(1, 0, 1, 3),
            (1, 1): _window(1, 1, 4, 6),
        }
        plan = types.SimpleNamespace(
            windows_by_key=self.windows,
            window_count_of=lambda _stream_id: 2,
            window_indices_of=self._window_indices_of,
            window_at=self.windows.__getitem__,
            windowed_stream_ids=self._windowed_stream_ids,
            windows_of=self._windows_of,
            commit_extent_of=self._commit_extent_of,
            total_windows=2,
            round_count_of=lambda _stream_id: 6,
        )
        tracker = types.SimpleNamespace(
            operation_by_id={1: self.operation},
            is_sealed=lambda _stream_id: True,
            has_unsealed_streams=lambda: False,
            blocking_operation_ids=set(),
        )
        self.store = _Store()
        stores = (self.store,)
        retention = types.SimpleNamespace(
            entry_stores=lambda: stores,
            release_hold_if_live=lambda _owner, _store: None,
            release_potential_strong=lambda _key: None,
            landing_stores=tuple,
        )
        self.release = _Release()
        self.completions = []
        self.ledger = committed_rounds.LogicalLedger()
        self.factory = _Factory(self.completions)
        self.results = operation_results.OperationResults()
        self.results.planner = plan
        self.results.tracker = tracker
        self.results.retention = retention
        self.results.ledger = self.ledger
        self.results.conditional_release = self.release
        self.results.factory = self.factory
        self.results.release_rule = release_rules.OwnStreams()
        self.delivered = result_ledger.ResultLedger()
        self.results.trace.operation_result_delivered.connect(
            self.delivered.operation_result_delivered
        )

    def _windows_of(self, _stream_id) -> list:
        windows = self.windows.values()
        return list(windows)

    def _window_indices_of(self, _stream_id) -> list:
        return [0, 1]

    def _commit_extent_of(self, _stream_id) -> tuple:
        windows = self.windows.values()
        commit_los = [window.commit_lo for window in windows]
        commit_his = [window.commit_hi for window in windows]
        commit_lo = min(commit_los)
        commit_hi = max(commit_his)
        return commit_lo, commit_hi

    def _windowed_stream_ids(self) -> list:
        return [1]

    def commit(
        self, key, bits, is_final=True, tier=window_records.WEAK_POOL_ID
    ):
        window = self.windows[key]
        window.committed = True
        self.results.install_window_contribution(window, bits)
        if is_final:
            window.published_request_key = window_records.DecoderRequestKey(
                key[0], key[1], tier, key[1]
            )
        self.results.note_window_committed(window, is_final)
        self.results.deliver_if_final(self.operation)


def test_the_result_is_the_xor_over_every_window_delivered_once():
    fixture = _Fixture()
    fixture.commit((1, 0), (1, 0))
    assert fixture.release.released == []
    fixture.commit((1, 1), (1, 1))
    assert fixture.release.released == [1]
    assert fixture.delivered.result_by_operation == {1: (0, 1)}
    assert fixture.store.closed == [1]
    assert fixture.completions == ["done"]
    fixture.results.deliver_if_final(fixture.operation)
    assert fixture.release.released == [1]


def test_a_delivery_reports_the_contributions_its_result_is_the_xor_of():
    fixture = _Fixture()
    heard = {}
    fixture.results.trace.contributions_delivered.connect(heard.__setitem__)
    fixture.commit((1, 0), (1, 0))
    fixture.commit((1, 1), (1, 1))
    assert list(heard) == [1]
    contributions = heard[1]
    owners = [contribution.owner_key for contribution in contributions]
    assert owners == [(1, 0), (1, 1)]
    assert fixture.delivered.result_by_operation == {1: (0, 1)}


def test_a_window_awaiting_strong_holds_the_operation_result():
    fixture = _Fixture()
    fixture.commit((1, 0), (1, 0))
    fixture.commit((1, 1), (1, 1), is_final=False)
    assert fixture.release.released == []
    assert fixture.completions == []
    window = fixture.windows[(1, 1)]
    fixture.results.replace_prediction((1, 1), (0, 0))
    window.published_request_key = window_records.DecoderRequestKey(
        1, 1, window_records.STRONG_POOL_ID, 7
    )
    fixture.results.deliver_if_final(fixture.operation)
    assert fixture.release.released == [1]
    assert fixture.delivered.result_by_operation == {1: (1, 0)}


def test_a_segment_releases_when_the_committed_prefix_covers_it():
    fixture = _Fixture()
    segment = program_records.Operation(
        2, "segment", (0,), blocked_by=1, stream_id=1
    )
    fixture.results.tracker.operation_by_id[2] = segment
    fixture.results.tracker.blocking_operation_ids.add(2)
    fixture.results.bind_stream_segment(2, 1, 0)
    fixture.results.bind_required_stream_end(2, 3)
    fixture.commit((1, 0), (1, 1))
    assert fixture.release.released[0] == 2
    assert fixture.delivered.result_by_operation[2] == (1, 1)
    assert fixture.results.committed_prefix_round_count(1) == 3


def test_a_later_segment_carries_every_correction_before_it():
    """Rounds 1-3 flip the observable and rounds 4-6 do not: still (1,)."""
    fixture = _Fixture()
    segment = program_records.Operation(
        2, "segment", (0,), blocked_by=1, stream_id=1
    )
    fixture.results.tracker.operation_by_id[2] = segment
    fixture.results.tracker.blocking_operation_ids.add(2)
    fixture.results.bind_stream_segment(2, 1, 3)
    fixture.results.bind_required_stream_end(2, 6)
    fixture.commit((1, 0), (1,))
    fixture.commit((1, 1), (0,))
    assert fixture.delivered.result_by_operation[2] == (1,)


def test_the_workload_is_not_final_until_every_stream_is_sealed():
    fixture = _Fixture()
    fixture.results.tracker.has_unsealed_streams = lambda: True
    fixture.commit((1, 0), (1, 0))
    fixture.commit((1, 1), (1, 1))
    assert fixture.completions == []
    fixture.results.tracker.has_unsealed_streams = lambda: False
    fixture.results.finish_workload_if_ready()
    assert fixture.completions == ["done"]
    fixture.results.finish_workload_if_ready()
    assert fixture.completions == ["done"]


def test_a_timing_only_window_leaves_no_result():
    fixture = _Fixture()
    fixture.commit((1, 0), None)
    fixture.commit((1, 1), (1, 1))
    assert fixture.release.released == [1]
    assert fixture.delivered.result_by_operation == {}
