"""Each checkpoint of a stream, closed: the outcome had it ended there.

Tan et al. measure the logical error rate of runs of several lengths
from one run: at a length they copy the state, treat the window there
as the last, decode it and read the copy out, and the run goes on
unchanged (2209.09219 lines 1003-1037). The source copies its state and
reads the copy out without noise (records/checkpoints.py); this listener
ends the decoding there, at zero simulated time, by one law:

- The windows that read only rounds before the checkpoint's are the
  same as in a stream that ends there, so their final answers stand:
  their observables, XORed, and the boundary the last of them hands on
  (windows/window_commits.py). A closed window, a parallel B block,
  also takes the boundary of the A block after it, so it stands only
  when that block does too.
- The windows laid after them that commit a round up to the
  checkpoint's are the stream's last windows had it ended there, each
  clipped to end on it, as a growing stream's tail closes
  (windows/region_streams.py RegionStream.close_tail). Each folds in
  the boundaries it takes, from the window before it, or for a B block
  from both sides, reads the copy's readout as its last detectors, and
  has the model it would have had ending there
  (observe/closing_windows.py). The machine's own decoders decide each
  as they decide a final window, at zero time and learning nothing
  (observe/closing_decisions.py).

The checkpoint fails when that prediction is not the copy's truth, and
it records the threshold the stream's edge read then. No component
reads any of this; the listener hears the source, the device, the
window side and the decoder managers through their trace sources.
"""

import dataclasses
from typing import Any, Optional

import decsim.observe.closing_decisions as closing_decisions
import decsim.observe.closing_windows as closing_windows
import decsim.records.checkpoints as checkpoint_records
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records


@dataclasses.dataclass(frozen=True)
class CheckpointOutcome:
    """One stream's checkpoint, closed.

    failed is whether the closed decoding missed the copy's truth, None
    when a window it kept answered no observable. windows is how many
    final windows it kept, escalated_windows how many of them the
    policy escalated. threshold is the one the stream's edge read when
    the checkpoint closed, in nats, None with no edge.
    """

    stream_id: Any  # an opaque identity
    round_index: int
    failed: Optional[bool]
    windows: int
    escalated_windows: int
    threshold: Optional[float]


class Checkpoints:
    """The checkpoints of a run's streams, each closed once it can be.

    every_rounds spaces them; a stream's last round is always one, so
    its last checkpoint is its own end. closer decides each closing
    window.
    """

    def __init__(
        self, every_rounds: int, closer: closing_decisions.ClosingDecisions
    ) -> None:
        self.every_rounds = every_rounds
        self.closer = closer
        # stream id -> _StreamCheckpoints, in the order streams were heard
        self.streams: dict = {}
        self.outcomes: list = []

    def checkpoint_reached(
        self, snapshot: checkpoint_records.CheckpointSnapshot
    ) -> None:
        """A round ran; keep it as a checkpoint when it is one."""
        is_spaced = snapshot.round_index % self.every_rounds == 0
        if not is_spaced and not snapshot.is_final:
            return
        stream = self._stream(snapshot.stream_id)
        stream.wait_for(snapshot, self.closer)
        self._close_ready(stream)

    def round_emitted(self, readout: round_records.QPUReadout) -> None:
        """A round left the QPU; its raw bits form the closing events."""
        stream = self._stream(readout.stream_id)
        stream.packets[readout.round_index] = readout.bits
        self._close_ready(stream)

    def window_planned(self, window: window_records.Window) -> None:
        """A stream laid a window."""
        stream = self._stream(window.stream_id)
        stream.chain.windows.append(window)

    def request_ended(
        self,
        job: decoding_records.DecodeJob,
        result: Optional[decoding_records.DecodeResult],
        outcome: decoding_records.RequestProcessingOutcome,
        decode_output_ticks: Optional[int],
    ) -> None:
        """A decode ended; a window may publish its result."""
        del outcome, decode_output_ticks
        # a self-contained job, a factory's or an idle region's, has no window
        if job.request_key is None:
            return
        stream = self._stream(job.stream_id)
        stream.chain.keep_result(job.request_key, result)
        self._close_ready(stream)

    def verdict_given(
        self,
        window_key: tuple,
        request_key: window_records.DecoderRequestKey,
        verdict: decoding_records.Verdict,
    ) -> None:
        """The policy answered a weak result; count the escalations."""
        del request_key
        if verdict is not decoding_records.Verdict.ESCALATE:
            return
        stream = self._stream(window_key[0])
        stream.chain.escalated.add(window_key)

    def window_final(self, window: window_records.Window) -> None:
        """A window's record is final."""
        stream = self._stream(window.stream_id)
        self._close_ready(stream)

    def rows(self) -> list:
        """One row a checkpoint round: each stream's outcome, then counts.

        failed and threshold hold each stream's, in the order the streams
        were heard, as a memory task's patches are; the counts add up the
        streams'.
        """
        order = list(self.streams)
        by_round: dict = {}
        for outcome in self.outcomes:
            closed = by_round.setdefault(outcome.round_index, {})
            closed[outcome.stream_id] = outcome
        rows = []
        for round_index in sorted(by_round):
            closed = by_round[round_index]
            outcomes = [closed[key] for key in order if key in closed]
            row = _row(round_index, outcomes)
            rows.append(row)
        return rows

    def _stream(self, stream_id: Any) -> "_StreamCheckpoints":
        stream = self.streams.get(stream_id)
        if stream is None:
            stream = _StreamCheckpoints(stream_id)
            self.streams[stream_id] = stream
        return stream

    def _close_ready(self, stream: "_StreamCheckpoints") -> None:
        """Close the stream's waiting checkpoints, oldest first, while ready."""
        while stream.pending:
            outcome = stream.close_oldest()
            if outcome is None:
                return
            self.outcomes.append(outcome)


class _StreamCheckpoints:
    """One stream's raw rounds, waiting checkpoints and window chain."""

    def __init__(self, stream_id: Any) -> None:
        self.stream_id = stream_id
        # round -> raw bits, from the earliest a closing window may read
        self.packets: dict = {}
        self.pending: list = []
        self.chain = _WindowChain()
        # the stream's closing windows, made at the first checkpoint,
        # which carries the stream's program
        self.closing: Optional[closing_windows.ClosingWindows] = None

    def wait_for(
        self,
        snapshot: checkpoint_records.CheckpointSnapshot,
        closer: closing_decisions.ClosingDecisions,
    ) -> None:
        """Keep the checkpoint until its windows are final."""
        if self.closing is None:
            self.closing = closing_windows.ClosingWindows(
                snapshot.program, closer
            )
        self.pending.append(snapshot)

    def close_oldest(self) -> Optional[CheckpointOutcome]:
        """The oldest checkpoint closed, or None while it cannot be."""
        snapshot = self.pending[0]
        round_index = snapshot.round_index
        if round_index not in self.packets:
            return None
        kept_and_tail = self.chain.closing_at(round_index)
        if kept_and_tail is None:
            return None
        kept, tail = kept_and_tail
        self.pending.pop(0)
        first_round = tail[0].start_round
        events = self.closing.events(snapshot, first_round, self.packets)
        observables = self.closing.tail_observables(
            snapshot, events, kept.boundary_ids, tail
        )
        threshold = self.closing.closer.threshold_of(self.stream_id)
        first_after_kept = kept.commit_hi + 1
        self._let_go_before(first_after_kept)
        prediction = closing_windows.xor_or_none(kept.observables, observables)
        failed = closing_windows.failed(prediction, snapshot)
        return CheckpointOutcome(
            self.stream_id,
            round_index,
            failed,
            kept.windows,
            kept.escalated_windows,
            threshold,
        )

    def _let_go_before(self, first_round: int) -> None:
        """Every later checkpoint closes from first_round or after."""
        reach = self.closing.reach
        earliest = first_round - reach
        let_go = [
            round_index
            for round_index in self.packets
            if round_index < earliest
        ]
        for round_index in let_go:
            del self.packets[round_index]


@dataclasses.dataclass
class _Kept:
    """The final windows a checkpoint keeps, folded in window order.

    observables is None once a kept window answered none; boundary_ids
    are the detectors the last one's commit hands on.
    """

    windows: int = 0
    escalated_windows: int = 0
    commit_hi: int = 0
    observables: Optional[tuple] = ()
    boundary_ids: tuple = ()


class _WindowChain:
    """A stream's windows, their final results, folded as checkpoints close.

    A window that read only rounds before a checkpoint is final the same
    way whether the stream ends there or goes on, so its answer folds
    into what every later checkpoint keeps, and it is let go.
    """

    def __init__(self) -> None:
        # the windows laid and not folded yet, in index order
        self.windows: list = []
        # window key -> {request key: result} until the window folds
        self.results: dict = {}
        self.escalated: set = set()
        self.kept = _Kept()

    def keep_result(
        self,
        request_key: window_records.DecoderRequestKey,
        result: Optional[decoding_records.DecodeResult],
    ) -> None:
        """Hold a decode's result until its window is final."""
        window_key = (request_key.stream_id, request_key.window_id)
        by_request = self.results.setdefault(window_key, {})
        by_request[request_key] = result

    def closing_at(self, round_index: int) -> Optional[tuple]:
        """(the kept windows, the tail) of a stream ending on round_index.

        The kept windows are the final ones whose answers rest on
        earlier rounds only; the tail is each laid window after them
        that commits a round up to round_index. None while the window
        committing the round is not laid, so the tail is not known, or
        while a kept window is not final.
        """
        if not self._holds(round_index):
            return None
        before = self._reading_before(round_index)
        kept_windows = _standing(before)
        if not all(self._is_final(window) for window in kept_windows):
            return None
        for window in kept_windows:
            self._fold(window)
        self.windows = self.windows[len(kept_windows) :]
        tail = self._committing_up_to(round_index)
        kept = dataclasses.replace(self.kept)
        return kept, tail

    def _holds(self, round_index: int) -> bool:
        """Whether a laid window commits the round or a later one."""
        return any(window.commit_hi >= round_index for window in self.windows)

    def _committing_up_to(self, round_index: int) -> list:
        """The laid windows whose commit begins at or before round_index."""
        return [
            window for window in self.windows if window.commit_lo <= round_index
        ]

    def _reading_before(self, round_index: int) -> list:
        """The laid windows whose last read round is before round_index."""
        return [
            window for window in self.windows if window.buffer_hi < round_index
        ]

    def _is_final(self, window: window_records.Window) -> bool:
        published = window.published_request_key
        by_request = self.results.get(window.key, {})
        return published in by_request

    def _fold(self, window: window_records.Window) -> None:
        by_request = self.results.pop(window.key)
        result = by_request[window.published_request_key]
        kept = self.kept
        kept.windows += 1
        kept.escalated_windows += window.key in self.escalated
        self.escalated.discard(window.key)
        kept.commit_hi = window.commit_hi
        kept.observables = closing_windows.xor_or_none(
            kept.observables, result.logical_observables
        )
        kept.boundary_ids = closing_windows.boundary_ids_of(result)


def _standing(reading_before: list) -> list:
    """The windows whose answers rest only on rounds before a checkpoint.

    reading_before are the windows that read only earlier rounds, a
    prefix in index order. A last one that is closed, a parallel B
    block, also takes the boundary of the A block after it, which reads
    further, so the B is decided again in the tail.
    """
    if not reading_before:
        return reading_before
    last_window = reading_before[-1]
    if last_window.closed_temporal_boundaries:
        return reading_before[:-1]
    return reading_before


def _row(round_index: int, outcomes: list) -> dict:
    windows = sum(outcome.windows for outcome in outcomes)
    escalated = sum(outcome.escalated_windows for outcome in outcomes)
    return {
        "round": round_index,
        "failed": [outcome.failed for outcome in outcomes],
        "windows": windows,
        "escalated_windows": escalated,
        "threshold": [outcome.threshold for outcome in outcomes],
    }
