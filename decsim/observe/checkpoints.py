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
  (windows/window_commits.py).
- The rounds after the last of them, with that boundary folded in and
  the copy's readout as their last detectors, are decoded as one window
  that ends the stream, by a fixed referee: PyMatching on the model the
  stream would have had ending there (qpu/moved_stream_models.py).

The checkpoint fails when that prediction is not the copy's truth. No
component reads any of this; the listener hears the source, the device,
the window side and the decoder managers through their trace sources.
"""

import dataclasses
from typing import Any, Optional

import decsim.ports as ports
import decsim.qpu.moved_stream_models as moved_stream_models
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
    policy escalated.
    """

    stream_id: Any  # an opaque identity
    round_index: int
    failed: Optional[bool]
    windows: int
    escalated_windows: int


class Checkpoints:
    """The checkpoints of a run's streams, each closed once it can be.

    every_rounds spaces them; a stream's last round is always one, so
    its last checkpoint is its own end. referee decodes each closing
    window.
    """

    def __init__(self, every_rounds: int, referee: ports.Decoder) -> None:
        self.every_rounds = every_rounds
        self.referee = referee
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
        stream.wait_for(snapshot, self.referee)
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

        failed holds each stream's, in the order the streams were heard,
        as a memory task's patches are; the counts add up the streams'.
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
        # the referee's decode of a closing window, made at the first
        # checkpoint, which carries the stream's program
        self.tail: Optional[_TailDecoder] = None

    def wait_for(
        self,
        snapshot: checkpoint_records.CheckpointSnapshot,
        referee: ports.Decoder,
    ) -> None:
        """Keep the checkpoint until its windows are final."""
        if self.tail is None:
            self.tail = _TailDecoder(snapshot.program, referee)
        self.pending.append(snapshot)

    def close_oldest(self) -> Optional[CheckpointOutcome]:
        """The oldest checkpoint closed, or None while it cannot be."""
        snapshot = self.pending[0]
        round_index = snapshot.round_index
        if round_index not in self.packets:
            return None
        kept = self.chain.kept_before(round_index)
        if kept is None:
            return None
        self.pending.pop(0)
        observables = self.tail.observables(snapshot, kept, self.packets)
        first_after_kept = kept.commit_hi + 1
        self._let_go_before(first_after_kept)
        failed = _failed(kept.observables, observables, snapshot)
        return CheckpointOutcome(
            self.stream_id,
            round_index,
            failed,
            kept.windows,
            kept.escalated_windows,
        )

    def _let_go_before(self, first_round: int) -> None:
        """Every later checkpoint closes from first_round or after."""
        reach = self.tail.reach
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

    def kept_before(self, round_index: int) -> Optional[_Kept]:
        """The final windows that read only rounds before round_index.

        None while a laid window that reads the round is not known, so
        more may read only earlier ones, or while one is not final.
        """
        before = self._reading_before(round_index)
        if len(before) == len(self.windows):
            return None
        if not all(self._is_final(window) for window in before):
            return None
        for window in before:
            self._fold(window)
        self.windows = self.windows[len(before) :]
        return dataclasses.replace(self.kept)

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
        kept.observables = _xor_or_none(
            kept.observables, result.logical_observables
        )
        kept.boundary_ids = _boundary_ids_of(result)


class _TailDecoder:
    """The referee's decode of the rounds after a checkpoint's kept windows."""

    def __init__(self, program, referee: ports.Decoder) -> None:
        requirement = referee.fault_model_requirement
        self.models = moved_stream_models.MovedStimModels(program, requirement)
        self.referee = referee
        # how many rounds back a round's detectors read, and one more
        self.reach = self.models.template.prefix_rounds

    def observables(
        self,
        snapshot: checkpoint_records.CheckpointSnapshot,
        kept: _Kept,
        packets: dict,
    ) -> Optional[tuple]:
        """The closing window's observables, its rounds and readout decoded."""
        first_round = kept.commit_hi + 1
        last_round = snapshot.round_index
        events = self._events(snapshot, first_round, packets)
        for detector_id in kept.boundary_ids:
            if detector_id in events:
                events[detector_id] ^= 1
        round_count = last_round - first_round + 1
        window = window_records.Window(
            snapshot.stream_id,
            kept.windows,
            first_round,
            last_round,
            last_round,
            round_count,
        )
        model = self.models.for_window_ending_at(window, last_round)
        syndrome = [events[detector_id] for detector_id in model.detector_ids]
        payload = round_records.QPUReadout(
            snapshot.stream_id, (), first_round, bits=syndrome
        )
        job = decoding_records.DecodeJob(
            snapshot.stream_id,
            kept.windows,
            round_count,
            detector_error_model=model,
            payloads=[payload],
            label=f"checkpoint {last_round}",
        )
        result = self.referee.decode(job)
        return result.logical_observables

    def _events(
        self,
        snapshot: checkpoint_records.CheckpointSnapshot,
        first_round: int,
        packets: dict,
    ) -> dict:
        """Detector id -> event, of the rounds and the copy's readout.

        The readout's detectors are the last of its round, in detector
        order, as the final fragment appends them to the repeated round.
        """
        last_round = snapshot.round_index
        table = self.models.template.table(
            last_round, is_final=True, live_reach=None
        )
        recipes = []
        for round_index in range(first_round, last_round):
            round_recipes = table.detectors_of_round(round_index)
            recipes.extend(round_recipes)
        last_recipes = table.detectors_of_round(last_round)
        readout_count = len(snapshot.readout_events)
        round_count = len(last_recipes) - readout_count
        recipes.extend(last_recipes[:round_count])
        events = {}
        for recipe in recipes:
            events[recipe.detector_index] = _formed(recipe, packets)
        readout_recipes = last_recipes[round_count:]
        readout = zip(readout_recipes, snapshot.readout_events, strict=True)
        for recipe, event in readout:
            events[recipe.detector_index] = event
        return events


def _formed(recipe, packets: dict) -> int:
    """A recipe's event: its reference parity XOR the raw bits it reads."""
    parity = recipe.reference_parity
    for round_index, slot in recipe.records:
        bits = packets[round_index]
        parity ^= int(bits[slot])
    return parity


def _failed(
    kept_observables: Optional[tuple],
    closing_observables: Optional[tuple],
    snapshot: checkpoint_records.CheckpointSnapshot,
) -> Optional[bool]:
    prediction = _xor_or_none(kept_observables, closing_observables)
    if prediction is None:
        return None
    return prediction != snapshot.observable_flips


def _xor_or_none(
    first: Optional[tuple], second: Optional[tuple]
) -> Optional[tuple]:
    """Two observable tuples XORed; an empty one is all zeros, None stays."""
    if first is None or second is None:
        return None
    if not first:
        return tuple(second)
    pairs = zip(first, second, strict=True)
    return tuple(left ^ right for left, right in pairs)


def _boundary_ids_of(result: decoding_records.DecodeResult) -> tuple:
    boundary = result.boundary_data
    if boundary is None:
        return ()
    return tuple(boundary.detector_ids)


def _row(round_index: int, outcomes: list) -> dict:
    windows = sum(outcome.windows for outcome in outcomes)
    escalated = sum(outcome.escalated_windows for outcome in outcomes)
    return {
        "round": round_index,
        "failed": [outcome.failed for outcome in outcomes],
        "windows": windows,
        "escalated_windows": escalated,
    }
