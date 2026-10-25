"""Each window's weak and strong answers, each checked against the truth.

A switching machine keeps the weak decoder's answer on a window or has
the strong decoder decode it again. Whether the choice mattered shows
only against the truth, and a window has a truth only through a
checkpoint (Tan et al. 2209.09219 lines 1003-1037; observe/checkpoints.py):
the logical outcome had the stream been read out after a round. A
window's check is the checkpoint at the first round where a stream that
ended there keeps the window's own answer, the round after the last it
reads, or the stream's last round for a window that reads it.

This listener keeps its own chain of answers for each path it is given,
at zero simulated time, and never feeds the machine. For each window in
order it decodes the window on the entry pool and on the edge's target
(observe/closing_decisions.py AnswerPair), the chain's boundary folded
in, closes the check once with each answer (observe/closing_windows.py:
the windows after it that commit a round up to the check's are cut
there and decided on the machine's own path), records one row, and
keeps one answer by its path:

- weak: always the weak answer, so the chain's outcome is the weak
  decoder's alone;
- strong: always the strong answer, the strong decoder's alone;
- best_of_two: the weak answer unless it fails the check and the
  strong one passes: a perfect chooser of each window, the per-window
  best-of-two line. Sheth et al. define the per-shot one, an ensemble
  right when at least one of its decoders is (1905.02345 lines 186-196,
  equation 1; experiments/paired_shots.py).

A static chain, one a threshold it is given, keeps the weak answer
unless the machine's own verdict row escalates it at that fixed
threshold (ClosingDecisions.escalates_at), and then the strong one. It
is checked only at its last window and keeps no rows: its outcome, the
windows it escalated and the windows it decided.

A chain's outcome is its last window's check, which has no window after
it. No component reads any of this; the listener hears the source, the
device and the window side through their trace sources.
"""

import dataclasses
from typing import Any, Optional

import decsim.escalation.threshold_sources as threshold_sources
import decsim.observe.closing_decisions as closing_decisions
import decsim.observe.closing_windows as closing_windows
import decsim.observe.settings as observe_settings
import decsim.records.checkpoints as checkpoint_records
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records

WEAK, STRONG, BEST_OF_TWO = observe_settings.ANSWER_CHECK_PATHS


@dataclasses.dataclass(frozen=True)
class AnswerCheck:
    """One window of one chain: its check and the answer the chain kept.

    weak_gap is the weak answer's confidence in nats, None when it gave
    none. weak_failed and strong_failed are each answer's check, None
    when a window of it answered no observable. same_logical_class is
    whether the two answers carry the same observables for the window
    itself, None when one carries none. When they do, only the boundary
    each hands on can make one check fail and the other pass, through
    the windows the check cuts. would_escalate is the machine's verdict
    on the weak answer, its threshold copied then.
    """

    window_index: int
    check_round: int
    weak_gap: Optional[float]
    weak_failed: Optional[bool]
    strong_failed: Optional[bool]
    same_logical_class: Optional[bool]
    would_escalate: bool
    kept: str

    @property
    def kept_failed(self) -> Optional[bool]:
        """The check of the answer the chain kept."""
        if self.kept == STRONG:
            return self.strong_failed
        return self.weak_failed


class AnswerChecks:
    """The answer checks of a run's streams: one chain a path and a threshold.

    thresholds are the static chains' thresholds, in decibels.
    """

    def __init__(
        self,
        paths: tuple,
        thresholds: tuple,
        closer: closing_decisions.ClosingDecisions,
    ) -> None:
        self.paths = paths
        self.thresholds = thresholds
        self.closer = closer
        # stream id -> _StreamChecks, in the order streams were heard
        self.streams: dict = {}

    def checkpoint_reached(
        self, snapshot: checkpoint_records.CheckpointSnapshot
    ) -> None:
        """A round ran; its copy may be a window's check."""
        stream = self._stream(snapshot.stream_id)
        stream.keep_snapshot(snapshot, self.closer)
        stream.check_ready()

    def round_emitted(self, readout: round_records.QPUReadout) -> None:
        """A round left the QPU; its raw bits form the windows' events."""
        stream = self._stream(readout.stream_id)
        stream.packets[readout.round_index] = readout.bits
        stream.check_ready()

    def window_planned(self, window: window_records.Window) -> None:
        """A stream laid a window."""
        stream = self._stream(window.stream_id)
        stream.windows.append(window)

    def window_final(self, window: window_records.Window) -> None:
        """A window's record is final, so its extent no longer moves."""
        stream = self._stream(window.stream_id)
        stream.final_keys.add(window.key)
        stream.check_ready()

    def rows(self) -> list:
        """One row a path: each stream's outcome and its windows' checks.

        failed and windows hold each stream's, in the order the streams
        were heard, as a memory task's patches are.
        """
        streams = self.streams.values()
        rows = []
        for path in self.paths:
            chains = [stream.chains[path] for stream in streams]
            windows = [
                [dataclasses.asdict(check) for check in chain.checks]
                for chain in chains
            ]
            row = {
                "path": path,
                "failed": [chain.failed for chain in chains],
                "windows": windows,
            }
            rows.append(row)
        return rows

    def threshold_rows(self) -> list:
        """One row a threshold: each stream's outcome and window counts.

        failed, escalated_windows and window_count hold each stream's,
        in the order the streams were heard.
        """
        streams = self.streams.values()
        rows = []
        for decibels in self.thresholds:
            chains = [stream.chains[decibels] for stream in streams]
            row = {
                "threshold_decibels": decibels,
                "failed": [chain.failed for chain in chains],
                "escalated_windows": [
                    chain.escalated_windows for chain in chains
                ],
                "window_count": [chain.window_count for chain in chains],
            }
            rows.append(row)
        return rows

    def _stream(self, stream_id: Any) -> "_StreamChecks":
        stream = self.streams.get(stream_id)
        if stream is None:
            stream = _StreamChecks(self.paths, self.thresholds)
            self.streams[stream_id] = stream
        return stream


@dataclasses.dataclass(frozen=True)
class _WindowCheck:
    """What a window's check closes on.

    The copy at the check's round, the window as decided there, its
    events, and the laid windows the check cuts after it.
    """

    snapshot: checkpoint_records.CheckpointSnapshot
    window: window_records.Window
    events: dict
    tail: list


@dataclasses.dataclass(frozen=True)
class _Answers:
    """One window's two answers on one boundary, and the job they answer."""

    job: decoding_records.DecodeJob
    pair: closing_decisions.AnswerPair


@dataclasses.dataclass(frozen=True)
class _Answered:
    """One window's answers on one boundary, each with its check's prediction.

    A prediction is the answer's observables XOR the tail's, closed on
    the boundary the answer hands on; a chain XORs in its own past.
    """

    pair: closing_decisions.AnswerPair
    weak_prediction: Optional[tuple]
    strong_prediction: Optional[tuple]


class _WindowAnswers:
    """One window's answers on each boundary a chain hands it.

    Chains on one boundary share the window's two answers, decoded once,
    and the tail closed behind each, closed only when a chain checks.
    """

    def __init__(
        self, closing: closing_windows.ClosingWindows, check: _WindowCheck
    ) -> None:
        self.closing = closing
        self.check = check
        # boundary ids -> _Answers
        self.answers: dict = {}
        # boundary ids -> _Answered
        self.checked: dict = {}

    def answers_on(self, boundary_ids: tuple) -> _Answers:
        """The two answers on the boundary, decoded once."""
        answers = self.answers.get(boundary_ids)
        if answers is None:
            check = self.check
            job = self.closing.job(
                check.snapshot, check.window, check.events, boundary_ids
            )
            pair = self.closing.closer.answer_pair(job)
            answers = _Answers(job, pair)
            self.answers[boundary_ids] = answers
        return answers

    def checked_on(self, boundary_ids: tuple) -> _Answered:
        """The two answers on the boundary, each with its check's prediction."""
        answered = self.checked.get(boundary_ids)
        if answered is None:
            answers = self.answers_on(boundary_ids)
            pair = answers.pair
            weak = self._prediction(pair.weak)
            strong = self._prediction(pair.strong)
            answered = _Answered(pair, weak, strong)
            self.checked[boundary_ids] = answered
        return answered

    def _prediction(self, result) -> Optional[tuple]:
        """The answer's observables XOR the tail closed behind it."""
        if result is None:
            return None
        check = self.check
        boundary_ids = closing_windows.boundary_ids_of(result)
        tail_observables = self.closing.tail_observables(
            check.snapshot, check.events, boundary_ids, check.tail
        )
        # an empty tail XORs nothing in
        answer = result.logical_observables
        return closing_windows.xor_or_none(tail_observables, answer)


@dataclasses.dataclass
class _Chain:
    """One path's answers so far: their observables and the last boundary.

    failed is the chain's outcome, None until its last window is checked.
    """

    path: str
    observables: Optional[tuple] = ()
    boundary_ids: tuple = ()
    checks: list = dataclasses.field(default_factory=list)
    failed: Optional[bool] = None

    def take(
        self, answers: _WindowAnswers, window_index: int, is_last: bool
    ) -> None:
        """Check the window and keep the path's answer."""
        answered = answers.checked_on(self.boundary_ids)
        snapshot = answers.check.snapshot
        _keep(self.path, self, answered, snapshot, window_index)
        if is_last:
            self.failed = self.checks[-1].kept_failed


@dataclasses.dataclass
class _StaticChain:
    """One fixed threshold's answers so far, kept by the verdict row.

    escalated_windows and window_count count the windows it escalated
    and decided; failed is its outcome, None until its last window.
    """

    threshold_nats: float
    observables: Optional[tuple] = ()
    boundary_ids: tuple = ()
    escalated_windows: int = 0
    window_count: int = 0
    failed: Optional[bool] = None

    def take(
        self, answers: _WindowAnswers, window_index: int, is_last: bool
    ) -> None:
        """Keep the answer the row gives at the threshold; check the last."""
        del window_index
        found = answers.answers_on(self.boundary_ids)
        closer = answers.closing.closer
        weak = found.pair.weak
        escalates = closer.escalates_at(found.job, weak, self.threshold_nats)
        result = weak
        if escalates:
            result = found.pair.strong
        if is_last:
            self.failed = self._last_failed(answers, escalates)
        self.window_count += 1
        self.escalated_windows += escalates
        self.observables = _observables_after(self.observables, result)
        self.boundary_ids = _boundary_after(result)

    def _last_failed(
        self, answers: _WindowAnswers, escalates: bool
    ) -> Optional[bool]:
        """The kept answer's check, with the chain's past XORed in."""
        answered = answers.checked_on(self.boundary_ids)
        prediction = answered.weak_prediction
        if escalates:
            prediction = answered.strong_prediction
        prediction = closing_windows.xor_or_none(self.observables, prediction)
        return closing_windows.failed(prediction, answers.check.snapshot)


class _StreamChecks:
    """One stream's rounds, copies, laid windows and chains."""

    def __init__(self, paths: tuple, thresholds: tuple) -> None:
        # round -> raw bits, from the earliest a window still to check reads
        self.packets: dict = {}
        # round -> the copy closed after it, from the next check's round
        self.snapshots: dict = {}
        # the laid windows not checked yet, in index order
        self.windows: list = []
        self.final_keys: set = set()
        # each path's chain under its name, each static chain under its
        # threshold in decibels
        self.chains: dict = {path: _Chain(path) for path in paths}
        for decibels in thresholds:
            threshold_nats = threshold_sources.decibels_to_nats(decibels)
            self.chains[decibels] = _StaticChain(threshold_nats)
        # made at the first copy, which carries the stream's program
        self.closing: Optional[closing_windows.ClosingWindows] = None

    def keep_snapshot(
        self,
        snapshot: checkpoint_records.CheckpointSnapshot,
        closer: closing_decisions.ClosingDecisions,
    ) -> None:
        """Hold the copy until the window it checks is checked."""
        if self.closing is None:
            self.closing = closing_windows.ClosingWindows(
                snapshot.program, closer
            )
        self.snapshots[snapshot.round_index] = snapshot

    def check_ready(self) -> None:
        """Check the windows in order while the next one can be."""
        while self.windows:
            window = self.windows[0]
            check_round = self._check_round(window)
            if check_round is None:
                return
            tail = self._tail_after(check_round)
            if tail is None:
                return
            self._check(window, check_round, tail)
            self.windows.pop(0)
            self._let_go(window, check_round)

    def _check_round(self, window: window_records.Window) -> Optional[int]:
        """The round of the window's check, None while it cannot be closed.

        The window's extent must be final, and the check's copy and raw
        round must have arrived.
        """
        if window.key not in self.final_keys:
            return None
        after_last_read = window.buffer_hi + 1
        check_round = after_last_read
        final_round = self._final_round()
        if final_round is not None:
            check_round = min(after_last_read, final_round)
        if check_round not in self.snapshots:
            return None
        if check_round not in self.packets:
            return None
        return check_round

    def _final_round(self) -> Optional[int]:
        """The stream's last round once its readout ran, else None.

        The copy of the last round is never let go: no check comes after
        it.
        """
        if not self.snapshots:
            return None
        latest = max(self.snapshots)
        if self.snapshots[latest].is_final:
            return latest
        return None

    def _tail_after(self, check_round: int) -> Optional[list]:
        """The laid windows after it that commit a round up to the check's.

        None while the window committing the check's round is not laid.
        """
        later = self.windows[1:]
        holds = any(laid.commit_hi >= check_round for laid in self.windows)
        if not holds:
            return None
        return [laid for laid in later if laid.commit_lo <= check_round]

    def _check(
        self, window: window_records.Window, check_round: int, tail: list
    ) -> None:
        """Every chain's take of the window, on answers they share."""
        snapshot = self.snapshots[check_round]
        events = self.closing.events(snapshot, window.start_round, self.packets)
        deciding = window
        if window.buffer_hi >= check_round:
            deciding = closing_windows.ending_on(window, check_round)
        is_last = not tail and check_round == self._final_round()
        check = _WindowCheck(snapshot, deciding, events, tail)
        answers = _WindowAnswers(self.closing, check)
        for chain in self.chains.values():
            chain.take(answers, window.window_index, is_last)

    def _let_go(self, window: window_records.Window, check_round: int) -> None:
        """Every later window reads from its own start, checked later.

        A later check's round is the same or later, as a later window
        reads the same rounds or later ones.
        """
        earliest = window.commit_hi + 1
        if self.windows:
            earliest = min(earliest, self.windows[0].start_round)
        earliest_packet = earliest - self.closing.reach
        _let_go_before(self.packets, earliest_packet)
        _let_go_before(self.snapshots, check_round)


def _keep(
    path: str,
    chain: _Chain,
    answered: _Answered,
    snapshot: checkpoint_records.CheckpointSnapshot,
    window_index: int,
) -> None:
    """Record the chain's check of the window and keep its path's answer."""
    pair = answered.pair
    weak_prediction = closing_windows.xor_or_none(
        chain.observables, answered.weak_prediction
    )
    strong_prediction = closing_windows.xor_or_none(
        chain.observables, answered.strong_prediction
    )
    weak_failed = closing_windows.failed(weak_prediction, snapshot)
    strong_failed = closing_windows.failed(strong_prediction, snapshot)
    kept = _kept_answer(path, weak_failed, strong_failed)
    weak_gap = _gap_of(pair.weak)
    same_logical_class = _same_logical_class(pair)
    check = AnswerCheck(
        window_index=window_index,
        check_round=snapshot.round_index,
        weak_gap=weak_gap,
        weak_failed=weak_failed,
        strong_failed=strong_failed,
        same_logical_class=same_logical_class,
        would_escalate=pair.would_escalate,
        kept=kept,
    )
    chain.checks.append(check)
    result = pair.weak
    if kept == STRONG:
        result = pair.strong
    chain.observables = _observables_after(chain.observables, result)
    chain.boundary_ids = _boundary_after(result)


def _kept_answer(
    path: str, weak_failed: Optional[bool], strong_failed: Optional[bool]
) -> str:
    """The answer the path keeps on the window."""
    if path == WEAK:
        return WEAK
    if path == STRONG:
        return STRONG
    strong_helps = weak_failed is True and strong_failed is False
    if strong_helps:
        return STRONG
    return WEAK


def _let_go_before(by_round: dict, first_round: int) -> None:
    """Forget every round before first_round."""
    old = [round_index for round_index in by_round if round_index < first_round]
    for round_index in old:
        del by_round[round_index]


def _same_logical_class(
    pair: closing_decisions.AnswerPair,
) -> Optional[bool]:
    """Whether both answers carry the same observables for the window."""
    if pair.weak is None or pair.strong is None:
        return None
    weak = pair.weak.logical_observables
    strong = pair.strong.logical_observables
    if weak is None or strong is None:
        return None
    return weak == strong


def _gap_of(result) -> Optional[float]:
    if result is None or result.soft_output is None:
        return None
    return result.soft_output.gap


def _observables_after(observables: Optional[tuple], result) -> Optional[tuple]:
    if result is None:
        return None
    return closing_windows.xor_or_none(observables, result.logical_observables)


def _boundary_after(result) -> tuple:
    if result is None:
        return ()
    return closing_windows.boundary_ids_of(result)
