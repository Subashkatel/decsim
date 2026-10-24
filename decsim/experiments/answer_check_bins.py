"""Windows binned by the weak answer's confidence, against the truth.

A confidence is useful to a switching machine when the windows under
its threshold are the ones whose weak answer is wrong and whose strong
answer is right. Each window's answer checks (observe/answer_checks.py)
say which of its two answers passed; this bins them by the weak gap.

A window's check also holds the chain's past answers, and a past that
is wrong turns which answer fails around. A row counts where the
answer the chain kept on the window before it passed, so the past is
right, as the first window's is. A check also decodes the windows cut
at its round, whose own errors fail both answers alike; they end with
the check, so the window after one still counts once its past passes
again. In each bin it gives how often the weak answer fails where the
strong one passes (escalation repairs), the strong fails where the
weak passes (escalation breaks), and both fail (no choice helps, or
the cut windows erred), each with its exact 95 percent interval
(failure_statistics.estimate, Clopper and Pearson for a fixed count).
"""

import dataclasses
import math
from collections.abc import Iterable, Mapping, Sequence

import decsim.experiments.failure_statistics as failure_statistics

FIXED_COUNT = failure_statistics.StopKind.CAP


@dataclasses.dataclass(frozen=True)
class GapBin:
    """The counted windows whose weak gap lies in [low_nats, high_nats).

    escalated counts the windows the machine's verdict would escalate.
    """

    low_nats: float
    high_nats: float
    windows: int
    escalated: int
    weak_only: failure_statistics.Estimate
    strong_only: failure_statistics.Estimate
    both: failure_statistics.Estimate


def gap_bins(
    streams: Iterable[Sequence[Mapping]], edges_nats: Sequence[float]
) -> list:
    """One GapBin a pair of neighbouring edges, over every stream's rows.

    streams holds each stream's rows of one path, in window order, as
    AnswerChecks.rows gives them. A row with no gap or no check is left
    out, and so is a row whose window before kept a failing answer.
    """
    counts = [_Counts() for _ in edges_nats[1:]]
    for rows in streams:
        _count_stream(counts, rows, edges_nats)
    bins = []
    for index, count in enumerate(counts):
        low = edges_nats[index]
        high = edges_nats[index + 1]
        gap_bin = count.binned(low, high)
        bins.append(gap_bin)
    return bins


@dataclasses.dataclass
class _Counts:
    windows: int = 0
    escalated: int = 0
    weak_only: int = 0
    strong_only: int = 0
    both: int = 0

    def add(self, row: Mapping) -> None:
        weak_failed = row["weak_failed"]
        strong_failed = row["strong_failed"]
        self.windows += 1
        self.escalated += row["would_escalate"]
        self.weak_only += weak_failed and not strong_failed
        self.strong_only += strong_failed and not weak_failed
        self.both += weak_failed and strong_failed

    def binned(self, low: float, high: float) -> GapBin:
        weak_only = _estimate(self.weak_only, self.windows)
        strong_only = _estimate(self.strong_only, self.windows)
        both = _estimate(self.both, self.windows)
        return GapBin(
            low,
            high,
            self.windows,
            self.escalated,
            weak_only,
            strong_only,
            both,
        )


def _count_stream(counts: list, rows: Sequence[Mapping], edges_nats) -> None:
    """Add the stream's counted rows to the bins their gaps fall in."""
    for row in _counted(rows):
        index = _bin_of(row["weak_gap"], edges_nats)
        if index is not None:
            counts[index].add(row)


def _counted(rows: Sequence[Mapping]) -> list:
    """The rows whose window before kept an answer that passed its check."""
    counted = []
    past_passed = True
    for row in rows:
        if past_passed and _is_whole(row):
            counted.append(row)
        kept_failed = row[f"{row['kept']}_failed"]
        past_passed = kept_failed is False
    return counted


def _is_whole(row: Mapping) -> bool:
    """Whether the row has a gap and both answers' checks."""
    has_gap = row["weak_gap"] is not None
    checked = row["weak_failed"] is not None
    checked = checked and row["strong_failed"] is not None
    return has_gap and checked


def _bin_of(gap: float, edges_nats: Sequence[float]):
    """The bin index holding the gap, None outside the edges.

    A last edge at infinity holds the infinite gap of a growth with no
    odd walk (confidence/cluster.py).
    """
    pairs = zip(edges_nats[:-1], edges_nats[1:], strict=True)
    for index, (low, high) in enumerate(pairs):
        below_high = gap < high or high == math.inf
        if low <= gap and below_high:
            return index
    return None


def _estimate(failures: int, windows: int) -> failure_statistics.Estimate:
    return failure_statistics.estimate(failures, windows, FIXED_COUNT)
