"""Policies for ordering, timing, and routing decode jobs.

A *job* is one window of syndrome data waiting to be decoded.
Schedulers pick the order jobs run. Deadline policies say when a job is due.
Lane policies pick which pool of decoder hardware runs it.
"""

from __future__ import annotations

from .message import DecodeJob


class FifoScheduler:
    """Decode in arrival order. One slow job blocks everything behind it."""

    def insert(self, queue: list, job: DecodeJob) -> None:
        """Put the job at the back of the queue."""
        queue.append(job)

    def pop(self, queue: list, now_ticks: int) -> DecodeJob:
        """Take the job that has waited longest."""
        return queue.pop(0)


class EarliestDeadlineScheduler:
    """Decode whatever is due soonest, even if it arrived late."""

    def insert(self, queue: list, job: DecodeJob) -> None:
        """Put the job in the queue."""
        queue.append(job)

    def pop(self, queue: list, now_ticks: int) -> DecodeJob:
        """Take the job with the nearest deadline."""
        queue.sort(key=lambda j: j.deadline)
        return queue.pop(0)


class WeightedUrgencyCostScheduler:
    """Trade off "due soonest" against "quickest to finish".

    Scores each job as w_u * urgency + w_c * cheapness, where urgency rises
    as the deadline nears and cheapness rises as the window gets smaller.
    Weights sum to 1: w_c=0 gives earliest-deadline, w_u=0 gives shortest-job.

    Triage steady-mode priority, arXiv:2605.04459 Eq. 2. Their emergency
    mode and Min-Degree-First are not implemented.
    """

    def __init__(self, w_u: float = 0.5, w_c: float = 0.5):
        """Set the urgency/cheapness split; the weights must sum to 1."""
        if abs(w_u + w_c - 1.0) > 1e-9:
            raise ValueError(f"w_u + w_c must be 1 (got {w_u} + {w_c})")
        self.w_u = float(w_u)
        self.w_c = float(w_c)

    def insert(self, queue: list, job: DecodeJob) -> None:
        """Put the job in the queue."""
        queue.append(job)

    def priority(self, job: DecodeJob, now_ticks: int) -> float:
        """Score the job now; higher runs sooner. Overdue jobs score highest."""
        slack = max(job.deadline - now_ticks, 1)
        return self.w_u / slack + self.w_c / max(job.n_rounds, 1)

    def pop(self, queue: list, now_ticks: int) -> DecodeJob:
        """Take the highest-scoring job; ties go to the oldest."""
        best = max(range(len(queue)),
                   key=lambda i: (
                       self.priority(queue[i], now_ticks),
                       -i,
                   ))
        return queue.pop(best)


class EnqueueTimeDeadline:
    """Give every job the same deadline, so none is more urgent than another."""

    def deadline(self, op, window, now: int, on_reaction_path: bool) -> int:
        """Due immediately."""
        return now


class ReactionPathDeadline:
    """Rush jobs the computation is waiting on; give the rest some slack."""

    def __init__(self, slack_ticks: int):
        """Set the slack granted to jobs not on the reaction path."""
        self.slack_ticks = int(slack_ticks)

    def deadline(self, op, window, now: int, on_reaction_path: bool) -> int:
        """Due now on the reaction path, now + slack otherwise."""
        return now if on_reaction_path else now + self.slack_ticks


class BufferExpiryDeadline:
    """Finish before the hardware overwrites the window's syndrome data.

    The buffer holds capacity_rounds rounds; once that many newer rounds
    arrive the window's first round is gone. So older windows are the urgent
    ones, not the fresh ones.
    """

    def __init__(self, capacity_rounds: int, round_ticks: int):
        """Set the buffer depth in rounds and how long one round takes."""
        self.capacity_rounds = int(capacity_rounds)
        self.round_ticks = int(round_ticks)

    def deadline(self, op, window, now: int, on_reaction_path: bool) -> int:
        """Tick when the window's first round gets overwritten."""
        first = getattr(window, "t_first_round", None)
        if first is None:
            raise RuntimeError(
                f"cannot stamp buffer-expiry deadline for window {window.key}: "
                "first-round arrival provenance is missing"
            )
        return first + self.capacity_rounds * self.round_ticks
