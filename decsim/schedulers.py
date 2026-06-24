"""Decode queue ordering policies."""

from __future__ import annotations

from .message import DecodeJob


class FifoScheduler:
    """First-in, first-out."""

    def insert(self, queue: list, job: DecodeJob) -> None:
        """Add a job to the back of the queue."""
        queue.append(job)

    def pop(self, queue: list) -> DecodeJob:
        """Take the oldest job (first in, first out)."""
        return queue.pop(0)


class EarliestDeadlineScheduler:
    """Send the job with the nearest deadline first."""

    def insert(self, queue: list, job: DecodeJob) -> None:
        """Add a job to the queue."""
        queue.append(job)

    def pop(self, queue: list) -> DecodeJob:
        """Take the job with the earliest deadline."""
        queue.sort(key=lambda j: j.deadline)
        return queue.pop(0)


class EnqueueTimeDeadline:
    """Default policy where every job deadline is its enqueue time."""

    def deadline(self, op, window, now: int, on_reaction_path: bool) -> int:
        """Return the enqueue time (all jobs equally urgent)."""
        return now


class ReactionPathDeadline:
    """Reaction-path-first deadline policy.

    A window whose result unblocks a waiting non-Clifford operation gets a tight
    deadline. Other windows get now + slack, so waiting non-reaction jobs still run.
    """

    def __init__(self, slack_ticks: int):
        self.slack_ticks = int(slack_ticks)

    def deadline(self, op, window, now: int, on_reaction_path: bool) -> int:
        """Tight deadline on the reaction path; now + slack off it."""
        return now if on_reaction_path else now + self.slack_ticks
