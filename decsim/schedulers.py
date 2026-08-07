"""How queued decode work is ordered, timed, and routed.

Three separate, independently swappable policy kinds live here. The decoder
manager holds one of each.

1. Schedulers (`insert` / `pop`) decide the ORDER jobs leave the queue.
2. Deadline policies (`deadline`) stamp each job with the tick by which it
   ought to be finished. Schedulers may then order by that stamp.
3. Lane policies (`pool_for`) decide WHICH pool of decoder hardware a job
   runs on.

Vocabulary used throughout: a *job* is one window of syndrome data waiting to
be decoded; *ticks* are simulator time units; a *pool* is a named group of
decoder units (e.g. a bank of FPGAs).
"""

from __future__ import annotations

from .message import DecodeJob


class FifoScheduler:
    """Decode in arrival order.

    The simplest policy and the usual baseline: no job ever overtakes an
    earlier one, so a single slow job holds up everything behind it.
    """

    def insert(self, queue: list, job: DecodeJob) -> None:
        """Put the job at the back of the queue."""
        queue.append(job)

    def pop(self, queue: list, now_ticks: int) -> DecodeJob:
        """Take the job that has been waiting longest."""
        return queue.pop(0)


class EarliestDeadlineScheduler:
    """Decode whichever job is closest to its deadline (classic EDF).

    Lets an urgent job overtake jobs that arrived earlier but have more time
    left, so work is likelier to finish before it is needed. Deadlines come
    from whichever deadline policy is installed.
    """

    def insert(self, queue: list, job: DecodeJob) -> None:
        """Put the job in the queue; ordering happens at pop time."""
        queue.append(job)

    def pop(self, queue: list, now_ticks: int) -> DecodeJob:
        """Take the job with the nearest deadline."""
        queue.sort(key=lambda j: j.deadline)
        return queue.pop(0)


class WeightedUrgencyCostScheduler:
    """Balance "due soonest" against "quickest to finish".

    Pure EDF always serves the most urgent job even when it is huge and will
    block the queue. This policy scores every waiting job on two things and
    serves the highest score:

        score = w_u * urgency + w_c * cheapness
        urgency   = 1 / time left until the deadline   (nearer  -> higher)
        cheapness = 1 / number of rounds in the window (smaller -> higher)

    The two weights must sum to 1 and set where you sit between the extremes:
    `w_c = 0` behaves like EarliestDeadlineScheduler, `w_u = 0` behaves like
    shortest-job-first. Ties are broken toward the job queued earliest.

    This is the decsim mapping of the Triage steady-mode priority function
    (arXiv:2605.04459 Eq. 2), with the paper's slices read as jobs and its
    decode cost read as window rounds.

    Steady mode only. Triage's emergency dual-mode scheduler and its
    Min-Degree-First policy are deliberately not implemented -- the latter
    needs decoding-graph degree, which decsim jobs do not carry.
    """

    def __init__(self, w_u: float = 0.5, w_c: float = 0.5):
        """Set the urgency/cheapness split. The two weights must sum to 1."""
        if abs(w_u + w_c - 1.0) > 1e-9:
            raise ValueError(f"w_u + w_c must be 1 (got {w_u} + {w_c})")
        self.w_u = float(w_u)
        self.w_c = float(w_c)

    def insert(self, queue: list, job: DecodeJob) -> None:
        """Put the job in the queue; scoring happens at pop time."""
        queue.append(job)

    def priority(self, job: DecodeJob, now_ticks: int) -> float:
        """Score this job right now. Higher means served sooner.

        Time left is floored at one tick so an overdue job scores high
        rather than dividing by zero or going negative.
        """
        slack = max(job.deadline - now_ticks, 1)
        return self.w_u / slack + self.w_c / max(job.n_rounds, 1)

    def pop(self, queue: list, now_ticks: int) -> DecodeJob:
        """Take the highest-scoring job, breaking ties toward the oldest."""
        best = max(range(len(queue)),
                   key=lambda i: (
                       self.priority(queue[i], now_ticks),
                       -i,
                   ))
        return queue.pop(best)


class EnqueueTimeDeadline:
    """Give every job the same deadline: the moment it was created.

    The default. Since no job is more urgent than any other, a deadline-aware
    scheduler falls back to arrival order.
    """

    def deadline(self, op, window, now: int, on_reaction_path: bool) -> int:
        """Return now, so all newly built jobs are equally urgent."""
        return now


class ReactionPathDeadline:
    """Rush the jobs the computation is actually waiting on.

    A window is "on the reaction path" when a later quantum operation cannot
    proceed until this decode returns. Those get an immediate deadline;
    everything else gets a fixed grace period.
    """

    def __init__(self, slack_ticks: int):
        """Set the grace period granted to jobs off the reaction path."""
        self.slack_ticks = int(slack_ticks)

    def deadline(self, op, window, now: int, on_reaction_path: bool) -> int:
        """Due immediately on the reaction path, now + slack otherwise."""
        return now if on_reaction_path else now + self.slack_ticks


class BufferExpiryDeadline:
    """Finish each decode before its syndrome data is overwritten.

    Hardware holds syndrome rounds in a buffer that fits only
    `capacity_rounds` of them. Once that many newer rounds arrive, the
    oldest is overwritten and any decode still needing it has lost its
    input. So the real deadline is when the window's FIRST round expires:

        deadline = arrival of first round + capacity_rounds * round_ticks

    Note this makes older windows MORE urgent than fresh ones, because their
    data expires sooner -- the reverse of handing everyone equal slack.

    Reaction-path urgency is deliberately ignored here; that belongs to
    ReactionPathDeadline. A window that never recorded when its first round
    arrived has no definable expiry, and asking for one raises.
    """

    def __init__(self, capacity_rounds: int, round_ticks: int):
        """Set the buffer depth in rounds and the duration of one round."""
        self.capacity_rounds = int(capacity_rounds)
        self.round_ticks = int(round_ticks)

    def deadline(self, op, window, now: int, on_reaction_path: bool) -> int:
        """Return the tick this window's oldest buffered round is overwritten."""
        first = getattr(window, "t_first_round", None)
        if first is None:
            raise RuntimeError(
                f"cannot stamp buffer-expiry deadline for window {window.key}: "
                "first-round arrival provenance is missing"
            )
        return first + self.capacity_rounds * self.round_ticks


class DistanceLanes:
    """Send jobs to different hardware pools based on code distance.

    Bigger code distances mean bigger, slower decodes, so it can pay to give
    them their own units instead of letting them clog the queue shared with
    small ones.

    `lanes` maps a distance to a pool name; `distance_of(job)` reports a
    job's distance, or None if it is unknown. A job whose distance is unknown
    or has no lane falls through to the default pool. A job carrying an
    explicit `hint` bypasses this entirely (see DecoderManager.pool_for), so
    lane routing never interferes with strong-decoder escalation.
    """

    def __init__(self, lanes: dict, distance_of):
        """Set the distance -> pool map and how to read a job's distance."""
        self.lanes = dict(lanes)
        self.distance_of = distance_of

    def pool_for(self, job: DecodeJob):
        """Return this job's pool name, or None to use the default pool."""
        return self.lanes.get(self.distance_of(job))
