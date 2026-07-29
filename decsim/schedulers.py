"""Decode queue ordering, deadline, and lane policies."""

from __future__ import annotations

from .message import DecodeJob


class FifoScheduler:
    """First-in, first-out."""

    def insert(self, queue: list, job: DecodeJob) -> None:
        """Add a job to the back of the queue."""
        queue.append(job)

    def pop(self, queue: list, now_ticks: int) -> DecodeJob:
        """Take the oldest job (first in, first out)."""
        return queue.pop(0)


class EarliestDeadlineScheduler:
    """Send the job with the nearest deadline first."""

    def insert(self, queue: list, job: DecodeJob) -> None:
        """Add a job to the queue."""
        queue.append(job)

    def pop(self, queue: list, now_ticks: int) -> DecodeJob:
        """Take the job with the earliest deadline."""
        queue.sort(key=lambda j: j.deadline)
        return queue.pop(0)


class WeightedUrgencyCostScheduler:
    """Triage steady-mode priority function at the job level.

    Pops the queued job maximizing
        P(job) = w_u * urgency + w_c * cost_efficiency
    with urgency = 1 / max(deadline - now, 1 tick) and
    cost_efficiency = 1 / n_rounds — the decsim mapping of Triage's
    Eq. 2 (arXiv:2605.04459: slices -> jobs, decode cost -> window
    rounds). w_u + w_c = 1. w_c = 0 degenerates to EDF ordering
    (strictly, nearest-deadline-first among unexpired jobs); w_u = 0
    to shortest-job-first.

    SCOPE: steady mode ONLY. Triage's dual-mode emergency scheduler
    (predictive causal-cone coloring) and the Min-Degree-First policy
    (needs decoding-graph degree, which decsim jobs do not carry) are
    NOT implemented; the paper remains a qualitative anchor for those.
    The decoder manager supplies the exact current dispatch tick to `pop`.
    """

    def __init__(self, w_u: float = 0.5, w_c: float = 0.5):
        if abs(w_u + w_c - 1.0) > 1e-9:
            raise ValueError(f"w_u + w_c must be 1 (got {w_u} + {w_c})")
        self.w_u = float(w_u)
        self.w_c = float(w_c)

    def insert(self, queue: list, job: DecodeJob) -> None:
        """Add a job to the queue."""
        queue.append(job)

    def priority(self, job: DecodeJob, now_ticks: int) -> float:
        """Triage Eq.2 mapped to decsim jobs (higher = served first)."""
        slack = max(job.deadline - now_ticks, 1)
        return self.w_u / slack + self.w_c / max(job.n_rounds, 1)

    def pop(self, queue: list, now_ticks: int) -> DecodeJob:
        """Take the highest-priority job."""
        best = max(range(len(queue)),
                   key=lambda i: (
                       self.priority(queue[i], now_ticks),
                       -i,
                   ))
        return queue.pop(best)


class EnqueueTimeDeadline:
    """Default deadline at window-job construction/logical admission."""

    def deadline(self, op, window, now: int, on_reaction_path: bool) -> int:
        """Return the policy-stamp tick (all newly built jobs equally urgent)."""
        return now


class ReactionPathDeadline:
    """Reaction-path windows get a tight deadline; others get now + slack."""

    def __init__(self, slack_ticks: int):
        self.slack_ticks = int(slack_ticks)

    def deadline(self, op, window, now: int, on_reaction_path: bool) -> int:
        """Tight deadline on the reaction path; now + slack off it."""
        return now if on_reaction_path else now + self.slack_ticks


class BufferExpiryDeadline:
    """Deadline = the tick the window's oldest buffered round expires.

    Models a bounded per-patch syndrome buffer of capacity_rounds rounds:
    the buffer entry holding the window's FIRST round is overwritten
    capacity_rounds * round_ticks after that round arrived, so the decode
    must complete by window.t_first_round + capacity_rounds * round_ticks.
    Larger/older windows therefore carry TIGHTER deadlines than fresh
    ones (their data expires sooner) — the opposite of uniform slack.

    Pure buffer semantics: on_reaction_path is ignored here (reaction
    tightening stays ReactionPathDeadline's job). A window without its
    first-round arrival stamp cannot define a buffer-expiry deadline.
    """

    def __init__(self, capacity_rounds: int, round_ticks: int):
        self.capacity_rounds = int(capacity_rounds)
        self.round_ticks = int(round_ticks)

    def deadline(self, op, window, now: int, on_reaction_path: bool) -> int:
        """Expiry tick of the window's first buffered round."""
        first = getattr(window, "t_first_round", None)
        if first is None:
            raise RuntimeError(
                f"cannot stamp buffer-expiry deadline for window {window.key}: "
                "first-round arrival provenance is missing"
            )
        return first + self.capacity_rounds * self.round_ticks


class ReservedCapacityLanes:
    """Route burst-hit patches to a reserved pool once activated.

    Inactive (the default), every job goes to the default pool and
    the reserved units sit idle — that idle slice IS the reservation;
    its cost is the premium an experiment must measure. activate()
    latches routing of the given patches' jobs to the reserved pool;
    extend() grows the hit set while active. patch_of(job) supplies
    the job's patch (None -> default pool). Explicit job.hint still
    wins (DecoderManager.pool_for precedence), and jobs already
    queued keep their enqueue-time placement.

    deactivate() releases the reservation (Gate 7 P16); the
    activation itself still latches until the caller's rule fires.
    """

    def __init__(self, pool: str, patch_of):
        self.pool = pool
        self.patch_of = patch_of
        self.active = False
        self.hit_patches = frozenset()

    def activate(self, patches) -> None:
        """Latch reserved routing for these patches."""
        self.active = True
        self.hit_patches = frozenset(patches)

    def extend(self, patches) -> None:
        """Grow the hit set (no-op unless active)."""
        if self.active:
            self.hit_patches = self.hit_patches | frozenset(patches)

    def deactivate(self) -> None:
        """Release the reservation: routing returns to default for
        all patches; extend() becomes a no-op until a fresh
        activate(). (Gate 7 P16 — the V18 latch's counterpart.)"""
        self.active = False
        self.hit_patches = frozenset()

    def pool_for(self, job: DecodeJob):
        """Reserved pool for active hit patches; None -> default."""
        if self.active and self.patch_of(job) in self.hit_patches:
            return self.pool
        return None


class DistanceLanes:
    """Assign decode jobs to unit pools ("lanes") keyed by code distance.

    lanes maps distance -> pool name; distance_of(job) supplies the
    job's distance (return None for unknown). Jobs whose distance has
    no lane — or whose distance is unknown — go to the default pool.
    An explicit job.hint always wins (DecoderManager.pool_for), so
    strong-decode routing is untouched by lane assignment.
    """

    def __init__(self, lanes: dict, distance_of):
        self.lanes = dict(lanes)
        self.distance_of = distance_of

    def pool_for(self, job: DecodeJob):
        """Lane pool name for the job, or None for the default pool."""
        return self.lanes.get(self.distance_of(job))
