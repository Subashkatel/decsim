"""Decoder models and routing helpers.

Every decoder here implements the Decoder port (protocols.py, port 8):
``latency(job)`` prices one window-job's compute as a service time in ticks
(the manager dispatches the job to a free unit and schedules completion that
many ticks later), and ``decode(job)`` produces the DecodeResult. Timing-only
decoders return empty results; data-path decoders also compute corrections.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

from .message import DecodeJob, DecodeResult
from .config import us

if TYPE_CHECKING:
    from .protocols import Decoder


class CodeRouter:
    """Route each job by code name, with a default decoder fallback."""

    def __init__(self, default, by_code: Optional[dict] = None):
        self.default = default
        self.by_code = dict(by_code) if by_code else {}

    def route(self, job: DecodeJob):
        """Pick the decoder for this job (by code; default when unmapped)."""
        return self.by_code.get(job.code, self.default)


class FunctionLatencyDecoder:
    """Timing-only decoder whose service time comes from a caller-supplied
    ``job -> microseconds`` function (any factors — n_rounds, spatial_nodes,
    code, attempt — are on the job). One-off models belong next to the
    experiment that uses them; the named classes below are the established
    parameterizations of this one."""

    def __init__(self, latency_us_for):
        self.latency_us_for = latency_us_for   # job -> microseconds

    def latency(self, job: DecodeJob) -> int:
        """Service time in ticks, priced by the caller's function."""
        return us(self.latency_us_for(job))

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Return an empty timing-only result."""
        return DecodeResult(job.op_id, job.window_id)


class PresetLatencyDecoder(FunctionLatencyDecoder):
    """Timing-only decoder with one fixed latency for every job
    (independent of window size)."""

    def __init__(self, latency_us: float = 1.0):
        self.latency_us = latency_us
        super().__init__(lambda job: self.latency_us)


class PerRoundDecoder(FunctionLatencyDecoder):
    """Timing-only decoder with linear cost per syndrome round:
    n_rounds * tau_us."""

    def __init__(self, tau_us: float = 1.0):
        self.tau_us = tau_us
        super().__init__(lambda job: job.n_rounds * self.tau_us)


class SwitchingDecoder:
    """Naive serial weak/strong switch: one job pays weak, handoff, and
    strong latency back-to-back on the SAME unit.

    This is the timing-level baseline for switching A/B studies. Unlike
    the routed path (SwitchingRouter + Switching strategy), the strong
    decode here is not a separate job, so it cannot queue for or contend
    over strong units."""

    def __init__(self, weak: "Decoder", strong: "Decoder", gamma_switch: float,
                 handoff_us: float = 0.5, seed: int = 0,
                 t_comm_weak_us: float = 0.0):
        self.weak = weak
        self.strong = strong
        self.gamma_switch = gamma_switch
        self.handoff = us(handoff_us)
        self.t_comm_weak = us(t_comm_weak_us)
        self.rng = random.Random(seed)
        self.switches = 0                      # diagnostic: how many jobs escalated

    def latency(self, job: DecodeJob) -> int:
        """Weak latency, plus handoff and strong latency when a switch is sampled.

        Sampling happens HERE, and the job is marked via job.hint so that
        decode() later follows the same weak/strong path it was priced for."""
        latency_ticks = self.t_comm_weak + self.weak.latency(job)
        if self.rng.random() < self.gamma_switch:
            job.hint = "strong"
            self.switches += 1
            # 2x: the handoff is a round trip (syndrome over to the strong
            # decoder, result back)
            latency_ticks += 2 * self.handoff + self.strong.latency(job)
        return latency_ticks

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Decode through the path sampled by latency()."""
        if job.hint == "strong":
            result = self.strong.decode(job)
            result.soft_output = 0.0
        else:
            result = self.weak.decode(job)
            result.soft_output = 1.0
        return result


class SwitchingRouter:
    """Route strong side jobs to the strong decoder and all other jobs to weak."""

    def __init__(self, weak: "Decoder", strong: "Decoder"):
        self.weak = weak
        self.strong = strong

    def route(self, job: DecodeJob):
        """Strong decoder for escalated jobs, weak decoder for everything else."""
        return self.strong if job.hint == "strong" else self.weak


class SampledConfidenceDecoder:
    """Pretends the decoder inside it is unsure about a random fraction of
    windows.

    Timing-only decoders produce no syndrome data, so there is nothing to
    compute a real confidence from; this wrapper asserts one as an
    experimental input. After the inner (weak) decode, a seeded coin sets
    ``result.soft_output``: 0.0 with ``escalation_probability`` (low
    confidence — the Switching strategy escalates the window), else 1.0
    (keep the weak result). ``probability_for`` replaces the flat rate
    with a per-job function (see switch_probability_per_round). Latency
    passes through to the inner decoder unchanged. Swap in a real
    soft-output decoder and the downstream pipeline behaves identically."""

    def __init__(self, inner: "Decoder", escalation_probability: float, seed: int = 0,
                 probability_for=None):
        self.inner = inner
        self.escalation_probability = escalation_probability
        self.probability_for = probability_for
        self.rng = random.Random(seed)

    def latency(self, job: DecodeJob) -> int:
        """The weak decode latency; the strong path is the cluster's separate parallel job."""
        return self.inner.latency(job)

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Weak-decode the window, then attach the sampled soft output."""
        result = self.inner.decode(job)
        if self.probability_for is not None:
            escalation_probability = self.probability_for(job)
        else:
            escalation_probability = self.escalation_probability
        result.soft_output = 0.0 if self.rng.random() < escalation_probability \
            else 1.0
        return result


def switch_probability_per_round(gamma_switch: float, d: int):
    """Per-window escalation probability that scales with window size.

    ``gamma_switch`` is the escalation rate per d rounds; a window
    committing more rounds is proportionally more likely to escalate."""

    def probability(job: DecodeJob) -> float:
        window = job.window
        commit_rounds = (window.commit_hi - window.commit_lo + 1) \
            if window is not None else job.n_rounds
        return gamma_switch * commit_rounds / d
    return probability
