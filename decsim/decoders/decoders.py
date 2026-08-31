"""Decoder models and routing helpers.

Every decoder here implements the Decoder protocol (protocols.py):
``latency(job)`` prices one window-job's compute as a service time in ticks
(the manager dispatches the job to a free unit and schedules completion that
many ticks later), and ``decode(job)`` produces the DecodeResult. Timing-only
decoders return empty results; data-path decoders also compute corrections.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from ..message import (
    DecodeJob,
    DecodeResult,
    RunSeedChild,
    RunSeedPathSegment,
    SoftOutput,
    SoftOutputSource,
)
from ..config import us
from ..seeding import _RandomSeedConsumer
from ..detector_error_model.fault_model_contracts import (
    DecoderFaultModelRequirement,
    NO_FAULT_MODEL_REQUIRED,
)

SAMPLED_CONFIDENCE_SOURCE = SoftOutputSource(
    method="sampled_confidence",
    cluster_origin="synthetic",
    growth_schedule="bernoulli_per_window",
    gap_units="branch_marker",
    correction="none",
    weight_step_natural_log=None,
    references=("controlled Bernoulli experimental input",),
)

if TYPE_CHECKING:
    from ..protocols import Decoder


def _check_probability(value, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or not 0 <= normalized <= 1:
        raise ValueError(f"{field_name} must be finite and in [0, 1]")
    return normalized


class CodeRouter:
    """Route each job by code name, with a default decoder fallback."""

    def __init__(self, default, by_code: Optional[dict] = None):
        self.default = default
        self.by_code = dict(by_code) if by_code else {}
        for decoder in (self.default, *self.by_code.values()):
            if decoder is not None:
                _decoder_fault_model_requirement(decoder)

    def run_seed_children(self):
        """Expose routed decoders under stable semantic paths."""
        children = [
            RunSeedChild(
                (RunSeedPathSegment("field", "default"),),
                self.default,
            ),
        ]
        for key, decoder in self.by_code.items():
            key_segment = (
                RunSeedPathSegment("none_key", None)
                if key is None
                else RunSeedPathSegment("string_key", key)
            )
            children.append(
                RunSeedChild(
                    (
                        RunSeedPathSegment("field", "by_code"),
                        key_segment,
                    ),
                    decoder,
                )
            )
        return tuple(children)

    def route(self, job: DecodeJob):
        """Pick the decoder for this job (by code; default when unmapped)."""
        return self.by_code.get(job.code, self.default)

    def fault_model_requirement_for(
        self,
        code: Optional[str],
    ) -> DecoderFaultModelRequirement:
        """Return only the requirement of the decoder selected for ``code``."""
        decoder = self.by_code.get(code, self.default)
        if decoder is None:
            return NO_FAULT_MODEL_REQUIRED
        return _decoder_fault_model_requirement(decoder)


class FunctionLatencyDecoder:
    """Timing-only decoder whose service time comes from a caller-supplied
    ``job -> microseconds`` function (any factors — n_rounds, spatial_nodes,
    code, attempt — are on the job). One-off models belong next to the
    experiment that uses them; the named classes below are the established
    parameterizations of this one."""

    fault_model_requirement = NO_FAULT_MODEL_REQUIRED

    def __init__(self, latency_us_for):
        self.latency_us_for = latency_us_for   # job -> microseconds

    def run_seed_children(self):
        """Expose the callback that controls simulated service time."""
        return (
            RunSeedChild(
                (RunSeedPathSegment("field", "latency_us_for"),),
                self.latency_us_for,
            ),
        )

    def latency(self, job: DecodeJob) -> int:
        """Service time in ticks, priced by the caller's function."""
        return us(self.latency_us_for(job))

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Return an empty timing-only result."""
        return DecodeResult(job.op_id, job.window_id)


class PresetLatencyDecoder:
    """Timing-only decoder with one fixed latency for every job
    (independent of window size)."""

    fault_model_requirement = NO_FAULT_MODEL_REQUIRED

    def __init__(self, latency_us: float = 1.0):
        self.latency_us = latency_us

    def latency(self, job: DecodeJob) -> int:
        return us(self.latency_us)

    def decode(self, job: DecodeJob) -> DecodeResult:
        return DecodeResult(job.op_id, job.window_id)


class PerRoundDecoder:
    """Timing-only decoder with linear cost per syndrome round:
    n_rounds * tau_us."""

    fault_model_requirement = NO_FAULT_MODEL_REQUIRED

    def __init__(self, tau_us: float = 1.0):
        self.tau_us = tau_us

    def latency(self, job: DecodeJob) -> int:
        return us(job.n_rounds * self.tau_us)

    def decode(self, job: DecodeJob) -> DecodeResult:
        return DecodeResult(job.op_id, job.window_id)


class PipelinedDecoder:
    """A pipelined decoder unit model: the DEC-003 pair of numbers.

    Results return after the inner model's latency; a new job may START
    on the same unit every ``initiation_interval_us``, with at most
    ``pipeline_depth`` decodes in flight at once (default: the fully
    occupied pipeline, ceil(latency / interval), computed per job). The
    non-pipelined models above keep their exact meaning: occupancy equal
    to latency is what "no initiation interval" means.

    Serves plain window and external decodes; the strong tier, gap
    siblings, and merged batches refuse a pipelined route until they get
    their own design pass."""

    def __init__(self, inner, initiation_interval_us: float,
                 pipeline_depth: Optional[int] = None):
        if not math.isfinite(initiation_interval_us) \
                or initiation_interval_us <= 0:
            raise ValueError("initiation_interval_us must be positive and finite")
        if us(initiation_interval_us) == 0:
            raise ValueError("initiation_interval_us is positive but rounds to zero ticks")
        if pipeline_depth is not None and pipeline_depth < 1:
            raise ValueError("pipeline_depth must be at least 1")
        if getattr(inner, "run", None) is not None:
            raise ValueError("PipelinedDecoder wraps an algorithm timing model, "
                             "not a staged DecoderEngine")
        self.inner = inner
        self.initiation_interval_us = initiation_interval_us
        self.pipeline_depth = pipeline_depth

    @property
    def fault_model_requirement(self):
        return getattr(self.inner, "fault_model_requirement",
                       NO_FAULT_MODEL_REQUIRED)

    def run_seed_children(self):
        """Expose the wrapped timing model to the seed graph."""
        return (RunSeedChild((RunSeedPathSegment("field", "inner"),),
                             self.inner),)

    def latency(self, job: DecodeJob) -> int:
        return self.inner.latency(job)

    def initiation_interval(self, job: DecodeJob) -> int:
        """Minimum ticks between consecutive starts on one unit."""
        return us(self.initiation_interval_us)

    def decode(self, job: DecodeJob) -> DecodeResult:
        return self.inner.decode(job)


class SwitchingRouter:
    """Route strong side jobs to the strong decoder and all other jobs to weak.

    An optional ``gap`` engine serves split-pair sibling jobs (hint
    "gap"): the second forced-class solve on its own decoder unit. Its
    presence is also the manager's signal that split-pair joins are on.
    """

    def __init__(self, weak: "Decoder", strong: "Decoder",
                 gap: "Decoder" = None):
        self.weak = weak
        self.strong = strong
        self.gap = gap
        _fault_model_requirement_for(weak, None)
        _fault_model_requirement_for(strong, None)
        if gap is not None:
            _fault_model_requirement_for(gap, None)

    def run_seed_children(self):
        """Expose every routed decoder tier."""
        children = [
            RunSeedChild(
                (RunSeedPathSegment("field", "weak"),),
                self.weak,
            ),
            RunSeedChild(
                (RunSeedPathSegment("field", "strong"),),
                self.strong,
            ),
        ]
        if self.gap is not None:
            children.append(RunSeedChild(
                (RunSeedPathSegment("field", "gap"),),
                self.gap,
            ))
        return tuple(children)

    def route(self, job: DecodeJob):
        """Strong decoder for escalated jobs, weak decoder for everything else."""
        if job.hint == "gap" and self.gap is not None:
            return self.gap
        return self.strong if job.hint == "strong" else self.weak

    def fault_model_requirement_for(
        self,
        code: Optional[str],
    ) -> DecoderFaultModelRequirement:
        """Join the weak and strong views that may own this code's window."""
        return _fault_model_requirement_for(
            self.weak,
            code,
        ).joined(_fault_model_requirement_for(self.strong, code))


def _decoder_fault_model_requirement(
    decoder,
) -> DecoderFaultModelRequirement:
    """Return one decoder's explicitly declared fault-model requirement."""
    return decoder.fault_model_requirement


def _fault_model_requirement_for(
    decoder_or_router,
    code: Optional[str],
) -> DecoderFaultModelRequirement:
    """Resolve either a leaf declaration or a code-aware routed declaration."""
    resolver = getattr(decoder_or_router, "fault_model_requirement_for", None)
    if resolver is not None:
        return resolver(code)
    return _decoder_fault_model_requirement(decoder_or_router)


class SampledConfidenceDecoder(_RandomSeedConsumer):
    """Pretends the decoder inside it is unsure about a random fraction of
    windows.

    Timing-only decoders produce no syndrome data, so there is nothing to
    compute a real confidence from; this wrapper asserts one as an
    experimental input. After the inner (weak) decode, a seeded coin sets a
    typed branch-marker confidence: gap 0.0 with
    ``escalation_probability`` (low confidence — the Switching policy
    escalates the window), else gap 1.0 (keep the weak result).
    ``probability_for`` replaces the flat rate
    with a per-job function (see switch_probability_per_round). Latency
    passes through to the inner decoder unchanged. Swap in a real
    soft-output decoder and the downstream pipeline behaves identically."""

    def __init__(self, inner: "Decoder", escalation_probability: float,
                 seed: Optional[int] = None,
                 probability_for=None):
        self.inner = inner
        self.escalation_probability = _check_probability(
            escalation_probability, "escalation_probability")
        self.probability_for = probability_for
        self.fault_model_requirement = _decoder_fault_model_requirement(inner)
        self._initialize_run_seed_state(seed)

    def run_seed_children(self):
        """Expose the inner decoder and optional probability callback."""
        children = [
            RunSeedChild(
                (RunSeedPathSegment("field", "inner"),),
                self.inner,
            ),
        ]
        if self.probability_for is not None:
            children.append(
                RunSeedChild(
                    (RunSeedPathSegment("field", "probability_for"),),
                    self.probability_for,
                )
            )
        return tuple(children)

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
        escalation_probability = _check_probability(
            escalation_probability, "probability_for result")
        self._mark_stochastic_use()
        confidence_gap = (
            0.0
            if self._rng.random() < escalation_probability
            else 1.0
        )
        result.soft_output = SoftOutput(
            gap=confidence_gap,
            source=SAMPLED_CONFIDENCE_SOURCE,
        )
        return result


def switch_probability_per_round(gamma_switch: float, d: int):
    """Per-window escalation probability that scales with window size.

    ``gamma_switch`` is the escalation rate per d rounds; a window
    committing more rounds is proportionally more likely to escalate."""

    gamma_switch = _check_probability(gamma_switch, "gamma_switch")
    if d <= 0:
        raise ValueError(f"d must be positive; got {d!r}")

    def probability(job: DecodeJob) -> float:
        window = job.window
        commit_rounds = (window.commit_hi - window.commit_lo + 1) \
            if window is not None else job.n_rounds
        if commit_rounds <= 0:
            raise ValueError(
                f"commit_rounds must be positive; got {commit_rounds!r}"
            )
        return _check_probability(
            gamma_switch * commit_rounds / d,
            "switch probability",
        )
    return probability
