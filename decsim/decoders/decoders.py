"""Timing-only decoders, the routers and the sampled-confidence wrapper.

Every decoder here implements the Decoder port (ports.py): latency(job)
prices one window job's compute as a service time in ticks (the manager
dispatches the job to a free unit and schedules completion that many
ticks later), and decode(job) produces the DecodeResult. Timing-only
decoders return empty results; data-path decoders also compute
corrections.
"""

import math
from typing import Optional

import decsim.config as config
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.message as message
import decsim.ports as ports
import decsim.seeding as seeding

SAMPLED_CONFIDENCE_SOURCE = message.SoftOutputSource(
    method="sampled_confidence",
    cluster_origin="synthetic",
    growth_schedule="bernoulli_per_window",
    gap_units="branch_marker",
    correction="none",
    weight_step_natural_log=None,
    references=("controlled Bernoulli experimental input",),
)


class CodeRouter:
    """Route each job by code name, with a default decoder fallback."""

    def __init__(self, default, by_code: Optional[dict] = None):
        self.default = default
        self.by_code = {}
        if by_code:
            self.by_code = dict(by_code)

    def run_seed_children(self) -> tuple:
        """The routed decoders under stable semantic paths."""
        default_path = (message.RunSeedPathSegment("field", "default"),)
        children = [message.RunSeedChild(default_path, self.default)]
        for key, decoder in self.by_code.items():
            path = _by_code_path(key)
            child = message.RunSeedChild(path, decoder)
            children.append(child)
        return tuple(children)

    def route(self, job: message.DecodeJob):
        """The decoder for this job: by code, the default when unmapped."""
        return self.by_code.get(job.code, self.default)

    def fault_model_requirement_for(
        self, code: Optional[str]
    ) -> fault_models.DecoderFaultModelRequirement:
        """Only the requirement of the decoder selected for ``code``."""
        decoder = self.by_code.get(code, self.default)
        if decoder is None:
            return fault_models.NO_FAULT_MODEL_REQUIRED
        return decoder.fault_model_requirement


class SwitchingRouter:
    """Route strong side jobs to the strong decoder and every other to weak.

    An optional ``gap`` engine serves split-pair sibling jobs (hint
    "gap"): the second forced-class solve on its own decoder unit. Its
    presence is also the manager's signal that split-pair joins are on.
    """

    def __init__(
        self,
        weak: ports.Decoder,
        strong: ports.Decoder,
        gap: Optional[ports.Decoder] = None,
    ):
        self.weak = weak
        self.strong = strong
        self.gap = gap

    def run_seed_children(self) -> tuple:
        """Every routed decoder tier under its own path."""
        weak_path = (message.RunSeedPathSegment("field", "weak"),)
        strong_path = (message.RunSeedPathSegment("field", "strong"),)
        children = [
            message.RunSeedChild(weak_path, self.weak),
            message.RunSeedChild(strong_path, self.strong),
        ]
        if self.gap is not None:
            gap_path = (message.RunSeedPathSegment("field", "gap"),)
            gap_child = message.RunSeedChild(gap_path, self.gap)
            children.append(gap_child)
        return tuple(children)

    def route(self, job: message.DecodeJob):
        """Strong decoder for escalated jobs, weak for everything else."""
        if job.hint == "gap" and self.gap is not None:
            return self.gap
        if job.hint == "strong":
            return self.strong
        return self.weak

    def fault_model_requirement_for(
        self, code: Optional[str]
    ) -> fault_models.DecoderFaultModelRequirement:
        """Join the weak and strong views that may own this code's window."""
        weak_requirement = _fault_model_requirement_for(self.weak, code)
        strong_requirement = _fault_model_requirement_for(self.strong, code)
        return weak_requirement.joined(strong_requirement)


class FunctionLatencyDecoder:
    """Timing-only decoder priced by a caller-supplied function.

    The function maps a job to microseconds; any factor (n_rounds,
    spatial_nodes, code, attempt) is on the job. One-off models belong
    next to the experiment that uses them; the named classes below are
    the established parameterizations of this one.
    """

    fault_model_requirement = fault_models.NO_FAULT_MODEL_REQUIRED

    def __init__(self, latency_us_for):
        self.latency_us_for = latency_us_for  # job -> microseconds

    def run_seed_children(self) -> tuple:
        """The callback that controls simulated service time."""
        path = (message.RunSeedPathSegment("field", "latency_us_for"),)
        child = message.RunSeedChild(path, self.latency_us_for)
        return (child,)

    def latency(self, job: message.DecodeJob) -> int:
        """Service time in ticks, priced by the caller's function."""
        microseconds = self.latency_us_for(job)
        return config.microseconds_to_ticks(microseconds)

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """An empty timing-only result."""
        return message.DecodeResult(job.op_id, job.window_id)


class PresetLatencyDecoder:
    """Timing-only decoder with one fixed latency for every job."""

    fault_model_requirement = fault_models.NO_FAULT_MODEL_REQUIRED

    def __init__(self, latency_us: float = 1.0):
        self.latency_us = latency_us

    def latency(self, job: message.DecodeJob) -> int:
        """The preset latency in ticks, whatever the job."""
        del job
        return config.microseconds_to_ticks(self.latency_us)

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """An empty timing-only result."""
        return message.DecodeResult(job.op_id, job.window_id)


class PerRoundDecoder:
    """Timing-only decoder with a linear cost per syndrome round."""

    fault_model_requirement = fault_models.NO_FAULT_MODEL_REQUIRED

    def __init__(self, tau_us: float = 1.0):
        self.tau_us = tau_us

    def latency(self, job: message.DecodeJob) -> int:
        """n_rounds times tau, in ticks."""
        microseconds = job.n_rounds * self.tau_us
        return config.microseconds_to_ticks(microseconds)

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """An empty timing-only result."""
        return message.DecodeResult(job.op_id, job.window_id)


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
    their own design pass.
    """

    def __init__(
        self,
        inner,
        initiation_interval_us: float,
        pipeline_depth: Optional[int] = None,
    ):
        if (
            not math.isfinite(initiation_interval_us)
            or initiation_interval_us <= 0
        ):
            raise ValueError(
                "initiation_interval_us must be positive and finite"
            )
        interval_ticks = config.microseconds_to_ticks(initiation_interval_us)
        if interval_ticks == 0:
            raise ValueError(
                "initiation_interval_us is positive but rounds to zero ticks"
            )
        if pipeline_depth is not None and pipeline_depth < 1:
            raise ValueError("pipeline_depth must be at least 1")
        run = getattr(inner, "run", None)
        if run is not None:
            raise ValueError(
                "PipelinedDecoder wraps an algorithm timing model, "
                "not a staged DecoderEngine"
            )
        self.inner = inner
        self.initiation_interval_us = initiation_interval_us
        self.pipeline_depth = pipeline_depth

    @property
    def fault_model_requirement(self):
        """The wrapped model's requirement; none when it declares none."""
        return getattr(
            self.inner,
            "fault_model_requirement",
            fault_models.NO_FAULT_MODEL_REQUIRED,
        )

    def run_seed_children(self) -> tuple:
        """The wrapped timing model under the segment inner."""
        path = (message.RunSeedPathSegment("field", "inner"),)
        child = message.RunSeedChild(path, self.inner)
        return (child,)

    def latency(self, job: message.DecodeJob) -> int:
        """The wrapped model's latency."""
        return self.inner.latency(job)

    def initiation_interval(self, job: message.DecodeJob) -> int:
        """Minimum ticks between consecutive starts on one unit."""
        del job
        return config.microseconds_to_ticks(self.initiation_interval_us)

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """The wrapped model's result."""
        return self.inner.decode(job)


class SampledConfidenceDecoder(seeding._RandomSeedConsumer):
    """Pretends the decoder inside it is unsure about a fraction of windows.

    Timing-only decoders produce no syndrome data, so there is nothing to
    compute a real confidence from; this wrapper asserts one as an
    experimental input. After the inner (weak) decode, a seeded coin sets
    a typed branch-marker confidence: gap 0.0 with
    ``escalation_probability`` (low confidence, the Switching policy
    escalates the window), else gap 1.0 (keep the weak result).
    ``probability_for`` replaces the flat rate with a per-job function
    (see switch_probability_per_round). Latency passes through to the
    inner decoder unchanged. Swap in a real soft-output decoder and the
    downstream pipeline behaves identically.
    """

    def __init__(
        self,
        inner: ports.Decoder,
        escalation_probability: float,
        seed: Optional[int] = None,
        probability_for=None,
    ):
        self.inner = inner
        self.escalation_probability = _check_probability(
            escalation_probability, "escalation_probability"
        )
        self.probability_for = probability_for
        self.fault_model_requirement = inner.fault_model_requirement
        self._initialize_run_seed_state(seed)

    def run_seed_children(self) -> tuple:
        """The inner decoder and the optional probability callback."""
        inner_path = (message.RunSeedPathSegment("field", "inner"),)
        children = [message.RunSeedChild(inner_path, self.inner)]
        if self.probability_for is not None:
            path = (message.RunSeedPathSegment("field", "probability_for"),)
            child = message.RunSeedChild(path, self.probability_for)
            children.append(child)
        return tuple(children)

    def latency(self, job: message.DecodeJob) -> int:
        """The weak decode latency; the strong path is a separate job."""
        return self.inner.latency(job)

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """Weak-decode the window, then attach the sampled soft output."""
        result = self.inner.decode(job)
        escalation_probability = self.escalation_probability
        if self.probability_for is not None:
            escalation_probability = self.probability_for(job)
        escalation_probability = _check_probability(
            escalation_probability, "probability_for result"
        )
        self._mark_stochastic_use()
        draw = self._rng.random()
        confidence_gap = 1.0
        if draw < escalation_probability:
            confidence_gap = 0.0
        result.soft_output = message.SoftOutput(
            gap=confidence_gap, source=SAMPLED_CONFIDENCE_SOURCE
        )
        return result


def switch_probability_per_round(gamma_switch: float, d: int):
    """Per-window escalation probability that scales with window size.

    ``gamma_switch`` is the escalation rate per d rounds; a window
    committing more rounds is proportionally more likely to escalate.
    """
    gamma_switch = _check_probability(gamma_switch, "gamma_switch")
    if d <= 0:
        raise ValueError(f"d must be positive; got {d!r}")

    def probability(job: message.DecodeJob) -> float:
        window = job.window
        commit_rounds = job.n_rounds
        if window is not None:
            commit_rounds = window.commit_hi - window.commit_lo + 1
        if commit_rounds <= 0:
            raise ValueError(
                f"commit_rounds must be positive; got {commit_rounds!r}"
            )
        scaled = gamma_switch * commit_rounds / d
        return _check_probability(scaled, "switch probability")

    return probability


def _check_probability(value, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or not 0 <= normalized <= 1:
        raise ValueError(f"{field_name} must be finite and in [0, 1]")
    return normalized


def _by_code_path(key) -> tuple:
    """The seed path of one by-code route: the field, then its key."""
    if key is None:
        key_segment = message.RunSeedPathSegment("none_key", None)
    else:
        key_segment = message.RunSeedPathSegment("string_key", key)
    return (message.RunSeedPathSegment("field", "by_code"), key_segment)


def _fault_model_requirement_for(
    decoder_or_router, code: Optional[str]
) -> fault_models.DecoderFaultModelRequirement:
    """A leaf's declaration, or a code-aware router's."""
    resolver = getattr(decoder_or_router, "fault_model_requirement_for", None)
    if resolver is not None:
        return resolver(code)
    return decoder_or_router.fault_model_requirement
