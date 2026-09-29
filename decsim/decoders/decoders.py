"""Timing-only decoders and the sampled-confidence wrapper.

Every decoder here is a row on the Decoder port (ports.py, the defaults
in decoder.py): latency(job) prices one window job's compute as a
service time in ticks, and decode(job) produces the DecodeResult, empty
for a timing-only row. A pipelined unit is the staged decoder's timing
(staged_decoder.py, UnitTiming), not a decoder.
"""

import math
from typing import Optional

import decsim.config as config
import decsim.decoders.decoder as decoder_module
import decsim.records.decoding as decoding_records
import decsim.records.seeds as seed_records
import decsim.seeding as seeding

SAMPLED_CONFIDENCE_SOURCE = decoding_records.SoftOutputSource(
    method="sampled_confidence",
    cluster_origin="synthetic",
    growth_schedule="bernoulli_per_window",
    gap_units="branch_marker",
    correction="none",
    weight_step_natural_log=None,
    references=("controlled Bernoulli experimental input",),
)


class FunctionLatencyDecoder(decoder_module.DecoderBase):
    """Timing-only decoder priced by a caller-supplied function.

    The function maps a job to microseconds; any factor (round_count,
    spatial_nodes, code, attempt) is on the job. One-off models belong
    next to the experiment that uses them; the named classes below are
    the established parameterizations of this one.
    """

    def __init__(self, latency_us_for):
        self.latency_us_for = latency_us_for  # job -> microseconds

    def run_seed_children(self) -> tuple:
        """The callback that controls simulated service time."""
        path = (seed_records.RunSeedPathSegment("field", "latency_us_for"),)
        child = seed_records.RunSeedChild(path, self.latency_us_for)
        return (child,)

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """Service time in ticks, priced by the caller's function."""
        microseconds = self.latency_us_for(job)
        return config.microseconds_to_ticks(microseconds)

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """An empty timing-only result."""
        return decoding_records.DecodeResult(job.operation_id, job.window_id)


class PresetLatencyDecoder(decoder_module.DecoderBase):
    """Timing-only decoder with one fixed latency for every job."""

    def __init__(self, latency_us: float = 1.0):
        self.latency_us = latency_us

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """The preset latency in ticks, whatever the job."""
        del job
        return config.microseconds_to_ticks(self.latency_us)

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """An empty timing-only result."""
        return decoding_records.DecodeResult(job.operation_id, job.window_id)


class PerRoundDecoder(decoder_module.DecoderBase):
    """Timing-only decoder with a linear cost per syndrome round."""

    def __init__(self, tau_us: float = 1.0):
        self.tau_us = tau_us

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """round_count times tau, in ticks."""
        microseconds = job.round_count * self.tau_us
        return config.microseconds_to_ticks(microseconds)

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """An empty timing-only result."""
        return decoding_records.DecodeResult(job.operation_id, job.window_id)


class SampledConfidenceDecoder(
    seeding._RandomSeedConsumer, decoder_module.DecoderBase
):
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
        inner: decoder_module.DecoderBase,
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
        inner_path = (seed_records.RunSeedPathSegment("field", "inner"),)
        children = [seed_records.RunSeedChild(inner_path, self.inner)]
        if self.probability_for is not None:
            path = (
                seed_records.RunSeedPathSegment("field", "probability_for"),
            )
            child = seed_records.RunSeedChild(path, self.probability_for)
            children.append(child)
        return tuple(children)

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """The weak decode latency; the strong path is a separate job."""
        return self.inner.latency(job)

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
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
        result.soft_output = decoding_records.SoftOutput(
            gap=confidence_gap, source=SAMPLED_CONFIDENCE_SOURCE
        )
        return result


def switch_probability_per_round(gamma_switch: float, d: int):
    """Per-window escalation probability that scales with window size.

    ``gamma_switch`` is the escalation rate per d rounds; a window
    committing more rounds is proportionally more likely to escalate.
    No shipped config names it; it stays for a study that samples
    switching at Toshio's rate (2510.25222 eq. (6), gamma_switch per d
    rounds), passed as SampledConfidenceDecoder's probability_for.
    """
    gamma_switch = _check_probability(gamma_switch, "gamma_switch")
    if d <= 0:
        raise ValueError(f"d must be positive; got {d!r}")

    def probability(job: decoding_records.DecodeJob) -> float:
        window = job.window
        commit_rounds = job.round_count
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
