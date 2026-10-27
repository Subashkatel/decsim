"""The verdict rows: what an escalate edge decides on each source result.

An edge's switching settings name one (SwitchingSettings.verdict). A row
keeps a result, which goes home as the window's answer, or escalates
its window to the edge's target pool. A row's reads_threshold says
whether it decides on the edge's threshold, which an online threshold
learns from by the windows it escalates.

    gap_threshold  keep a gap at or above the threshold, the default
                   (Toshio et al. 2510.25222 Sec. III A)
    status         escalate a result its backend did not report as
                   succeeded: nonconverged, invalid or failed
    always         escalate every window
    never          keep every window
    bernoulli      escalate each window independently at a rate,
                   drawn from the run's seed (Toshio et al. Sec. III C)
    backlog_bernoulli
                   bernoulli at a rate the outer loop steps by the
                   source pool's backlog (target_steps.py)
    agreement      keep an ensemble's answer at or above a share of its
                   members agreeing; all of them by default (Shutty et
                   al. 2401.12434 Sec. V)
"""

import dataclasses
from typing import Optional

import decsim.escalation.target_steps as target_steps
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.seeding as seeding

KEEP = decoding_records.Verdict.KEEP
ESCALATE = decoding_records.Verdict.ESCALATE


class GapThreshold:
    """Keep a result whose confidence gap is at or above the threshold.

    A result with no soft output escalates: no confidence is not a
    confident answer.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The gap row, which reads the edge's confidence and threshold."""

        name = "gap_threshold"
        reads_threshold = True
        reads_consensus = False
        reads_source_queue = False

        def build(self) -> "GapThreshold":
            """A verdict on the gap against the threshold."""
            return GapThreshold()

    def decide(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        threshold: ports.ThresholdSource,
    ) -> decoding_records.Verdict:
        """Keep at a gap at or above the threshold; escalate otherwise."""
        if result.soft_output is None:
            return ESCALATE
        if threshold.decide_keep(job, result):
            return KEEP
        return ESCALATE


class Status:
    """Escalate a result its backend did not report as succeeded.

    A best-effort correction (nonconverged, low confidence, invalid) or
    none at all goes to the target; a succeeded one is kept, as BP+OSD
    runs OSD only when BP does not converge (Roffe et al. 2005.07016).
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The status row, which takes no setting."""

        name = "status"
        reads_threshold = False
        reads_consensus = False
        reads_source_queue = False

        def build(self) -> "Status":
            """A verdict on the backend's reported status."""
            return Status()

    def decide(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        threshold: ports.ThresholdSource,
    ) -> decoding_records.Verdict:
        """Keep a succeeded result; escalate any other."""
        del job, threshold
        if result.logical_observables is None:
            return ESCALATE
        status = result.decode_status
        if status is None:
            return KEEP
        if status is decoding_records.BackendDecodeStatus.SUCCEEDED:
            return KEEP
        return ESCALATE


class Agreement:
    """Keep an ensemble's answer when enough of its members agree.

    Harmony's layered decoding (Shutty et al. 2401.12434 Sec. V) runs
    the second pass "if and only if the first pass ensemble fails to
    yield a consensus prediction": minimum 1.0, the default, escalates
    on any dissent. A lower minimum keeps a majority short of
    unanimity. It reads the consensus row's soft output, and a result
    with none escalates.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """minimum is the share of members that must vote with the majority."""

        minimum: float = 1.0
        name = "agreement"
        reads_threshold = False
        reads_consensus = True
        reads_source_queue = False

        def __post_init__(self) -> None:
            if not 0.0 < self.minimum <= 1.0:
                raise ValueError(
                    "switching.verdict.minimum must be in (0, 1], got "
                    f"{self.minimum}"
                )

        def build(self) -> "Agreement":
            """A verdict on the ensemble's agreement."""
            return Agreement(self.minimum)

    def __init__(self, minimum: float) -> None:
        self.minimum = minimum

    def decide(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        threshold: ports.ThresholdSource,
    ) -> decoding_records.Verdict:
        """Keep at an agreement at or above the minimum; escalate otherwise."""
        del job, threshold
        soft_output = result.soft_output
        if soft_output is None:
            return ESCALATE
        if soft_output.gap >= self.minimum:
            return KEEP
        return ESCALATE


class Always:
    """Escalate every window: the target re-decodes them all."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The always row, which takes no setting."""

        name = "always"
        reads_threshold = False
        reads_consensus = False
        reads_source_queue = False

        def build(self) -> "Always":
            """A verdict that escalates every window."""
            return Always()

    def decide(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        threshold: ports.ThresholdSource,
    ) -> decoding_records.Verdict:
        """Escalate."""
        del job, result, threshold
        return ESCALATE


class Never:
    """Keep every window: the source's answer is always the final one."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The never row, which takes no setting."""

        name = "never"
        reads_threshold = False
        reads_consensus = False
        reads_source_queue = False

        def build(self) -> "Never":
            """A verdict that keeps every window."""
            return Never()

    def decide(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        threshold: ports.ThresholdSource,
    ) -> decoding_records.Verdict:
        """Keep."""
        del job, result, threshold
        return KEEP


class Bernoulli(seeding.GeneratorSeedConsumer):
    """Escalate each window independently with probability rate.

    Toshio et al. 2510.25222 Sec. III C model the backlog with switching
    "independent and identical between different small windows", at
    γ_switch per window. This row is that model: it ignores the result,
    so a capacity study sets the escalation rate directly. A run seeds
    it at the edge's policy path, so a seed repeats its draws. A change
    of conditions mid-stream gives later_rate from window
    from_window on, as a fixed threshold escalates more when p rises.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """rate is the probability that a window escalates."""

        rate: float
        later_rate: Optional[float] = None
        from_window: Optional[int] = None
        name = "bernoulli"
        reads_threshold = False
        reads_consensus = False
        reads_source_queue = False

        def __post_init__(self) -> None:
            _check_rate(self.rate, "rate")
            if (self.later_rate is None) != (self.from_window is None):
                raise ValueError(
                    "switching.verdict.later_rate and from_window go "
                    f"together, got {self.later_rate!r} and "
                    f"{self.from_window!r}"
                )
            if self.later_rate is not None:
                _check_rate(self.later_rate, "later_rate")

        def build(self) -> "Bernoulli":
            """A verdict that escalates at the rate."""
            return Bernoulli(self.rate, self.later_rate, self.from_window)

    def __init__(
        self,
        rate: float,
        later_rate: Optional[float] = None,
        from_window: Optional[int] = None,
    ) -> None:
        seeding.GeneratorSeedConsumer.__init__(self)
        self.rate = rate
        self.later_rate = later_rate
        self.from_window = from_window

    def decide(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        threshold: ports.ThresholdSource,
    ) -> decoding_records.Verdict:
        """Escalate with probability rate, later_rate from its window on."""
        del result, threshold
        rate = self.rate
        changed = self.from_window is not None
        if changed and job.window_id >= self.from_window:
            rate = self.later_rate
        generator = self.generator()
        draw = generator.random()
        if draw < rate:
            return ESCALATE
        return KEEP


class BacklogBernoulli(seeding.GeneratorSeedConsumer):
    """Bernoulli at the share the outer loop sets from the backlog.

    The timing-only form of that loop, as Toshio et al. 2510.25222 Sec.
    III C model switching: no gap is read, each window escalates at the
    loop's current share, and the loop (target_steps.TargetSteps) reads
    the windows waiting for the source pool before each decision.
    source_queue is the source pool's manager, bound by the build as the
    deadline's is.
    """

    source_queue = ports.Port(ports.DecodeQueue)

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The outer loop's settings."""

        steps: target_steps.TargetSteps.Settings
        name = "backlog_bernoulli"
        reads_threshold = False
        reads_consensus = False
        reads_source_queue = True

        def build(self) -> "BacklogBernoulli":
            """A verdict at the loop's start share."""
            steps = self.steps.build()
            return BacklogBernoulli(steps)

    def __init__(self, steps: target_steps.TargetSteps) -> None:
        seeding.GeneratorSeedConsumer.__init__(self)
        self.steps = steps

    def decide(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        threshold: ports.ThresholdSource,
    ) -> decoding_records.Verdict:
        """Read the backlog, then escalate at the loop's share."""
        del job, result, threshold
        waiting = self.source_queue.waiting_jobs()
        self.steps.observe(waiting)
        generator = self.generator()
        draw = generator.random()
        if draw < self.steps.target:
            return ESCALATE
        return KEEP


def _check_rate(rate, name: str) -> None:
    """A rate is a probability."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(
            f"switching.verdict.{name} must be in [0, 1], got {rate}"
        )
