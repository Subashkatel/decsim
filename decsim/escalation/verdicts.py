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
"""

import dataclasses

import decsim.ports as ports
import decsim.records.decoding as decoding_records

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


class Always:
    """Escalate every window: the target re-decodes them all."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The always row, which takes no setting."""

        name = "always"
        reads_threshold = False

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
