"""Pure five-stage decoder cycle cards and cycle-to-tick arithmetic.

This is a fresh decsim design, not copied external code. The fixed stage shape
comes from the IF/ID/EX/MEM/WB structure in Ripes ``rv5s.h:45,351-359``. The
sum-then-convert-once arithmetic follows XQsim ``xqsim.py:216-229``. Numeric
card provenance is carried by every configured value and serialized for reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Optional

from .config import us


class DecoderPipelineStage(str, Enum):
    """The fixed five-stage decoder budget vocabulary, in pipeline order."""

    FETCH = "fetch"
    DECODE = "decode"
    EXECUTE = "execute"
    MEMORY = "memory"
    WRITEBACK = "writeback"


class CycleQuantityBasis(str, Enum):
    """Whether configured cycles are charged once or once per input round."""

    PER_JOB = "per_job"
    PER_ROUND = "per_round"


def _whole_cycles(value, field_name: str) -> int:
    """Normalize one semantically integral cycle count to an exact integer."""
    try:
        normalized = int(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field_name} must be a finite whole number") from error
    if normalized != value:
        raise ValueError(f"{field_name} must be a finite whole number")
    return normalized


def _provenance_line(value: str, field_name: str) -> str:
    """Return a stripped, nonempty provenance line."""
    stripped = value.strip()
    if not stripped or "\n" in value or "\r" in value:
        raise ValueError(f"{field_name} must be a stripped nonempty single line")
    return stripped


def _optional_provenance_line(
    value: Optional[str],
    field_name: str,
) -> Optional[str]:
    if value is None:
        return None
    return _provenance_line(value, field_name)


@dataclass(frozen=True)
class StageCycleConfig:
    """One stage's exact cycle budget and number provenance."""

    cycles: int
    basis: CycleQuantityBasis
    source: Optional[str] = None
    zero_justification: Optional[str] = None

    def __post_init__(self) -> None:
        cycles = _whole_cycles(self.cycles, "cycles")
        if cycles < 0:
            raise ValueError("cycles must be nonnegative")
        if (
            self.basis is not CycleQuantityBasis.PER_JOB
            and self.basis is not CycleQuantityBasis.PER_ROUND
        ):
            raise ValueError(f"unknown cycle quantity basis {self.basis!r}")

        source = _optional_provenance_line(self.source, "source")
        zero_justification = _optional_provenance_line(
            self.zero_justification,
            "zero_justification",
        )
        if cycles == 0:
            if source is not None or zero_justification is None:
                raise ValueError(
                    "zero cycles require only a zero_justification"
                )
        elif source is None or zero_justification is not None:
            raise ValueError("positive cycles require only a source")

        object.__setattr__(self, "cycles", cycles)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "zero_justification",
            zero_justification,
        )

    def to_json_value(self) -> dict:
        return {
            "cycles": self.cycles,
            "basis": self.basis.value,
            "source": self.source,
            "zero_justification": self.zero_justification,
        }


@dataclass(frozen=True)
class DecoderCycleModel:
    """A frozen five-stage service-time card with sourced clock conversion."""

    fetch: StageCycleConfig
    decode: StageCycleConfig
    execute: StageCycleConfig
    memory: StageCycleConfig
    writeback: StageCycleConfig
    frequency_mhz: float
    frequency_source: str
    include_inner_latency: bool
    model_name: str
    inner_latency_source: Optional[str] = None
    inner_latency_zero_justification: Optional[str] = None

    def __post_init__(self) -> None:
        frequency_mhz = float(self.frequency_mhz)
        if not math.isfinite(frequency_mhz) or frequency_mhz <= 0.0:
            raise ValueError("frequency_mhz must be finite and positive")
        if us(1.0 / frequency_mhz) == 0:
            raise ValueError(
                f"frequency_mhz={frequency_mhz!r} is too high: one cycle "
                "must convert to at least one tick"
            )
        frequency_source = _provenance_line(
            self.frequency_source,
            "frequency_source",
        )
        inner_latency_source = _optional_provenance_line(
            self.inner_latency_source,
            "inner_latency_source",
        )
        inner_latency_zero_justification = _optional_provenance_line(
            self.inner_latency_zero_justification,
            "inner_latency_zero_justification",
        )
        if self.include_inner_latency is True:
            if (
                inner_latency_source is None
                or inner_latency_zero_justification is not None
            ):
                raise ValueError(
                    "included inner latency requires only an "
                    "inner_latency_source"
                )
        elif self.include_inner_latency is False:
            if (
                inner_latency_source is not None
                or inner_latency_zero_justification is None
            ):
                raise ValueError(
                    "excluded inner latency requires only an "
                    "inner_latency_zero_justification"
                )
        else:
            raise ValueError("include_inner_latency must be an exact bool")

        object.__setattr__(self, "frequency_mhz", frequency_mhz)
        object.__setattr__(self, "frequency_source", frequency_source)
        object.__setattr__(
            self,
            "inner_latency_source",
            inner_latency_source,
        )
        object.__setattr__(
            self,
            "inner_latency_zero_justification",
            inner_latency_zero_justification,
        )

    def _stage_configs(self) -> tuple:
        return (
            (DecoderPipelineStage.FETCH, self.fetch),
            (DecoderPipelineStage.DECODE, self.decode),
            (DecoderPipelineStage.EXECUTE, self.execute),
            (DecoderPipelineStage.MEMORY, self.memory),
            (DecoderPipelineStage.WRITEBACK, self.writeback),
        )

    def stage_cycles(self, job: Any) -> tuple:
        """Return immutable ``(stage, charged_cycles)`` rows in fixed order."""
        rows = []
        for stage, stage_config in self._stage_configs():
            if stage_config.basis is CycleQuantityBasis.PER_JOB:
                charged_cycles = stage_config.cycles
            else:
                round_count = _whole_cycles(job.n_rounds, "job.n_rounds")
                if round_count < 0:
                    raise ValueError("job.n_rounds must be nonnegative")
                charged_cycles = stage_config.cycles * round_count
            rows.append((stage, charged_cycles))
        return tuple(rows)

    def total_cycles(self, job: Any) -> int:
        """Return the exact integer sum of all five charged stage budgets."""
        return sum(cycles for _stage, cycles in self.stage_cycles(job))

    def latency_ticks(self, job: Any, inner_latency_ticks: int) -> int:
        """Convert the aggregate cycle budget once and optionally add inner time."""
        if not self.include_inner_latency and inner_latency_ticks != 0:
            raise ValueError(
                "inner_latency_ticks must be zero when inner latency is excluded"
            )
        total_cycles = self.total_cycles(job)
        cycle_latency_ticks = us(total_cycles / self.frequency_mhz)
        if total_cycles > 0 and cycle_latency_ticks == 0:
            raise RuntimeError("positive decoder work converted to zero ticks")
        if self.include_inner_latency:
            return inner_latency_ticks + cycle_latency_ticks
        return cycle_latency_ticks

    def to_json_value(self) -> dict:
        stages = []
        for stage, stage_config in self._stage_configs():
            stage_value = {"stage": stage.value}
            stage_value.update(stage_config.to_json_value())
            stages.append(stage_value)
        value = {
            "model_name": self.model_name,
            "stages": stages,
            "frequency_mhz": self.frequency_mhz,
            "frequency_source": self.frequency_source,
            "include_inner_latency": self.include_inner_latency,
        }
        if self.include_inner_latency:
            value["inner_latency_source"] = self.inner_latency_source
        else:
            value["inner_latency_zero_justification"] = (
                self.inner_latency_zero_justification
            )
        return value


def delegated_latency_model() -> DecoderCycleModel:
    """Return the zero-cycle card that exactly delegates existing latency."""
    def delegated_zero(stage_name: str) -> StageCycleConfig:
        return StageCycleConfig(
            cycles=0,
            basis=CycleQuantityBasis.PER_JOB,
            zero_justification=(
                f"The default card assigns no {stage_name} cycles; the wrapped "
                "decoder's existing latency model prices the whole decode job."
            ),
        )

    return DecoderCycleModel(
        fetch=delegated_zero("FETCH"),
        decode=delegated_zero("DECODE"),
        execute=delegated_zero("EXECUTE"),
        memory=delegated_zero("MEMORY"),
        writeback=delegated_zero("WRITEBACK"),
        frequency_mhz=250.0,
        frequency_source=(
            "tmp/references/papers/2108.06569v1.txt:848 reports 250 MHz for "
            "the [d=3,m=2] FPGA; this default card multiplies zero cycles, so "
            "the frequency cannot affect latency."
        ),
        include_inner_latency=True,
        inner_latency_source=(
            "This card prices no cycles itself and delegates the whole decode "
            "price to the wrapped decoder's existing latency model."
        ),
        model_name="delegated_latency",
    )


def lookup_table_published_total_model() -> DecoderCycleModel:
    """Return LILLIPUT's distance-3, two-round, embedded-memory FPGA card.

    Using this card for another distance, round count, or memory configuration
    is a modelling error. The seven published cycles are declared in EXECUTE as
    one aggregate total, not as a per-stage decomposition.
    """
    end_to_end_zero = (
        "tmp/references/papers/2108.06569v1.txt:830-831 reports an end-to-end "
        "seven-cycle total from syndrome arrival that already includes this "
        "stage's work."
    )
    return DecoderCycleModel(
        fetch=StageCycleConfig(
            0,
            CycleQuantityBasis.PER_JOB,
            zero_justification=end_to_end_zero,
        ),
        decode=StageCycleConfig(
            0,
            CycleQuantityBasis.PER_JOB,
            zero_justification=end_to_end_zero,
        ),
        execute=StageCycleConfig(
            7,
            CycleQuantityBasis.PER_JOB,
            source=(
                "tmp/references/papers/2108.06569v1.txt:830-831 publishes a "
                "7-cycle end-to-end total per decode step and does not "
                "decompose it; this card declares EXECUTE as the aggregate "
                "stage rather than apportioning the total."
            ),
        ),
        memory=StageCycleConfig(
            0,
            CycleQuantityBasis.PER_JOB,
            zero_justification=(
                "tmp/references/papers/2108.06569v1.txt:828-829 reports one "
                "LUT access but not its cycle cost; :805 identifies embedded "
                "memory, so no external-memory term from :834-838 applies, "
                "and the end-to-end total already includes MEMORY work."
            ),
        ),
        writeback=StageCycleConfig(
            0,
            CycleQuantityBasis.PER_JOB,
            zero_justification=end_to_end_zero,
        ),
        frequency_mhz=250.0,
        frequency_source=(
            "tmp/references/papers/2108.06569v1.txt:848 reports 250 MHz for "
            "the [d=3,m=2] embedded-memory FPGA configuration."
        ),
        include_inner_latency=False,
        inner_latency_zero_justification=(
            "tmp/references/papers/2108.06569v1.txt:830-831 publishes seven "
            "cycles for the whole decode step, so adding the wrapped decoder's "
            "latency would double-count it."
        ),
        model_name="lilliput_d3_m2_embedded_memory",
    )
