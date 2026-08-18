"""Five-stage decoder cycle cards and cycle-to-tick arithmetic.

This is a fresh decsim design, not copied external code. The fixed stage shape
comes from the IF/ID/EX/MEM/WB structure in Ripes ``rv5s.h:45,351-359``. The
sum-then-convert-once arithmetic follows XQsim ``xqsim.py:216-229``. Each
cycle budget and the decoder frequency retain a serialized source.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

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
    """Return one semantically integral cycle count as an exact integer."""
    try:
        normalized = int(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field_name} must be a finite whole number") from error
    if normalized != value:
        raise ValueError(f"{field_name} must be a finite whole number")
    return normalized


@dataclass(frozen=True)
class StageCycleConfig:
    """One stage's cycle budget, quantity basis, and source."""

    cycles: int
    basis: CycleQuantityBasis
    source: str

    def __post_init__(self) -> None:
        cycles = _whole_cycles(self.cycles, "cycles")
        if cycles < 0:
            raise ValueError("cycles must be nonnegative")

    def to_json_value(self) -> dict:
        return {
            "cycles": self.cycles,
            "basis": self.basis.value,
            "source": self.source,
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

    def __post_init__(self) -> None:
        if not math.isfinite(self.frequency_mhz) or self.frequency_mhz <= 0.0:
            raise ValueError("frequency_mhz must be finite and positive")
        if us(1.0 / self.frequency_mhz) == 0:
            raise ValueError(
                f"frequency_mhz={self.frequency_mhz!r} is too high: one cycle "
                "must convert to at least one tick"
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
        """Return the sum of all five charged stage budgets."""
        return sum(cycles for _stage, cycles in self.stage_cycles(job))

    def latency_ticks(self, job: Any, inner_latency_ticks: int) -> int:
        """Convert the aggregate cycle budget once and optionally add inner time."""
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
        return {
            "model_name": self.model_name,
            "stages": stages,
            "frequency_mhz": self.frequency_mhz,
            "frequency_source": self.frequency_source,
            "include_inner_latency": self.include_inner_latency,
        }
