"""Prebuilt decoder cycle cards and their numeric sources."""

from __future__ import annotations

from .decoder_cycle_model import (
    CycleQuantityBasis,
    DecoderCycleModel,
    StageCycleConfig,
)


def delegated_latency_model() -> DecoderCycleModel:
    """Return the zero-cycle card that exactly delegates existing latency."""
    def delegated_zero(stage_name: str) -> StageCycleConfig:
        return StageCycleConfig(
            cycles=0,
            basis=CycleQuantityBasis.PER_JOB,
            source=(
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
            source=end_to_end_zero,
        ),
        decode=StageCycleConfig(
            0,
            CycleQuantityBasis.PER_JOB,
            source=end_to_end_zero,
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
            source=(
                "tmp/references/papers/2108.06569v1.txt:828-829 reports one "
                "LUT access but not its cycle cost; :805 identifies embedded "
                "memory, so no external-memory term from :834-838 applies, "
                "and the end-to-end total already includes MEMORY work."
            ),
        ),
        writeback=StageCycleConfig(
            0,
            CycleQuantityBasis.PER_JOB,
            source=end_to_end_zero,
        ),
        frequency_mhz=250.0,
        frequency_source=(
            "tmp/references/papers/2108.06569v1.txt:848 reports 250 MHz for "
            "the [d=3,m=2] embedded-memory FPGA configuration."
        ),
        include_inner_latency=False,
        model_name="lilliput_d3_m2_embedded_memory",
    )
