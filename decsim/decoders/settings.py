"""The settings of the decoder tiers, their manager and the escalation.

A decoder tier is one unit pool: its algorithm, its unit count, its
input memory and the cycle-priced stages around the algorithm (Toshio
arXiv 2510.25222: lightweight decoders decode constantly, a separate
accurate decoder is invoked on demand). The escalation says whether and
when a window is decoded again by the strong tier.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any, Optional, Union

import decsim.config as config
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.belief_propagation_osd.decoder as belief_propagation_osd
import decsim.decoders.decoder_memory as decoder_memory_module
import decsim.decoders.tesseract.decoder as tesseract
import decsim.decoders.union_find.decoder as union_find
import decsim.ports as ports
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)
from decsim.decoders.relay_belief_propagation import (
    decoder as relay_belief_propagation,
)

# weak_decoder.kind and strong_decoder.kind name one of these rows. A
# named row decodes every window for real and is charged its measured
# wall clock; a number instead of a name is a fixed core latency in
# microseconds on the MWPM path (decsim/build/decoders.py). Every row is
# one class on the Decoder port (decsim/decoders/decoder.py); sinter's
# BUILT_IN_DECODERS is the shape.
DECODERS = {
    "pymatching": minimum_weight_perfect_matching.PyMatchingDecoder,
    "unweighted_pymatching": (
        minimum_weight_perfect_matching.UnweightedPyMatchingDecoder
    ),
    "belief_matching": belief_matching.BeliefMatchingDecoder,
    "union_find": union_find.UnionFindDecoder,
    "tesseract": tesseract.TesseractDecoder,
    "relay_bp": relay_belief_propagation.RelayBeliefPropagationDecoder,
    "bposd": belief_propagation_osd.BeliefPropagationOsdDecoder,
}

DECODER_MANAGER_KEYS = ("bulk_strong", "clock", "dispatch_cycles")

# weak_decoder.input and strong_decoder.input name one of these rows:
# how a tier's unit gets the rounds it decodes. copy moves them over the
# tier's input link into the unit's own memory, which is what a hardware
# decoder does when its input is off-chip (Collision Clustering's Init
# unit loads the syndrome into the storage elements, 2309.05558 lines
# 268-271, and the strong hop is a transfer of the assigned data,
# Toshio 2510.25222 lines 1248-1250). in_place leaves the rounds in the
# store and reads them where they are, which is what a decoder with its
# input on-chip does: AFS's processing elements "can directly access the
# data stored on-chip" (2001.06598 lines 528-531). The default is copy,
# which is what every hop of this tree does today.
DECODER_INPUTS = {"copy": True, "in_place": False}

# weak_decoder.boundary_fold names one of these rows: how the window's
# boundary mask is folded into the input the decode reads. copy
# duplicates the landed input and XORs the mask into the duplicate, so
# the unit's stored rounds stay raw, which is what a software decoder
# does (cuda-q QEC keeps the raw rounds and rebuilds the window syndrome
# each time, sliding_window.cpp:283-292). in_place XORs the mask into
# the unit's own memory, which is what a hardware decoder does: AFS's
# processing elements write on-chip memory directly (2001.06598 lines
# 528-531) and Helios keeps its shared memory in registers with a single
# writer (2301.08419 lines 632-640). One input read by two jobs has no
# single writer, so in_place is refused there by name.
DECODER_BOUNDARY_FOLDS = {"copy": True, "in_place": False}

# <tier>_decoder.engine.detection_event_latency_cycles, the fixed
# latency of the tier's event-detection stage: Yang et al. 2605.04892
# lines 1273-1275, "All variables are stored in FPGA registers, enabling
# fully pipelined operation. The total latency of the preprocessing
# stage for syndrome calculation is fixed at 20 ns (5 FPGA clock
# cycles)", counted inside the decoder's own subtotal (Table I, lines
# 1049-1052).
DETECTION_EVENT_LATENCY_CYCLES = 5

# <tier>_decoder.engine.detection_event_cycles_per_round, the pipelined
# stage's rate: fully pipelined (Yang 2605.04892 line 1273) means the
# stage takes a new round every clock, which is the rate the fetch stage
# already reads at (LILLIPUT 2108.06569 line 593, the FIFO of the last m
# rounds).
DETECTION_EVENT_CYCLES_PER_ROUND = 1


@dataclasses.dataclass(frozen=True)
class DecoderSettings:
    """The yaml's `weak_decoder` and `strong_decoder` sections.

    Table rows (DECODERS, above): pymatching, belief_matching (the two
    tiers of the decoder-switching setting, decoded per window and
    charged their measured wall clock), or a number, a fixed core
    latency in microseconds on the MWPM path. The card prices the
    algorithm stage only; the fetch and release stages are cycles of the
    engine's clock, resolved to a frequency once at load.
    unit_memory_rounds is the input SRAM per unit (None is unbounded); a
    unit overlaps input transfer with compute only when two windows fit.
    input names a row of DECODER_INPUTS (above): whether this tier's
    unit is given a copy of the rounds it reads or reads them where the
    store keeps them. boundary_fold names a row of
    DECODER_BOUNDARY_FOLDS: whether the window's boundary mask is XORed
    into a duplicate of the landed input or into the unit's own memory.
    result_blocks_unit says when a unit's compute goes back to its pool:
    false at the decode's end, which is Chen's frame manager taking the
    correction without blocking the decoder (2605.30765 lines
    1618-1620), or true at the window's commit, which is Riverlane's
    polled status register, the decoder holding its output until the
    reader takes it (2410.05202 lines 1256-1259). It is read on the tier
    that decodes the plan's windows.
    detection_event_latency_cycles and detection_event_cycles_per_round
    price this tier's own event-detection logic, charged only when the
    rounds reach it raw (controller.detection_events_formed_at decoder).
    The stage is pipelined, so a job forming n rounds pays the fixed
    latency once and the rate for every round after the first: Yang et
    al. 2605.04892 lines 1273-1275 store "All variables ... in FPGA
    registers, enabling fully pipelined operation" and fix "The total
    latency of the preprocessing stage for syndrome calculation ... at
    20 ns (5 FPGA clock cycles)", which is the latency default; the rate
    default is one round a clock. A null latency is uncharged and the
    rate is then read by nobody.
    kind None is no decoder at all, right for a run that plans no
    windows. A Python-built decoder is routed as it is, with no engine
    stages around it.
    """

    kind: Union[str, float, None] = None
    units: int = 1
    input: str = "copy"
    boundary_fold: str = "copy"
    result_blocks_unit: bool = False
    unit_memory_rounds: Optional[int] = None
    fetch_cycles_per_round: int = 1
    release_cycles_per_job: int = 1
    detection_event_latency_cycles: Optional[int] = (
        DETECTION_EVENT_LATENCY_CYCLES
    )
    detection_event_cycles_per_round: int = DETECTION_EVENT_CYCLES_PER_ROUND
    engine_megahertz: Optional[float] = None
    decoder: Optional[ports.Decoder] = None

    @classmethod
    def from_yaml(
        cls,
        section: Mapping,
        clocks: config.ClockSettings,
        section_name: str,
    ) -> "DecoderSettings":
        """A tier section: kind, units, unit memory and the engine card."""
        engine = section["engine"]
        engine_megahertz = clocks.megahertz(engine["clock"])
        unit_memory_rounds = section["unit_memory_rounds"]
        if unit_memory_rounds is not None and unit_memory_rounds < 1:
            raise ValueError(
                "unit_memory_rounds must be at least one round, or null "
                f"for an unbounded unit memory (got {unit_memory_rounds})"
            )
        input_kind = section.get("input", "copy")
        boundary_fold = section.get("boundary_fold", "copy")
        formation_latency = _formation_latency_cycles(engine)
        formation_rate = _formation_cycles_per_round(engine)
        result_blocks_unit = section.get("result_blocks_unit", False)
        _check_boolean(section_name, "result_blocks_unit", result_blocks_unit)
        return cls(
            kind=section["kind"],
            units=section["units"],
            input=input_kind,
            boundary_fold=boundary_fold,
            result_blocks_unit=result_blocks_unit,
            unit_memory_rounds=unit_memory_rounds,
            fetch_cycles_per_round=engine["fetch_cycles_per_round"],
            release_cycles_per_job=engine["release_cycles_per_job"],
            detection_event_latency_cycles=formation_latency,
            detection_event_cycles_per_round=formation_rate,
            engine_megahertz=engine_megahertz,
        )


@dataclasses.dataclass(frozen=True)
class DecoderManagerSettings:
    """The yaml's `decoder_manager` section, and the manager's Python knobs.

    bulk_strong serves the strong pool's queued re-decodes as one merged
    batch (Toshio 2510.25222 Sec. III C, the strong decoder processes its
    assigned data in bulk); timing-only, since a batch carries no
    accuracy-bearing result, and serial only. dispatch_cycles prices the
    manager's own work per dispatch on the clock the section names,
    charged before the job's input is asked for; Caune et al. 2410.05202
    lines 519-526 and 636-641 measure 250 to 370 control cycles per
    decode on the control system's own clock. It is zero by default, so a
    run that does not model that work is unchanged. The rest are Python
    objects. A router picks the decoder for each job
    (CodeRouter by code name, SwitchingRouter by tier); given, it
    replaces the one the root builds from the two tier sections. The
    scheduler orders the ready queue (FifoScheduler by default),
    unit_pools names each pool's unit count (built from the tiers' units
    by default) and decoder_memory bounds each pool's input memory in
    rounds (built from the active tier's unit_memory_rounds by default).
    """

    router: Optional[Any] = None
    scheduler: Optional[Any] = None
    unit_pools: Optional[Mapping[str, int]] = None
    decoder_memory: Optional[decoder_memory_module.DecoderMemoryConfig] = None
    bulk_strong: bool = False
    dispatch_microseconds: float = 0.0

    @classmethod
    def from_yaml(
        cls, section: Mapping, clocks: config.ClockSettings
    ) -> "DecoderManagerSettings":
        """The `decoder_manager` section: its knobs, cycles resolved once."""
        unknown = set(section) - set(DECODER_MANAGER_KEYS)
        if unknown:
            listed = sorted(unknown)
            raise ValueError(
                f"decoder_manager does not know {listed}; its keys are "
                f"{list(DECODER_MANAGER_KEYS)}"
            )
        bulk_strong = section.get("bulk_strong", False)
        if bulk_strong not in (True, False):
            raise ValueError(
                "decoder_manager.bulk_strong must be true or false, got "
                f"{bulk_strong!r}"
            )
        dispatch_microseconds = _dispatch_microseconds(section, clocks)
        return cls(
            bulk_strong=bulk_strong,
            dispatch_microseconds=dispatch_microseconds,
        )

    def dispatch_ticks(self) -> int:
        """The manager's per-dispatch cost in ticks."""
        return config.microseconds_to_ticks(self.dispatch_microseconds)


def _dispatch_microseconds(
    section: Mapping, clocks: config.ClockSettings
) -> float:
    """dispatch_cycles on the section's clock; zero when it names none."""
    cycles = section.get("dispatch_cycles", 0)
    if cycles == 0:
        return 0.0
    if "clock" not in section:
        raise ValueError(
            "decoder_manager.dispatch_cycles needs a clock: name the "
            "domain its cycles are counted in"
        )
    clock = section["clock"]
    return clocks.microseconds(cycles, clock)


def _formation_latency_cycles(engine: Mapping) -> Optional[int]:
    """The fixed latency of the tier's event-detection stage; null is off."""
    if "detection_event_latency_cycles" not in engine:
        return DETECTION_EVENT_LATENCY_CYCLES
    return engine["detection_event_latency_cycles"]


def _formation_cycles_per_round(engine: Mapping) -> int:
    """The rate that stage accepts rounds at, one a clock by default."""
    if "detection_event_cycles_per_round" not in engine:
        return DETECTION_EVENT_CYCLES_PER_ROUND
    return engine["detection_event_cycles_per_round"]


def _check_boolean(section_name: str, key: str, value) -> None:
    """A yaml key that is written true or false, checked where it enters."""
    if value is True or value is False:
        return
    raise ValueError(
        f"{section_name}.{key} {value!r} is not a boolean; write true or false"
    )
