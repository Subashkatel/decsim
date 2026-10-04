"""The settings of the decoder tiers, their manager and the escalation.

A decoder tier is one unit pool: its algorithm, its unit count, its
input memory and the cycle-priced stages around the algorithm (Toshio
arXiv 2510.25222: lightweight decoders decode constantly, a separate
accurate decoder is invoked on demand). The escalation says whether and
when a window is decoded again by the strong tier.
"""

import dataclasses
from typing import Optional, Protocol

import decsim.config as config
import decsim.decoders.schedulers as schedulers
import decsim.ports as ports
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)

# the engine's four stage costs, each a cycle count
ENGINE_CYCLE_KEYS = (
    "fetch_cycles_per_round",
    "fetch_cycles_per_job",
    "release_cycles_per_job",
    "release_cycles_per_round",
)


@dataclasses.dataclass(frozen=True)
class UnitMemorySettings:
    """The input memory of one decoder unit.

    A window's rounds are copied into it before that unit decodes them,
    and freed when the decode ends. bits is one unit's capacity, in bits;
    None models no capacity
    at all, an ideal memory nothing ever fills. A memory is its own
    object carrying its own size, the way gem5 declares a cache's
    capacity on the memory rather than on the compute beside it
    (`size = Param.MemorySize("Capacity")`, gem5
    src/mem/cache/Cache.py) and gem5-Aladdin's systolic array declares
    its private scratchpad
    (`size = Param.Int(32768, "Size of the scratchpad in bytes.")`,
    src/systolic_array/SystolicArray.py). The unit is bits because the
    rest of the tree counts bits (payload_bits, bits_per_cycle) and a
    syndrome round is not byte aligned.

    word_bits is what one read of that memory moves: the fetch stage
    reads each round in whole words, one word a cycle, beside its per-job
    and per-round cycles, as gem5's crossbar charges divCeil(size, width)
    per packet (src/mem/xbar.cc:135) and Helios loads a round a byte a
    clock (Helios_scalable_QEC control_node_single_FPGA.v lines 35-36
    and 152-167). None keeps the fetch at its per-round cycles alone. A
    tier that reads in place has no memory of its own, so it takes none.
    """

    bits: Optional[int] = None
    word_bits: Optional[int] = None

    def __post_init__(self) -> None:
        config.check_capacity_bits("unit_memory.bits", self.bits)
        _check_word_bits("unit_memory.word_bits", self.word_bits)


@dataclasses.dataclass(frozen=True)
class EngineSettings:
    """One tier's engine card: the stages around the algorithm.

    clock is the domain the stages count on, None for the machine's; the
    four stage costs price the fetch and release stages once a job and
    once a round, in cycles of that clock.
    """

    clock: Optional[config.Clock] = None
    fetch_cycles_per_round: int = 1
    fetch_cycles_per_job: int = 0
    release_cycles_per_job: int = 1
    release_cycles_per_round: int = 0

    def __post_init__(self) -> None:
        for key in ENGINE_CYCLE_KEYS:
            cycles = getattr(self, key)
            config.check_cycles(f"engine.{key}", cycles)


@dataclasses.dataclass(frozen=True)
class DecoderPoolSettings:
    """One tier's pool of decoder units.

    algorithm is a decoder row's Settings record (decsim/decoders/). The
    card prices the algorithm stage only; the fetch and release stages
    are cycles of the engine's clock, each stage priced once a job and
    once a round, because a unit's load and write-out carry a header of
    their own beside their per-round bytes (Helios 2301.08419v2's
    controller takes one header byte and then a round's bytes and a
    loading cycle, and streams three header bytes and a round's
    correction bytes back).
    unit_count is the count of identical decoding engines inside this
    tier's one chip, gem5's FUDesc.count ("number of these FU's available",
    gem5 src/cpu/FuncUnit.py): every unit has its own input memory and
    all of them share the tier's links. Hardware holds several engines
    on a chip too: AFS "uses L/N decoder blocks to perform error
    correction over the L logical qubits" (2001.06598 lines 1049-1052),
    and Yang et al. instantiate an X-type and a Z-type decoder in one
    FPGA (2605.04892 lines 986-988). Each paper fixes its count; sweeping
    it is this simulator's own use of the field. The chip count is one
    and is not a setting.
    unit_memory is the input SRAM of one unit, sized in bits
    (UnitMemorySettings, above); a unit overlaps input transfer with
    compute only when two windows fit.
    copies_input says whether this tier's unit is given a copy of the
    rounds it reads, or reads them where the store keeps them.
    copies_boundary_fold says whether the window's boundary mask is
    XORed into a duplicate of the landed input, or into the unit's own
    memory, which needs that copy.
    result_blocks_unit says when a unit's compute goes back to its pool:
    False at the decode's end, or when the confidence walk charged on
    the unit ends, which is Chen's frame manager taking the
    correction without blocking the decoder (2605.30765 lines
    1618-1620), or True when the result is read, at the window's commit
    or, for a forced-class solve, when the confidence join holds it,
    which is a unit with no output buffer, stalled by back-pressure
    until its output is taken; Bascones et al. 2605.01035 lines 607-609
    size FIFOs after their decoder's tiles "to avoid stalling the U, V
    tiles outputs", and True is that decoder without them. No referent
    found measures a whole decoder unit held this way. It is read on
    the tier that decodes the plan's windows.
    The algorithm's build makes the unit's decoder
    (UnionFindDecoder.Settings and the rest). A run with no decoder on a
    tier leaves the machine's slot None. Every decoder runs between the
    engine's stages.
    """

    algorithm: ports.DecoderSettings
    unit_count: int = 1
    engine: EngineSettings = EngineSettings()
    # built per pool, since its check is defined below this class
    unit_memory: UnitMemorySettings = dataclasses.field(
        default_factory=UnitMemorySettings
    )
    copies_input: bool = True
    copies_boundary_fold: bool = True
    result_blocks_unit: bool = False

    def __post_init__(self) -> None:
        _check_unit_count(self.unit_count)
        _check_flag("copies_input", self.copies_input)
        _check_flag("copies_boundary_fold", self.copies_boundary_fold)
        _check_flag("result_blocks_unit", self.result_blocks_unit)
        _check_word_has_a_memory(self)


class SchedulerSettings(Protocol):
    """A ready-queue rule's settings record (schedulers.py), which builds it."""

    def build(self) -> schedulers.Scheduler:
        """A fresh rule, one per manager."""


@dataclasses.dataclass(frozen=True)
class DecoderManagerSettings:
    """The decoder manager's knobs.

    bulk_strong serves the strong pool's queued re-decodes as one merged
    batch (Toshio 2510.25222 Sec. III C, the strong decoder processes its
    assigned data in bulk); timing-only, since a batch carries no
    accuracy-bearing result, and serial only. dispatch_cycles prices the
    manager's own work per dispatch on its clock,
    charged before the job's input is asked for; Caune et al. 2410.05202
    lines 519-526 and 636-641 measure 250 to 370 control cycles per
    decode on the control system's own clock. It is zero by default, so a
    run that does not model that work is unchanged. scheduler is the
    settings record of the rule that orders a ready queue
    (FifoScheduler's by default); each manager builds its own, since the
    chip's and the host's queues are separate hardware (LATTE 2509.03954
    lines 20-25 and 718-722), as gem5 gives every object its own copy of
    a SimObject parameter (src/python/m5/SimObject.py:775-782).
    """

    scheduler: SchedulerSettings = schedulers.FifoScheduler.Settings()
    bulk_strong: bool = False
    dispatch_cycles: int = 0
    clock: Optional[config.Clock] = None

    def __post_init__(self) -> None:
        config.check_cycles(
            "decoder_manager.dispatch_cycles", self.dispatch_cycles
        )


def _check_word_has_a_memory(pool: DecoderPoolSettings) -> None:
    """A word width prices reads of the unit's own memory, which in_place lacks.

    Under input in_place the unit reads the store's words, and the store
    prices that read (syndrome_buffer/round_output.py); a width here would
    price the same bits twice.
    """
    if pool.copies_input or pool.unit_memory.word_bits is None:
        return
    raise ValueError(
        "a decoder pool's unit_memory.word_bits prices reads of the unit's "
        "own memory, and a pool that reads its input in place "
        "(copies_input False) reads the rounds where the store keeps them, "
        "which the store prices; leave word_bits out, or copy the input"
    )


def _check_word_bits(key: str, word_bits) -> None:
    """A word is a whole number of bits, at least one; None has none."""
    if word_bits is None:
        return
    if config.is_whole_count(word_bits):
        return
    raise ValueError(
        f"{key} must be a whole number of bits, at least one, or None "
        f"(got {word_bits!r})"
    )


def _check_flag(name: str, value) -> None:
    """A pool's yes-or-no field is True or False, nothing that reads as one."""
    if value is True or value is False:
        return
    raise ValueError(
        f"a decoder pool's {name} {value!r} is not a flag; give True or False"
    )


def _check_unit_count(unit_count) -> None:
    """A pool's engine count is a whole number, at least one."""
    if config.is_whole_count(unit_count):
        return
    raise ValueError(
        "a decoder pool's unit_count must be a whole number of engines, "
        f"at least one (got {unit_count!r})"
    )


def linear_decoder_pool(
    decode_microseconds_per_round: float,
    clock: config.Clock,
    *,
    solves_per_window: int,
) -> DecoderPoolSettings:
    """One unit of Toshio et al.'s linear decoder: T_dec(r) = tau_dec r.

    decode_microseconds_per_round is Toshio's tau_dec (2510.25222 lines
    968-971, 1309-1311): one unit's whole time a round of a window, its
    soft output included (lines 602-604). A window solved
    solves_per_window times, two under the complementary gap, pays
    tau_dec / solves_per_window a round on each solve. The fetch stage
    carries it, as its cycles scale with the job's rounds; every other
    stage and the matching row cost nothing. T_comm (lines 1035-1036) is
    a link's: controller_to_weak_buffer for the weak decoder,
    weak_decoder_to_strong_decoder for the strong one. The paper's
    values are at lines 1109-1114 and 1125-1133.
    """
    config.check_whole_count("solves_per_window", solves_per_window, "solves")
    decode_ticks = config.microseconds_to_ticks(decode_microseconds_per_round)
    solve_share_ticks = clock.period_ticks * solves_per_window
    cycles, remainder_ticks = divmod(decode_ticks, solve_share_ticks)
    if remainder_ticks:
        raise ValueError(
            f"a per-round decode time of "
            f"{decode_microseconds_per_round} us over {solves_per_window} "
            f"solves is not a whole number of cycles of a "
            f"{clock.period_ticks}-tick clock"
        )
    engine = EngineSettings(
        clock=clock,
        fetch_cycles_per_round=cycles,
        fetch_cycles_per_job=0,
        release_cycles_per_job=0,
        release_cycles_per_round=0,
    )
    matching = minimum_weight_perfect_matching.PyMatchingDecoder
    algorithm = matching.Settings(preset_latency_microseconds=0.0)
    return DecoderPoolSettings(algorithm=algorithm, engine=engine)
