"""The settings of the decoder tiers, their manager and the escalation.

A decoder tier is one unit pool: its algorithm, its unit count, its
input memory and the cycle-priced stages around the algorithm (Toshio
arXiv 2510.25222: lightweight decoders decode constantly, a separate
accurate decoder is invoked on demand). The escalation says whether and
when a window is decoded again by the strong tier.
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.config as config
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.belief_propagation_osd.decoder as belief_propagation_osd
import decsim.decoders.dispatch_steps.decoder as dispatch_steps
import decsim.decoders.measured_table.decoder as measured_table
import decsim.decoders.schedulers as schedulers
import decsim.decoders.tesseract.decoder as tesseract
import decsim.decoders.union_find.cycle_count as cycle_count
import decsim.decoders.union_find.decoder as union_find
import decsim.ports as ports
import decsim.tables as tables
from decsim.decoders.minimum_weight_perfect_matching import (
    decoder as minimum_weight_perfect_matching,
)
from decsim.decoders.relay_belief_propagation import (
    decoder as relay_belief_propagation,
)

# weak_decoder.kind and strong_decoder.kind name one of these rows. A
# named row decodes every window for real and is charged its measured
# wall clock, except measured_table and dispatch_steps, which are
# charged a GPU's measured time for decsim's own Relay-BP decode, whole
# or step by step (measured_table/, dispatch_steps/); a
# number instead of a name is a fixed core latency in microseconds on
# the MWPM path (PyMatchingDecoder.Settings). Every row is one class on
# the Decoder port (decsim/decoders/decoder.py) whose Settings record
# builds it; sinter's BUILT_IN_DECODERS is the shape.
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
    "measured_table": measured_table.MeasuredTableDecoder,
    "dispatch_steps": dispatch_steps.DispatchStepsDecoder,
}

# The keys every row of a <tier>_decoder section shares; any other key
# is the row's own (its Settings, decsim/tables.py row_settings).
DECODER_KEYS = (
    "kind",
    "units",
    "input",
    "boundary_fold",
    "result_blocks_unit",
    "unit_memory",
    "engine",
)

# The keys a <tier>_decoder section cannot leave out: the row, the unit
# count, the unit's memory and the card its stages are priced on.
_REQUIRED_DECODER_KEYS = ("kind", "units", "unit_memory", "engine")

DECODER_MANAGER_KEYS = ("bulk_strong", "clock", "dispatch_cycles")

# <tier>_decoder.unit_memory's keys.
UNIT_MEMORY_KEYS = ("bits", "word_bits")

# <tier>_decoder.engine's four stage keys, each with what it prices, so
# a card that leaves one out is refused by name and by what is missing.
ENGINE_CYCLE_KEYS = {
    "fetch_cycles_per_round": "the fetch stage's cost for each round",
    "fetch_cycles_per_job": "the fetch stage's cost once a job",
    "release_cycles_per_job": "the release stage's cost once a job",
    "release_cycles_per_round": "the release stage's cost for each round",
}

# Every key a <tier>_decoder.engine card may hold; any other is refused.
_ENGINE_KEYS = (
    "clock",
    *ENGINE_CYCLE_KEYS,
)

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
# each time, sliding_window.cpp:287-293). in_place XORs the mask into
# the unit's own memory, which is what a hardware decoder does: AFS's
# processing elements write on-chip memory directly (2001.06598 lines
# 528-531) and Helios keeps its shared memory in registers with a single
# writer (2301.08419 lines 632-640). One input read by two jobs has no
# single writer, so in_place is refused there by name.
DECODER_BOUNDARY_FOLDS = {"copy": True, "in_place": False}


@dataclasses.dataclass(frozen=True)
class UnitMemorySettings:
    """The yaml's `<tier>_decoder.unit_memory` section.

    The input memory of one decoder unit: a window's rounds are copied
    into it before that unit decodes them, and freed when the decode
    ends. bits is one unit's capacity, in bits; null models no capacity
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
    and 152-167). null keeps the fetch at its per-round cycles alone. A
    tier that reads in place has no memory of its own, so it takes none.
    """

    bits: Optional[int] = None
    word_bits: Optional[int] = None

    def __post_init__(self) -> None:
        config.check_capacity_bits("unit_memory.bits", self.bits)
        _check_word_bits("unit_memory.word_bits", self.word_bits)

    @classmethod
    def from_yaml(
        cls, section: Mapping, section_name: str
    ) -> "UnitMemorySettings":
        """The unit_memory block's two keys."""
        block_name = f"{section_name}.unit_memory"
        tables.refuse_unknown_keys(block_name, section, UNIT_MEMORY_KEYS)
        values = {
            "bits": section.get("bits"),
            "word_bits": section.get("word_bits"),
        }
        return tables.section_record(section_name, cls, values)


@dataclasses.dataclass(frozen=True)
class EngineSettings:
    """The yaml's `<tier>_decoder.engine` card.

    clock is the domain the stages count on, resolved to its Clock once
    at load; the four stage keys price the fetch and release stages
    once a job and once a round, in cycles of that clock.
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


_CLOCK_250_MEGAHERTZ = config.Clock.from_megahertz(250.0)
# The shipped baselines' engine: one round a cycle in and ten cycles to
# write the correction out. No source states either count.
ESTIMATED_ENGINE = EngineSettings(
    clock=_CLOCK_250_MEGAHERTZ,  # 2108.06569 Table 4
    fetch_cycles_per_round=1,  # estimate
    fetch_cycles_per_job=0,
    release_cycles_per_job=10,  # estimate
    release_cycles_per_round=0,
)
# Helios inside RISC-Q's root node: Helios's controller takes a header
# byte and then a loading cycle a round (control_node_single_FPGA.v at
# 2dda998), the aggregator having assembled the whole frame (Liu et al.
# 2603.16203 lines 668-670), and the root's error distributor sends the
# correction on in one message.
RISC_Q_HELIOS_ENGINE = EngineSettings(
    clock=cycle_count.HELIOS.clock,
    fetch_cycles_per_round=1,  # control_node_single_FPGA.v:170-182
    fetch_cycles_per_job=1,  # header byte, control_node_single_FPGA.v:137
    release_cycles_per_job=3,  # 25 ns, 2603.16203 line 902, rounded up
    release_cycles_per_round=0,  # one message a job, 2603.16203 line 902
)
# A decoder priced by a measured time that already holds the syndrome's
# copies to and from the device (measured_table/measurements.py) pays no
# stage of its own, so its clock times nothing.
MEASURED_TIME_ENGINE = EngineSettings(
    clock=_CLOCK_250_MEGAHERTZ,  # estimate
    fetch_cycles_per_round=0,
    fetch_cycles_per_job=0,
    release_cycles_per_job=0,
    release_cycles_per_round=0,
)


@dataclasses.dataclass(frozen=True)
class DecoderPoolSettings:
    """One tier's pool of decoder units: the yaml's `<tier>_decoder` section.

    Table rows (DECODERS, above): pymatching, belief_matching (the two
    tiers of the decoder-switching setting, decoded per window and
    charged their measured wall clock), or a number, a fixed core
    latency in microseconds on the MWPM path. The card prices the
    algorithm stage only; the fetch and release stages are cycles of the
    engine's clock, resolved to that domain's Clock once at load, each
    stage priced once a job and once a round, because a unit's load and
    write-out carry a header of their own beside their per-round bytes
    (Helios 2301.08419v2's controller takes one header byte and then a
    round's bytes and a loading cycle, and streams three header bytes
    and a round's correction bytes back).
    unit_count is the count of identical decoding engines inside this
    tier's one chip, gem5's FUDesc.count ("number of these FU's available",
    gem5 src/cpu/FuncUnit.py): every unit has its own input memory and
    all of them share the tier's links. Hardware holds several engines
    on a chip too: AFS "uses L/N decoder blocks to perform error
    correction over the L logical qubits" (2001.06598 lines 1049-1052),
    and Yang et al. instantiate an X-type and a Z-type decoder in one
    FPGA (2605.04892 lines 986-988). Each paper fixes its count; sweeping
    it is this simulator's own use of the key. The chip count is one and
    is not a key.
    unit_memory is the input SRAM of one unit, sized in bits
    (UnitMemorySettings, above); a unit overlaps input transfer with
    compute only when two windows fit.
    copies_input says whether this tier's unit is given a copy of the
    rounds it reads, or reads them where the store keeps them
    (DECODER_INPUTS, above, names the two). copies_boundary_fold says
    whether the window's boundary mask is XORed into a duplicate of the
    landed input, or into the unit's own memory, which needs that copy
    (DECODER_BOUNDARY_FOLDS).
    result_blocks_unit says when a unit's compute goes back to its pool:
    false at the decode's end, or when the confidence walk charged on
    the unit ends, which is Chen's frame manager taking the
    correction without blocking the decoder (2605.30765 lines
    1618-1620), or true when the result is read, at the window's commit
    or, for a forced-class solve, when the confidence join holds it,
    which is a unit with no output buffer, stalled by back-pressure
    until its output is taken; Bascones et al. 2605.01035 lines 607-609
    size FIFOs after their decoder's tiles "to avoid stalling the U, V
    tiles outputs", and true is that decoder without them. No referent
    found measures a whole decoder unit held this way. It is read on
    the tier that decodes the plan's windows.
    algorithm is the row's own Settings record, whose build makes the
    unit's decoder (UnionFindDecoder.Settings and the rest); the section
    fills it from its keys outside DECODER_KEYS (union_find's weight_step
    and cycle_count). A run with no decoder on a tier leaves the
    machine's slot None. Every decoder runs between the engine's stages.
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
        _check_fold_has_a_memory(self)
        _check_word_has_a_memory(self)

    @classmethod
    def from_yaml(
        cls,
        section: Mapping,
        clocks: config.ClockSettings,
        section_name: str,
    ) -> Optional["DecoderPoolSettings"]:
        """A tier section: kind, units, unit memory and the engine card.

        kind null is no decoder, so the section fills no slot.
        """
        tables.refuse_missing_keys(
            section_name, section, _REQUIRED_DECODER_KEYS
        )
        algorithm = _algorithm(section, clocks, section_name)
        engine_section = _block(section, section_name, "engine")
        engine = _engine_card(engine_section, clocks, section_name)
        memory_section = _block(section, section_name, "unit_memory")
        unit_memory = UnitMemorySettings.from_yaml(memory_section, section_name)
        copies_input = _copy_row(section, section_name, "input", DECODER_INPUTS)
        copies_boundary_fold = _copy_row(
            section, section_name, "boundary_fold", DECODER_BOUNDARY_FOLDS
        )
        result_blocks_unit = section.get("result_blocks_unit", False)
        if algorithm is None:
            return None
        return cls(
            algorithm=algorithm,
            unit_count=section["units"],
            engine=engine,
            unit_memory=unit_memory,
            copies_input=copies_input,
            copies_boundary_fold=copies_boundary_fold,
            result_blocks_unit=result_blocks_unit,
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
    objects. scheduler is the class of the rule that orders a ready queue
    (FifoScheduler by default); each manager builds its own, since the
    chip's and the host's queues are separate hardware (LATTE 2509.03954
    lines 20-25 and 718-722), as gem5 gives every object its own copy of
    a SimObject parameter (src/python/m5/SimObject.py:775-782).
    """

    scheduler: type = schedulers.FifoScheduler
    bulk_strong: bool = False
    dispatch_cycles: int = 0
    clock: Optional[config.Clock] = None

    def __post_init__(self) -> None:
        config.check_cycles(
            "decoder_manager.dispatch_cycles", self.dispatch_cycles
        )

    @classmethod
    def from_yaml(
        cls, section: Mapping, clocks: config.ClockSettings
    ) -> "DecoderManagerSettings":
        """The `decoder_manager` section: its knobs, cycles resolved once."""
        tables.refuse_unknown_keys(
            "decoder_manager", section, DECODER_MANAGER_KEYS
        )
        bulk_strong = section.get("bulk_strong", False)
        _check_boolean("decoder_manager", "bulk_strong", bulk_strong)
        dispatch_cycles = section.get("dispatch_cycles", 0)
        clock = _dispatch_clock(section, clocks, dispatch_cycles)
        return cls(
            bulk_strong=bulk_strong,
            dispatch_cycles=dispatch_cycles,
            clock=clock,
        )


def _dispatch_clock(
    section: Mapping, clocks: config.ClockSettings, dispatch_cycles: int
) -> Optional[config.Clock]:
    """The clock dispatch_cycles are counted on; None when none is named.

    A named clock is resolved whether or not a cycle is charged, so a
    domain the clocks section does not have is refused either way.
    """
    if "clock" in section:
        return clocks.clock(section["clock"])
    if dispatch_cycles == 0:
        return None
    raise ValueError(
        "decoder_manager.dispatch_cycles needs a clock: name the "
        "domain its cycles are counted in"
    )


def _block(section: Mapping, section_name: str, key: str) -> Mapping:
    """A nested block of the section, which is a mapping of its own keys.

    `unit_memory: 4096` reads as a capacity to a user and as no mapping
    to the reader, so it is refused with the form it takes.
    """
    block = section[key]
    if isinstance(block, Mapping):
        return block
    raise ValueError(
        f"{section_name}.{key} holds {block!r}; it is a mapping of its "
        "keys, as in configs/reference.yaml"
    )


def _engine_card(
    engine: Mapping, clocks: config.ClockSettings, section_name: str
) -> EngineSettings:
    """The engine card: its clock and every stage it prices."""
    card_name = f"{section_name}.engine"
    tables.refuse_unknown_keys(card_name, engine, _ENGINE_KEYS)
    cycles = _engine_stage_cycles(engine, section_name)
    clock = _engine_clock(engine, clocks, section_name)
    values = {"clock": clock, **cycles}
    return tables.section_record(section_name, EngineSettings, values)


def _engine_clock(
    engine: Mapping, clocks: config.ClockSettings, section_name: str
) -> config.Clock:
    """The domain the engine card's cycles count on; a card must name it."""
    if "clock" not in engine:
        raise ValueError(
            f"{section_name}.engine needs clock, the domain its stage "
            "cycles are counted in"
        )
    return clocks.clock(engine["clock"])


def _engine_stage_cycles(engine: Mapping, section_name: str) -> dict:
    """The engine card's four stage keys, by their own names."""
    cycles = {}
    for key in ENGINE_CYCLE_KEYS:
        cycles[key] = _engine_cycles(engine, section_name, key)
    return cycles


def _engine_cycles(engine: Mapping, section_name: str, key: str) -> int:
    """One stage key of the engine card; a card that omits it is refused."""
    if key not in engine:
        priced = ENGINE_CYCLE_KEYS[key]
        raise ValueError(
            f"{section_name}.engine needs {key}, {priced} in cycles of "
            "its clock"
        )
    return engine[key]


def _check_boolean(section_name: str, key: str, value) -> None:
    """A yaml key that is written true or false, checked where it enters."""
    if value is True or value is False:
        return
    raise ValueError(
        f"{section_name}.{key} {value!r} is not a boolean; write true or false"
    )


def _copy_row(
    section: Mapping, section_name: str, key: str, table: dict
) -> bool:
    """Whether a copy-or-in-place key copies; an absent key copies."""
    name = section.get(key, "copy")
    return tables.row(table, f"{section_name}.{key}", name)


def _check_fold_has_a_memory(pool: DecoderPoolSettings) -> None:
    """Folding into the unit's memory needs the unit to hold a copy.

    A tier that reads its input in place holds no rounds of its own, so
    there is no single-writer memory to XOR the mask into (Helios
    2301.08419 lines 632-640).
    """
    if pool.copies_input or pool.copies_boundary_fold:
        return
    raise ValueError(
        "a decoder pool that folds the boundary into the unit's memory "
        "(copies_boundary_fold False, boundary_fold in_place) needs the "
        "unit's own copy of the rounds, and one that reads its input in "
        "place (copies_input False) has none; fold into a copy, or copy "
        "the input"
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
    """A word is a whole number of bits, at least one; null has none."""
    if word_bits is None:
        return
    if config.is_whole_count(word_bits):
        return
    raise ValueError(
        f"{key} must be a whole number of bits, at least one, or null "
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
        "a decoder pool's unit_count (<tier>_decoder.units) must be a whole "
        f"number of engines, at least one (got {unit_count!r})"
    )


def _algorithm(
    section: Mapping, clocks: config.ClockSettings, section_name: str
):
    """The Settings record of the row the tier's kind names.

    A number is a fixed core latency on the MWPM path and takes no keys;
    a row reads the keys its Settings declares when the record reads
    yaml, and takes none otherwise; kind null is no decoder.
    """
    kind = section["kind"]
    if kind is None:
        tables.refuse_unknown_keys(section_name, section, DECODER_KEYS)
        return None
    if not isinstance(kind, str):
        latency = _latency_settings(kind, section_name)
        tables.refuse_unknown_keys(section_name, section, DECODER_KEYS)
        return latency
    row = tables.row(DECODERS, f"{section_name}.kind", kind)
    if not hasattr(row.Settings, "from_yaml"):
        tables.refuse_unknown_keys(section_name, section, DECODER_KEYS)
        return row.Settings()
    return tables.row_settings(
        row, section_name, section, DECODER_KEYS, clocks, section_name
    )


def _latency_settings(kind, section_name: str):
    """A kind that is no row's name is a core latency on the MWPM path."""
    matching = minimum_weight_perfect_matching.PyMatchingDecoder
    try:
        return matching.Settings(preset_latency_microseconds=kind)
    except ValueError:
        rows = sorted(DECODERS)
        raise ValueError(
            f"{section_name}.kind {kind!r} is neither a row nor a latency; "
            f"name one of {rows}, or write a finite nonnegative number of "
            "microseconds"
        ) from None


def toshio_decoder_pool(
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
            f"Toshio's per-round decode time of "
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
