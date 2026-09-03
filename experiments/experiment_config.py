"""One yaml file is one experiment; this module is the only yaml reader.

The loader turns the file into the small frozen cards below and checks
the names it knows, so every other module works with typed fields instead
of dict lookups. `extends: other.yaml` starts from that file (same folder)
and overrides the top-level keys this file names.
"""

import dataclasses
import math
from pathlib import Path
from typing import Optional, Union

import yaml

# A decoder unit's algorithm: a named real algorithm, decoded per window
# and charged its measured wall clock, or a number, a fixed core latency
# in microseconds (a hypothetical core). The names are the two tiers of
# the decoder-switching setting (Toshio arXiv 2510.25222: MWPM weak,
# belief-matching strong).
ALGORITHMS = ("pymatching", "belief_matching")
AlgorithmCard = Union[float, str]

DECODE_PATHS = ("weak_baseline", "strong_only", "switching")
# The decoder unit each decode_path decodes on. A decode_path's unit must
# be defined; the other tier's card may be omitted. Switching decodes
# every window on the weak tier first (its unit is the active one) and
# requires the strong tier too, for escalation.
DECODE_PATH_TIER = {
    "weak_baseline": "weak",
    "strong_only": "strong",
    "switching": "weak",
}
TRACE_MODES = ("off", "print", "file", "both")
SCHEMES = ("sliding", "parallel", "sandwich", "naive_online")
# How an idle patch's rounds are charged. Idle rounds are decoder workload
# in every reference system (SWIPER ISCA 2025, XQsim, Terhal backlog), so
# separate_decode_jobs is the default; ignore is the optimistic card for
# active-path latency studies; extend_stream folds them into a live stream.
IDLE_POLICIES = ("separate_decode_jobs", "ignore", "extend_stream")
LINK_PATHS = (
    "qpu_to_controller",
    "controller_to_weak_buffer",
    "controller_to_strong_buffer",
    "weak_buffer_to_weak_decoder",
    "weak_decoder_to_strong_decoder",
    "strong_buffer_to_strong_decoder",
    "decoder_to_decoder",
    "weak_decoder_to_frame",
    "strong_decoder_to_frame",
    "frame_to_controller",
    "controller_to_qpu",
)
THRESHOLD_SOURCES = ("fixed", "table", "online")
GAP_COMPUTATIONS = ("serial", "parallel_pair", "split_pair")
SWITCHING_KEYS = (
    "gap_threshold_db",
    "double_window",
    "gap_computation",
    "gap_units",
    "threshold_source",
    "threshold_table",
    "threshold_column",
    "online",
)
ONLINE_KEYS = (
    "target_escalation_rate",
    "step_db",
    "audit_rate",
    "kept_bad_budget",
    "adjust_factor",
    "min_escalation_rate",
    "max_escalation_rate",
)
SWEEP_KEYS = (
    "physical_error_probability",
    "distance",
    "round_period_us",
    "shots",
)
# Decibels are 10 log10 of the likelihood ratio; matching weights are its
# natural log: nats = decibels * ln(10) / 10.
LN_TEN = math.log(10.0)


@dataclasses.dataclass(frozen=True)
class LinkCard:
    """One path's numbers, in cycles of a named clock domain.

    Propagation latency, per-lane capacity (None = unbounded) times a lane
    count, and an optional fixed per-transfer DMA setup cost. The loader
    resolves the microsecond fields from the clocks card once, so
    everything downstream keeps reading microseconds while the yaml speaks
    cycles (XQsim's shape: domain labels with frequencies over one tick
    core).
    """

    latency_cycles: float
    clock: str
    bits_per_cycle: Optional[float]
    channels: int
    setup_cycles_per_transfer: Optional[float]
    latency_microseconds: float  # latency_cycles / clock megahertz
    bits_per_microsecond: Optional[float]  # bits_per_cycle x channels x MHz
    setup_microseconds_per_transfer: Optional[float]


@dataclasses.dataclass(frozen=True)
class EngineCard:
    """The decoder engine's fetch and release stage costs.

    In cycles of a named clock domain; the loader resolves the frequency
    once, like the links.
    """

    clock: str
    fetch_cycles_per_round: int
    release_cycles_per_job: int
    megahertz: float


@dataclasses.dataclass(frozen=True)
class DecoderUnitCard:
    """One decoder tier: its algorithm, its unit pool, and its unit memory.

    unit_memory_rounds is the input SRAM size per unit in rounds (None =
    unbounded). A unit overlaps input transfer with compute only when two
    windows fit in its SRAM.
    """

    algorithm: AlgorithmCard
    units: int
    unit_memory_rounds: Optional[int]
    engine: EngineCard


@dataclasses.dataclass(frozen=True)
class DecoderCard:
    """The two tiers of the decoder-switching architecture.

    Two distinct units (Toshio arXiv 2510.25222: lightweight decoders
    decode constantly, a separate accurate decoder is invoked on demand).
    A config defines the units its decode_path uses; the decode_path
    picks the active one.
    """

    weak: Optional[DecoderUnitCard]
    strong: Optional[DecoderUnitCard]


@dataclasses.dataclass(frozen=True)
class OnlineThresholdCard:
    """The online calibrator's knobs (threshold_source: online).

    Two loops around the live threshold: a rate tracker steps it toward
    target_escalation_rate on every window (step_decibels per event), and
    a randomized audit lane strong-decodes audit_rate of the kept
    windows; one revised audit multiplies the target by adjust_factor,
    and only ceil(3 / kept_bad_budget) consecutive clean audits divide it
    back. The target stays inside [min_escalation_rate,
    max_escalation_rate]; the max is the Theorem 1 backlog cap. Defaults
    are the validated drift-replay configuration.
    """

    target_escalation_rate: float
    step_decibels: float
    step_nats: float
    audit_rate: float
    kept_bad_budget: float
    adjust_factor: float
    min_escalation_rate: float
    max_escalation_rate: float


@dataclasses.dataclass(frozen=True)
class SwitchingCard:
    """The escalation decision's knobs.

    Keep the weak result when its complementary gap is at or above the
    threshold, in decibels (the paper's unit; Toshio 2510.25222 uses gth =
    20 dB). The core compares in natural-log weight units; the loader
    stores both.

    threshold_source picks where the threshold comes from: "fixed"
    (gap_threshold_decibels as given), "table" (looked up per sweep point
    in an offline calibration table, threshold_table's threshold_column;
    the table must be calibrated for this run's window geometry, since a
    threshold does not transfer between geometries), or "online"
    (gap_threshold_decibels is only the starting point and the online
    card's two-loop controller adapts it across the point's shots; serial
    switching only).

    double_window selects the paper's Sec. III C scheme (the weak chain
    keeps committing past an escalated strong window); without it,
    escalation is serial and boundaries are held until results are final.
    gap_computation picks the weak unit's gap engine: "serial" runs the
    forced solves after the decode on one core, "parallel_pair" runs the
    two forced-class solves on two cores inside the unit and charges the
    slower core plus the join, "split_pair" sends the second forced solve
    to its own decoder unit pool (gap_units of them) with its own
    syndrome transfer, and the window's decision waits at the join.
    split_pair is the option for a unit that cannot hold two cores.
    """

    gap_threshold_decibels: Optional[float]  # None exactly for source table
    gap_threshold_nats: Optional[float]
    threshold_source: str
    threshold_table: Optional[str]  # table source: the csv path
    threshold_column: Optional[str]  # table source: which computed column
    online: Optional[OnlineThresholdCard]
    double_window: bool
    gap_computation: str
    gap_units: int


@dataclasses.dataclass(frozen=True)
class BuffersCard:
    """Syndrome-path store capacities, in rounds (None = unbounded).

    weak_buffer_rounds bounds Buffer 0, the upstream round store; a full
    Buffer 0 refuses the next round and the packing overflow policy
    decides what happens (fail-stop by default). strong_buffer_rounds
    bounds syndrome buffer 1, the strong-side store; overflowing it is a
    hard error. packing_rounds_in_flight bounds the packing stage's
    assembly workspace, the rounds in flight through the stage at once.
    """

    weak_buffer_rounds: Optional[int]
    strong_buffer_rounds: Optional[int]
    packing_rounds_in_flight: Optional[int]


@dataclasses.dataclass(frozen=True)
class WindowingCard:
    """The window scheme and its commit and buffer sizes (None = d)."""

    scheme: str
    commit_rounds: Optional[int]
    buffer_rounds: Optional[int]


@dataclasses.dataclass(frozen=True)
class ControllerCard:
    """Controller work per round, in cycles of a named clock domain.

    The loader resolves the microsecond fields once, like the link cards.
    """

    readout_to_bits_cycles: float
    packing_cycles_per_round: float
    decision_to_pulse_cycles: float
    clock: str
    readout_to_bits_microseconds: float
    packing_microseconds_per_round: float
    decision_to_pulse_microseconds: float


@dataclasses.dataclass(frozen=True)
class SweepBlock:
    """One cross product of the three axes, `shots` seeds per point.

    The algorithm is not a sweep axis: it is structure, fixed per unit on
    the decoder card; comparing algorithms is comparing configs. Distance
    is an axis because the papers' LER plots are one curve per d (Toshio
    2510.25222 sweeps d at fixed p; threshold plots sweep p per d).
    """

    physical_error_probabilities: tuple
    distances: tuple
    round_periods_microseconds: tuple
    shots: int


@dataclasses.dataclass(frozen=True)
class RoundsCard:
    """rounds_per_shot: a fixed count, or per-distance scaling.

    "10d" is ten rounds per unit of code distance, the 10 d-round memory
    experiment of Toshio 2510.25222 Fig. 4.
    """

    fixed: Optional[int]
    per_distance: Optional[int]

    def rounds_for(self, distance: int) -> int:
        """The rounds one shot runs at this distance."""
        if self.fixed is not None:
            return self.fixed
        return self.per_distance * distance

    def __str__(self) -> str:
        if self.fixed is not None:
            return str(self.fixed)
        return f"{self.per_distance}d"


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Everything one yaml file says, as typed cards."""

    name: str  # the yaml stem; suffixes the run folder
    decode_path: str  # weak_baseline | strong_only | switching
    circuit: str  # stim generator task
    rounds_per_shot: RoundsCard  # fixed count or per-distance ("10d")
    windowing: WindowingCard
    sweep: tuple  # of SweepBlock
    controller: ControllerCard
    # clock domain name -> MHz; links price their cycles on the domain
    # they name
    clocks: dict
    links: dict  # path -> LinkCard | None (reference card)
    buffers: BuffersCard
    decoder: DecoderCard
    # the escalation threshold; present exactly when decode_path is
    # switching
    switching: Optional[SwitchingCard]
    # off | print | file | both: the engine narrator, live on screen
    # and/or one log file per shot in the run dir's trace/
    trace: str
    # add component I/O lines to the trace: what each store and unit
    # received, holds, and emitted
    log_component_io: bool
    # none | tesseract: re-decode every window with the Tesseract referee
    # and count disagreements (never priced)
    check_windows_with: str
    # separate_decode_jobs | ignore | extend_stream: how an idle patch's
    # rounds are charged (inert while the workload is a single always-busy
    # operation)
    idle_policy: str
    # resolved from pauli_frame.write_cycles on its named clock
    pauli_frame_commit_microseconds: float
    # the yaml files this config was read from, nearest first (an extends
    # chain)
    config_files: tuple

    @property
    def active_decoder(self) -> DecoderUnitCard:
        """The unit the decode_path decodes on (present, enforced at load)."""
        tier = DECODE_PATH_TIER[self.decode_path]
        return getattr(self.decoder, tier)


def load_experiment(path) -> ExperimentConfig:
    """Read one yaml file, its extends chain applied, into cards."""
    path = Path(path)
    raw, config_files = _raw_yaml(path)
    decode_path = _require(raw["decode_path"], DECODE_PATHS, "decode_path")
    clocks = dict(raw["clocks"])
    raw_trace = raw.get("trace", "off")
    if raw_trace is False:
        raw_trace = "off"  # yaml 1.1 reads a bare `off` as boolean False
    trace = _require(raw_trace, TRACE_MODES, "trace")
    raw_log_component_io = raw.get("log_component_io", False)
    log_component_io = _require(
        raw_log_component_io, (True, False), "log_component_io"
    )
    raw_idle_policy = raw.get("idle_policy", "separate_decode_jobs")
    idle_policy = _require(raw_idle_policy, IDLE_POLICIES, "idle_policy")
    raw_check = raw.get("check_windows_with", "none")
    check_windows_with = _require(
        raw_check, ("none", "tesseract"), "check_windows_with"
    )
    rounds_per_shot = _rounds_card(raw["rounds_per_shot"])
    windowing = _windowing_card(raw["windowing"])
    sweep = _sweep_blocks(raw["sweep"])
    controller = _controller_card(raw["controller"], clocks)
    links = _link_cards(raw["links"], clocks)
    buffers = _buffers_card(raw["buffers"])
    decoder = _decoder_card(raw["decoder"], clocks, decode_path)
    raw_switching = raw.get("switching")
    switching = _switching_card(raw_switching, decode_path)
    pauli_frame_commit_microseconds = _pauli_frame_commit_microseconds(
        raw["pauli_frame"], clocks
    )
    return ExperimentConfig(
        name=path.stem,
        decode_path=decode_path,
        circuit=raw["circuit"],
        rounds_per_shot=rounds_per_shot,
        windowing=windowing,
        sweep=sweep,
        controller=controller,
        clocks=clocks,
        links=links,
        buffers=buffers,
        decoder=decoder,
        switching=switching,
        trace=trace,
        log_component_io=log_component_io,
        idle_policy=idle_policy,
        check_windows_with=check_windows_with,
        pauli_frame_commit_microseconds=pauli_frame_commit_microseconds,
        config_files=config_files,
    )


def _raw_yaml(path: Path) -> tuple:
    """The file's keys with its `extends` chain applied, and its files.

    The files come this file first, then the base it extends, and so on.
    A key this file names replaces the base's key whole: a child that
    declares `sweep` ignores the base's sweep entirely.
    """
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    base_name = raw.pop("extends", None)
    if base_name is None:
        return raw, (path,)
    base_path = path.parent / base_name
    base, base_paths = _raw_yaml(base_path)
    base.update(raw)
    return base, (path,) + base_paths


def _link_cards(raw_links: dict, clocks: dict) -> dict:
    cards = {}
    for link_path in LINK_PATHS:
        raw_card = raw_links.get(link_path)
        cards[link_path] = _link_card(raw_card, clocks)
    return cards


def _link_card(card: Optional[dict], clocks: dict) -> Optional[LinkCard]:
    if card is None:
        return None
    clock = card["clock"]
    megahertz = clocks[clock]
    latency_cycles = card["latency_cycles"]
    bits_per_cycle = card["bits_per_cycle"]
    channels = card.get("channels", 1)
    setup_cycles = card.get("setup_cycles_per_transfer")
    bits_per_microsecond = None
    if bits_per_cycle is not None:
        bits_per_microsecond = bits_per_cycle * channels * megahertz
    setup_microseconds = None
    if setup_cycles is not None:
        setup_microseconds = setup_cycles / megahertz
    latency_microseconds = latency_cycles / megahertz
    return LinkCard(
        latency_cycles=latency_cycles,
        clock=clock,
        bits_per_cycle=bits_per_cycle,
        channels=channels,
        setup_cycles_per_transfer=setup_cycles,
        latency_microseconds=latency_microseconds,
        bits_per_microsecond=bits_per_microsecond,
        setup_microseconds_per_transfer=setup_microseconds,
    )


def _controller_card(card: dict, clocks: dict) -> ControllerCard:
    clock = card["clock"]
    megahertz = clocks[clock]
    input_cycles = card["readout_to_bits_cycles"]
    pack_cycles = card["packing_cycles_per_round"]
    output_cycles = card["decision_to_pulse_cycles"]
    readout_to_bits_microseconds = input_cycles / megahertz
    packing_microseconds_per_round = pack_cycles / megahertz
    decision_to_pulse_microseconds = output_cycles / megahertz
    return ControllerCard(
        readout_to_bits_cycles=input_cycles,
        packing_cycles_per_round=pack_cycles,
        decision_to_pulse_cycles=output_cycles,
        clock=clock,
        readout_to_bits_microseconds=readout_to_bits_microseconds,
        packing_microseconds_per_round=packing_microseconds_per_round,
        decision_to_pulse_microseconds=decision_to_pulse_microseconds,
    )


def _decoder_unit(
    card: Optional[dict], clocks: dict, tier: str
) -> Optional[DecoderUnitCard]:
    if card is None:
        return None
    algorithm = card["algorithm"]
    is_named = isinstance(algorithm, str)
    if is_named and algorithm not in ALGORITHMS:
        raise ValueError(
            f"decoder.{tier}.algorithm is a number (fixed core latency, us) "
            f"or one of {ALGORITHMS}, got {algorithm!r}"
        )
    raw_engine = card["engine"]
    engine_clock = raw_engine["clock"]
    engine = EngineCard(
        clock=engine_clock,
        fetch_cycles_per_round=raw_engine["fetch_cycles_per_round"],
        release_cycles_per_job=raw_engine["release_cycles_per_job"],
        megahertz=clocks[engine_clock],
    )
    return DecoderUnitCard(
        algorithm=algorithm,
        units=card["units"],
        unit_memory_rounds=card["unit_memory_rounds"],
        engine=engine,
    )


def _decoder_card(raw_decoder, clocks: dict, decode_path: str) -> DecoderCard:
    """One unit card per tier; the decode_path's tier is required."""
    unknown = set(raw_decoder) - {"weak", "strong"}
    if unknown:
        listed = sorted(unknown)
        raise ValueError(
            f"decoder does not know {listed}; its keys are the tiers: weak, "
            "strong"
        )
    raw_weak = raw_decoder.get("weak")
    raw_strong = raw_decoder.get("strong")
    weak = _decoder_unit(raw_weak, clocks, "weak")
    strong = _decoder_unit(raw_strong, clocks, "strong")
    card = DecoderCard(weak=weak, strong=strong)
    tier = DECODE_PATH_TIER[decode_path]
    if getattr(card, tier) is None:
        raise ValueError(
            f"decode_path {decode_path} decodes on decoder.{tier}, which "
            "this config does not define"
        )
    if decode_path == "switching" and card.strong is None:
        raise ValueError(
            "decode_path switching escalates to decoder.strong, which this "
            "config does not define"
        )
    return card


def _switching_card(raw_switching, decode_path: str) -> Optional[SwitchingCard]:
    """The switching card, present exactly when the decode_path escalates."""
    if decode_path != "switching":
        if raw_switching is not None:
            raise ValueError(
                f"decode_path {decode_path} never escalates; drop the "
                "switching card"
            )
        return None
    if raw_switching is None:
        raise ValueError(
            "decode_path switching needs a switching card with gap_threshold_db"
        )
    unknown = set(raw_switching) - set(SWITCHING_KEYS)
    if unknown:
        listed = sorted(unknown)
        known = _listed(SWITCHING_KEYS)
        raise ValueError(
            f"switching does not know {listed}; its keys are {known}"
        )
    raw_source = raw_switching.get("threshold_source", "fixed")
    threshold_source = _require(
        raw_source, THRESHOLD_SOURCES, "switching.threshold_source"
    )
    gap_threshold_decibels = _gap_threshold_decibels(
        raw_switching, threshold_source
    )
    gap_threshold_nats = None
    if gap_threshold_decibels is not None:
        gap_threshold_nats = _decibels_to_nats(gap_threshold_decibels)
    threshold_column = None
    if threshold_source == "table":
        raw_column = raw_switching.get("threshold_column", "gth_eq4_wilson")
        threshold_column = str(raw_column)
    online = _online_card(raw_switching, threshold_source)
    gap_computation, gap_units = _gap_computation(raw_switching)
    raw_double_window = raw_switching.get("double_window", False)
    double_window = _require(
        raw_double_window, (True, False), "switching.double_window"
    )
    _check_serial_only(gap_computation, threshold_source, double_window)
    threshold_table = raw_switching.get("threshold_table")
    return SwitchingCard(
        gap_threshold_decibels=gap_threshold_decibels,
        gap_threshold_nats=gap_threshold_nats,
        threshold_source=threshold_source,
        threshold_table=threshold_table,
        threshold_column=threshold_column,
        online=online,
        double_window=double_window,
        gap_computation=gap_computation,
        gap_units=gap_units,
    )


def _gap_threshold_decibels(
    raw_switching: dict, threshold_source: str
) -> Optional[float]:
    """The card's threshold; the table source computes it instead."""
    if threshold_source == "table":
        if "gap_threshold_db" in raw_switching:
            raise ValueError(
                "threshold_source table computes the threshold from "
                "threshold_table; drop gap_threshold_db"
            )
        if "threshold_table" not in raw_switching:
            raise ValueError(
                "threshold_source table needs threshold_table, the "
                "calibration csv path (calibrate offline for this run's "
                "window geometry)"
            )
        return None
    has_table_key = "threshold_table" in raw_switching
    has_column_key = "threshold_column" in raw_switching
    if has_table_key or has_column_key:
        raise ValueError(
            "threshold_table/threshold_column belong to threshold_source "
            f"table; the source is {threshold_source}"
        )
    if "gap_threshold_db" not in raw_switching:
        raise ValueError(
            "decode_path switching needs a switching card with gap_threshold_db"
        )
    return float(raw_switching["gap_threshold_db"])


def _online_card(
    raw_switching: dict, threshold_source: str
) -> Optional[OnlineThresholdCard]:
    """The online card, present exactly for the online source."""
    if "online" in raw_switching and threshold_source != "online":
        raise ValueError(
            "the online card belongs to threshold_source online; the "
            f"source is {threshold_source}"
        )
    if threshold_source != "online":
        return None
    raw_online = raw_switching.get("online") or {}
    return _online_threshold_card(raw_online)


def _gap_computation(raw_switching: dict) -> tuple:
    """(gap_computation, gap_units); gap_units belongs to split_pair."""
    raw_computation = raw_switching.get("gap_computation", "serial")
    gap_computation = _require(
        raw_computation, GAP_COMPUTATIONS, "switching.gap_computation"
    )
    raw_units = raw_switching.get("gap_units", 1)
    gap_units = int(raw_units)
    if gap_units < 1:
        raise ValueError(
            f"switching.gap_units needs at least 1 unit (got {gap_units})"
        )
    if "gap_units" in raw_switching and gap_computation != "split_pair":
        raise ValueError(
            "switching.gap_units sizes the split_pair sibling pool; "
            f"gap_computation is {gap_computation!r}, so drop the key"
        )
    return gap_computation, gap_units


def _check_serial_only(
    gap_computation: str, threshold_source: str, double_window: bool
) -> None:
    """split_pair and online calibration are validated for serial switching."""
    if not double_window:
        return
    if gap_computation == "split_pair":
        raise ValueError(
            "split_pair is validated for serial switching only: a strong "
            "window absorbing windows whose gap join is still open is not "
            "supported yet"
        )
    if threshold_source == "online":
        raise ValueError(
            "threshold_source online is serial-only: an audit label "
            "compares one window's weak and strong committed "
            "observables, and a double-window strong result owns a "
            "larger extent than the audited window"
        )


def _online_threshold_card(raw_online) -> OnlineThresholdCard:
    """The online card, defaults from the validated drift replay."""
    unknown = set(raw_online) - set(ONLINE_KEYS)
    if unknown:
        listed = sorted(unknown)
        known = _listed(ONLINE_KEYS)
        raise ValueError(
            f"switching.online does not know {listed}; its keys are {known}"
        )
    target_escalation_rate = _online_float(
        raw_online, "target_escalation_rate", 1e-3
    )
    step_decibels = _online_float(raw_online, "step_db", 0.25)
    audit_rate = _online_float(raw_online, "audit_rate", 0.01)
    kept_bad_budget = _online_float(raw_online, "kept_bad_budget", 2e-4)
    adjust_factor = _online_float(raw_online, "adjust_factor", 2.0)
    min_escalation_rate = _online_float(raw_online, "min_escalation_rate", 1e-5)
    max_escalation_rate = _online_float(raw_online, "max_escalation_rate", 0.30)
    if step_decibels <= 0:
        raise ValueError(
            f"switching.online.step_db must be positive (got {step_decibels})"
        )
    if not 0 < audit_rate < 1:
        raise ValueError(
            f"switching.online.audit_rate must be in (0, 1) (got {audit_rate})"
        )
    if not 0 < kept_bad_budget < 1:
        raise ValueError(
            "switching.online.kept_bad_budget must be in (0, 1) "
            f"(got {kept_bad_budget})"
        )
    if adjust_factor <= 1:
        raise ValueError(
            "switching.online.adjust_factor must exceed 1 "
            f"(got {adjust_factor})"
        )
    _check_escalation_rates(
        target_escalation_rate, min_escalation_rate, max_escalation_rate
    )
    step_nats = _decibels_to_nats(step_decibels)
    return OnlineThresholdCard(
        target_escalation_rate=target_escalation_rate,
        step_decibels=step_decibels,
        step_nats=step_nats,
        audit_rate=audit_rate,
        kept_bad_budget=kept_bad_budget,
        adjust_factor=adjust_factor,
        min_escalation_rate=min_escalation_rate,
        max_escalation_rate=max_escalation_rate,
    )


def _online_float(raw_online: dict, key: str, default: float) -> float:
    raw = raw_online.get(key, default)
    return float(raw)


def _check_escalation_rates(
    target_escalation_rate: float,
    min_escalation_rate: float,
    max_escalation_rate: float,
) -> None:
    """0 < min <= max <= 1, and the target inside [min, max]."""
    if not 0 < min_escalation_rate <= max_escalation_rate <= 1:
        raise ValueError(
            "switching.online needs 0 < min_escalation_rate <= "
            f"max_escalation_rate <= 1 (got {min_escalation_rate} and "
            f"{max_escalation_rate})"
        )
    if not min_escalation_rate <= target_escalation_rate <= max_escalation_rate:
        raise ValueError(
            "switching.online.target_escalation_rate must lie inside "
            "[min_escalation_rate, max_escalation_rate] "
            f"(got {target_escalation_rate})"
        )


def _sweep_blocks(raw_sweep: list) -> tuple:
    blocks = []
    for index, block in enumerate(raw_sweep, start=1):
        sweep_block = _sweep_block(block, index)
        blocks.append(sweep_block)
    return tuple(blocks)


def _sweep_block(block: dict, index: int) -> SweepBlock:
    unknown = set(block) - set(SWEEP_KEYS)
    if unknown:
        listed = sorted(unknown)
        raise ValueError(
            f"sweep block {index} does not know {listed}; its axes "
            "are physical_error_probability, distance and round_period_us, "
            "plus shots (the algorithm lives on the decoder card, not in "
            "the sweep)"
        )
    return SweepBlock(
        physical_error_probabilities=tuple(block["physical_error_probability"]),
        distances=tuple(block["distance"]),
        round_periods_microseconds=tuple(block["round_period_us"]),
        shots=block["shots"],
    )


def _windowing_card(raw_windowing: dict) -> WindowingCard:
    scheme = _require(raw_windowing["scheme"], SCHEMES, "windowing.scheme")
    return WindowingCard(
        scheme=scheme,
        commit_rounds=raw_windowing["commit_rounds"],
        buffer_rounds=raw_windowing["buffer_rounds"],
    )


def _buffers_card(raw_buffers: dict) -> BuffersCard:
    return BuffersCard(
        weak_buffer_rounds=raw_buffers["weak_buffer_rounds"],
        strong_buffer_rounds=raw_buffers["strong_buffer_rounds"],
        packing_rounds_in_flight=raw_buffers["packing_rounds_in_flight"],
    )


def _rounds_card(value) -> RoundsCard:
    if isinstance(value, int):
        return RoundsCard(fixed=value, per_distance=None)
    if _is_per_distance_text(value):
        per_distance = int(value[:-1])
        return RoundsCard(fixed=None, per_distance=per_distance)
    raise ValueError(
        "rounds_per_shot is a round count or '<n>d' (rounds per unit of "
        f"distance), got {value!r}"
    )


def _is_per_distance_text(value) -> bool:
    """True for "<n>d": digits then a d."""
    if not isinstance(value, str):
        return False
    if not value.endswith("d"):
        return False
    digits = value[:-1]
    return digits.isdigit()


def _pauli_frame_commit_microseconds(card: dict, clocks: dict) -> float:
    megahertz = clocks[card["clock"]]
    return card["write_cycles"] / megahertz


def _decibels_to_nats(decibels: float) -> float:
    scaled = decibels * LN_TEN
    return scaled / 10.0


def _listed(keys: tuple) -> str:
    """The keys as prose: a, b and c."""
    leading = ", ".join(keys[:-1])
    return f"{leading} and {keys[-1]}"


def _require(value, allowed: tuple, key: str):
    if value not in allowed:
        raise ValueError(f"{key} must be one of {allowed}, got {value!r}")
    return value
