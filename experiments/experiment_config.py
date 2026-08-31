"""One yaml file is one experiment; this module is the only yaml reader.

The loader turns the file into the small frozen cards below and validates
the names it knows, so every other module works with typed fields instead
of dict lookups. `extends: other.yaml` starts from that file (same folder)
and overrides the top-level keys this file names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml

# A decoder unit's algorithm: a named real algorithm, decoded per window
# and charged its measured wall clock, or a number, a fixed core latency in
# us (a hypothetical core). The names are the two tiers of the decoder-
# switching setting (Toshio arXiv 2510.25222: MWPM weak, belief-matching
# strong).
ALGORITHMS = ("pymatching", "belief_matching")
AlgorithmCard = Union[float, str]

MODES = ("weak_baseline", "strong_only", "switching")
# The decoder unit each mode decodes on. A mode's unit must be defined;
# the other tier's card may be omitted. Switching decodes every window
# on the weak tier first (its unit is the active one) and requires the
# strong tier too, for escalation.
MODE_TIER = {"weak_baseline": "weak", "strong_only": "strong",
             "switching": "weak"}
TRACE_MODES = ("off", "print", "file", "both")
SCHEMES = ("sliding", "parallel", "sandwich", "naive_online")
# How an idle patch's rounds are charged. Idle rounds are decoder workload
# in every reference system (SWIPER ISCA 2025, XQsim, Terhal backlog), so
# separate_decode_jobs is the default; ignore is the optimistic card for
# active-path latency studies; extend_stream folds them into a live stream.
IDLE_POLICIES = ("separate_decode_jobs", "ignore", "extend_stream")
LINK_PATHS = ("qc", "cwb", "csb", "wbd", "wsd", "sbd",
              "dd", "wdo", "do", "oc", "cq")


@dataclass(frozen=True)
class LinkCard:
    """One path's numbers, in cycles of a named clock domain: propagation
    latency, per-channel capacity (None = unbounded) times a channel count,
    and an optional fixed per-transfer DMA setup cost. The loader resolves
    the microsecond fields from the clocks card once, so everything
    downstream keeps reading microseconds while the yaml speaks cycles
    (XQsim's shape: domain labels with frequencies over one tick core)."""
    latency_cycles: float
    clock: str
    bits_per_cycle: Optional[float]
    channels: int
    transfer_overhead_cycles: Optional[float]
    latency_us: float                     # latency_cycles / clock MHz
    bits_per_us: Optional[float]          # bits_per_cycle x channels x MHz
    transfer_overhead_us: Optional[float]


@dataclass(frozen=True)
class EngineCard:
    """The decoder engine's fetch/release stage costs, in cycles of a named
    clock domain; the loader resolves the frequency once, like the links."""
    clock: str
    fetch_cycles_per_round: int
    release_cycles_per_job: int
    frequency_mhz: float


@dataclass(frozen=True)
class DecoderUnitCard:
    """One decoder tier: its algorithm, its unit pool, and per unit the
    input SRAM size in rounds (None = unbounded). A unit overlaps input
    transfer with compute only when two windows fit in its SRAM."""
    algorithm: AlgorithmCard
    units: int
    unit_buffer_size: Optional[int]
    engine: EngineCard


@dataclass(frozen=True)
class DecoderCard:
    """The two tiers of the decoder-switching architecture, as two distinct
    units (Toshio arXiv 2510.25222: lightweight decoders decode constantly,
    a separate accurate decoder is invoked on demand). A config defines the
    units its mode uses; the mode picks the active one."""
    weak: Optional[DecoderUnitCard]
    strong: Optional[DecoderUnitCard]


@dataclass(frozen=True)
class OnlineThresholdCard:
    """The online calibrator's knobs (threshold_source: online).

    Two loops around the live threshold: a rate tracker steps it toward
    ``target_escalation_rate`` on every window (``step_db`` per event),
    and a randomized audit lane strong-decodes ``audit_rate`` of the
    KEPT windows; one revised audit multiplies the target by
    ``adjust_factor``, and only ceil(3 / kept_bad_budget) consecutive
    clean audits divide it back. The target stays inside
    [min_escalation_rate, max_escalation_rate]; the max is the
    Theorem 1 backlog cap. Defaults are the validated drift-replay
    configuration."""
    target_escalation_rate: float
    step_db: float
    step_nats: float
    audit_rate: float
    kept_bad_budget: float
    adjust_factor: float
    min_escalation_rate: float
    max_escalation_rate: float


@dataclass(frozen=True)
class SwitchingCard:
    """The escalation decision's knobs: keep the weak result when its
    complementary gap is at or above the threshold, in decibels (the
    paper's unit; Toshio 2510.25222 uses gth = 20 dB). The core compares
    in natural-log weight units; the loader stores both.

    threshold_source picks where the threshold comes from: "fixed"
    (gap_threshold_db as given), "table" (looked up per sweep point in
    an offline calibration table, threshold_table's threshold_column;
    the table must be calibrated for this run's window geometry, since
    a threshold does not transfer between geometries), or "online"
    (gap_threshold_db is only the starting point and the online card's
    two-loop controller adapts it across the point's shots; serial
    switching only).

    double_window selects the paper's Sec. III C scheme (the weak chain
    keeps committing past an escalated strong window); without it,
    escalation is serial and boundaries are held until results are
    final. gap_computation picks the weak unit's gap engine: "serial"
    runs the forced solves after the decode on one core,
    "parallel_pair" runs the two forced-class solves on two cores
    inside the unit and charges the slower core plus the join,
    "split_pair" sends the second forced solve to its own decoder unit
    pool (gap_units of them) with its own syndrome transfer, and the
    window's decision waits at the join. split_pair is the option for a
    unit that cannot hold two cores."""
    gap_threshold_db: Optional[float]   # None exactly when source is table
    gap_threshold_nats: Optional[float]
    threshold_source: str
    threshold_table: Optional[str]      # table source: the csv path
    threshold_column: Optional[str]     # table source: which computed column
    online: Optional[OnlineThresholdCard]
    double_window: bool
    gap_computation: str
    gap_units: int


@dataclass(frozen=True)
class BuffersCard:
    """Syndrome-path store capacities, in rounds (None = unbounded).

    ``buffer_0_size`` bounds Buffer 0, the upstream round store; a full
    Buffer 0 refuses the next round and the packing overflow policy decides
    what happens (fail-stop by default). ``buffer_1_size`` bounds syndrome
    buffer 1, the strong-side store; overflowing it is a hard error.
    ``packing_workspace_size`` bounds the packing stage's assembly
    workspace, the rounds in flight through the stage at once."""
    buffer_0_size: Optional[int]
    buffer_1_size: Optional[int]
    packing_workspace_size: Optional[int]


@dataclass(frozen=True)
class WindowingCard:
    """The window scheme and its commit/buffer sizes (None = d)."""
    scheme: str
    commit_rounds: Optional[int]
    buffer_rounds: Optional[int]


@dataclass(frozen=True)
class ControllerCard:
    """Controller work per round, in cycles of a named clock domain; the
    loader resolves the microsecond fields once, like the link cards."""
    measurement_signal_to_classical_bits_cycles: float
    t_pack_cycles: float
    instruction_or_decision_to_analog_control_pulse_cycles: float
    clock: str
    measurement_signal_to_classical_bits_us: float
    t_pack_us: float
    instruction_or_decision_to_analog_control_pulse_us: float


@dataclass(frozen=True)
class SweepBlock:
    """One cross product of the three axes, `shots` seeds per point. The
    algorithm is not a sweep axis: it is structure, fixed per unit on the
    decoder card; comparing algorithms is comparing configs. Distance is
    an axis because the papers' LER plots are one curve per d (Toshio
    2510.25222 sweeps d at fixed p; threshold plots sweep p per d)."""
    physical_error_probabilities: tuple
    distances: tuple
    round_periods_us: tuple
    shots: int


@dataclass(frozen=True)
class RoundsCard:
    """rounds_per_shot: a fixed count, or per-distance scaling ("10d" =
    ten rounds per unit of code distance, the 10 d-round memory experiment
    of Toshio 2510.25222 Fig. 4)."""
    fixed: Optional[int]
    per_distance: Optional[int]

    def rounds_for(self, distance: int) -> int:
        if self.fixed is not None:
            return self.fixed
        return self.per_distance * distance

    def __str__(self) -> str:
        if self.fixed is not None:
            return str(self.fixed)
        return f"{self.per_distance}d"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str                       # the yaml stem; suffixes the run folder
    mode: str                       # weak_baseline | strong_only
    code_task: str                  # stim generator task
    rounds_per_shot: RoundsCard     # fixed count or per-distance ("10d")
    windowing: WindowingCard
    sweep: tuple                    # of SweepBlock
    controller: ControllerCard
    clocks: dict                    # clock domain name -> MHz; links price
                                    # their cycles on the domain they name
    links: dict                     # path -> LinkCard | None (reference card)
    buffers: BuffersCard
    decoder: DecoderCard
    switching: Optional[SwitchingCard]  # the escalation threshold; present
                                    # exactly when mode is switching
    trace: str                      # off | print | file | both: the engine
                                    # narrator, live on screen and/or one log
                                    # file per shot in the run dir's trace/
    trace_io: bool                  # add component I/O lines to the trace:
                                    # what each store and unit received,
                                    # holds, and emitted
    verify_windows: str             # none | tesseract: re-decode every
                                    # window with the Tesseract referee and
                                    # count disagreements (never priced)
    idle_policy: str                # separate_decode_jobs | ignore |
                                    # extend_stream: how an idle patch's
                                    # rounds are charged (inert while the
                                    # workload is a single always-busy op)
    pauli_frame_commit_us: float    # resolved from pauli_frame.commit_cycles
                                    # on its named clock
    config_files: tuple             # the yaml files this config was read
                                    # from, nearest first (an extends chain)

    @property
    def active_decoder(self) -> DecoderUnitCard:
        """The unit the mode decodes on: weak_baseline reads decoder.weak,
        strong_only reads decoder.strong (present, enforced at load)."""
        return getattr(self.decoder, MODE_TIER[self.mode])


def _link_card(card: Optional[dict], clocks: dict) -> Optional[LinkCard]:
    if card is None:
        return None
    clock = card["clock"]
    megahertz = clocks[clock]
    latency_cycles = card["latency_cycles"]
    bits_per_cycle = card["bits_per_cycle"]
    channels = card.get("channels", 1)
    overhead_cycles = card.get("transfer_overhead_cycles")
    return LinkCard(
        latency_cycles=latency_cycles, clock=clock,
        bits_per_cycle=bits_per_cycle, channels=channels,
        transfer_overhead_cycles=overhead_cycles,
        latency_us=latency_cycles / megahertz,
        bits_per_us=(None if bits_per_cycle is None
                     else bits_per_cycle * channels * megahertz),
        transfer_overhead_us=(None if overhead_cycles is None
                              else overhead_cycles / megahertz))


def _controller_card(card: dict, clocks: dict) -> ControllerCard:
    clock = card["clock"]
    megahertz = clocks[clock]
    input_cycles = card["measurement_signal_to_classical_bits_cycles"]
    pack_cycles = card["t_pack_cycles"]
    output_cycles = card["instruction_or_decision_to_analog_control_pulse_cycles"]
    return ControllerCard(
        measurement_signal_to_classical_bits_cycles=input_cycles,
        t_pack_cycles=pack_cycles,
        instruction_or_decision_to_analog_control_pulse_cycles=output_cycles,
        clock=clock,
        measurement_signal_to_classical_bits_us=input_cycles / megahertz,
        t_pack_us=pack_cycles / megahertz,
        instruction_or_decision_to_analog_control_pulse_us=
            output_cycles / megahertz)


def _decoder_unit(card: Optional[dict], clocks: dict,
                  tier: str) -> Optional[DecoderUnitCard]:
    if card is None:
        return None
    algorithm = card["algorithm"]
    if isinstance(algorithm, str) and algorithm not in ALGORITHMS:
        raise ValueError(
            f"decoder.{tier}.algorithm is a number (fixed core latency, us) "
            f"or one of {ALGORITHMS}, got {algorithm!r}")
    engine = card["engine"]
    engine_clock = engine["clock"]
    return DecoderUnitCard(
        algorithm=algorithm, units=card["units"],
        unit_buffer_size=card["unit_buffer_size"],
        engine=EngineCard(
            clock=engine_clock,
            fetch_cycles_per_round=engine["fetch_cycles_per_round"],
            release_cycles_per_job=engine["release_cycles_per_job"],
            frequency_mhz=clocks[engine_clock]))


def _decoder_card(raw_decoder, clocks: dict, mode: str) -> DecoderCard:
    """The decoder card: one unit card per tier, the mode's tier required."""
    unknown = set(raw_decoder) - {"weak", "strong"}
    if unknown:
        raise ValueError(f"decoder does not know {sorted(unknown)}; its keys "
                         f"are the tiers: weak, strong")
    card = DecoderCard(
        weak=_decoder_unit(raw_decoder.get("weak"), clocks, "weak"),
        strong=_decoder_unit(raw_decoder.get("strong"), clocks, "strong"))
    tier = MODE_TIER[mode]
    if getattr(card, tier) is None:
        raise ValueError(f"mode {mode} decodes on decoder.{tier}, "
                         f"which this config does not define")
    if mode == "switching" and card.strong is None:
        raise ValueError("mode switching escalates to decoder.strong, "
                         "which this config does not define")
    return card


def _switching_card(raw_switching, mode: str) -> Optional[SwitchingCard]:
    """The switching card, present exactly when the mode escalates."""
    if mode != "switching":
        if raw_switching is not None:
            raise ValueError(f"mode {mode} never escalates; drop the "
                             f"switching card")
        return None
    if raw_switching is None:
        raise ValueError("mode switching needs a switching card with "
                         "gap_threshold_db")
    unknown = set(raw_switching) - {"gap_threshold_db", "double_window",
                                    "gap_computation", "gap_units",
                                    "threshold_source", "threshold_table",
                                    "threshold_column", "online"}
    if unknown:
        raise ValueError(f"switching does not know {sorted(unknown)}; its "
                         f"keys are gap_threshold_db, double_window, "
                         f"gap_computation, gap_units, threshold_source, "
                         f"threshold_table, threshold_column and online")
    import math
    threshold_source = _require(
        raw_switching.get("threshold_source", "fixed"),
        ("fixed", "table", "online"), "switching.threshold_source")
    if threshold_source == "table":
        if "gap_threshold_db" in raw_switching:
            raise ValueError(
                "threshold_source table computes the threshold from "
                "threshold_table; drop gap_threshold_db")
        if "threshold_table" not in raw_switching:
            raise ValueError(
                "threshold_source table needs threshold_table, the "
                "calibration csv path (calibrate offline for this run's "
                "window geometry)")
        gap_threshold_db = None
        gap_threshold_nats = None
    else:
        if "threshold_table" in raw_switching or (
                "threshold_column" in raw_switching):
            raise ValueError(
                f"threshold_table/threshold_column belong to "
                f"threshold_source table; the source is {threshold_source}")
        if "gap_threshold_db" not in raw_switching:
            raise ValueError("mode switching needs a switching card with "
                             "gap_threshold_db")
        gap_threshold_db = float(raw_switching["gap_threshold_db"])
        # decibels are 10 log10 of the likelihood ratio; matching weights
        # are its natural log
        gap_threshold_nats = gap_threshold_db * math.log(10.0) / 10.0
    threshold_table = raw_switching.get("threshold_table")
    threshold_column = None
    if threshold_source == "table":
        threshold_column = str(
            raw_switching.get("threshold_column", "gth_eq4_wilson"))
    if "online" in raw_switching and threshold_source != "online":
        raise ValueError(
            f"the online card belongs to threshold_source online; the "
            f"source is {threshold_source}")
    online = None
    if threshold_source == "online":
        online = _online_threshold_card(raw_switching.get("online") or {})
    gap_computation = _require(
        raw_switching.get("gap_computation", "serial"),
        ("serial", "parallel_pair", "split_pair"),
        "switching.gap_computation")
    gap_units = int(raw_switching.get("gap_units", 1))
    if gap_units < 1:
        raise ValueError("switching.gap_units needs at least 1 unit "
                         f"(got {gap_units})")
    if "gap_units" in raw_switching and gap_computation != "split_pair":
        raise ValueError(
            "switching.gap_units sizes the split_pair sibling pool; "
            f"gap_computation is {gap_computation!r}, so drop the key")
    double_window = _require(
        raw_switching.get("double_window", False),
        (True, False), "switching.double_window")
    if gap_computation == "split_pair" and double_window:
        raise ValueError(
            "split_pair is validated for serial switching only: a strong "
            "window absorbing windows whose gap join is still open is not "
            "supported yet")
    if threshold_source == "online" and double_window:
        raise ValueError(
            "threshold_source online is serial-only: an audit label "
            "compares one window's weak and strong committed "
            "observables, and a double-window strong result owns a "
            "larger extent than the audited window")
    return SwitchingCard(gap_threshold_db=gap_threshold_db,
                         gap_threshold_nats=gap_threshold_nats,
                         threshold_source=threshold_source,
                         threshold_table=threshold_table,
                         threshold_column=threshold_column,
                         online=online,
                         double_window=double_window,
                         gap_computation=gap_computation,
                         gap_units=gap_units)


def _online_threshold_card(raw_online) -> OnlineThresholdCard:
    """The online card, defaults from the validated drift replay."""
    import math
    unknown = set(raw_online) - {
        "target_escalation_rate", "step_db", "audit_rate",
        "kept_bad_budget", "adjust_factor", "min_escalation_rate",
        "max_escalation_rate"}
    if unknown:
        raise ValueError(
            f"switching.online does not know {sorted(unknown)}; its keys "
            f"are target_escalation_rate, step_db, audit_rate, "
            f"kept_bad_budget, adjust_factor, min_escalation_rate and "
            f"max_escalation_rate")
    target_escalation_rate = float(
        raw_online.get("target_escalation_rate", 1e-3))
    step_db = float(raw_online.get("step_db", 0.25))
    audit_rate = float(raw_online.get("audit_rate", 0.01))
    kept_bad_budget = float(raw_online.get("kept_bad_budget", 2e-4))
    adjust_factor = float(raw_online.get("adjust_factor", 2.0))
    min_escalation_rate = float(raw_online.get("min_escalation_rate", 1e-5))
    max_escalation_rate = float(raw_online.get("max_escalation_rate", 0.30))
    if step_db <= 0:
        raise ValueError(f"switching.online.step_db must be positive "
                         f"(got {step_db})")
    if not 0 < audit_rate < 1:
        raise ValueError(f"switching.online.audit_rate must be in (0, 1) "
                         f"(got {audit_rate})")
    if not 0 < kept_bad_budget < 1:
        raise ValueError(f"switching.online.kept_bad_budget must be in "
                         f"(0, 1) (got {kept_bad_budget})")
    if adjust_factor <= 1:
        raise ValueError(f"switching.online.adjust_factor must exceed 1 "
                         f"(got {adjust_factor})")
    if not 0 < min_escalation_rate <= max_escalation_rate <= 1:
        raise ValueError(
            f"switching.online needs 0 < min_escalation_rate <= "
            f"max_escalation_rate <= 1 (got {min_escalation_rate} and "
            f"{max_escalation_rate})")
    if not min_escalation_rate <= target_escalation_rate <= (
            max_escalation_rate):
        raise ValueError(
            f"switching.online.target_escalation_rate must lie inside "
            f"[min_escalation_rate, max_escalation_rate] "
            f"(got {target_escalation_rate})")
    return OnlineThresholdCard(
        target_escalation_rate=target_escalation_rate,
        step_db=step_db,
        step_nats=step_db * math.log(10.0) / 10.0,
        audit_rate=audit_rate,
        kept_bad_budget=kept_bad_budget,
        adjust_factor=adjust_factor,
        min_escalation_rate=min_escalation_rate,
        max_escalation_rate=max_escalation_rate)


def _sweep_block(block: dict, index: int) -> SweepBlock:
    unknown = set(block) - {"physical_error_probability", "distance",
                            "round_period_us", "shots"}
    if unknown:
        raise ValueError(
            f"sweep block {index} does not know {sorted(unknown)}; its axes "
            f"are physical_error_probability, distance and round_period_us, "
            f"plus shots (the algorithm lives on the decoder card, not in "
            f"the sweep)")
    return SweepBlock(
        physical_error_probabilities=tuple(block["physical_error_probability"]),
        distances=tuple(block["distance"]),
        round_periods_us=tuple(block["round_period_us"]),
        shots=block["shots"])


def _rounds_card(value) -> RoundsCard:
    if isinstance(value, int):
        return RoundsCard(fixed=value, per_distance=None)
    if isinstance(value, str) and value.endswith("d") and value[:-1].isdigit():
        return RoundsCard(fixed=None, per_distance=int(value[:-1]))
    raise ValueError(f"rounds_per_shot is a round count or '<n>d' (rounds "
                     f"per unit of distance), got {value!r}")


def _pauli_frame_commit_us(card: dict, clocks: dict) -> float:
    return card["commit_cycles"] / clocks[card["clock"]]


def _require(value, allowed: tuple, key: str):
    if value not in allowed:
        raise ValueError(f"{key} must be one of {allowed}, got {value!r}")
    return value


def _raw_yaml(path: Path) -> tuple:
    """The file's keys with its `extends` chain applied, and the files that
    produced them (this file first, then the base it extends, and so on).

    A key this file names replaces the base's key whole: a child that
    declares `sweep` ignores the base's sweep entirely.
    """
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    base_name = raw.pop("extends", None)
    if base_name is None:
        return raw, (path,)
    base, base_paths = _raw_yaml(path.parent / base_name)
    base.update(raw)
    return base, (path,) + base_paths


def load_experiment(path) -> ExperimentConfig:
    path = Path(path)
    raw, config_files = _raw_yaml(path)
    windowing = raw["windowing"]
    controller = raw["controller"]
    buffers = raw["buffers"]
    mode = _require(raw["mode"], MODES, "mode")
    clocks = dict(raw["clocks"])
    links = {link_path: _link_card(raw["links"].get(link_path), clocks)
             for link_path in LINK_PATHS}
    raw_trace = raw.get("trace", "off")
    if raw_trace is False:
        raw_trace = "off"     # yaml 1.1 reads a bare `off` as boolean False
    sweep = tuple(_sweep_block(block, index)
                  for index, block in enumerate(raw["sweep"], start=1))
    return ExperimentConfig(
        name=path.stem,
        mode=mode,
        code_task=raw["code_task"],
        rounds_per_shot=_rounds_card(raw["rounds_per_shot"]),
        windowing=WindowingCard(
            scheme=_require(windowing["scheme"], SCHEMES, "windowing.scheme"),
            commit_rounds=windowing["commit_rounds"],
            buffer_rounds=windowing["buffer_rounds"]),
        sweep=sweep,
        controller=_controller_card(controller, clocks),
        clocks=clocks,
        links=links,
        buffers=BuffersCard(
            buffer_0_size=buffers["buffer_0_size"],
            buffer_1_size=buffers["buffer_1_size"],
            packing_workspace_size=buffers["packing_workspace_size"]),
        decoder=_decoder_card(raw["decoder"], clocks, mode),
        switching=_switching_card(raw.get("switching"), mode),
        trace=_require(raw_trace, TRACE_MODES, "trace"),
        trace_io=_require(raw.get("trace_io", False), (True, False),
                          "trace_io"),
        idle_policy=_require(raw.get("idle_policy", "separate_decode_jobs"),
                             IDLE_POLICIES, "idle_policy"),
        verify_windows=_require(raw.get("verify_windows", "none"),
                                ("none", "tesseract"), "verify_windows"),
        pauli_frame_commit_us=_pauli_frame_commit_us(raw["pauli_frame"],
                                                      clocks),
        config_files=config_files)
