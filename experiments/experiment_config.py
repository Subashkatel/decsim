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

MODES = ("weak_baseline", "strong_only")
# The decoder unit each mode decodes on. A mode's unit must be defined;
# the other tier's card may be omitted.
MODE_TIER = {"weak_baseline": "weak", "strong_only": "strong"}
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
    t_binary_availability_cycles: float
    t_pack_cycles: float
    clock: str
    t_binary_availability_us: float
    t_pack_us: float


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
    binary_cycles = card["t_binary_availability_cycles"]
    pack_cycles = card["t_pack_cycles"]
    return ControllerCard(
        t_binary_availability_cycles=binary_cycles,
        t_pack_cycles=pack_cycles, clock=clock,
        t_binary_availability_us=binary_cycles / megahertz,
        t_pack_us=pack_cycles / megahertz)


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
    return card


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

