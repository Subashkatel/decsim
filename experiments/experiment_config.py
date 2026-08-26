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

MEASURED = "measured"
# A decoder algorithm card: a fixed core latency in microseconds, or
# MEASURED to charge the real PyMatching wall clock of every call.
AlgorithmCard = Union[float, str]

MODES = ("weak_baseline", "strong_only")
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
    """One path's numbers: propagation latency, capacity (None = unbounded),
    and an optional fixed per-transfer DMA setup cost."""
    latency_us: float
    bits_per_us: Optional[float]
    transfer_overhead_us: Optional[float]


@dataclass(frozen=True)
class EngineCard:
    """The decoder engine's clock and its fetch/release stage costs."""
    frequency_mhz: float
    fetch_cycles_per_round: int
    release_cycles_per_job: int


@dataclass(frozen=True)
class DecoderCard:
    """The unit pool: how many units and, per unit, the input SRAM size in
    rounds (None = unbounded). A unit overlaps input transfer with compute
    only when two windows fit in its SRAM."""
    units: int
    unit_buffer_size: Optional[int]
    engine: EngineCard


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
    t_binary_availability_us: float
    t_pack_us: float


@dataclass(frozen=True)
class SweepBlock:
    """One cross product of the three axes, `shots` seeds per point."""
    physical_error_probabilities: tuple
    algorithm_latencies_us: tuple
    round_periods_us: tuple
    shots: int


@dataclass(frozen=True)
class ExperimentConfig:
    name: str                       # the yaml file's stem; names the results folder
    mode: str                       # weak_baseline | strong_only
    code_task: str                  # stim generator task
    distance: int
    rounds_per_shot: int
    windowing: WindowingCard
    sweep: tuple                    # of SweepBlock
    controller: ControllerCard
    links: dict                     # path -> LinkCard | None (None = reference card)
    buffers: BuffersCard
    decoder: DecoderCard
    trace: str                      # off | print | file | both: the engine
                                    # narrator, live on screen and/or one log
                                    # file per shot in results/<name>/trace/
    trace_io: bool                  # add component I/O lines to the trace:
                                    # what each store and unit received,
                                    # holds, and emitted
    idle_policy: str                # separate_decode_jobs | ignore |
                                    # extend_stream: how an idle patch's
                                    # rounds are charged (inert while the
                                    # workload is a single always-busy op)
    pauli_frame_commit_us: float
    config_files: tuple             # the yaml files this config was read
                                    # from, nearest first (an extends chain)

    @property
    def results_dir(self) -> Path:
        return Path("experiments/results") / self.name


def _link_card(card: Optional[dict]) -> Optional[LinkCard]:
    if card is None:
        return None
    return LinkCard(latency_us=card["latency_us"],
                    bits_per_us=card["bits_per_us"],
                    transfer_overhead_us=card.get("transfer_overhead_us"))


def _buffer_size(value, key: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"{key} must be a positive round count or null, got {value!r}")
    return value


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
    decoder = raw["decoder"]
    engine = decoder["engine"]
    links = {link_path: _link_card(raw["links"].get(link_path))
             for link_path in LINK_PATHS}
    raw_trace = raw.get("trace", "off")
    if raw_trace is False:
        raw_trace = "off"     # yaml 1.1 reads a bare `off` as boolean False
    sweep = tuple(
        SweepBlock(physical_error_probabilities=tuple(block["physical_error_probability"]),
                   algorithm_latencies_us=tuple(block["algorithm_latency_us"]),
                   round_periods_us=tuple(block["round_period_us"]),
                   shots=block["shots"])
        for block in raw["sweep"])
    return ExperimentConfig(
        name=path.stem,
        mode=_require(raw["mode"], MODES, "mode"),
        code_task=raw["code_task"],
        distance=raw["distance"],
        rounds_per_shot=raw["rounds_per_shot"],
        windowing=WindowingCard(
            scheme=_require(windowing["scheme"], SCHEMES, "windowing.scheme"),
            commit_rounds=windowing["commit_rounds"],
            buffer_rounds=windowing["buffer_rounds"]),
        sweep=sweep,
        controller=ControllerCard(
            t_binary_availability_us=controller["t_binary_availability_us"],
            t_pack_us=controller["t_pack_us"]),
        links=links,
        buffers=BuffersCard(
            buffer_0_size=_buffer_size(
                buffers["buffer_0_size"], "buffers.buffer_0_size"),
            buffer_1_size=_buffer_size(
                buffers["buffer_1_size"], "buffers.buffer_1_size"),
            packing_workspace_size=_buffer_size(
                buffers["packing_workspace_size"],
                "buffers.packing_workspace_size")),
        decoder=DecoderCard(
            units=decoder["units"],
            unit_buffer_size=_buffer_size(
                decoder["unit_buffer_size"], "decoder.unit_buffer_size"),
            engine=EngineCard(
                frequency_mhz=engine["frequency_mhz"],
                fetch_cycles_per_round=engine["fetch_cycles_per_round"],
                release_cycles_per_job=engine["release_cycles_per_job"])),
        trace=_require(raw_trace, TRACE_MODES, "trace"),
        trace_io=_require(raw.get("trace_io", False), (True, False), "trace_io"),
        idle_policy=_require(raw.get("idle_policy", "separate_decode_jobs"),
                             IDLE_POLICIES, "idle_policy"),
        pauli_frame_commit_us=raw["pauli_frame"]["commit_us"],
        config_files=config_files)
