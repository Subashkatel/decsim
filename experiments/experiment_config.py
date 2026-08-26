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
SCHEMES = ("sliding", "parallel", "sandwich", "naive_online")
LINK_PATHS = ("qc", "c2b", "copy_out", "cwd", "wsd", "csd",
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
    units: int
    engine: EngineCard


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
    decoder: DecoderCard
    decoder_memory_rounds: Optional[int]   # per unit; None = unbounded
    pauli_frame_commit_us: float

    @property
    def results_dir(self) -> Path:
        return Path("experiments/results") / self.name


def _link_card(card: Optional[dict]) -> Optional[LinkCard]:
    if card is None:
        return None
    return LinkCard(latency_us=card["latency_us"],
                    bits_per_us=card["bits_per_us"],
                    transfer_overhead_us=card.get("transfer_overhead_us"))


def _require(value, allowed: tuple, key: str):
    if value not in allowed:
        raise ValueError(f"{key} must be one of {allowed}, got {value!r}")
    return value


def _raw_yaml(path: Path) -> dict:
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    base_name = raw.pop("extends", None)
    if base_name is None:
        return raw
    base = _raw_yaml(path.parent / base_name)
    base.update(raw)
    return base


def load_experiment(path) -> ExperimentConfig:
    path = Path(path)
    raw = _raw_yaml(path)
    windowing = raw["windowing"]
    controller = raw["controller"]
    decoder = raw["decoder"]
    engine = decoder["engine"]
    links = {link_path: _link_card(raw["links"].get(link_path))
             for link_path in LINK_PATHS}
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
        decoder=DecoderCard(
            units=decoder["units"],
            engine=EngineCard(
                frequency_mhz=engine["frequency_mhz"],
                fetch_cycles_per_round=engine["fetch_cycles_per_round"],
                release_cycles_per_job=engine["release_cycles_per_job"])),
        decoder_memory_rounds=raw["decoder_memory_rounds"],
        pauli_frame_commit_us=raw["pauli_frame"]["commit_us"])
