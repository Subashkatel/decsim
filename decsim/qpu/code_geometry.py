"""The code cards: the numbers the simulator needs from a QEC code.

A card is a small frozen record, not a stabilizer code. The simulator
prices decoder timing, so all it takes from a code is its distance, its
window sizes, the size of the decoding graph per round, and the syndrome
bits per round. The numbers can be set by hand or copied from an upstream
tool's output; decsim never imports such a tool.

The rotated surface-code card follows Stim's generated
``surface_code:rotated_memory_z`` circuit (Stim, src/stim/gen/
gen_surface_code.cc): a distance-d patch has d*d data qubits and d*d - 1
measure qubits, and every round reads out every measure qubit once. The
bivariate-bicycle card follows Bravyi et al., High-threshold and
low-overhead fault-tolerant quantum memory (Nature 627, 778, 2024; arXiv
2308.07915): a [[n, k, d]] code whose round measures n/2 X checks and n/2
Z checks, the [[144, 12, 12]] gross code by default.
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class CodeModel(Protocol):
    """A code card: window sizes, cycle length, graph size, syndrome width."""

    name: str
    distance: int

    def rounds_per_logical_cycle(self) -> int:
        """Syndrome rounds per logical cycle."""

    def round_period_us(self) -> Optional[float]:
        """The card's own round period, or None for the run's cadence."""

    def commit_rounds(self) -> int:
        """Rounds committed per decode window."""

    def buffer_rounds(self) -> int:
        """Look-ahead rounds per decode window."""

    def spatial_nodes(self, num_patches: int) -> int:
        """The per-round graph size a latency model prices this card at."""

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Syndrome bits one round of this many patches produces."""


@dataclasses.dataclass(frozen=True)
class SurfaceCodeModel:
    """Timing and sizing card of one rotated surface-code patch."""

    distance: int = 3
    # None: the run's cadence.
    round_microseconds: Optional[float] = None
    # None: the distance.
    commit_rounds_override: Optional[int] = None
    buffer_rounds_override: Optional[int] = None

    def __post_init__(self) -> None:
        round_microseconds = _optional_float(self.round_microseconds)
        object.__setattr__(self, "round_microseconds", round_microseconds)

    @property
    def name(self) -> str:
        """The routing and readout identity of this card."""
        return f"rotated surface code (d={self.distance})"

    def rounds_per_logical_cycle(self) -> int:
        """Syndrome rounds per logical cycle: the distance."""
        return self.distance

    def round_period_us(self) -> Optional[float]:
        """The card's own round period, or None for the run's cadence."""
        return self.round_microseconds

    def commit_rounds(self) -> int:
        """Rounds committed per decode window; the distance by default."""
        if self.commit_rounds_override is not None:
            return self.commit_rounds_override
        return self.distance

    def buffer_rounds(self) -> int:
        """Look-ahead rounds per decode window; the distance by default."""
        if self.buffer_rounds_override is not None:
            return self.buffer_rounds_override
        return self.distance

    def spatial_nodes(self, num_patches: int) -> int:
        """Per-round graph size for a latency model: d*d per patch, plus a seam.

        A size knob, not the detector count: a rotated patch contributes
        d*d - 1 detector nodes per round (Stim's bulk layer, read off
        stim.Circuit.generated at d=3, 5 and 7 as 8, 24 and 48), one
        fewer per patch than this returns. No shipped decoder row reads
        the number; the reader is a caller-supplied latency function
        (decoders/decoders.py FunctionLatencyDecoder), where the
        difference is a scale factor and reaches no correction. The seam
        strip is a heuristic for a multi-patch operation.
        """
        node_count_per_patch = self.distance * self.distance
        seam_node_count = 0
        if num_patches > 1:
            seam_node_count = self.distance
        patch_node_count = num_patches * node_count_per_patch
        return patch_node_count + seam_node_count

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Bits read out per round: the d*d - 1 stabilizers of every patch."""
        qubit_count = self.distance * self.distance
        stabilizer_count = qubit_count - 1
        return num_patches * stabilizer_count


@dataclasses.dataclass(frozen=True)
class BivariateBicycleCodeModel:
    """Timing and sizing card of one bivariate-bicycle CSS code.

    One modeled round is one complete extraction cycle: n/2 X checks and
    n/2 Z checks. The detector error model owns the exact window-local
    detector rows. The qubit counts are the card's own keys (Settings);
    the distance comes from the sweep, as the surface card's does.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The card's own keys: n and k of the [[n, k, d]] code."""

        qubit_count: int = 144
        logical_qubit_count: int = 12

        def __post_init__(self) -> None:
            _require_positive_int(self.qubit_count, "qubit_count")
            _require_positive_int(
                self.logical_qubit_count, "logical_qubit_count"
            )
            if self.qubit_count % 2:
                raise ValueError(
                    f"qubit_count must be even; got {self.qubit_count!r}"
                )
            if self.logical_qubit_count > self.qubit_count:
                raise ValueError(
                    "logical_qubit_count must not exceed qubit_count; got "
                    f"logical_qubit_count={self.logical_qubit_count!r}, "
                    f"qubit_count={self.qubit_count!r}"
                )

        @classmethod
        def from_yaml(
            cls, section: Mapping
        ) -> "BivariateBicycleCodeModel.Settings":
            """The qpu section's qubit counts; absent is the gross code's."""
            return cls(**section)

    settings: Settings = dataclasses.field(default_factory=Settings)
    distance: int = 12
    # None: the run's cadence.
    round_microseconds: Optional[float] = None
    commit_rounds_override: Optional[int] = None
    buffer_rounds_override: Optional[int] = None

    def __post_init__(self) -> None:
        _require_positive_int(self.distance, "distance")
        qubit_count = self.settings.qubit_count
        if self.distance > qubit_count:
            raise ValueError(
                "distance must not exceed qubit_count; got "
                f"distance={self.distance!r}, qubit_count={qubit_count!r}"
            )
        _check_window_overrides(
            self.commit_rounds_override, self.buffer_rounds_override
        )
        round_microseconds = _optional_float(self.round_microseconds)
        object.__setattr__(self, "round_microseconds", round_microseconds)

    @property
    def name(self) -> str:
        """The routing and readout identity of this card."""
        qubit_count = self.settings.qubit_count
        logical_qubit_count = self.settings.logical_qubit_count
        return (
            f"bivariate-bicycle code [[{qubit_count},"
            f"{logical_qubit_count},{self.distance}]]"
        )

    def rounds_per_logical_cycle(self) -> int:
        """Syndrome rounds per logical cycle: the distance."""
        return self.distance

    def round_period_us(self) -> Optional[float]:
        """The card's own round period, or None for the run's cadence."""
        return self.round_microseconds

    def commit_rounds(self) -> int:
        """Rounds committed per decode window; the distance by default."""
        if self.commit_rounds_override is None:
            return self.distance
        return self.commit_rounds_override

    def buffer_rounds(self) -> int:
        """Look-ahead rounds per decode window; none by default."""
        if self.buffer_rounds_override is None:
            return 0
        return self.buffer_rounds_override

    def spatial_nodes(self, num_patches: int) -> int:
        """Per-round graph size for a latency model: n per patch."""
        return num_patches * self.settings.qubit_count

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Bits read out per round: the n X-plus-Z checks of every patch."""
        return num_patches * self.settings.qubit_count


def _check_window_overrides(
    commit_rounds_override: Optional[int],
    buffer_rounds_override: Optional[int],
) -> None:
    """A commit override is positive and a buffer override not negative."""
    if commit_rounds_override is not None:
        _require_positive_int(commit_rounds_override, "commit_rounds_override")
    if buffer_rounds_override is None:
        return
    if buffer_rounds_override < 0:
        raise ValueError("buffer_rounds_override must be nonnegative")


def _require_positive_int(value, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be positive; got {value!r}")


def _optional_float(value) -> Optional[float]:
    if value is None:
        return None
    return float(value)
