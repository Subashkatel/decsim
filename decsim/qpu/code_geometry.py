"""The code cards: the numbers the simulator needs from a QEC code.

A card is a small frozen record, not a stabilizer code: decsim prices
decoder timing, so it takes a distance, window sizes, the graph size per
round and the syndrome bits per round, and never imports an upstream
tool. The rotated surface card follows Stim's
surface_code:rotated_memory_z (src/stim/gen/gen_surface_code.cc): d*d
data and d*d - 1 measure qubits, each measure qubit read once a round.
The bivariate-bicycle card follows Bravyi et al. (Nature 627, 778, 2024;
arXiv 2308.07915): n/2 X and n/2 Z checks a round, the [[144, 12, 12]]
gross code by default.
"""

import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class SurfaceCodeModel:
    """The code card of one rotated surface-code patch."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The surface card has no keys beyond the qpu's distance."""

        def build(
            self,
            distance: Optional[int],
            commit_rounds_override: Optional[int],
            buffer_rounds_override: Optional[int],
        ) -> "SurfaceCodeModel":
            """The card at the distance, three when None, and window sizes."""
            arguments = _card_arguments(
                distance, commit_rounds_override, buffer_rounds_override
            )
            return SurfaceCodeModel(**arguments)

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

    def spatial_nodes(self, patch_count: int) -> int:
        """Per-round graph size for a latency model: d*d per patch, plus a seam.

        A size knob, not the detector count: a patch has d*d - 1 detectors a
        round (Stim's bulk layer: 8, 24 and 48 at d=3, 5, 7). No shipped decoder
        row reads it; for a supplied latency model the difference is a scale
        factor and reaches no correction. The seam is the one line of d qubits a
        two-patch lattice-surgery merge adds (Horsman et al. 1111.4022 Sec.
        3.1); for more patches the paper gives no count.
        """
        node_count_per_patch = self.distance * self.distance
        seam_node_count = 0
        if patch_count > 1:
            seam_node_count = self.distance
        patch_node_count = patch_count * node_count_per_patch
        return patch_node_count + seam_node_count

    def syndrome_bits_per_round(self, patch_count: int) -> int:
        """Bits read out per round: the d*d - 1 stabilizers of every patch."""
        qubit_count = self.distance * self.distance
        stabilizer_count = qubit_count - 1
        return patch_count * stabilizer_count

    def data_bits_per_readout(self, patch_count: int) -> int:
        """Bits the final readout adds: the d*d data qubits of every patch."""
        qubit_count = self.distance * self.distance
        return patch_count * qubit_count


@dataclasses.dataclass(frozen=True)
class BivariateBicycleCodeModel:
    """The code card of one bivariate-bicycle CSS code.

    One round is a whole extraction cycle, n/2 X and n/2 Z checks; the
    detector error model owns the window rows. The distance comes from the
    sweep, as the surface card's does.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The card's own keys.

        qubit_count is n and logical_qubit_count is k in [[n, k, d]]. The
        defaults are the gross code [[144, 12, 12]] (2308.07915v2 lines
        180-184); its distance is twelve when the run names none.
        """

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

        def build(
            self,
            distance: Optional[int],
            commit_rounds_override: Optional[int],
            buffer_rounds_override: Optional[int],
        ) -> "BivariateBicycleCodeModel":
            """The card at the distance, twelve when None, and window sizes."""
            arguments = _card_arguments(
                distance, commit_rounds_override, buffer_rounds_override
            )
            return BivariateBicycleCodeModel(settings=self, **arguments)

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

    def spatial_nodes(self, patch_count: int) -> int:
        """Per-round graph size for a latency model: n per patch."""
        return patch_count * self.settings.qubit_count

    def syndrome_bits_per_round(self, patch_count: int) -> int:
        """Bits read out per round: the n X-plus-Z checks of every patch."""
        return patch_count * self.settings.qubit_count

    def data_bits_per_readout(self, patch_count: int) -> int:
        """Bits the final readout adds: the n data qubits of every patch."""
        return patch_count * self.settings.qubit_count


def _card_arguments(
    distance: Optional[int],
    commit_rounds_override: Optional[int],
    buffer_rounds_override: Optional[int],
) -> dict:
    """A card's constructor keywords; a distance of None is the card's own."""
    arguments = {
        "commit_rounds_override": commit_rounds_override,
        "buffer_rounds_override": buffer_rounds_override,
    }
    if distance is not None:
        arguments["distance"] = distance
    return arguments


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
