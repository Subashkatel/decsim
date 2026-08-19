"""Noise presets for Stim-backed circuit generation."""

from __future__ import annotations

from dataclasses import dataclass

from .surface_code import _check_probability, generate_circuit


@dataclass(frozen=True)
class NoiseModel:
    """Physical error rates carried by each generated Stim circuit."""

    p_data: float = 0.0
    p_meas: float = 0.0
    p_clifford: float = 0.0
    p_reset: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("p_data", "p_meas", "p_clifford", "p_reset"):
            object.__setattr__(
                self,
                field_name,
                _check_probability(getattr(self, field_name), field_name),
            )

    @classmethod
    def circuit_level(cls, p: float) -> "NoiseModel":
        """Set data, measurement, Clifford, and reset noise to the same rate."""
        return cls(p_data=p, p_meas=p, p_clifford=p, p_reset=p)

    @classmethod
    def phenomenological(cls, p: float) -> "NoiseModel":
        """Use measurement noise p and data depolarization 1.5*p."""
        normalized_p = _check_probability(p, "phenomenological p")
        return cls(p_data=1.5 * normalized_p, p_meas=normalized_p)

    def circuit(self, code_task: str = "surface_code:rotated_memory_z", *,
                distance: int, rounds: int, **kwargs):
        """Build a noisy ``stim.Circuit`` with this model's channel rates."""
        return generate_circuit(
            code_task, distance=distance, rounds=rounds,
            after_clifford_depolarization=self.p_clifford,
            before_round_data_depolarization=self.p_data,
            before_measure_flip_probability=self.p_meas,
            after_reset_flip_probability=self.p_reset,
            **kwargs)
