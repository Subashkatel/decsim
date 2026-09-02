"""Noise presets for circuits built with the vendored generator.

A NoiseModel holds the four physical error rates the generator takes
(after a Clifford, before a round on data, before a measurement, after a
reset) and builds a stim.Circuit with them. Nothing in decsim uses it.
"""

import dataclasses

import decsim.qpu.stimcircuits.surface_code as surface_code


@dataclasses.dataclass(frozen=True)
class NoiseModel:
    """The four physical error rates of one generated circuit."""

    p_data: float = 0.0
    p_meas: float = 0.0
    p_clifford: float = 0.0
    p_reset: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("p_data", "p_meas", "p_clifford", "p_reset"):
            value = getattr(self, field_name)
            checked = surface_code._check_probability(value, field_name)
            object.__setattr__(self, field_name, checked)

    @classmethod
    def circuit_level(cls, p: float) -> "NoiseModel":
        """The same rate on data, measurement, Clifford and reset noise."""
        return cls(p_data=p, p_meas=p, p_clifford=p, p_reset=p)

    @classmethod
    def phenomenological(cls, p: float) -> "NoiseModel":
        """Measurement noise p and data depolarization 1.5 p."""
        checked = surface_code._check_probability(p, "phenomenological p")
        data_rate = 1.5 * checked
        return cls(p_data=data_rate, p_meas=checked)

    def circuit(
        self,
        code_task: str = "surface_code:rotated_memory_z",
        *,
        distance: int,
        rounds: int,
        **generator_arguments,
    ):
        """A noisy stim.Circuit with this model's rates."""
        return surface_code.generate_circuit(
            code_task,
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=self.p_clifford,
            before_round_data_depolarization=self.p_data,
            before_measure_flip_probability=self.p_meas,
            after_reset_flip_probability=self.p_reset,
            **generator_arguments,
        )
