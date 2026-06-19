from __future__ import annotations

from dataclasses import dataclass

from .surface_code import generate_circuit

# ==================================================================================
# NOISE MODEL
# Physical noise for a memory / code-capacity experiment, carried as DATA ON THE
# CIRCUIT -- the stim/sinter convention: noise travels with each experiment's circuit,
# never in a global timing config (SimConfig stays purely timing/latency). This is the
# single place the project's noise conventions (incl. the phenomenological 1.5x factor,
# previously copy-pasted three different ways across experiments) are written down.
# ==================================================================================


@dataclass(frozen=True)
class NoiseModel:
    """The physical error rates of one memory experiment, as the four stim circuit-level
    noise channels. Build the noisy circuit with ``.circuit(code_task, distance=d, rounds=R)``;
    sweep a parameter by building one NoiseModel (and one circuit) per point.

    Channels (mirror ``stimcircuits.generate_circuit`` / stim's circuit-level model):
      ``p_data``     -- before_round_data_depolarization: DEPOLARIZE1 on every data qubit
                        at the start of each stabilizer round.
      ``p_meas``     -- before_measure_flip_probability: the measurement readout flips.
      ``p_clifford`` -- after_clifford_depolarization: DEPOLARIZE1/DEPOLARIZE2 after each
                        single/two-qubit gate.
      ``p_reset``    -- after_reset_flip_probability: the state-preparation (reset) flips.

    Prefer the named constructors over raw fields so the convention is centralized:
      ``NoiseModel.circuit_level(p)``    -- all four channels at the same rate p (the standard
                                            circuit-level model; what stim.Circuit.generated(p,p,p,p)
                                            produces).
      ``NoiseModel.phenomenological(p)`` -- data depolarization 1.5*p and measurement flip p, no
                                            gate or reset noise. The SINGLE-BASIS memory convention:
                                            a depolarizing data error is detected by the one-basis
                                            decoding graph with probability 2/3, so 1.5*p delivers
                                            an effective data-error rate p ((2/3)(1.5p)=p). Using a
                                            bare p here would be only ~0.67p effective -- the footgun
                                            this constructor removes.

    All defaults are 0.0, so ``NoiseModel()`` is noiseless (no detection events) and leaves
    timing-only runs unchanged.
    """
    p_data: float = 0.0
    p_meas: float = 0.0
    p_clifford: float = 0.0
    p_reset: float = 0.0

    @classmethod
    def circuit_level(cls, p: float) -> "NoiseModel":
        """Standard circuit-level model: every noise channel at the same physical rate p."""
        return cls(p_data=p, p_meas=p, p_clifford=p, p_reset=p)

    @classmethod
    def phenomenological(cls, p: float) -> "NoiseModel":
        """Single-basis memory: data depolarization 1.5*p (effective data-error rate p on the
        one-basis decoding graph) and measurement flip p; no gate or reset noise."""
        return cls(p_data=1.5 * p, p_meas=p)

    def circuit(self, code_task: str = "surface_code:rotated_memory_z", *,
                distance: int, rounds: int, **kw):
        """Build the noisy ``stim.Circuit`` for this experiment via the decsim-vendored
        generator. Extra keyword arguments pass straight through to ``generate_circuit``
        (e.g. ``x_distance``/``z_distance``, ``exclude_other_basis_detectors``)."""
        return generate_circuit(
            code_task, distance=distance, rounds=rounds,
            after_clifford_depolarization=self.p_clifford,
            before_round_data_depolarization=self.p_data,
            before_measure_flip_probability=self.p_meas,
            after_reset_flip_probability=self.p_reset,
            **kw)
