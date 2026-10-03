"""Ordinary Stim fragments for a repeated physical measurement program.

A final round replaces a bulk round, so a producer may combine its last
check readout and destructive data measurement in one physical instruction.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import stim


@dataclass(frozen=True)
class RepeatedStimCircuit:
    """One preparation, repeatable protection, and actual final readout.

    The first and repeated fragments leave the data live. The final
    fragment contains a complete last round and destructive readout;
    single_round contains preparation and readout for a one-round run.
    All fragments use the same physical qubit and logical observable ids.
    A declared period binds duration-dependent physics to the QPU cadence,
    and the live source checks it against the run's period before it
    executes. None leaves cadence independent of the circuit's noise
    probabilities and skips that check: every fragment, preparation and
    the terminal readout included, is then charged one round period.
    """

    # Stim's circuits have no hash; equal records hold equal circuits, so
    # the hash leaves them out and still agrees with equality.
    first_round: stim.Circuit = field(hash=False)
    repeated_round: stim.Circuit = field(hash=False)
    final_round: stim.Circuit = field(hash=False)
    single_round: stim.Circuit = field(hash=False)
    round_period_microseconds: Optional[float] = None

    def __post_init__(self) -> None:
        period = self.round_period_microseconds
        if period is not None:
            if not math.isfinite(period) or period <= 0:
                raise ValueError(
                    "round_period_microseconds must be finite and positive"
                )
        for name in (
            "first_round",
            "repeated_round",
            "final_round",
            "single_round",
        ):
            circuit = getattr(self, name)
            copied = circuit.copy()
            object.__setattr__(self, name, copied)

    def round_circuit(self, round_index: int, is_final: bool) -> stim.Circuit:
        """The physical instructions of the requested one-based round."""
        name = _round_name(round_index, is_final)
        circuit = getattr(self, name)
        return circuit.copy()

    def assemble(self, round_count: int) -> tuple[stim.Circuit, dict[int, int]]:
        """Describe a complete physical history and its measurement schedule."""
        circuit = stim.Circuit()
        measurement_rounds = {}
        after_last_round = round_count + 1
        for round_index in range(1, after_last_round):
            is_final = round_index == round_count
            fragment = self.round_circuit(round_index, is_final)
            first_measurement = circuit.num_measurements
            circuit += fragment
            after_last_measurement = circuit.num_measurements
            for measurement in range(first_measurement, after_last_measurement):
                measurement_rounds[measurement] = round_index
        return circuit, measurement_rounds


def _round_name(round_index: int, is_final: bool) -> str:
    if round_index == 1:
        if is_final:
            return "single_round"
        return "first_round"
    if is_final:
        return "final_round"
    return "repeated_round"
