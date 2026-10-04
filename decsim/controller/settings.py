"""The controller's settings."""

import dataclasses
from typing import Optional

import decsim.config as config


@dataclasses.dataclass(frozen=True)
class ControllerSettings:
    """The controller's costs, in cycles of its clock.

    Readout to bits and packing are charged per round on the way in,
    decision to pulse per decision on the way out. Zero means the work sits
    inside the round period, as Google's 921 ns cycle holds its 500 ns
    measurement (2207.06431). Points: 40 ns in-FPGA discrimination (Fermilab
    2406.18807); 20 ns to compute a syndrome (Yang 2605.04892); 125 ns at
    USTC (2110.07965); 155 ns root to leaf (Liu et al. 2603.16203).

    decision_to_pulse is the control processor's issue pipeline: 8 cycles
    traced on QubiC's core (Fruitwala 2404.15260 Sec. III and IV), with
    gem5 MinorCPU stage delays where the paper is silent (1 latch, 1
    compare, 1 jump redirect, 3 fetch to execute, 1 pulse register, 1
    strobe). The taken branch is the cost because both paths are padded to
    it (Caune 2410.05202), and one core per qubit runs the same branch, so
    it does not grow with the patch. QICK measures 36 clocks on its deeper
    tProcessor (2110.00557).

    packing_rounds_in_flight bounds the rounds in the packing stage
    (round_assembly.RoundsInFlight); None is unbounded. clock is the domain
    all three costs are charged on, None the machine's; a zero cost is not
    rounded up to an edge.
    """

    clock: Optional[config.Clock] = None
    readout_to_bits_cycles: int = 0
    packing_cycles_per_round: int = 0
    decision_to_pulse_cycles: int = 0
    packing_rounds_in_flight: Optional[int] = None

    def __post_init__(self) -> None:
        config.check_cycles(
            "controller.readout_to_bits_cycles", self.readout_to_bits_cycles
        )
        config.check_cycles(
            "controller.packing_cycles_per_round",
            self.packing_cycles_per_round,
        )
        config.check_cycles(
            "controller.decision_to_pulse_cycles",
            self.decision_to_pulse_cycles,
        )
        self._check_rounds_in_flight()

    def _check_rounds_in_flight(self) -> None:
        """Refuse a bound that is not a whole count of at least one round.

        A round enters whole, and a bound below one admits none, so every round
        would wait for ever.
        """
        bound = self.packing_rounds_in_flight
        if bound is None:
            return
        if config.is_whole_count(bound):
            return
        raise ValueError(
            "controller.packing_rounds_in_flight must be a whole count of "
            f"rounds, at least one, or None for no bound (got {bound!r})"
        )
