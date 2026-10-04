"""The controller's settings.

The controller charges three per-round costs and bounds its packing
workspace.
"""

import dataclasses
from typing import Optional

import decsim.config as config


@dataclasses.dataclass(frozen=True)
class ControllerSettings:
    """The controller's costs, in cycles of its clock.

    The three costs are charged per round on the way in (readout to bits,
    packing) and per decision on the way out (decision to pulse); zero
    means the work sits inside the round period, as Google's 921 ns cycle
    holds its 500 ns measurement (2207.06431). Points: 40 ns in-FPGA
    discrimination (Fermilab 2406.18807); 20 ns to compute a syndrome
    from the bit strings (Yang 2605.04892); 125 ns at USTC (2110.07965),
    155 ns root to leaf in Liu et al. (2603.16203). decision_to_pulse
    is the control processor's issue pipeline, the decision at the core
    to the pulse trigger: 8 cycles traced on QubiC's core (Fruitwala
    2404.15260 Sec. III and IV) with gem5's MinorCPU stage delays where
    the paper is silent: 1 the result latched with its ready signal, 1
    the compare, 1 the taken jump's redirect, 3 the target fetched,
    decoded and reaching execute, 1 the pulse register written, 1 the
    strobe on the next edge. The taken branch is the cost because both
    paths are padded to it (Caune 2410.05202: the qubit idles for the
    gate's duration when the result is 0), and one core per qubit runs
    the same branch on the broadcast bits, so the count does not grow
    with the patch. QICK measures 16 clocks for the conditional
    evaluation and the jump and 20 for the next pulse on its deeper
    tProcessor (2110.00557 lines 893-900), 36 in all.
    packing_rounds_in_flight bounds the rounds in flight through the
    packing stage at once, each from its emission, in emission order,
    until the windows hear of it (round_assembly.RoundsInFlight); None is
    unbounded.
    clock is the domain all three cycle counts are charged on; a cost of
    zero cycles is uncharged rather than rounded up to the next edge.
    clock None is the machine's clock.
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
        """The packing stage's bound is a whole count of rounds, or None.

        A round enters the stage whole, so the bound counts whole rounds,
        and a bound below one admits no round, so every round would wait
        for ever. gem5's integer parameters refuse a value outside
        their range where the configuration is read
        (src/python/m5/params/param_types.py:230-235, CheckedInt._check).
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
