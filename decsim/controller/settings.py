"""The controller's settings, and the idle policy it relays through.

The controller charges three per-round costs and bounds its packing
workspace; the idle policy says how an idle patch's rounds are charged.
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.config as config
import decsim.controller.syndrome_packing as syndrome_packing
import decsim.ports as ports


@dataclasses.dataclass(frozen=True)
class ControllerSettings:
    """The yaml's `controller` section, in microseconds.

    The three costs are charged per round on the way in (readout to bits,
    packing) and per decision on the way out (decision to pulse); zero
    means the work sits inside the round period, as Google's 921 ns cycle
    holds its 500 ns measurement (2207.06431). Points: 40 ns in-FPGA
    discrimination (Fermilab 2406.18807); 20 ns to compute a syndrome
    from the bit strings (Yang 2605.04892); a 42 ns conditional jump and
    a 52 ns next pulse on QICK (2110.00557 Table II), 125 ns at USTC
    (2110.07965), 155 ns root to leaf in Liu et al. (2603.16203).
    packing_rounds_in_flight bounds the packing stage's assembly
    workspace, the rounds in flight through the stage at once; None is
    unbounded. The packing policy is a Python-only knob.
    """

    readout_to_bits_microseconds: float = 0.0
    packing_microseconds_per_round: float = 0.0
    decision_to_pulse_microseconds: float = 0.0
    packing_rounds_in_flight: Optional[int] = None
    packing_policy: syndrome_packing.SyndromePackingPolicy = (
        syndrome_packing.SyndromePackingPolicy()
    )

    def __post_init__(self) -> None:
        config.check_duration(
            "readout_to_bits_microseconds", self.readout_to_bits_microseconds
        )
        config.check_duration(
            "packing_microseconds_per_round",
            self.packing_microseconds_per_round,
        )
        config.check_duration(
            "decision_to_pulse_microseconds",
            self.decision_to_pulse_microseconds,
        )

    @classmethod
    def from_yaml(
        cls, section: Mapping, clocks: config.ClockSettings
    ) -> "ControllerSettings":
        """The `controller` section: cycles of its clock, resolved once."""
        clock = section["clock"]
        readout_cycles = section["readout_to_bits_cycles"]
        packing_cycles = section["packing_cycles_per_round"]
        decision_cycles = section["decision_to_pulse_cycles"]
        readout_microseconds = clocks.microseconds(readout_cycles, clock)
        packing_microseconds = clocks.microseconds(packing_cycles, clock)
        decision_microseconds = clocks.microseconds(decision_cycles, clock)
        packing_rounds_in_flight = section.get("packing_rounds_in_flight")
        return cls(
            readout_to_bits_microseconds=readout_microseconds,
            packing_microseconds_per_round=packing_microseconds,
            decision_to_pulse_microseconds=decision_microseconds,
            packing_rounds_in_flight=packing_rounds_in_flight,
        )

    def readout_to_bits_ticks(self) -> int:
        """The readout classification cost in ticks."""
        return config.microseconds_to_ticks(self.readout_to_bits_microseconds)

    def packing_ticks(self) -> int:
        """The packet assembly cost per round in ticks."""
        return config.microseconds_to_ticks(self.packing_microseconds_per_round)

    def decision_to_pulse_ticks(self) -> int:
        """The decision to pulse cost in ticks."""
        return config.microseconds_to_ticks(self.decision_to_pulse_microseconds)


@dataclasses.dataclass(frozen=True)
class IdlePolicySettings:
    """The yaml's `idle_policy` value: how an idle patch's rounds are charged.

    Table rows (decsim/machine.py): separate_decode_jobs, ignore,
    extend_stream. Idle rounds are decoder workload in every reference
    system (SWIPER ISCA 2025, XQsim, Terhal backlog), so
    separate_decode_jobs is the default; ignore is the optimistic card
    for active-path latency studies; extend_stream folds them into a
    live stream. A Python-built policy is used as it is.
    """

    kind: str = "separate_decode_jobs"
    policy: Optional[ports.IdlePolicy] = None
