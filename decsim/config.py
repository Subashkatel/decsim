"""Core non-link timing and integer-tick conversions."""
from __future__ import annotations

from dataclasses import dataclass

TICKS_PER_MICROSECOND = 1_000_000


def microseconds_to_ticks(microseconds: float) -> int:
    """Convert microseconds to integer ticks."""
    return int(round(microseconds * TICKS_PER_MICROSECOND))


def microseconds(ticks: int) -> float:
    """Convert integer ticks to microseconds, three decimals."""
    return round(ticks / TICKS_PER_MICROSECOND, 3)


def format_ticks(ticks: int) -> str:
    """Format ticks as microseconds for readability in logs."""
    return f"{ticks / TICKS_PER_MICROSECOND:7.3f} us"


@dataclass(frozen=True)
class TimingConfig:
    """Run-wide non-link timing quantities, expressed in microseconds."""

    # QEC round period (one syndrome-extraction cycle). Published cadences:
    # Google 921 ns (2207.06431) and 1.1 us (2408.13687); Krinner 1.1 us
    # (2112.03708); Yang 1.25 us (2605.04892); USTC 3.9 us effective
    # (2110.07965).
    round_us: float = 1.1
    # Analog readout acquisition / discrimination, represented by its
    # latency and the classified bits it produces (not an ADC waveform).
    # Zero means the classification sits inside the round period, as
    # Google's 921 ns cycle holds its 500 ns measurement; a separate
    # point: 40 ns in-FPGA discrimination (Fermilab 2406.18807).
    measurement_signal_to_classical_bits_us: float = 0.0
    # Controller packet assembly, charged once per completed round.
    # Yang 2605.04892 compute the syndrome from the bit strings in 20 ns.
    t_pack_us: float = 0.0
    # Online sequencer/branch plus waveform-command generation; CQ
    # transport remains a separate link latency. Points: QICK conditional
    # jump 42 ns and next pulse 52 ns (2110.00557 Table II); USTC 125 ns
    # (2110.07965); Liu et al. root to leaf 155 ns (2603.16203).
    instruction_or_decision_to_analog_control_pulse_us: float = 0.0

    def __post_init__(self) -> None:
        import math
        for name, value in (
            ("round_us", self.round_us),
            ("measurement_signal_to_classical_bits_us",
             self.measurement_signal_to_classical_bits_us),
            ("t_pack_us", self.t_pack_us),
            ("instruction_or_decision_to_analog_control_pulse_us",
             self.instruction_or_decision_to_analog_control_pulse_us),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
            if value > 0 and microseconds_to_ticks(value) == 0:
                raise ValueError(f"{name} is positive but rounds to zero ticks")

    def ticks(self, name: str) -> int:
        """Return one named non-link timing quantity in integer ticks."""
        values = {
            "measurement_signal_to_classical_bits":
                self.measurement_signal_to_classical_bits_us,
            "t_pack": self.t_pack_us,
            "instruction_or_decision_to_analog_control_pulse":
                self.instruction_or_decision_to_analog_control_pulse_us,
        }
        return microseconds_to_ticks(values[name])
