"""Core non-link timing and integer-tick conversions."""
from __future__ import annotations

from dataclasses import dataclass

TICKS_PER_US = 1_000_000


def us(microseconds: float) -> int:
    """Convert microseconds to integer ticks."""
    return int(round(microseconds * TICKS_PER_US))


def microseconds(ticks: int) -> float:
    """Convert integer ticks to microseconds, three decimals."""
    return round(ticks / TICKS_PER_US, 3)


def fmt(ticks: int) -> str:
    """Format ticks as microseconds for readability in logs."""
    return f"{ticks / TICKS_PER_US:7.3f} us"


@dataclass(frozen=True)
class TimingConfig:
    """Run-wide non-link timing quantities, expressed in microseconds."""

    round_us: float = 1.1          # QEC round period (one syndrome-extraction cycle)
    # Analog readout acquisition / discrimination, represented by
    # its latency and the classified bits it produces (not an ADC waveform).
    measurement_signal_to_classical_bits_us: float = 0.0
    t_pack_us: float = 0.0         # controller packet assembly before WBD send
    # Online sequencer/branch plus waveform-command generation.
    # CQ transport remains a separate link latency.
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
            if value > 0 and us(value) == 0:
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
        return us(values[name])
