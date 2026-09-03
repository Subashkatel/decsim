"""The tick and the run-wide timing that no link owns.

One microsecond is TICKS_PER_MICROSECOND ticks; every duration in the
machine is an integer count of them. TimingConfig holds the round period
and the controller's three per-round costs.
"""

import dataclasses
import math

TICKS_PER_MICROSECOND = 1_000_000


def microseconds_to_ticks(microseconds: float) -> int:
    """Round a duration in microseconds to whole ticks."""
    scaled = microseconds * TICKS_PER_MICROSECOND
    rounded = round(scaled)
    return int(rounded)


def microseconds(ticks: int) -> float:
    """A tick count as microseconds, to three decimals."""
    exact = ticks / TICKS_PER_MICROSECOND
    return round(exact, 3)


def format_ticks(ticks: int) -> str:
    """A tick count as the microsecond stamp every log line carries."""
    return f"{ticks / TICKS_PER_MICROSECOND:7.3f} us"


@dataclasses.dataclass(frozen=True)
class TimingConfig:
    """Run-wide timing outside the links, in microseconds.

    The round period is one syndrome extraction cycle. Published cadences:
    Google 921 ns (2207.06431) and 1.1 us (2408.13687), Krinner 1.1 us
    (2112.03708), Yang 1.25 us (2605.04892), USTC 3.9 us effective
    (2110.07965). The three controller costs are charged per round on
    the way in and per decision on the way out; zero means the work sits
    inside the round period, as Google's 921 ns cycle holds its 500 ns
    measurement. Points: 40 ns in-FPGA discrimination (Fermilab
    2406.18807); 20 ns to compute a syndrome from the bit strings (Yang
    2605.04892); a 42 ns conditional jump and a 52 ns next pulse on QICK
    (2110.00557 Table II), 125 ns at USTC (2110.07965), 155 ns root to
    leaf in Liu et al. (2603.16203).
    """

    round_period_microseconds: float = 1.1
    readout_to_bits_microseconds: float = 0.0
    packing_microseconds_per_round: float = 0.0
    decision_to_pulse_microseconds: float = 0.0

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            _check_duration(field.name, value)

    def ticks(self, name: str) -> int:
        """One controller cost by name, in ticks."""
        values = {
            "readout_to_bits": self.readout_to_bits_microseconds,
            "packing": self.packing_microseconds_per_round,
            "decision_to_pulse": self.decision_to_pulse_microseconds,
        }
        return microseconds_to_ticks(values[name])


def _check_duration(name: str, value: float) -> None:
    """Refuse a duration the yaml or a front call cannot mean."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    ticks = microseconds_to_ticks(value)
    if value > 0 and ticks == 0:
        raise ValueError(f"{name} is positive but rounds to zero ticks")
