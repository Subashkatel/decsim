"""The L5 numbers of one round store: occupancy over time, residence per round.

A listener on the store's round_stored and round_released events. It
integrates the occupancy as a step function of time and sums the
residence of every round that left, so the sample-path form of Little's
law can be checked exactly: over a run in which every stored round is
released, the occupancy integral equals the residence sum (Stidham
1974, "A last word on L = lambda W"; Ciw's and gem5's queues as the
oracles). Built only
when the observation section asks; the store runs without it.
"""

from typing import Optional


class RoundStoreOccupancy:
    """Occupancy integral, peak, residence sum and arrivals of one store."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.stored_tick_by_key: dict = {}
        self.occupancy = _StepIntegral()
        self.residence_sum = 0
        self.arrivals = 0

    def round_stored(self, round_key, _packet) -> None:
        """One more round in the store from now on."""
        self.arrivals += 1
        self.stored_tick_by_key[round_key] = self.engine.now
        count = len(self.stored_tick_by_key)
        self.occupancy.observe(self.engine.now, count)

    def round_released(self, round_key) -> None:
        """One round fewer; its residence is the time it was stored."""
        stored_tick = self.stored_tick_by_key.pop(round_key)
        self.residence_sum += self.engine.now - stored_tick
        count = len(self.stored_tick_by_key)
        self.occupancy.observe(self.engine.now, count)

    @property
    def peak(self) -> int:
        """The most rounds stored at once."""
        return self.occupancy.peak

    @property
    def integral(self) -> int:
        """The occupancy integrated over the ticks since the first store."""
        return self.occupancy.area

    def time_average(self) -> float:
        """The mean occupancy over the ticks since the first store."""
        return self.occupancy.time_average()


class _StepIntegral:
    """The integral of a step function sampled at every change."""

    def __init__(self) -> None:
        self.first_tick: Optional[int] = None
        self.last_tick: Optional[int] = None
        self.last_value = 0
        self.area = 0
        self.peak = 0

    def observe(self, tick: int, value: int) -> None:
        """The value holds from this tick until the next observation."""
        if self.first_tick is None:
            self.first_tick = tick
            self.last_tick = tick
        assert tick >= self.last_tick, "occupancy observations run backwards"
        self.area += self.last_value * (tick - self.last_tick)
        self.last_tick = tick
        self.last_value = value
        self.peak = max(self.peak, value)

    def time_average(self) -> float:
        """The area over the span; zero before the first observation."""
        if self.first_tick is None:
            return 0.0
        span = self.last_tick - self.first_tick
        if span == 0:
            return 0.0
        return self.area / span
