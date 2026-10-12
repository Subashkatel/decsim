"""The region of interest: the run's statistics restart after a warm-up.

A long run starts with empty queues, so its first rounds measure the
fill rather than the steady state. gem5 marks the region it measures
with m5.stats.reset() at the start and m5.stats.dump() at the end
(src/python/m5/stats/__init__.py:392-450; WorkBeginExitHandler and
WorkEndExitHandler, src/python/gem5/simulate/exit_handler.py:291-315).
Here the region begins when the QPU reads out the first round past the
warm-up, of any operation, and runs to the end of the run: the registry
resets (statistics/registry.py) and with it the backlog peak.

What the region covers is the registry's statistics and what is read
from them: the statistic rows, the queue peaks, the busy fractions and
offered loads over the rounds read out since, and the backlog peak.
The windows' records are kept whole: the latency samples follow
observation.first_recorded_round, and the logical outcome is the
shot's.
"""

from typing import Any

import decsim.records.rounds as round_records
import decsim.statistics.registry as statistics_registry


class RegionOfInterest:
    """Resets the run's statistics once, at the first round of the region."""

    def __init__(
        self, registry: statistics_registry.Registry, warmup_rounds: int
    ) -> None:
        self.registry = registry
        self.warmup_rounds = warmup_rounds
        self.has_begun = False

    def round_read_out(self, readout: round_records.QPUReadout) -> None:
        """A round read out; the first past the warm-up begins the region."""
        if self.has_begun:
            return
        if readout.round_index <= self.warmup_rounds:
            return
        self.has_begun = True
        self.registry.reset()

    def check_begun(self, rounds_read_out: Any) -> None:
        """Refuse a run that ended inside its warm-up.

        rounds_read_out is the last round of each operation read out.
        """
        if self.has_begun:
            return
        raise ValueError(
            f"observation.warmup_rounds is {self.warmup_rounds}, but the "
            f"run read out rounds only up to {rounds_read_out}; no region "
            "of interest began"
        )
