"""The two questions the machine asks a burst detector about a window.

The switching policy escalates every window a published flag meets,
and the strong window model may raise the flagged region's priors. Both
rows keep one FlagLog per operation and answer the same way.
"""

from collections.abc import Mapping
from typing import Optional

import decsim.burst_detectors.flag_log as flag_log
import decsim.engine as engine_module
import decsim.records.windows as window_records


class BurstWindows:
    """A row's answers about a window, read off its operations' flags.

    operations maps an operation id to that operation's counters or
    chart bank: each keeps its flags in a FlagLog named flags and names
    a flag's region with region_of(episode).
    """

    def __init__(
        self, operations: Mapping, engine: engine_module.Engine
    ) -> None:
        self.operations = operations
        self.engine = engine

    def is_burst_window(self, window: window_records.Window) -> bool:
        """Whether a flag published by now meets the window's rounds."""
        episode = self._published_episode(window)
        return episode is not None

    def with_burst_priors(self, window: window_records.Window, model):
        """The model with the meeting flag's region raised; unmet, unchanged."""
        episode = self._published_episode(window)
        if episode is None:
            return model
        operation = self.operations[window.operation_id]
        region = operation.region_of(episode)
        return region.raised(model)

    def _published_episode(
        self, window: window_records.Window
    ) -> Optional[flag_log.Episode]:
        """The flag published by now that meets the window, or None."""
        operation = self.operations[window.operation_id]
        return operation.flags.episode_meeting(window, self.engine.now)
