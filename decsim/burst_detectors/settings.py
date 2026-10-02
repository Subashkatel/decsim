"""The burst_detector section: the row that watches for bursts, and its keys.

A burst of errors raises the detection rate of the stabilisers it
covers for hundreds of rounds (Google 2408.13687 lines 386-391 and
2101-2119). A row of BURST_DETECTORS reads each round's bulk detection
events as they are formed; its class is the only place its keys are
written (sinter's BUILT_IN_DECODERS shape).
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.burst_detectors.event_count.detector as event_count
import decsim.config as config
import decsim.tables as tables
from decsim.burst_detectors.masked_regional_cusum import (
    detector as masked_regional_cusum,
)

# burst_detector.kind names one of these rows: none builds no detector,
# event_count is the simple count baseline, masked_regional_cusum a
# CUSUM bank over regions of the checks.
BURST_DETECTORS = {
    "none": None,
    "event_count": event_count.EventCountBurstDetector,
    "masked_regional_cusum": (
        masked_regional_cusum.MaskedRegionalCusumBurstDetector
    ),
}


@dataclasses.dataclass(frozen=True)
class BurstDetectorSettings:
    """The yaml's `burst_detector` section, which names one row.

    The row none, the default, builds no detector and takes no keys, so
    a run without the section is the machine without a detector.
    """

    kind: str = "none"
    row_settings: Optional[object] = None

    @classmethod
    def from_yaml(
        cls, section: Mapping, clocks: config.ClockSettings
    ) -> "BurstDetectorSettings":
        """The kind and the keys its row declares."""
        kind = section.get("kind", "none")
        row = tables.row(BURST_DETECTORS, "burst_detector.kind", kind)
        row_settings = tables.row_settings(
            row, "burst_detector", section, ("kind",), clocks
        )
        return cls(kind=kind, row_settings=row_settings)
