"""The burst_detector section: the row that watches for bursts, and its keys.

A burst of errors raises the detection rate of the stabilisers it
covers for hundreds of rounds (Google 2408.13687 lines 386-391 and
2101-2119). A row of BURST_DETECTORS reads each round's bulk detection
events as they are formed; its class is the only place its keys are
written (sinter's BUILT_IN_DECODERS shape). The row's record rides on
the switching slot (escalation/settings.py), the only slot that sends
a flagged window to the strong tier, and the catch deadline on the
observation settings, since only the shot's measurement reads it.
"""

from collections.abc import Mapping

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


def detector_from_yaml(section: Mapping, clocks: config.ClockSettings):
    """The detector row's record the section names; None for the row none.

    The row none, the default, takes no keys, so a run without the
    section is the machine without a detector.
    """
    kind = section.get("kind", "none")
    row = tables.row(BURST_DETECTORS, "burst_detector.kind", kind)
    return tables.row_settings(
        row, "burst_detector", section, ("kind",), clocks
    )


def refuse_a_detector_without_switching(section: Mapping, switching) -> None:
    """A flagged window goes to the strong tier, which only switching has."""
    kind = section.get("kind", "none")
    if kind == "none" or switching is not None:
        return
    raise ValueError(
        f"burst_detector.kind {kind} sends a burst's windows to the "
        "strong decoder, which only escalation.kind switching does; "
        "write burst_detector: {kind: none}, or escalate with switching"
    )
