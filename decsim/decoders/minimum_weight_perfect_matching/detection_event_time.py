"""Correlated matching's time from a window's detection events: a fitted law.

Sparse Blossom's time grows with the detection events it matches: about
linearly below threshold, and faster than linearly in dense noise
(Higgott and Gidney 2303.15933 Sec. 5 and Fig. 12, O(n^1.32) above
threshold). DetectionEventTime prices one decode by such a law, fitted
to decsim's own correlated matching timed on decsim's SD6 windows on
one pinned core (experiments/correlated_matching_timing), then scaled
so that its mean over calm windows at p = 0.003 is 2604.12841 Table
IX's time per round times the window's rounds. The law keeps the shape
measured here; the scale is the paper's Grace CPU. No host time enters.
"""

import dataclasses
import math
from typing import Optional

import numpy

import decsim.config as config


@dataclasses.dataclass(frozen=True)
class DetectionEventTime:
    """A decode's time as a law in its window's detection events.

    microseconds = scale x (intercept_microseconds
                            + coefficient_microseconds x events ** exponent)

    events is the count of the syndrome's fired rows, the residual of
    the window's pinned faces included, so a window in a burst is slow
    exactly when it holds many events. An exponent of 1 is a line. scale
    is the factor that makes the law's mean over the campaign's calm
    windows equal Table IX's time (1 keeps the measured host's time).
    source names the campaign's folder and date the law was fitted from.
    """

    intercept_microseconds: float
    coefficient_microseconds: float
    exponent: float
    scale: float
    source: str

    def __post_init__(self) -> None:
        config.check_microseconds(
            "intercept_microseconds", self.intercept_microseconds
        )
        config.check_microseconds(
            "coefficient_microseconds", self.coefficient_microseconds
        )
        _check_positive("exponent", self.exponent)
        _check_positive("scale", self.scale)

    def decode_ticks(self, detection_events: Optional[numpy.ndarray]) -> int:
        """The ticks of a decode of this syndrome; no window costs nothing."""
        if detection_events is None:
            return 0
        events = numpy.count_nonzero(detection_events)
        growth = events**self.exponent
        event_microseconds = self.coefficient_microseconds * growth
        law_microseconds = self.intercept_microseconds + event_microseconds
        microseconds = self.scale * law_microseconds
        return config.microseconds_to_ticks(microseconds)


def _check_positive(name: str, value: object) -> None:
    is_finite = config.is_number(value) and math.isfinite(value)
    if is_finite and value > 0:
        return
    raise ValueError(f"{name} must be a finite positive number (got {value!r})")
