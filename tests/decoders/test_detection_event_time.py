"""The fitted correlated matching time against its closed form.

DetectionEventTime's docstring states the law, scale x (intercept +
coefficient x events ** exponent), with events the syndrome's fired rows;
each test writes that law out by hand at a few event counts.
"""

import math

import numpy
import pytest

import decsim.config as config
from decsim.decoders.minimum_weight_perfect_matching import (
    detection_event_time,
)

LAW = detection_event_time.DetectionEventTime(
    intercept_microseconds=4.0,
    coefficient_microseconds=0.25,
    exponent=1.5,
    scale=2.0,
    source="a test's own numbers",
)


def test_a_syndrome_is_priced_by_the_law_at_its_fired_rows():
    syndrome = numpy.zeros(50, dtype=numpy.uint8)
    syndrome[[3, 7, 11, 20, 40, 41, 42, 49, 0]] = 1
    microseconds = 2.0 * (4.0 + 0.25 * 9**1.5)
    expected = config.microseconds_to_ticks(microseconds)
    assert LAW.decode_ticks(syndrome) == expected


def test_a_quiet_syndrome_costs_the_scaled_intercept():
    syndrome = numpy.zeros(50, dtype=numpy.uint8)
    expected = config.microseconds_to_ticks(8.0)
    assert LAW.decode_ticks(syndrome) == expected


def test_a_decode_with_no_window_costs_nothing():
    assert LAW.decode_ticks(None) == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("intercept_microseconds", -1.0),
        ("coefficient_microseconds", math.nan),
        ("exponent", 0.0),
        ("scale", -2.0),
        ("scale", math.inf),
    ],
)
def test_a_law_that_is_no_time_is_refused(field, value):
    numbers = {
        "intercept_microseconds": 4.0,
        "coefficient_microseconds": 0.25,
        "exponent": 1.5,
        "scale": 2.0,
    }
    numbers[field] = value
    with pytest.raises(ValueError, match=field):
        detection_event_time.DetectionEventTime(
            **numbers, source="a test's own numbers"
        )
