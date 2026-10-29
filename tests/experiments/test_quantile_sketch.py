"""A stream's percentiles off logarithmic buckets (quantile_sketch.py).

The referent is the shot's exact percentile of the same values,
report.percentile_of_counts: the sketch picks the same rank and answers
within its relative accuracy (DDSketch, Masson et al. 2019, Sec. 2).
"""

import collections
import random

import pytest

import decsim.experiments.quantile_sketch as quantile_sketch
import decsim.experiments.report as report


def _exact(values: list, fraction: float) -> float:
    counts = collections.Counter(values)
    return report.percentile_of_counts(counts, fraction)


@pytest.mark.parametrize("fraction", [0.0, 0.5, 0.9, 0.99, 1.0])
def test_a_percentile_is_within_the_accuracy_of_the_exact_one(fraction):
    generator = random.Random(5)
    # values over many decades, as reaction times run from ns to ms
    values = [generator.lognormvariate(0.0, 2.0) for _ in range(20_000)]
    sketch = quantile_sketch.QuantileSketch()
    for value in values:
        sketch.add(value)

    estimate = sketch.percentile(fraction)

    exact = _exact(values, fraction)
    allowed = quantile_sketch.DEFAULT_RELATIVE_ACCURACY * exact
    error = estimate - exact
    assert abs(error) <= allowed


def test_zero_is_answered_exactly():
    sketch = quantile_sketch.QuantileSketch()
    for value in (0.0, 0.0, 0.0, 2.0):
        sketch.add(value)

    median = sketch.percentile(0.5)

    assert median == 0.0


def test_the_reported_percentiles_are_named_as_a_shots():
    sketch = quantile_sketch.QuantileSketch()
    sketch.add(1.0)

    percentiles = sketch.percentiles()

    assert list(percentiles) == ["p50", "p90", "p99"]


def test_the_memory_is_a_count_per_bucket_used():
    sketch = quantile_sketch.QuantileSketch()
    for step in range(1, 1_000_001):
        value = step * 1e-3
        sketch.add(value)

    # values over six decades at 1% accuracy fill about 700 buckets
    assert len(sketch.counts) < 1_000


def test_a_negative_value_is_refused():
    sketch = quantile_sketch.QuantileSketch()

    with pytest.raises(ValueError, match="not negative"):
        sketch.add(-1.0)


@pytest.mark.parametrize("accuracy", [0.0, 1.0, -0.1])
def test_an_accuracy_outside_zero_and_one_is_refused(accuracy):
    with pytest.raises(ValueError, match="between 0 and 1"):
        quantile_sketch.QuantileSketch(accuracy)
