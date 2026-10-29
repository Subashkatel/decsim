"""Percentiles of a long stream of values, in bounded memory.

A shot keeps every latency sample, so its percentiles are exact
(report.percentile_of_counts). A stream of 10^6 rounds folds its
windows as it goes (stream_measures.py) and cannot keep them. This keeps
each value's bucket instead, DDSketch's logarithmic buckets (Masson,
Rim and Lee, PVLDB 12(12), 2019, Sec. 2): with gamma = (1 + accuracy) /
(1 - accuracy), a positive value x falls in bucket i = ceil(log_gamma x),
the bucket (gamma^(i-1), gamma^i], and the bucket answers for its
values with 2 gamma^i / (gamma + 1), within accuracy of each of them
relatively. Zero keeps a bucket of its own and answers 0. The memory is
one count per bucket used, a few hundred buckets for values from 1 ns to
1 s at 1% accuracy.

A percentile picks the same rank a shot does, numpy's nearest method
(report.percentile_of_counts), and answers with that rank's bucket, so a
stream's percentile is within accuracy of the shot's exact one for the
same values.
"""

import collections
import math

import decsim.experiments.report as report

# relative accuracy of every percentile: within 1% of the exact value
DEFAULT_RELATIVE_ACCURACY = 0.01
# the percentiles a stream reports, by the name sweep.csv gives them
REPORTED_PERCENTILES = report.SUMMARY_PERCENTILES


class QuantileSketch:
    """Counts of values by logarithmic bucket, read as percentiles."""

    def __init__(
        self, relative_accuracy: float = DEFAULT_RELATIVE_ACCURACY
    ) -> None:
        _check_accuracy(relative_accuracy)
        self.relative_accuracy = relative_accuracy
        self.gamma = (1 + relative_accuracy) / (1 - relative_accuracy)
        self.log_gamma = math.log(self.gamma)
        # bucket index -> values in it; None is the bucket of zero
        self.counts: collections.Counter = collections.Counter()

    def add(self, value: float) -> None:
        """One more value; a negative one is refused."""
        bucket = self._bucket_of(value)
        self.counts[bucket] += 1

    def percentile(self, fraction: float) -> float:
        """The value at fraction of the values, within the accuracy."""
        answers = {}
        for bucket, count in self.counts.items():
            answer = self._answer_of(bucket)
            answers[answer] = count
        return report.percentile_of_counts(answers, fraction)

    def percentiles(self) -> dict:
        """The reported percentiles by name: p50, p90 and p99."""
        return {
            name: self.percentile(fraction)
            for name, fraction in REPORTED_PERCENTILES
        }

    def _bucket_of(self, value: float):
        if value < 0:
            raise ValueError(f"a sketched value is not negative: {value}")
        if value == 0:
            return None
        scaled = math.log(value) / self.log_gamma
        return math.ceil(scaled)

    def _answer_of(self, bucket) -> float:
        if bucket is None:
            return 0.0
        upper = self.gamma**bucket
        return 2 * upper / (self.gamma + 1)


def _check_accuracy(relative_accuracy: float) -> None:
    if 0 < relative_accuracy < 1:
        return
    raise ValueError(
        "a sketch's relative accuracy is between 0 and 1, not "
        f"{relative_accuracy}"
    )
