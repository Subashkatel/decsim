"""The confidence wrappers: a weak decoder that also reports its soft output.

Each wrapper is a row of the Decoder port over a base decoder and a
ConfidenceSignal (decsim/ports.py): the correction and the observables
are the base's, and the result carries the signal's soft output. The
root selects one by escalation.gap_computation: SoftOutputDecoder
(serial) evaluates the signal on the same core; ParallelGapDecoder
(parallel_pair) charges the signal's two forced solves as two cores
plus a join; SplitGapDecoder (split_pair) solves one forced class here
while a GapHalfDecoder on its own unit solves the other, and the decoder
manager's GapJoins builds the gap when both halves land (Toshio et al.
2510.25222 Sec. III A: the weak decoder computes its soft output during
decoding). The pair and the split read the complementary gap's two
forced-class solves, so they take that signal.
"""

import time
import weakref
from typing import Optional

import decsim.decoders.decoder as decoder_module
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoding as decoding_records
import decsim.records.seeds as seed_records

# the cache value meaning "this model was inspected and the signal has
# no metric for it"; distinct from a missing key, which means "not built
# yet"
_NO_METRIC = object()


def cached_metric_for_model(cache: dict, signal, model):
    """The signal's metric for one window model, or None when it has none.

    Metric construction builds two matching graphs, which is setup work
    like the base decoder's own graph build (cached and pre-warmed in
    PyMatchingDecoder, never charged to decode time), so every gap
    engine shares this cache instead of rebuilding per decode. Entries
    are id-keyed and weakref-evicted exactly like that cache: CPython
    recycles id() values, and a stale hit would serve the wrong graph.
    """
    if model is None:
        return None
    model_identity = id(model)
    cached = cache.get(model_identity)
    if cached is _NO_METRIC:
        return None
    if cached is not None:
        return cached
    metric = signal.metric_for(model)
    if metric is None:
        metric = _NO_METRIC
    cache[model_identity] = metric
    weakref.finalize(model, cache.pop, model_identity, None)
    if metric is _NO_METRIC:
        return None
    return metric


class SoftOutputDecoder(decoder_module.DecoderBase):
    """The base decode with one signal's confidence attached.

    Timing is the base decoder's; on the measured path the soft output
    is part of the weak decoder's real work, so the measured time is the
    base's own backend call plus the timed gap solve, never the base's
    untimed setup (graph build and warm-up).
    """

    def __init__(self, base: decoder_module.DecoderBase, signal) -> None:
        self.base = base
        self.signal = signal
        self.fault_model_requirement = base.fault_model_requirement.joined(
            signal.fault_model_requirement
        )
        self._metrics_by_model_identity: dict = {}

    def run_seed_children(self) -> tuple:
        """The base decoder and the signal.

        The signal's seed segment keeps the name the results carry.
        """
        base_segment = seed_records.RunSeedPathSegment("field", "base")
        base_child = seed_records.RunSeedChild((base_segment,), self.base)
        signal_segment = seed_records.RunSeedPathSegment("field", "metric_cls")
        signal_child = seed_records.RunSeedChild((signal_segment,), self.signal)
        return (base_child, signal_child)

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """The base decoder's timing; the soft output adds no latency."""
        return self.base.latency(job)

    def occupancy(self, job: decoding_records.DecodeJob) -> Optional[int]:
        """The base decoder's occupancy; None when it is measured."""
        return self.base.occupancy(job)

    def pipeline_depth(self, job: decoding_records.DecodeJob) -> int:
        """The base decoder's pipeline depth."""
        return self.base.pipeline_depth(job)

    def cancel(self, job: decoding_records.DecodeJob) -> None:
        """Stop the base decoder's job."""
        self.base.cancel(job)

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """The base decode with the soft output attached when available."""
        result, _elapsed_nanoseconds = self.decode_timed(job)
        return result

    def decode_timed(self, job: decoding_records.DecodeJob) -> tuple:
        """The base's measured call plus the timed soft-output evaluation."""
        metric = self._metric_for(job.detector_error_model)
        result, base_nanoseconds = self.base.decode_timed(job)
        started = time.perf_counter_ns()
        if metric is not None:
            syndrome = decoder_module.payload_syndrome(job)
            result.soft_output = metric.evaluate(syndrome)
        finished = time.perf_counter_ns()
        evaluate_nanoseconds = finished - started
        return result, base_nanoseconds + evaluate_nanoseconds

    def _metric_for(self, model):
        """This window model's cached metric; None when the signal has none."""
        return cached_metric_for_model(
            self._metrics_by_model_identity, self.signal, model
        )


class ParallelGapDecoder(SoftOutputDecoder):
    """The paired-core weak unit: two forced-class solves side by side.

    The gap's two forced-class solves run on two matching cores, joined
    by a subtract-compare. Accuracy is unchanged by construction: the
    committed correction and observables still come from the base
    decode, and the pair only supplies the soft output. What changes is
    the modelled cost: the unit's wall clock is the slower forced solve
    plus combine_nanoseconds for the join, never the serial sum, and the
    hardware bill is two matching cores plus a syndrome broadcast inside
    one decoder unit; the manager still schedules one job on one unit.
    The card-latency path is untouched: a priced card already describes
    the whole unit, so the pair changes its cost sheet, not its timing.

    The winning forced class is a whole-window statement and is not
    compared against the result's logical_observables: those are the
    owned-region parity contribution for the sliding-window XOR chain,
    a different object that legitimately disagrees whenever the
    minimum-weight solution's flips straddle the ownership boundary.
    The serial metric has the same semantics (its gap also describes
    the whole-window class); the whole-window consistency invariant
    min(w_forced) == w_plain is pinned at the metric level.
    """

    def __init__(
        self,
        base: decoder_module.DecoderBase,
        signal,
        combine_nanoseconds: int = 0,
    ) -> None:
        SoftOutputDecoder.__init__(self, base, signal)
        self.combine_nanoseconds = combine_nanoseconds

    def decode_timed(self, job: decoding_records.DecodeJob) -> tuple:
        """The base decode for the payload; the pair for the gap and time."""
        result, base_nanoseconds = self.base.decode_timed(job)
        metric = self._metric_for(job.detector_error_model)
        if metric is None:
            return result, base_nanoseconds
        syndrome = decoder_module.payload_syndrome(job)
        paired = metric.paired_evaluate(syndrome)
        result.soft_output = paired.soft_output
        slower_solve_nanoseconds = max(paired.forced_solve_nanoseconds)
        return result, slower_solve_nanoseconds + self.combine_nanoseconds


class SplitGapDecoder(SoftOutputDecoder):
    """The weak half of a split gap pair.

    This unit solves one forced class (class 0) and the base decode,
    while a sibling job on a separate decoder unit solves the other
    class. The gap does not exist until both halves report, so this
    decoder attaches no soft output; it stamps its forced weight on the
    result (gap_half_weight) and the decoder manager's join builds the
    SoftOutput when the sibling lands. The unit's charged time is the
    forced solve (the base decode re-derives the same winning-class
    answer for the simulator's accuracy artifacts and is not charged,
    exactly as in ParallelGapDecoder).
    """

    FORCED_CLASS = 0

    def decode_timed(self, job: decoding_records.DecodeJob) -> tuple:
        """The base decode for the payload; this half's solve for the time."""
        result, base_nanoseconds = self.base.decode_timed(job)
        metric = self._metric_for(job.detector_error_model)
        if metric is None:
            return result, base_nanoseconds
        syndrome = decoder_module.payload_syndrome(job)
        weight, elapsed_nanoseconds = metric.forced_class_solve(
            syndrome, self.FORCED_CLASS
        )
        result.gap_half_weight = weight
        return result, elapsed_nanoseconds


class GapHalfDecoder(decoder_module.DecoderBase):
    """The sibling half of a split gap pair.

    One forced-class solve (class 1) on its own decoder unit, no
    correction, no observables. Its result exists only to carry a
    weight to the join; it never touches the strong requests or the
    Pauli frame (the manager routes gap_sibling_for completions to the
    join before any of that). Always measured on the host clock: the
    forced solve is the unit's time.
    """

    FORCED_CLASS = 1

    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED

    def __init__(self, signal) -> None:
        self.signal = signal
        self._metrics_by_model_identity: dict = {}

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """A measured decoder has no latency before its call."""
        del job
        raise NotImplementedError(
            "the gap half is measured on the host clock; the unit holds it "
            "for the forced solve's time"
        )

    def occupancy(self, job: decoding_records.DecodeJob) -> Optional[int]:
        """None: the unit cannot say in advance when it frees."""
        del job
        return None

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """The forced-class weight on an otherwise empty result."""
        result, _elapsed_nanoseconds = self.decode_timed(job)
        return result

    def decode_timed(self, job: decoding_records.DecodeJob) -> tuple:
        """The forced solve and its wall clock; nothing without a metric."""
        result = decoding_records.DecodeResult(job.operation_id, job.window_id)
        metric = cached_metric_for_model(
            self._metrics_by_model_identity,
            self.signal,
            job.detector_error_model,
        )
        if metric is None:
            return result, 0
        syndrome = decoder_module.payload_syndrome(job)
        weight, elapsed_nanoseconds = metric.forced_class_solve(
            syndrome, self.FORCED_CLASS
        )
        result.gap_half_weight = weight
        return result, elapsed_nanoseconds
