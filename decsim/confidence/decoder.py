"""Attach a typed confidence record to each committed decoder window."""

from typing import Optional

import decsim.decoders.decoder as decoder_module
import decsim.decoders.decoder as decoder_module
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.message as message

# cache value meaning "this model was inspected and has no usable
# observable"; distinct from a missing key, which means "not built yet"
_NO_OBSERVABLE = object()


def cached_metric_for_model(cache: dict, metric_cls, model):
    """One gap metric per window model, or None without an observable.

    Metric construction builds two matching graphs, which is setup work
    like the base decoder's own graph build (cached and pre-warmed in
    PyMatchingDecoder, never charged to decode time), so every gap
    engine shares this cache instead of rebuilding per decode. Entries
    are id-keyed and weakref-evicted exactly like that cache: CPython
    recycles id() values, and a stale hit would serve the wrong graph.
    """
    import weakref

    if model is None:
        return None
    model_identity = id(model)
    cached = cache.get(model_identity)
    if cached is _NO_OBSERVABLE:
        return None
    if cached is not None:
        return cached
    faults = model.require_faults(fault_models.FaultRepresentation.GRAPHLIKE)
    observables = faults.observables.toarray()
    has_one_observable = observables.shape[0] == 1 and observables.any()
    metric = _NO_OBSERVABLE
    if has_one_observable:
        metric = metric_cls.from_window_model(model)
    cache[model_identity] = metric
    weakref.finalize(model, cache.pop, model_identity, None)
    if metric is _NO_OBSERVABLE:
        return None
    return metric


class SoftOutputDecoder(decoder_module.DecoderBase):
    """Attach one configured metric's confidence without changing hard output.

    ``metric_cls`` is a configured factory instance declaring ``source`` and
    ``from_window_model(model)``. Timing is the base decoder's; on the
    measured path the soft output is part of the weak decoder's real
    work (the paper's weak decoder computes it during decoding), so the
    measured time is the base's own backend call plus the timed gap
    solve, never the base's untimed setup (graph build and warm-up).
    """

    def __init__(self, base: decoder_module.DecoderBase, metric_cls):
        if isinstance(metric_cls, type):
            raise TypeError(
                "SoftOutputDecoder requires a configured metric factory "
                "instance, not a metric class"
            )
        source = getattr(metric_cls, "source", None)
        if not isinstance(source, message.SoftOutputSource):
            raise TypeError(
                "configured metric factory must declare one SoftOutputSource"
            )
        builder = getattr(metric_cls, "from_window_model", None)
        if not callable(builder):
            raise TypeError(
                "configured metric factory must build from a window model"
            )
        try:
            base_requirement = base.fault_model_requirement
            metric_requirement = metric_cls.fault_model_requirement
        except AttributeError as error:
            raise TypeError(
                "base decoder and metric factory must declare "
                "fault_model_requirement"
            ) from error
        if not isinstance(
            base_requirement, fault_models.DecoderFaultModelRequirement
        ):
            raise TypeError(
                "base decoder fault_model_requirement must be a "
                "DecoderFaultModelRequirement"
            )
        if not isinstance(
            metric_requirement, fault_models.DecoderFaultModelRequirement
        ):
            raise TypeError(
                "metric factory fault_model_requirement must be a "
                "DecoderFaultModelRequirement"
            )
        self.base = base
        self.metric_cls = metric_cls
        self.fault_model_requirement = base_requirement.joined(
            metric_requirement
        )
        self._metrics_by_model_identity: dict = {}

    def run_seed_children(self):
        """Expose the base decoder and configured confidence builder."""
        base_path = (message.RunSeedPathSegment("field", "base"),)
        metric_path = (message.RunSeedPathSegment("field", "metric_cls"),)
        return (
            message.RunSeedChild(base_path, self.base),
            message.RunSeedChild(metric_path, self.metric_cls),
        )

    def latency(self, job: message.DecodeJob) -> int:
        """The base decoder's timing; the soft output adds no latency."""
        return self.base.latency(job)

    def occupancy(self, job: message.DecodeJob) -> Optional[int]:
        """The base decoder's occupancy; None when it is measured."""
        return self.base.occupancy(job)

    def pipeline_depth(self, job: message.DecodeJob) -> int:
        """The base decoder's pipeline depth."""
        return self.base.pipeline_depth(job)

    def cancel(self, job: message.DecodeJob) -> None:
        """Stop the base decoder's job."""
        self.base.cancel(job)

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """Run the base decode, then attach the soft output when available."""
        result, _elapsed_ns = self.decode_timed(job)
        return result

    def decode_timed(self, job: message.DecodeJob) -> tuple:
        """The base's measured call plus the timed soft-output evaluation."""
        import time

        metric = self._metric_for(job.dem)
        result, base_decode_ns = self.base.decode_timed(job)
        started_ns = time.perf_counter_ns()
        if metric is not None:
            syndrome = decoder_module.payload_syndrome(job)
            result.soft_output = metric.evaluate(syndrome)
        finished_ns = time.perf_counter_ns()
        evaluate_ns = finished_ns - started_ns
        return result, base_decode_ns + evaluate_ns

    def _metric_for(self, model):
        """This window model's cached metric, or None without an observable."""
        return cached_metric_for_model(
            self._metrics_by_model_identity, self.metric_cls, model
        )


class ParallelGapDecoder(SoftOutputDecoder):
    """The paired-core weak unit: two forced-class solves side by side.

    The gap's two forced-class solves run on two matching cores, joined
    by a subtract-compare.

    Accuracy is unchanged by construction: the committed correction and
    observables still come from the base decode, and the pair only
    supplies the soft output. What changes is the modelled cost. The
    unit's wall clock is the slower forced solve plus ``combine_ns``
    for the join, never the serial sum, and the hardware bill is two
    matching cores plus a syndrome broadcast inside one decoder unit;
    the manager still schedules one job on one unit. The card-latency
    path is untouched: a priced card already describes the whole unit,
    so the pair changes its cost sheet, not its timing.

    The winning forced class is a WHOLE-WINDOW statement and is not
    compared against the result's ``logical_observables``: those are
    the owned-region parity contribution for the sliding-window XOR
    chain, a different object that legitimately disagrees whenever the
    minimum-weight solution's flips straddle the ownership boundary.
    The serial metric has the same semantics (its gap also describes
    the whole-window class); test_parallel_gap pins the two engines to
    each other, and the whole-window consistency invariant
    min(w_forced) == w_plain is pinned at the metric level.
    """

    def __init__(
        self,
        base: decoder_module.DecoderBase,
        metric_cls,
        combine_ns: int = 0,
    ):
        SoftOutputDecoder.__init__(self, base, metric_cls)
        if combine_ns < 0:
            raise ValueError(
                "combine_ns models join hardware and cannot be negative"
            )
        self.combine_ns = combine_ns

    def decode_timed(self, job: message.DecodeJob) -> tuple:
        """Base decode for the payload; paired solves for gap and time."""
        result, base_decode_ns = self.base.decode_timed(job)
        metric = self._metric_for(job.dem)
        if metric is None:
            return result, base_decode_ns
        syndrome = decoder_module.payload_syndrome(job)
        paired = metric.paired_evaluate(syndrome)
        result.soft_output = paired.soft_output
        slower_solve_ns = max(paired.forced_solve_ns)
        return result, slower_solve_ns + self.combine_ns


class SplitGapDecoder(SoftOutputDecoder):
    """The weak half of a split gap pair.

    This unit solves ONE forced class (class 0) and the base decode,
    while a sibling job on a separate decoder unit solves the other
    class.

    The gap does not exist until both halves report, so this decoder
    attaches no soft output; it stamps its forced weight on the result
    (``gap_half_weight``) and the DecoderManager's join builds the
    SoftOutput when the sibling lands. The unit's charged time is the
    forced solve (the base decode re-derives the same winning-class
    answer for the simulator's accuracy artifacts and is not charged,
    exactly as in ParallelGapDecoder).
    """

    FORCED_CLASS = 0

    def decode_timed(self, job: message.DecodeJob) -> tuple:
        """Base decode for the payload; this half's forced solve for time."""
        result, base_decode_ns = self.base.decode_timed(job)
        metric = self._metric_for(job.dem)
        if metric is None:
            return result, base_decode_ns
        syndrome = decoder_module.payload_syndrome(job)
        weight, elapsed_ns = metric.forced_class_solve(
            syndrome, self.FORCED_CLASS
        )
        result.gap_half_weight = weight
        return result, elapsed_ns


class GapHalfDecoder(decoder_module.DecoderBase):
    """The sibling half of a split gap pair.

    One forced-class solve (class 1) on its own decoder unit, no
    correction, no observables.

    Its result exists only to carry a weight to the join; it never
    touches the strong ledger or the Pauli frame (the manager routes
    ``gap_sibling_for`` completions to the join before any of that).
    Always measured on the host clock: the forced solve is the unit's
    time.
    """

    FORCED_CLASS = 1

    fault_model_requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED

    def __init__(self, metric_cls):
        self.metric_cls = metric_cls
        self._metrics_by_model_identity: dict = {}

    def latency(self, job: message.DecodeJob) -> int:
        """A measured decoder has no latency before its call."""
        del job
        raise NotImplementedError(
            "the gap half is measured on the host clock; the unit holds it "
            "for the forced solve's time"
        )

    def occupancy(self, job: message.DecodeJob) -> Optional[int]:
        """None: the unit cannot say in advance when it frees."""
        del job
        return None

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """The forced-class weight on an otherwise empty result."""
        result, _elapsed_ns = self.decode_timed(job)
        return result

    def decode_timed(self, job: message.DecodeJob) -> tuple:
        """The forced solve and its wall clock; nothing without a metric."""
        result = message.DecodeResult(job.op_id, job.window_id)
        metric = cached_metric_for_model(
            self._metrics_by_model_identity, self.metric_cls, job.dem
        )
        if metric is None:
            return result, 0
        syndrome = decoder_module.payload_syndrome(job)
        weight, elapsed_ns = metric.forced_class_solve(
            syndrome, self.FORCED_CLASS
        )
        result.gap_half_weight = weight
        return result, elapsed_ns
