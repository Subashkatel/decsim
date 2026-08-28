"""Attach a typed confidence record to each committed decoder window."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..decoders.window_decode_results import payload_syndrome
from ..message import (
    DecodeJob,
    DecodeResult,
    RunSeedChild,
    RunSeedPathSegment,
    SoftOutputSource,
)
from ..detector_error_model.fault_model_contracts import DecoderFaultModelRequirement

if TYPE_CHECKING:
    from ..protocols import Decoder

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

    from ..detector_error_model.fault_model_contracts import (
        FaultRepresentation,
    )

    if model is None:
        return None
    cached = cache.get(id(model))
    if cached is not None:
        return None if cached is _NO_OBSERVABLE else cached
    faults = model.require_faults(FaultRepresentation.GRAPHLIKE)
    observables = faults.observables.toarray()
    has_one_observable = observables.shape[0] == 1 and observables.any()
    metric = (metric_cls.from_window_model(model)
              if has_one_observable else _NO_OBSERVABLE)
    cache[id(model)] = metric
    weakref.finalize(model, cache.pop, id(model), None)
    return None if metric is _NO_OBSERVABLE else metric


class SoftOutputDecoder:
    """Attach one configured metric's confidence without changing hard output.

    ``metric_cls`` is a configured factory instance declaring ``source`` and
    ``from_window_model(model)``.
    """

    def __init__(self, base: "Decoder", metric_cls):
        if isinstance(metric_cls, type):
            raise TypeError(
                "SoftOutputDecoder requires a configured metric factory "
                "instance, not a metric class"
            )
        if not isinstance(
            getattr(metric_cls, "source", None),
            SoftOutputSource,
        ):
            raise TypeError(
                "configured metric factory must declare one SoftOutputSource"
            )
        if not callable(getattr(metric_cls, "from_window_model", None)):
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
        if not isinstance(base_requirement, DecoderFaultModelRequirement):
            raise TypeError(
                "base decoder fault_model_requirement must be a "
                "DecoderFaultModelRequirement"
            )
        if not isinstance(metric_requirement, DecoderFaultModelRequirement):
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
        return (
            RunSeedChild(
                (RunSeedPathSegment("field", "base"),),
                self.base,
            ),
            RunSeedChild(
                (RunSeedPathSegment("field", "metric_cls"),),
                self.metric_cls,
            ),
        )

    def latency(self, job: DecodeJob) -> int:
        """Timing is the base decoder's; the soft output adds no modelled latency."""
        return self.base.latency(job)

    @property
    def measures_wall_clock(self):
        return getattr(self.base, "measures_wall_clock", False)

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Run the base decode, then attach the soft output when available.

        The soft output is part of the weak decoder's real work (the paper's
        weak decoder computes it during decoding), so last_decode_ns is
        the base decoder's OWN measured decode plus the timed gap solves.
        Wrapping base.decode in this wrapper's stopwatch instead would
        re-capture the base's untimed setup (graph build and warm-up),
        which the base deliberately keeps out of its measurement."""
        import time
        metric = self._metric_for(job.dem)
        result = self.base.decode(job)
        base_decode_ns = getattr(self.base, "last_decode_ns", None) or 0
        started_ns = time.perf_counter_ns()
        if metric is not None:
            result.soft_output = metric.evaluate(payload_syndrome(job))
        evaluate_ns = time.perf_counter_ns() - started_ns
        self.last_decode_ns = base_decode_ns + evaluate_ns
        return result

    def _metric_for(self, model):
        """This window model's cached metric, or None without an observable."""
        return cached_metric_for_model(
            self._metrics_by_model_identity, self.metric_cls, model)


class ParallelGapDecoder(SoftOutputDecoder):
    """The paired-core weak unit: the gap's two forced-class solves run
    on two matching cores side by side, joined by a subtract-compare.

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

    def __init__(self, base: "Decoder", metric_cls, combine_ns: int = 0):
        super().__init__(base, metric_cls)
        if combine_ns < 0:
            raise ValueError("combine_ns models join hardware and cannot "
                             "be negative")
        self.combine_ns = combine_ns

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Base decode for the payload; paired solves for gap and time."""
        result = self.base.decode(job)
        metric = self._metric_for(job.dem)
        if metric is None:
            self.last_decode_ns = getattr(self.base, "last_decode_ns", 0)
            return result
        paired = metric.paired_evaluate(payload_syndrome(job))
        result.soft_output = paired.soft_output
        self.last_decode_ns = max(paired.forced_solve_ns) + self.combine_ns
        return result


class SplitGapDecoder(SoftOutputDecoder):
    """The weak half of a split gap pair: this unit solves ONE forced
    class (class 0) and the base decode, while a sibling job on a
    separate decoder unit solves the other class.

    The gap does not exist until both halves report, so this decoder
    attaches no soft output; it stamps its forced weight on the result
    (``gap_half_weight``) and the DecoderManager's join builds the
    SoftOutput when the sibling lands. The unit's charged time is the
    forced solve (the base decode re-derives the same winning-class
    answer for the simulator's accuracy artifacts and is not charged,
    exactly as in ParallelGapDecoder).
    """

    FORCED_CLASS = 0

    def decode(self, job: DecodeJob) -> DecodeResult:
        result = self.base.decode(job)
        metric = self._metric_for(job.dem)
        if metric is None:
            self.last_decode_ns = getattr(self.base, "last_decode_ns", 0)
            return result
        weight, elapsed_ns = metric.forced_class_solve(
            payload_syndrome(job), self.FORCED_CLASS)
        result.gap_half_weight = weight
        self.last_decode_ns = elapsed_ns
        return result


class GapHalfDecoder:
    """The sibling half of a split gap pair: one forced-class solve
    (class 1) on its own decoder unit, no correction, no observables.

    Its result exists only to carry a weight to the join; it never
    touches the strong ledger or the Pauli frame (the manager routes
    ``gap_sibling_for`` completions to the join before any of that).
    """

    FORCED_CLASS = 1

    def __init__(self, metric_cls):
        from ..detector_error_model.fault_model_contracts import (
            GRAPHLIKE_FAULT_MODEL_REQUIRED,
        )
        self.metric_cls = metric_cls
        self.fault_model_requirement = GRAPHLIKE_FAULT_MODEL_REQUIRED
        self.measures_wall_clock = True
        self.last_decode_ns = 0
        self._metrics_by_model_identity: dict = {}

    def latency(self, job: DecodeJob) -> int:
        raise RuntimeError("measured wall-clock timing needs the "
                           "DecoderEngine: it decodes first and charges "
                           "the measured time")

    def decode(self, job: DecodeJob) -> DecodeResult:
        result = DecodeResult(job.op_id, job.window_id)
        metric = cached_metric_for_model(
            self._metrics_by_model_identity, self.metric_cls, job.dem)
        if metric is None:
            self.last_decode_ns = 0
            return result
        weight, elapsed_ns = metric.forced_class_solve(
            payload_syndrome(job), self.FORCED_CLASS)
        result.gap_half_weight = weight
        self.last_decode_ns = elapsed_ns
        return result
