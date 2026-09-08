"""One window's two forced-class solves, joined into its confidence.

The window side submits two jobs of one window, each pinned to one
logical class, at one instant; this object is both jobs' on_decoded. It
holds the first solve until the other arrives, the way a reservation
station holds a value until its tag matches (Tomasulo 1967, IBM Journal
of R&D, the tag here being the window key and the forced class), asks
the signal for the gap between the two weights, and hands the lighter
class's correction to the window committer, which applies the threshold.
The join is outside the decoder manager on purpose: every fine-grained
referent puts a two-result comparison in the producer or the consumer
and none puts it in the scheduler (gem5's SplitDataRequest counting its
own halves, src/cpu/o3/lsq.cc; RMT's store comparator beside the store
queue, Mukherjee et al. ISCA 2002).
"""

import dataclasses

import decsim.decoders.decode_queue as decode_queue_module
import decsim.observe.trace_source as trace_source
import decsim.records.decoding as decoding_records


@dataclasses.dataclass(frozen=True)
class HeldForcedSolve:
    """One class's finished solve, waiting for the other class's."""

    job: decoding_records.DecodeJob
    result: decoding_records.DecodeResult


class ForcedClassGapJoin:
    """Both forced-class jobs' on_decoded: the window's gap and its answer.

    Trace source: solve_held(job, result) when a window's first solve
    waits for the other, so the trace shows the held half and the join.
    """

    def __init__(self, engine, signal, committer, decode_queue) -> None:
        self.engine = engine
        self.signal = signal
        self.committer = committer
        self.decode_queue = decode_queue
        # window key -> the first solve of that window that finished
        self.held_by_window: dict[tuple, HeldForcedSolve] = {}
        self.solve_held = trace_source.TraceSource()

    def accept_result(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """One forced-class solve finished: hold it, or join the pair."""
        key = (job.operation_id, job.window_id)
        held = self.held_by_window.pop(key, None)
        if held is None:
            self._hold(key, job, result)
            return
        second = HeldForcedSolve(job, result)
        self._join(held, second)

    def unresolved_windows(self) -> list:
        """The windows whose second solve never arrived, sorted."""
        return sorted(self.held_by_window)

    def _hold(
        self,
        key: tuple,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """Keep the solve until the other class reports."""
        self.held_by_window[key] = HeldForcedSolve(job, result)
        forced_class = job.forced_logical_class
        self.engine.log(
            decode_queue_module.LOG_SOURCE,
            f"GAP HOLD {job.label}: class {forced_class} waits for the "
            "other class",
        )
        self.solve_held.fire(job, result)

    def _join(self, first: HeldForcedSolve, second: HeldForcedSolve) -> None:
        """Subtract the two weights and commit the lighter class's answer."""
        weights = [first.result.forced_class_weight]
        weights.append(second.result.forced_class_weight)
        soft_output = self.signal.soft_output_for(weights)
        lighter = _lighter_of(first, second)
        companion = second
        if lighter is second:
            companion = first
        lighter.result.soft_output = soft_output
        gap_text = _gap_text(soft_output)
        self.engine.log(
            decode_queue_module.LOG_SOURCE,
            f"GAP JOIN {lighter.job.label}: {gap_text}",
        )
        self.decode_queue.close_companion_request(
            companion.job, companion.result
        )
        self.committer.accept_result(lighter.job, lighter.result)


def _lighter_of(
    first: HeldForcedSolve, second: HeldForcedSolve
) -> HeldForcedSolve:
    """The solve of the class the decoder is in: the smaller weight.

    The unconstrained minimum weight is the minimum over the classes, so
    the lighter forced solve is the decoder's own answer. A missing
    weight keeps the first solve, whose result then carries no gap.
    """
    first_weight = first.result.forced_class_weight
    second_weight = second.result.forced_class_weight
    if first_weight is None or second_weight is None:
        return first
    if second_weight < first_weight:
        return second
    return first


def _gap_text(soft_output) -> str:
    """The joined gap for the log; a window without one says so."""
    if soft_output is None:
        return "no gap (a forced solve reported no weight)"
    return f"gap {soft_output.gap:.3f}"
