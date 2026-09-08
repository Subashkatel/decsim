"""One window's solves, joined into its confidence.

The window side submits a window's solves at one instant, one job per
class the signal needs forced (two for the complementary gap, none for
the cluster gap, which reads one ordinary decode); this object is every
one of those jobs' on_decoded. It holds each solve until the window's
last one arrives, the way a reservation station holds a value until its
tag matches (Tomasulo 1967, IBM Journal of R&D, the tag here being the
window key), asks the signal for the window's confidence, and hands the
answering solve to the window committer, which applies the threshold.
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
class HeldSolve:
    """One finished solve of a window, waiting for that window's others."""

    job: decoding_records.DecodeJob
    result: decoding_records.DecodeResult


class WindowGapJoin:
    """Every solve's on_decoded: the window's confidence and its answer.

    Trace source: solve_held(job, result) when a solve waits for the
    window's others, so the trace shows the held solve and the join.
    """

    def __init__(self, engine, signal, committer, decode_queue) -> None:
        self.engine = engine
        self.signal = signal
        self.committer = committer
        self.decode_queue = decode_queue
        # window key -> the solves of that window that have finished
        self.held_by_window: dict[tuple, list] = {}
        self.solve_held = trace_source.TraceSource()

    @property
    def solves_per_window(self) -> int:
        """Solves this signal reads: one per forced class, else one decode."""
        return len(self.signal.forced_logical_classes) or 1

    def accept_result(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """One solve finished: hold it, or join the window's solves."""
        key = (job.operation_id, job.window_id)
        held = self.held_by_window.setdefault(key, [])
        solve = HeldSolve(job, result)
        held.append(solve)
        if len(held) < self.solves_per_window:
            self._hold(job, result)
            return
        del self.held_by_window[key]
        self._join(held)

    def unresolved_windows(self) -> list:
        """The windows whose remaining solves never arrived, sorted."""
        return sorted(self.held_by_window)

    def _hold(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """Keep the solve until the window's others report."""
        forced_class = job.forced_logical_class
        self.engine.log(
            decode_queue_module.LOG_SOURCE,
            f"GAP HOLD {job.label}: class {forced_class} waits for the "
            "other class",
        )
        self.solve_held.fire(job, result)

    def _join(self, held: list) -> None:
        """Ask the signal for the window's gap and commit its answer."""
        solves = []
        for solve in held:
            solves.append(solve.result)
        soft_output = self.signal.soft_output_for(tuple(solves))
        answer = _answering_solve(held)
        answer.result.soft_output = soft_output
        gap_text = _gap_text(soft_output)
        self.engine.log(
            decode_queue_module.LOG_SOURCE,
            f"GAP JOIN {answer.job.label}: {gap_text}",
        )
        for solve in held:
            if solve is not answer:
                self.decode_queue.close_companion_request(
                    solve.job, solve.result
                )
        self.committer.accept_result(answer.job, answer.result)


def _answering_solve(held: list) -> HeldSolve:
    """The solve that carries the window's correction: the lightest one.

    The unconstrained minimum weight is the minimum over the classes, so
    the lightest forced solve is the decoder's own answer. A solve with
    no weight (a signal that forces no class, or a window that pins no
    observable) leaves the first solve answering, and its result then
    carries whatever gap the signal reported.
    """
    answer = held[0]
    answer_weight = answer.result.forced_class_weight
    if answer_weight is None:
        return answer
    for solve in held[1:]:
        weight = solve.result.forced_class_weight
        if weight is None:
            return answer
        if weight < answer_weight:
            answer = solve
            answer_weight = weight
    return answer


def _gap_text(soft_output) -> str:
    """The joined gap for the log; a window without one says so."""
    if soft_output is None:
        return "no gap (a solve reported no evidence)"
    return f"gap {soft_output.gap:.3f}"
