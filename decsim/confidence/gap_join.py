"""One window's solves, joined into its confidence.

The window side submits one job per class the signal forces (two for the
complementary gap, one plain decode for the cluster gap), and this
object is each job's on_decoded. It holds the solves until the window's
last one arrives, as a reservation station holds a value until its tag
matches (Tomasulo 1967; the tag is the window key), asks the signal for
the confidence, and hands the answering solve to the window verdict. The
join sits outside the decoder manager because the referents put a
two-result comparison in the producer or the consumer, never in the
scheduler (gem5's SplitDataRequest, src/cpu/o3/lsq.cc; RMT's store
comparator, Mukherjee et al. ISCA 2002).
"""

import dataclasses

import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.log_sources as log_sources
import decsim.trace_source as trace_source


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

    signal = ports.Port(ports.ConfidenceSignal)
    verdict = ports.Port(ports.WindowVerdict)
    decode_queue = ports.Port(ports.DecodeQueue)

    def __init__(self, engine) -> None:
        self.engine = engine
        # window key -> the solves of that window that have finished
        self.held_by_window: dict[tuple, list] = {}
        self.trace = _TraceSources()

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
        """Keep the solve and release its unit's result.

        The first weight waits while the second solve runs, as a
        reservation station captures a result and frees its unit
        (Tomasulo 1967; Toshio et al. 2510.25222 lines 482-494). A unit
        that blocked on its result would wait on itself when both solves
        share it.
        """
        forced_class = job.forced_logical_class
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"GAP HOLD {job.label}: class {forced_class} waits for the "
            "other class",
        )
        self.trace.solve_held.fire(job, result)
        self.decode_queue.read_result(job)

    def _join(self, held: list) -> None:
        """Ask the signal for the gap and commit the window's answer.

        The soft output goes on the answering solve. The ticks go on the
        solve that arrived last, whose unit the manager has not given
        back yet, so the unit that produced the evidence carries the
        signal's work.
        """
        solves = []
        for solve in held:
            solves.append(solve.result)
        computation = self.signal.compute(tuple(solves))
        answer = _answering_solve(held)
        answer.result.soft_output = computation.soft_output
        gap_text = _gap_text(computation.soft_output)
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"GAP JOIN {answer.job.label}: {gap_text}",
        )
        delivering = held[-1]
        self.decode_queue.charge_soft_output(delivering.job, computation.ticks)
        if computation.ticks <= 0:
            self._answer(held, answer)
            return
        self.engine.schedule(
            computation.ticks,
            lambda: self._answer(held, answer),
            label=f"soft_output_done({answer.job.label})",
        )

    def _answer(self, held: list, answer: HeldSolve) -> None:
        """Close the window's other solves and hand its answer on."""
        for solve in held:
            if solve is not answer:
                self.decode_queue.close_companion_request(
                    solve.job, solve.result
                )
        self.verdict.accept_result(answer.job, answer.result)


def _answering_solve(held: list) -> HeldSolve:
    """The solve that carries the window's correction: the lightest one.

    The unconstrained minimum weight is the minimum over the classes, so
    the lightest forced solve is the decoder's own answer. A solve with
    no weight leaves the first solve answering.
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


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the join reports, as one member (gem5's stats Group)."""

    solve_held: trace_source.TraceSource = trace_source.new_source()
