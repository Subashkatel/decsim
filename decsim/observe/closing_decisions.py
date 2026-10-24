"""A checkpoint's closing window, decided as the machine decides a final one.

A checkpoint is the machine's own answer had the run ended there (Tan
et al. 2209.09219 lines 1003-1037), so its closing window takes the
machine's path. The decoder of its stream's entry pool decodes it.
Where an escalate edge leaves that pool, the edge's confidence and
verdict read the result, the threshold copied at its value now, and an
escalated window is decoded again on the edge's target: the same
window, as a redo window decodes a final one
(escalation/strong_window_shapes.py RedoWindow; a stream refuses the
other shapes, experiments/stream_command.py).

All of it takes no simulated time and teaches nothing: no threshold
learns, no audit draws, and each pool's decoder is a fresh one built
from the pool's settings, so no unit's counters move and the run goes
on as if it were never asked. At zero time no deadline is missed, so a
final window the machine escalated only for a late answer
(escalation/deadlines.py) is kept here.
"""

import dataclasses
from typing import Any, Optional

import decsim.confidence.gap_join as gap_join
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models
import decsim.windows.escalate_edges as escalate_edges
import decsim.windows.routers as routers

ESCALATE = decoding_records.Verdict.ESCALATE


class ClosingDecisions:
    """The machine's path for a closing window, taken at zero time.

    decoder_by_pool holds a fresh decoder for each pool that has one,
    signal_by_pool the confidence signal of each pool an edge leaves,
    and requirement what a window model must offer all of them.
    """

    def __init__(
        self,
        router: routers.Router,
        edges: escalate_edges.EscalateEdges,
        decoder_by_pool: dict,
        signal_by_pool: dict,
        requirement: fault_models.DecoderFaultModelRequirement,
    ) -> None:
        self.router = router
        self.edges = edges
        self.decoder_by_pool = decoder_by_pool
        self.signal_by_pool = signal_by_pool
        self.requirement = requirement

    def decide(
        self, job: decoding_records.DecodeJob
    ) -> Optional[decoding_records.DecodeResult]:
        """The window's final result; None when its pool has no decoder."""
        pool = self.router.pool_of_stream(job.stream_id)
        result = self._decoded(pool, job)
        edge = self.edges.leaving(pool)
        while self._escalates(edge, job, result):
            pool = edge.target
            result = self._decoded(pool, job)
            edge = self.edges.leaving(pool)
        return result

    def threshold_of(self, stream_id: Any) -> Optional[float]:
        """The threshold the stream's entry edge reads now; None for no edge."""
        pool = self.router.pool_of_stream(stream_id)
        edge = self.edges.leaving(pool)
        if edge is None:
            return None
        return edge.policy.threshold_now()

    def _decoded(
        self, pool: Any, job: decoding_records.DecodeJob
    ) -> Optional[decoding_records.DecodeResult]:
        """The pool's answer, its solves joined when an edge reads them."""
        decoder = self.decoder_by_pool.get(pool)
        if decoder is None:
            return None
        signal = self.signal_by_pool.get(pool)
        if signal is None:
            return decoder.decode(job)
        held = _solves(decoder, job, signal)
        answer, _computation = gap_join.joined_answer(signal, held)
        return answer.result

    def _escalates(
        self,
        edge: Optional[ports.EscalateEdge],
        job: decoding_records.DecodeJob,
        result: Optional[decoding_records.DecodeResult],
    ) -> bool:
        if edge is None or result is None:
            return False
        verdict = edge.policy.verdict_without_learning(job, result)
        return verdict is ESCALATE


def _solves(
    decoder: ports.Decoder,
    job: decoding_records.DecodeJob,
    signal: ports.ConfidenceSignal,
) -> list:
    """One solve per class the signal pins, or the one plain decode.

    The window side submits the same jobs (windows/decode_requests.py).
    """
    classes = signal.forced_logical_classes
    if not classes:
        result = decoder.decode(job)
        return [gap_join.HeldSolve(job, result)]
    held = []
    for forced_class in classes:
        forced = dataclasses.replace(job, forced_logical_class=forced_class)
        result = decoder.decode(forced)
        solve = gap_join.HeldSolve(forced, result)
        held.append(solve)
    return held
