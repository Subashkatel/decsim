"""What a finished decode means and where its result goes.

As gem5's commit stage retires what executed (src/cpu/o3/commit.hh:286),
a weak result reaches its destination through the job's own on_decoded,
and the window side answers with the verdict once it holds the window's
whole answer, telling this ledger the request is resolved. A strong
result teaches the policy and becomes one completion per member request,
delivered to each destination waiting for it now and otherwise left in
the output slot of the unit that produced it (decoder_unit.py). Every
request's terminal outcome goes out on request_ended.
"""

import dataclasses
from typing import TYPE_CHECKING, Optional

import decsim.decoders.strong_requests as strong_requests_module
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.trace_source as trace_source

if TYPE_CHECKING:
    import decsim.decoders.decoder_manager as decoder_manager_module


class DecodeOutcomes:
    """Delivers weak results, concludes strong ones, reports every end.

    Trace sources: verdict_given(window_key, request_key, verdict) as
    the window side answers each weak request; request_ended(job,
    result, outcome, decode_output_ticks) at every terminal outcome.
    """

    def __init__(
        self,
        engine: engine_module.Engine,
        manager: "decoder_manager_module.DecoderManager",
    ) -> None:
        self.engine = engine
        self.manager = manager
        self.trace = _TraceSources()

    def deliver_weak(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """Hand one finished weak decode to the destination that asked.

        The destination is the job's on_decoded: the window committer,
        or the confidence join in front of it when the answer takes two
        forced-class solves. The verdict comes back later, through
        resolve_weak_request.
        """
        job.on_decoded(job, result)

    def resolve_weak_request(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        verdict: decoding_records.Verdict,
    ) -> None:
        """The window side decided this request: close the attempt.

        A revision along an escalate edge ended its request when it was
        delivered (complete_strong); only its attempt on the next edge
        closes here.
        """
        key = _destination_of(job)
        request_key = job.request_key
        self.trace.verdict_given.fire(key, request_key, verdict)
        self.manager.strong_requests.resolve_weak(key, request_key.pool)
        if job.strong_decode_for is not None:
            return
        is_escalated = verdict is decoding_records.Verdict.ESCALATE
        outcomes = decoding_records.RequestProcessingOutcome
        processing = outcomes.PRIMARY_FORWARDED_FOR_DELIVERY
        if is_escalated:
            processing = outcomes.WEAK_AWAITED_STRONG
        self.trace.request_ended.fire(job, result, processing, self.engine.now)

    def discard_weak_request(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """The decode read a boundary that changed: close its attempt.

        The window side decodes the window again on the boundary it
        holds now (windows/input_checks.py), so no verdict is given.
        """
        key = _destination_of(job)
        self.manager.strong_requests.resolve_weak(key, job.request_key.pool)
        outcome = decoding_records.RequestProcessingOutcome.WEAK_DISCARDED_STALE
        self.trace.request_ended.fire(job, result, outcome, self.engine.now)

    def close_companion_request(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """The forced-class solve whose window is answered by the other one."""
        outcomes = decoding_records.RequestProcessingOutcome
        self.trace.request_ended.fire(
            job,
            result,
            outcomes.WEAK_FORCED_CLASS_COMPANION,
            self.engine.now,
        )

    def conclude_strong(
        self, job: decoding_records.DecodeJob, deliveries: tuple
    ) -> None:
        """The strong decode ended: teach the policy, deliver each completion.

        deliveries are the per-request completions taken before the
        job left its slot; each reaches its destination now or waits
        in the output slot of the unit that produced it. The policy
        hears each request's own result under its window, so a batch's
        members are audited one by one (OnlineThreshold keys its audits
        by window).
        """
        self.manager.strong_requests.finish_service(job)
        # a strong decode runs on a pool an edge targets, and each such
        # pool's manager has a learner (build/escalation.py)
        learner = self.manager.strong_result_learner
        for held in deliveries:
            window_key = held.request_job.strong_decode_for
            learner.learn_from_strong_result(window_key, held.result)
        for held in deliveries:
            self.complete_strong(held)

    def complete_strong(
        self, completion: strong_requests_module.StrongCompletion
    ) -> None:
        """Deliver a strong result to the destination that waits for it.

        A destination whose selection is still on its way leaves the
        result in the output slot of the unit that produced it, as a
        sender keeps a packet until the far side accepts it (gem5
        port.hh:244-255).
        """
        strong_requests = self.manager.strong_requests
        if strong_requests.drop_abandoned(completion):
            self.trace.request_ended.fire(
                completion.request_job,
                completion.result,
                decoding_records.RequestProcessingOutcome.STRONG_COMPLETED_DISCARDED,
                completion.decode_output_ticks,
            )
            return
        if not strong_requests.complete(completion):
            self._hold_at_the_unit(completion)
            return
        request_job = completion.request_job
        request_job.on_decoded(request_job, completion.result)
        self.trace.request_ended.fire(
            completion.request_job,
            completion.result,
            decoding_records.RequestProcessingOutcome.STRONG_FORWARDED_FOR_DELIVERY,
            completion.decode_output_ticks,
        )

    def _hold_at_the_unit(
        self, completion: strong_requests_module.StrongCompletion
    ) -> None:
        """Leave the result in its producing unit's output slot."""
        unit = completion.unit
        request_key = completion.request_job.request_key
        window_key = (request_key.stream_id, request_key.window_id)
        # every finished strong result was produced by a unit, and the
        # unit is read at the decode's end, before its slot frees
        unit.hold_output(window_key, completion)

    def report_request(
        self,
        job: decoding_records.DecodeJob,
        result: Optional[decoding_records.DecodeResult],
        outcome: decoding_records.RequestProcessingOutcome,
        decode_output_ticks: Optional[int],
    ) -> None:
        """A request ended outside a conclusion: cancelled or withdrawn."""
        self.trace.request_ended.fire(job, result, outcome, decode_output_ticks)


def _destination_of(job: decoding_records.DecodeJob) -> tuple:
    """The window a job decodes for: its own, or the one it re-decodes."""
    if job.strong_decode_for is not None:
        return job.strong_decode_for
    return (job.stream_id, job.window_id)


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the decode outcomes reports, as one member."""

    verdict_given: trace_source.TraceSource = trace_source.new_source()
    request_ended: trace_source.TraceSource = trace_source.new_source()
