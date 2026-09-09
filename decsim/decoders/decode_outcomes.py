"""What a finished decode means and where its result goes.

gem5's commit stage retires what executed (src/cpu/o3/commit.hh:286);
here a weak result reaches its destination through the job's own
on_decoded (SimPy's callback on the event), and the window side answers
with the verdict once it holds the window's whole answer: it applies
the confidence threshold, commits the result as final or provisionally,
asks the strong tier itself, and tells this ledger the request is
resolved. A strong result teaches the policy and becomes one completion
per member request, delivered to each destination that waits for it now
and left in the output slot of the unit that produced it when the
destination's selection is still crossing the weak-to-strong link
(decoders/decoder_unit.py). Every terminal outcome goes out on
request_ended and service_ended; the record ledger listens.
"""

import dataclasses
from typing import Callable, Optional

import decsim.decoders.strong_requests as strong_requests_module
import decsim.records.decoding as decoding_records
import decsim.trace_source as trace_source


class DecodeOutcomes:
    """Delivers weak results, concludes strong ones, reports every end.

    A weak result goes to the destination the job carries and the
    window side answers with the verdict; a strong result teaches the
    policy and becomes one completion per member request.

    Trace sources: verdict_given(window_key, request_key, verdict) as
    the window side answers each weak request; request_ended(job,
    result, outcome, decode_output_ticks) and service_ended(job, tick)
    at every terminal outcome.
    """

    def __init__(
        self,
        engine,
        escalation_policy,
        strong_requests: strong_requests_module.StrongRequests,
        cancel_strong: Callable[[tuple], None],
    ) -> None:
        self.engine = engine
        self.escalation_policy = escalation_policy
        self.strong_requests = strong_requests
        # the manager's cancel, for a weak result the policy keeps
        self.cancel_strong = cancel_strong
        self.trace = _TraceSources()

    def deliver_weak(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """Hand one finished weak decode to the destination that asked.

        The destination is the job's own on_decoded: the window
        committer, or the confidence join in front of it when the
        window's answer takes two forced-class solves. The verdict on
        the window comes back later, through resolve_weak_request.

        The service ends when the decode and the confidence its evidence
        fed are both done: the join charges the signal's own computation
        on this unit while the delivery runs (decision D8), and a run
        whose signal only subtracts charges nothing.
        """
        job.on_decoded(job, result)
        service_ended_ticks = self.engine.now + job.soft_output_ticks
        self.trace.service_ended.fire(job, service_ended_ticks)

    def resolve_weak_request(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        verdict: decoding_records.Verdict,
    ) -> None:
        """The window side decided this weak request: close the attempt.

        A kept result cancels a live strong request; an escalated one
        has already asked the strong tier through the window side.
        """
        key = (job.operation_id, job.window_id)
        self.trace.verdict_given.fire(key, job.request_key, verdict)
        self.strong_requests.resolve_weak(key)
        is_escalated = verdict is decoding_records.Verdict.ESCALATE
        if not is_escalated:
            self.cancel_strong(key)  # no-op unless one is live or done
        outcomes = decoding_records.RequestProcessingOutcome
        processing = outcomes.PRIMARY_FORWARDED_FOR_DELIVERY
        if is_escalated:
            processing = outcomes.WEAK_AWAITED_STRONG
        self.trace.request_ended.fire(job, result, processing, self.engine.now)

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
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        deliveries: tuple,
    ) -> None:
        """The strong decode ended: teach the policy, deliver each completion.

        deliveries are the per-request completions taken before the
        job left its slot; each reaches its destination now or waits
        in the output slot of the unit that produced it.
        """
        self.strong_requests.finish_service(job)
        self.escalation_policy.learn_from_strong_result(
            job.strong_decode_for, result
        )
        for held in deliveries:
            self.complete_strong(held)
        self.trace.service_ended.fire(job, self.engine.now)

    def complete_strong(
        self, completion: strong_requests_module.StrongCompletion
    ) -> None:
        """Deliver a strong result to the destination that waits for it.

        A destination whose selection is still on its way leaves the
        result in the output slot of the unit that produced it, the way
        a sender keeps the packet until the far side accepts it (gem5
        port.hh:244-255).
        """
        if not self.strong_requests.complete(completion):
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
        window_key = (request_key.operation_id, request_key.window_id)
        # every finished strong result was produced by a unit, and the
        # unit is read at the decode's end, before its slot frees
        assert unit is not None, f"strong result for {window_key} has no unit"
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

    def report_service(self, job: decoding_records.DecodeJob) -> None:
        """A physical decode ended outside its conclusion: aborted."""
        self.trace.service_ended.fire(job, self.engine.now)


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the decode outcomes reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    verdict_given: trace_source.TraceSource = trace_source.new_source()
    request_ended: trace_source.TraceSource = trace_source.new_source()
    service_ended: trace_source.TraceSource = trace_source.new_source()
