"""What a finished decode means and where its result goes.

gem5's commit stage retires what executed (src/cpu/o3/commit.hh:286);
here a weak result reaches its destination through the job's own
on_decoded (SimPy's callback on the event), and the window side answers
with the verdict once it holds the window's whole answer: it applies
the confidence threshold, commits the result as final or provisionally,
asks the strong tier itself, and tells this ledger the request is
resolved. A strong result teaches the policy and becomes one completion
per member request, delivered to each destination that waits for it now
and held for one whose selection is still crossing the weak-to-strong
link (StrongRequests). Every terminal outcome goes out on request_ended
and service_ended; the record ledger listens.
"""

import dataclasses
from typing import Callable, Optional

import decsim.decoders.strong_requests as strong_requests_module
import decsim.observe.trace_source as trace_source
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records


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
        """
        job.on_decoded(job, result)
        self.trace.service_ended.fire(job, self.engine.now)

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
            self.cancel_strong(key)  # no-op unless one is live/held
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
        unit was freed; each reaches its destination now or is held.
        """
        self.strong_requests.finish_service(job)
        self.escalation_policy.learn_from_strong_result(
            job.strong_decode_for, result
        )
        for held in deliveries:
            self.complete_strong(held)
        self.trace.service_ended.fire(job, self.engine.now)

    def complete_strong(
        self, held: strong_requests_module.HeldStrongCompletion
    ) -> None:
        """Deliver a strong result to the destination that waits for it.

        The ledger holds it when the demand is still on its way.
        """
        if not self.strong_requests.complete(held):
            return
        request_job = held.request_job
        request_job.on_decoded(request_job, held.result)
        self.trace.request_ended.fire(
            held.request_job,
            held.result,
            decoding_records.RequestProcessingOutcome.STRONG_FORWARDED_FOR_DELIVERY,
            held.decode_output_ticks,
        )

    def select_strong_result(
        self, key: tuple, request_key: window_records.DecoderRequestKey
    ) -> None:
        """The selection landed: a held strong completion reaches its window."""
        held = self.strong_requests.select(key, request_key)
        if held is not None:
            self.complete_strong(held)

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
