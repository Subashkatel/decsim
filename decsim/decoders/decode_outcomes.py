"""What a finished decode means and where its result goes.

gem5's commit stage retires what executed (src/cpu/o3/commit.hh:286);
here a weak result is put to the escalation policy for its verdict,
keep or escalate, and then reaches its destination through the job's
own on_decoded (SimPy's callback on the event) with the verdict on the
job, so the window side commits it as final or provisionally and asks
the strong tier itself; a strong result teaches the policy and becomes
one completion per member request, delivered to each destination that
waits for it now and held for one whose selection is still crossing the
weak-to-strong link (StrongRequests). Every terminal outcome is
reported to the record ledger.
"""

from typing import Callable, Optional

import decsim.decoders.strong_requests as strong_requests_module
import decsim.message as message


class DecodeOutcomes:
    """Concludes weak and strong results and reports every terminal one."""

    def __init__(
        self,
        engine,
        escalation_policy,
        strong_requests: strong_requests_module.StrongRequests,
        records,
        cancel_strong: Callable[[tuple], None],
    ) -> None:
        self.engine = engine
        self.escalation_policy = escalation_policy
        self.strong_requests = strong_requests
        # the listener on the two terminal callbacks
        self.records = records
        # the manager's cancel, for a weak result the policy keeps
        self.cancel_strong = cancel_strong

    def conclude_weak(
        self, job: message.DecodeJob, result: message.DecodeResult
    ) -> None:
        """Put the weak result to the policy and deliver it with the verdict.

        A kept result cancels a live strong request; an escalated one
        rides to its destination with awaiting_strong_result set, so the
        window side asks the strong tier for the window and commits the
        result provisionally.
        """
        key = (job.op_id, job.window_id)
        verdict = self.escalation_policy.verdict_for_weak_result(job, result)
        self.strong_requests.resolve_weak(key)
        is_escalated = verdict is message.Verdict.ESCALATE
        if not is_escalated:
            self.cancel_strong(key)  # no-op unless one is live/held
        job.awaiting_strong_result = is_escalated  # BEFORE the commit callback
        job.on_decoded(job, result)
        processing = (
            message.RequestProcessingOutcome.PRIMARY_FORWARDED_FOR_DELIVERY
        )
        if is_escalated:
            processing = message.RequestProcessingOutcome.WEAK_AWAITED_STRONG
        self.records.request_ended(job, result, processing, self.engine.now)
        self.records.service_ended(job, self.engine.now)

    def conclude_strong(
        self,
        job: message.DecodeJob,
        result: message.DecodeResult,
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
        self.records.service_ended(job, self.engine.now)

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
        self.records.request_ended(
            held.request_job,
            held.result,
            message.RequestProcessingOutcome.STRONG_FORWARDED_FOR_DELIVERY,
            held.decode_output_ticks,
        )

    def select_strong_result(
        self, key: tuple, request_key: message.DecoderRequestKey
    ) -> None:
        """The selection landed: a held strong completion reaches its window."""
        held = self.strong_requests.select(key, request_key)
        if held is not None:
            self.complete_strong(held)

    def report_request(
        self,
        job: message.DecodeJob,
        result: Optional[message.DecodeResult],
        outcome: message.RequestProcessingOutcome,
        decode_output_ticks: Optional[int],
    ) -> None:
        """A request ended outside a conclusion: cancelled or withdrawn."""
        self.records.request_ended(job, result, outcome, decode_output_ticks)

    def report_service(self, job: message.DecodeJob) -> None:
        """A physical decode ended outside its conclusion: aborted."""
        self.records.service_ended(job, self.engine.now)
