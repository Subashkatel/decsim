"""What a finished decode means and where its result goes.

gem5's commit stage retires what executed (src/cpu/o3/commit.hh:286);
here a weak result is put to the escalation policy, which keeps it or
asks for the strong tier, and then reaches its destination through the
job's own on_decoded (SimPy's callback on the event); a strong result
becomes one completion per member request, delivered to each
destination that waits for it now and held for one whose selection is
still crossing the weak-to-strong link (StrongRequests). Every terminal
outcome is reported to the record ledger. The escalation's services
seam stays until slice 7 makes the policy a port that only decides.
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
        services,
        strong_requests: strong_requests_module.StrongRequests,
        records,
        cancel_strong: Callable[[tuple], None],
    ) -> None:
        self.engine = engine
        self.escalation_policy = escalation_policy
        # the EscalationServices seam (the window manager's escalation),
        # bound by the root after the window side is built (slice 7)
        self.services = services
        self.strong_requests = strong_requests
        # the listener on the two terminal callbacks
        self.records = records
        # the manager's cancel, for a weak result the policy keeps
        self.cancel_strong = cancel_strong

    def conclude_weak(
        self, job: message.DecodeJob, result: message.DecodeResult
    ) -> None:
        """Decide and deliver a weak outcome.

        The policy keeps the result (a live strong request is cancelled)
        or holds it for the strong tier (the selection is sent and the
        strong result is awaited); either way the destination hears now
        through job.on_decoded, with awaiting_strong_result set first.
        """
        key = (job.op_id, job.window_id)
        outcome = message.DecodeOutcome(job, result)
        directive = self.escalation_policy.on_decode_outcome(
            outcome, self.services
        )
        self.strong_requests.resolve_weak(key)
        awaiting = directive.directive is message.Directive.AWAIT_STRONG
        if directive.directive is message.Directive.FINALIZE:
            self.cancel_strong(key)  # no-op unless one is live/held
        if awaiting:
            self._await_strong_result(job, key, directive)
        job.awaiting_strong_result = awaiting  # BEFORE the commit callback
        job.on_decoded(job, result)
        processing = (
            message.RequestProcessingOutcome.PRIMARY_FORWARDED_FOR_DELIVERY
        )
        if awaiting:
            processing = message.RequestProcessingOutcome.WEAK_AWAITED_STRONG
        self.records.request_ended(job, result, processing, self.engine.now)
        self.records.service_ended(job, self.engine.now)

    def conclude_strong(
        self,
        job: message.DecodeJob,
        result: message.DecodeResult,
        deliveries: tuple,
    ) -> None:
        """The strong decode ended: tell the policy, deliver each completion.

        deliveries are the per-request completions taken before the
        unit was freed; each reaches its destination now or is held.
        """
        self.strong_requests.finish_service(job)
        outcome = message.DecodeOutcome(job, result)
        # FINALIZE_STRONG
        self.escalation_policy.on_decode_outcome(outcome, self.services)
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
        request_job.on_decoded(request_job, held.completion.result)
        self.records.request_ended(
            held.request_job,
            held.completion.result,
            message.RequestProcessingOutcome.STRONG_FORWARDED_FOR_DELIVERY,
            held.decode_output_ticks,
        )

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

    def _await_strong_result(
        self,
        job: message.DecodeJob,
        key: tuple,
        directive: message.OutcomeDirective,
    ) -> None:
        """Send the escalation; the strong result is selected on delivery."""
        serial_job = None
        if directive.extra is not None:
            serial_job = directive.extra.job
        strong_request_key = directive.strong_request_key
        deferred = serial_job is None and strong_request_key is not None
        if serial_job is None and not deferred:
            (carrier,) = self.strong_requests.carriers_for(key)
            strong_request_key = carrier.request_job.request_key
        self.services.prepare_strong_selection(
            job,
            strong_request_key,
            serial_job,
            deferred=deferred,
            on_selection_delivered=lambda: self._select_strong_result(
                key, strong_request_key
            ),
        )
        self.strong_requests.begin_selection(key, strong_request_key)

    def _select_strong_result(
        self, key: tuple, request_key: message.DecoderRequestKey
    ) -> None:
        """Make one strong completion eligible once the selection landed."""
        held = self.strong_requests.select(key, request_key)
        if held is not None:
            self.complete_strong(held)
