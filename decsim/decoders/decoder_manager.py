"""The decoder manager: the facade that gives ready windows a decoder unit.

A job waits in the WaitingJobs of its pool, the DecodeDispatcher places
it on the DecoderUnit the DecoderPool offers, the DecodeService stages
its input into that unit's memory and starts the routed decoder once the
input landed and the window owes no boundary, the GapJoins spawn the
split-gap sibling at service start, and the StrongRequests say which
destination waits for which strong result. The facade admits, cancels,
withdraws, releases and settles, and wires the rest, the shape of
gem5's cache (BaseCache owns its MSHR queue, write buffer and tags,
each one job, and implements the ports: src/mem/cache/base.hh). The
outcome of a finished decode (what the policy makes of it, where it
goes) and the switching study's terminal records are still here until
structural D moves them to DecodeOutcomes and the record ledger.

Wide state recorded: twelve attributes, the six components, the pool
they share, the engine, the escalation policy and its services seam
(slice 7), and the two record lists.
"""

import dataclasses
import enum
from typing import Callable, Optional

import decsim.decoders.decode_dispatch as decode_dispatch
import decsim.decoders.decode_queue as decode_queue
import decsim.decoders.decode_service as decode_service
import decsim.decoders.decoder_memory as decoder_memory_module
import decsim.decoders.decoder_memory_transfer as staging_module
import decsim.decoders.decoder_pool as decoder_pool_module
import decsim.decoders.gap_joins as gap_joins_module
import decsim.decoders.strong_requests as strong_requests_module
import decsim.message as message


class RequestProcessingOutcome(enum.Enum):
    """How one decode request ended, for the switching study's records."""

    PRIMARY_FORWARDED_FOR_DELIVERY = "primary_forwarded_for_delivery"
    WEAK_AWAITED_STRONG = "weak_awaited_strong"
    STRONG_FORWARDED_FOR_DELIVERY = "strong_forwarded_for_delivery"
    STRONG_COMPLETED_DISCARDED = "strong_completed_discarded"
    STRONG_CANCELLED_BEFORE_DISPATCH = "strong_cancelled_before_dispatch"
    STRONG_CANCELLED_WHILE_STAGED = "strong_cancelled_while_staged"
    STRONG_CANCELLED_DURING_SERVICE = "strong_cancelled_during_service"
    STRONG_CANCELLED_MEMBER_SERVICE_CONTINUED = (
        "strong_cancelled_member_service_continued"
    )
    WEAK_WITHDRAWN_FOR_STRONG_WINDOW = "weak_withdrawn_for_strong_window"


@dataclasses.dataclass(frozen=True)
class TerminalRequestRecord:
    """One decode request at its end: identity, input, ticks, outcome."""

    request_key: message.DecoderRequestKey
    input_round_lo: int
    input_round_hi: int
    input_round_count: int
    syndrome_bit_count: Optional[int]
    syndrome_weight: Optional[int]
    created_ticks: int
    admitted_ticks: Optional[int]
    ready_ticks: int
    dispatch_ticks: Optional[int]
    decode_output_ticks: Optional[int]
    service_key: Optional[message.DecoderServiceKey]
    soft_output: Optional[message.SoftOutput]
    terminal_processing_outcome: RequestProcessingOutcome


@dataclasses.dataclass(frozen=True)
class TerminalServiceRecord:
    """One decode service at its end: the requests it served and its ticks."""

    service_key: message.DecoderServiceKey
    pool: str
    original_request_keys: tuple[message.DecoderRequestKey, ...]
    completed_request_keys: tuple[message.DecoderRequestKey, ...]
    cancelled_request_keys: tuple[message.DecoderRequestKey, ...]
    input_round_count: int
    dispatch_ticks: int
    terminal_ticks: int
    service_ticks: int


class DecoderManager:
    """Admits, cancels, withdraws, releases and settles every decode."""

    def __init__(
        self,
        engine,
        *,
        router,
        scheduler,
        unit_pools: Optional[dict] = None,
        num_units: int = 1,
        bulk_strong: bool = False,
        capture_enabled: bool = False,
        decoder_memory: Optional[
            decoder_memory_module.DecoderMemoryConfig
        ] = None,
        escalation_policy,
        services,
        link=None,
    ):
        self.engine = engine
        if unit_pools is None:
            unit_pools = {"default": num_units}
        self.pool = decoder_pool_module.DecoderPool(
            router, unit_pools, decoder_memory
        )
        self.strong_requests = strong_requests_module.StrongRequests()
        self.queue = decode_queue.WaitingJobs(
            engine,
            scheduler,
            self.pool.units_by_pool,
            self.strong_requests,
            is_bulk_strong=bulk_strong,
        )
        # a router with a gap route turns the split-pair joins on; the
        # sibling's input rides the link, None sends nothing
        is_gap_split = router.gap is not None
        self.gap_joins = gap_joins_module.GapJoins(
            engine, link, self.enqueue, is_gap_split
        )
        transport = staging_module.CancellableDecoderMemoryTransfer(engine)
        staging = staging_module.DecoderInputStaging(transport, engine)
        self.service = decode_service.DecodeService(
            engine,
            self.pool,
            staging,
            self.strong_requests,
            self.gap_joins,
            on_completed=self.decode_completed,
            dispatch=self.dispatch,
        )
        self.dispatcher = decode_dispatch.DecodeDispatcher(
            self.queue, self.pool, self.service
        )
        self.escalation_policy = escalation_policy
        # the EscalationServices seam (the window manager)
        self.services = services
        self._terminal_request_records = None
        self._terminal_service_records = None
        if capture_enabled:
            self._terminal_request_records = []
            self._terminal_service_records = []

    # ---------------------------------------------------------- admission

    def enqueue(
        self, job: message.DecodeJob, send_input=None, on_decoded=None
    ) -> None:
        """Admit once and queue the request; its rounds stay in Buffer 0.

        ``send_input(on_landed)`` is called at dispatch, after a unit is
        assigned, to send the input over its link; it calls ``on_landed``
        at the delivery and returns the delay the link expects (the
        accelerator pattern: invoke the unit, then DMA its input into that
        unit's memory, then compute; Aladdin aladdin_sys_connection.h and
        dma_interface.h). ``None`` means the job carries no syndrome data.
        ``on_decoded(job, result)`` is where the result goes.
        """
        _refuse_spent_job(job)
        self.strong_requests.admit(job, self.engine.now)
        job.submitted = True
        job.send_input = send_input
        job.on_decoded = on_decoded
        is_strong = job.strong_decode_for is not None
        if is_strong and not self.strong_requests.is_live_request(job):
            # cancelled across the link; its credits may admit another
            self.service.release_input(job)
            self.dispatcher.run()
            return
        self.queue.add(job)
        self.dispatcher.run()

    def submit_decode(
        self,
        round_count: int,
        on_done: Callable[[], None],
        label: str = "external",
        code: Optional[str] = None,
        spatial_nodes: Optional[int] = None,
        hint: Optional[str] = None,
    ) -> None:
        """Queue a self-contained decode: a factory correction, an idle decode.

        The job carries no syndrome data, so it skips input transport and
        storage: its decoder-input round demand is zero and it charges no
        round credits.
        """
        job = message.DecodeJob(
            op_id=-1,
            window_id=0,
            n_rounds=round_count,
            ready_time=self.engine.now,
            on_done=on_done,
            label=label,
            code=code,
            spatial_nodes=spatial_nodes,
            hint=hint,
        )
        self.queue.add_quietly(job)
        self.dispatcher.run()

    def dispatch(self) -> None:
        """Place every waiting job that fits; the service's trigger."""
        self.dispatcher.run()

    def release_parked(self, window_key: tuple) -> None:
        """The window's last boundary arrived: start its parked decode.

        A job still in transfer passes the gate at its own landing instead.
        """
        for job in self.service.parked_jobs():
            key = (job.op_id, job.window_id)
            if key != window_key:
                continue
            if _is_boundary_owed(job):
                continue  # another dependency still owed
            job.is_parked = False
            self.service.restart_parked(job)
        self.dispatcher.run()  # a started decode may admit blocked stages

    def withdraw_window(self, window_key: tuple) -> None:
        """Take back one window's submitted, not-yet-started weak decode.

        The window is being rewritten (a strong window absorbs it), so its
        raw input is superseded. The attempt closes in the ledger and the
        caller resubmits a fresh job if the window is rebuilt.
        """
        job = self._find_window_job(window_key)
        if job is None:
            raise RuntimeError(
                f"no withdrawable decode for window {window_key}"
            )
        if job.service_started or job.completed:
            raise RuntimeError(
                f"{job.label} cannot be withdrawn: its decode already started"
            )
        job.cancelled = True
        was_queued = self.queue.remove(job)
        if was_queued:
            self.service.release_input(job)
        elif job.unit is not None:
            self.service.evict(job)
        else:
            raise RuntimeError(
                f"{job.label} is neither queued nor resident; nothing to "
                "withdraw"
            )
        self.strong_requests.resolve_weak(window_key)
        self._record_request(
            job,
            None,
            RequestProcessingOutcome.WEAK_WITHDRAWN_FOR_STRONG_WINDOW,
            None,
        )
        self.engine.log(
            decode_queue.LOG_SOURCE,
            f"WITHDRAW {job.label} (invalidated before start)",
        )
        self.dispatcher.run()

    # ------------------------------------------------- the strong requests

    def check_strong_route(
        self, weak_job: message.DecodeJob, strong_job: message.DecodeJob
    ) -> None:
        """Refuse a strong job that would route back to the weak decoder."""
        strong_decoder = self.pool.decoder_for(strong_job)
        weak_decoder = self.pool.decoder_for(weak_job)
        if strong_decoder is weak_decoder:
            raise RuntimeError(
                "Strong job routes to the same decoder as the weak job; "
                "pass a router (e.g. SwitchingRouter) that sends "
                "hint='strong' to a distinct decoder."
            )

    def cancel_strong(self, key: tuple) -> None:
        """Cancel an unneeded strong re-decode wherever it is.

        Queued, crossing the link, running, or held. A cancel ends one
        request; it passes no verdict on the destination. A destination
        that keeps its weak result stops being a consumer because its weak
        decode has resolved, and a destination still waiting keeps its
        demand, so the cancelled request can be replaced in either
        position.
        """
        held = self.strong_requests.take_held(key)
        if held is not None:
            self._record_request(
                held.request_job,
                held.completion.result,
                RequestProcessingOutcome.STRONG_COMPLETED_DISCARDED,
                held.decode_output_ticks,
            )
        live = self.strong_requests.take_live(key)
        if live is None:
            if held is not None:
                self.strong_requests.counts.cancelled += 1
            return
        job = live.service_job
        if job.unit is not None and not job.service_started:
            self._cancel_staged_strong(live)
        elif job.pool is None:
            self._cancel_queued_strong(live)
        else:
            self._cancel_running_strong(live)
        self.strong_requests.counts.cancelled += 1
        self.dispatcher.run()  # returned credits may admit a waiting head

    def admitted_strong_work_snapshot(self) -> tuple:
        """Each physical strong job once in its phase."""
        queue_memberships = {}
        for queued_job in self.queue.jobs():
            identity = id(queued_job)
            memberships = queue_memberships.setdefault(identity, [])
            memberships.append(queued_job)
        return self.strong_requests.snapshot(queue_memberships)

    # ------------------------------------------------- records and settling

    def terminal_request_records_snapshot(self) -> tuple:
        """The switching study's request records so far."""
        return tuple(self._terminal_request_records)

    def terminal_service_records_snapshot(self) -> tuple:
        """The switching study's service records so far."""
        return tuple(self._terminal_service_records)

    def check_decode_work_settled(self) -> None:
        """Require every admitted decode to reach a final result.

        Decoder-input storage settles with it, configured or not: nothing
        may still hold rounds, wait for round credits, or hold returned
        credits nobody drained. Every stored input is released
        unconditionally, so a leak on either path is a real defect rather
        than a tolerated one.
        """
        self.service.check_settled()
        unresolved_joins = self.gap_joins.unresolved_windows()
        if unresolved_joins:
            raise RuntimeError(
                f"run ended with split-gap joins unresolved: {unresolved_joins}"
            )
        unsettled = self.strong_requests.unsettled()
        held = self.service.units_holding_rounds()
        if held:
            unsettled["decoder memory still holding rounds"] = held
        if unsettled:
            detail = _unsettled_text(unsettled)
            raise RuntimeError(
                f"the run ended with decode work unsettled ({detail}): "
                "every window is final once the simulation is quiescent"
            )

    # ------------------------------------------------- outcomes

    def decode_completed(self, job: message.DecodeJob, result) -> None:
        """One decode finished: free the unit and settle the outcome."""
        if job.cancelled:
            self.service.discard_cancelled(job)
            self.dispatcher.run()
            return
        if job.gap_sibling_for is not None:
            self._gap_half_done(job, result)
            return
        if job.strong_decode_for is not None:
            self._strong_decode_done(job, result)
            return
        if job.on_done is not None:
            self._external_decode_done(job)
            return
        self._weak_decode_done(job, result)

    def _gap_half_done(self, job: message.DecodeJob, result) -> None:
        job.completed = True
        self.service.free(job)
        self.service.release_input(job)
        self.engine.log(
            decode_queue.LOG_SOURCE, f"DECODE DONE {job.label} (gap half)"
        )
        sibling_weight = None
        if result is not None:
            sibling_weight = result.gap_half_weight
        joined = self.gap_joins.sibling_done(
            job.gap_sibling_for, sibling_weight
        )
        if joined is not None:
            held_job, held_result = joined
            self._conclude_weak(held_job, held_result)
        self.dispatcher.run()

    def _strong_decode_done(self, job: message.DecodeJob, result) -> None:
        _validate_logical_observables(job, result)
        self.service.release_input(job)
        deliveries = self.strong_requests.deliveries_for(
            job, result, self.engine.now
        )
        job.completed = True
        self.service.free(job)
        members = self.strong_requests.members_of(job)
        self.service.release_inputs(members)
        self.strong_requests.finish_service(job)
        outcome = message.DecodeOutcome(job, result)
        # FINALIZE_STRONG
        self.escalation_policy.on_decode_outcome(outcome, self.services)
        for held in deliveries:
            self._complete_strong_result(held)
        self._record_service(job)
        self.dispatcher.run()

    def _external_decode_done(self, job: message.DecodeJob) -> None:
        job.completed = True
        self.service.free(job)
        # An external job that carried syndrome payloads through the
        # transport holds stored input like any other request, so its
        # credits come back before its callback runs and before the
        # same-tick drain; a self-contained submit_decode job holds none
        # and this is a no-op.
        self.service.release_input(job)
        pool_tag = decode_queue.pool_tag_of(job.pool)
        free_now = self.pool.free_count(job.pool)
        self.engine.log(
            decode_queue.LOG_SOURCE,
            f"DECODE DONE {job.label} ({pool_tag}units free now {free_now})",
        )
        job.on_done()
        self.dispatcher.run()

    def _weak_decode_done(self, job: message.DecodeJob, result) -> None:
        _validate_logical_observables(job, result)
        job.completed = True
        self.service.free(job)
        key = (job.op_id, job.window_id)
        if key in self.gap_joins.joins_by_window:
            # the unit is free either way; the OUTCOME waits at the join
            self.service.release_input(job)
        joined_result = self.gap_joins.take_weak_result(job, result)
        if joined_result is None:
            self.dispatcher.run()
            return
        self._conclude_weak(job, joined_result)

    def _conclude_weak(self, job: message.DecodeJob, result) -> None:
        """The weak outcome's decision and delivery.

        In split-pair mode this runs at the gap join, otherwise straight
        from decode end.
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
        processing = RequestProcessingOutcome.PRIMARY_FORWARDED_FOR_DELIVERY
        if awaiting:
            processing = RequestProcessingOutcome.WEAK_AWAITED_STRONG
        self._record_request(job, result, processing, self.engine.now)
        self._record_service(job)
        self.service.release_input(job)
        self.dispatcher.run()

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
        """Make one strong completion eligible only after WSD delivery."""
        held = self.strong_requests.select(key, request_key)
        if held is not None:
            self._complete_strong_result(held)

    def _complete_strong_result(
        self, held: strong_requests_module.HeldStrongCompletion
    ) -> None:
        """Deliver a strong result to the destination that waits for it.

        The ledger holds it when the demand is still on its way.
        """
        if not self.strong_requests.complete(held):
            return
        request_job = held.request_job
        request_job.on_decoded(request_job, held.completion.result)
        self._record_request(
            held.request_job,
            held.completion.result,
            RequestProcessingOutcome.STRONG_FORWARDED_FOR_DELIVERY,
            held.decode_output_ticks,
        )

    # ------------------------------------------------- cancelling strong

    def _cancel_staged_strong(
        self, live: strong_requests_module.LiveStrongRequest
    ) -> None:
        """Cancelled in its slot before its decode started."""
        job = live.service_job
        job.cancelled = True
        self.service.evict(job)
        self.service.release_input(live.request_job)
        self._record_request(
            live.request_job,
            None,
            RequestProcessingOutcome.STRONG_CANCELLED_WHILE_STAGED,
            None,
        )

    def _cancel_queued_strong(
        self, live: strong_requests_module.LiveStrongRequest
    ) -> None:
        job = live.service_job
        self.service.cancel_input(job)
        self.queue.remove(job)
        # Credits belong to the original request, never to a batch service
        # job, and the request is its own service job before dispatch.
        self.service.release_input(live.request_job)
        self._record_request(
            live.request_job,
            None,
            RequestProcessingOutcome.STRONG_CANCELLED_BEFORE_DISPATCH,
            None,
        )

    def _cancel_running_strong(
        self, live: strong_requests_module.LiveStrongRequest
    ) -> None:
        job = live.service_job
        job.service_cancelled_request_keys.add(live.request_job.request_key)
        if self.strong_requests.has_survivors(job):
            self.service.release_input(live.request_job)
            self._record_request(
                live.request_job,
                None,
                RequestProcessingOutcome.STRONG_CANCELLED_MEMBER_SERVICE_CONTINUED,
                None,
            )
            return
        job.cancelled = True
        self.service.abort(job)
        self.service.release_input(live.request_job)
        self._record_service(job)
        self.dispatcher.run()
        self._record_request(
            live.request_job,
            None,
            RequestProcessingOutcome.STRONG_CANCELLED_DURING_SERVICE,
            None,
        )

    def _find_window_job(self, window_key: tuple):
        candidates = self.queue.jobs()
        residents = self.service.resident_jobs()
        candidates.extend(residents)
        for job in candidates:
            if _is_live_weak_window_job(job, window_key):
                return job
        return None

    # ------------------------------------------------- records, private

    def _record_request(
        self,
        job: message.DecodeJob,
        result: Optional[message.DecodeResult],
        outcome: RequestProcessingOutcome,
        decode_output_ticks: Optional[int],
    ) -> None:
        if self._terminal_request_records is None:
            return
        window = job.window
        local_fragments = ()
        if job.decoder_input is not None:
            fragments = gap_joins_module.fragments_in(job.decoder_input)
            local_fragments = tuple(fragments)
        bit_count, weight = _syndrome_bit_count_and_weight(local_fragments)
        soft_output = None
        if result is not None:
            soft_output = result.soft_output
        record = TerminalRequestRecord(
            job.request_key,
            window.start_round,
            window.buffer_hi,
            job.n_rounds,
            bit_count,
            weight,
            job.request_created_ticks,
            job.request_admitted_ticks,
            job.ready_time,
            job.service_dispatch_ticks,
            decode_output_ticks,
            job.service_key,
            soft_output,
            outcome,
        )
        self._terminal_request_records.append(record)

    def _record_service(self, job: message.DecodeJob) -> None:
        if self._terminal_service_records is None:
            return
        if job.service_key is None:
            return
        original = job.service_original_request_keys
        cancelled = []
        completed = []
        for key in original:
            if key in job.service_cancelled_request_keys:
                cancelled.append(key)
            else:
                completed.append(key)
        dispatch = job.service_dispatch_ticks
        service_ticks = self.engine.now - dispatch
        record = TerminalServiceRecord(
            job.service_key,
            job.pool,
            original,
            tuple(completed),
            tuple(cancelled),
            job.n_rounds,
            dispatch,
            self.engine.now,
            service_ticks,
        )
        self._terminal_service_records.append(record)


def _refuse_spent_job(job: message.DecodeJob) -> None:
    """A DecodeJob is submitted once; a live or terminal one is refused."""
    spent = _spent_state(job)
    if spent is None:
        return
    raise RuntimeError(
        f"decode job {job.label!r} for window "
        f"({job.op_id}, {job.window_id}) has already been {spent}: a "
        "DecodeJob is submitted once, build a new one"
    )


def _spent_state(job: message.DecodeJob) -> Optional[str]:
    if job.cancelled:
        return "cancelled"
    if job.completed:
        return "completed"
    if job.submitted:
        return "admitted"
    return None


def _is_boundary_owed(job: message.DecodeJob) -> bool:
    if job.gate is None:
        return False
    return not job.gate.may_start(job)


def _validate_logical_observables(
    job: message.DecodeJob, result: message.DecodeResult
) -> None:
    logical_observables = result.logical_observables
    if logical_observables is None:
        return
    for observable_index, bit in enumerate(logical_observables):
        if bit not in (0, 1):
            raise ValueError(
                f"job ({job.op_id}, {job.window_id}) "
                f"logical_observables index {observable_index} must be "
                f"0 or 1, got {bit}"
            )


def _is_live_weak_window_job(job: message.DecodeJob, window_key: tuple) -> bool:
    key = (job.op_id, job.window_id)
    if key != window_key:
        return False
    if job.strong_decode_for is not None:
        return False
    if job.on_done is not None:
        return False
    return not job.cancelled


def _syndrome_bit_count_and_weight(fragments: tuple) -> tuple:
    """(bit count, set bits) of the landed input; None when bits are unknown."""
    if not fragments:
        return None, None
    bit_count = 0
    weight = 0
    for fragment in fragments:
        if fragment.bits is None:
            return None, None
        bit_count += len(fragment.bits)
        weight += sum(fragment.bits)
    return bit_count, weight


def _unsettled_text(unsettled: dict) -> str:
    parts = []
    for state, keys in unsettled.items():
        parts.append(f"{state}: {keys}")
    return "; ".join(parts)
