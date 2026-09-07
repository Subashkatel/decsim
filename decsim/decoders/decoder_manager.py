"""The decoder manager: the facade that gives ready windows a decoder unit.

A job waits in the WaitingJobs of its pool, the DecodeDispatcher places
it on the DecoderUnit the DecoderPool offers, the DecodeService stages
its input into that unit's memory and starts the routed decoder once the
input landed and the window owes no boundary, the GapJoins spawn the
split-gap sibling at service start, the StrongRequests say which
destination waits for which strong result, and the DecodeOutcomes
decide what a finished decode means and deliver it through the job's
on_decoded. The facade implements the DecodeQueue port (admits, cancels,
withdraws, releases, awaits and accepts a strong selection, settles) and
wires the six, the shape of gem5's
cache (BaseCache owns its MSHR queue, write buffer and tags, each one
job, and implements the ports: src/mem/cache/base.hh). One job reads as
enqueue, dispatcher.run, service.dispatch_to, service.begin,
decode_completed, outcomes.conclude_weak, job.on_decoded.
"""

from typing import Callable, Optional

import decsim.decoders.decode_dispatch as decode_dispatch
import decsim.decoders.decode_outcomes as decode_outcomes
import decsim.decoders.decode_queue as decode_queue
import decsim.decoders.decode_service as decode_service
import decsim.decoders.decoder_memory as decoder_memory_module
import decsim.decoders.decoder_memory_transfer as staging_module
import decsim.decoders.decoder_pool as decoder_pool_module
import decsim.decoders.gap_joins as gap_joins_module
import decsim.decoders.strong_requests as strong_requests_module
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records


class DecoderManager:
    """Admits, cancels, withdraws, releases and settles every decode.

    Seven attributes: the six one-job components the module docstring
    names, and the engine it logs and reads the clock on.
    """

    def __init__(
        self,
        engine,
        *,
        router,
        scheduler,
        unit_pools: Optional[dict] = None,
        num_units: int = 1,
        bulk_strong: bool = False,
        decoder_memory: Optional[
            decoder_memory_module.DecoderMemoryConfig
        ] = None,
        escalation_policy,
        link=None,
    ):
        if unit_pools is None:
            unit_pools = {"default": num_units}
        self.engine = engine
        pool = decoder_pool_module.DecoderPool(
            router, unit_pools, decoder_memory
        )
        self.strong_requests = strong_requests_module.StrongRequests()
        self.queue = decode_queue.WaitingJobs(
            engine,
            scheduler,
            pool.units_by_pool,
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
            pool,
            staging,
            self.strong_requests,
            self.gap_joins,
            on_completed=self.decode_completed,
            dispatch=self.dispatch,
        )
        self.dispatcher = decode_dispatch.DecodeDispatcher(
            self.queue, pool, self.service
        )
        self.outcomes = decode_outcomes.DecodeOutcomes(
            engine,
            escalation_policy,
            self.strong_requests,
            cancel_strong=self.cancel_strong,
        )

    @property
    def pool(self) -> decoder_pool_module.DecoderPool:
        """The pool the dispatcher and the service share."""
        return self.service.pool

    def copy_sources(self) -> list:
        """The copy_made sources of the manager's own hops, in hop order."""
        return [self.service.staging.copy_made, self.gap_joins.copy_made]

    # ---------------------------------------------------------- admission

    def enqueue(
        self, job: decoding_records.DecodeJob, send_input=None, on_decoded=None
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

    def enqueue_without_input(
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
        job = decoding_records.DecodeJob(
            operation_id=-1,
            window_id=0,
            round_count=round_count,
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
            key = (job.operation_id, job.window_id)
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
        self.outcomes.report_request(
            job,
            None,
            decoding_records.RequestProcessingOutcome.WEAK_WITHDRAWN_FOR_STRONG_WINDOW,
            None,
        )
        self.engine.log(
            decode_queue.LOG_SOURCE,
            f"WITHDRAW {job.label} (invalidated before start)",
        )
        self.dispatcher.run()

    # ------------------------------------------------- the strong requests

    def await_strong_result(
        self, window_key: tuple, request_key: window_records.DecoderRequestKey
    ) -> None:
        """The window asked for this request's strong result.

        Its selection is on the weak-to-strong link; the ledger holds
        the result for the window until the selection lands.
        """
        self.strong_requests.begin_selection(window_key, request_key)

    def accept_selection(
        self, window_key: tuple, request_key: window_records.DecoderRequestKey
    ) -> None:
        """The selection landed: the request's result may reach the window."""
        self.outcomes.select_strong_result(window_key, request_key)

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
            self.outcomes.report_request(
                held.request_job,
                held.result,
                decoding_records.RequestProcessingOutcome.STRONG_COMPLETED_DISCARDED,
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

    # ------------------------------------------------------------ settling

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

    # ------------------------------------------------- the decode's end

    def decode_completed(self, job: decoding_records.DecodeJob, result) -> None:
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

    def _gap_half_done(self, job: decoding_records.DecodeJob, result) -> None:
        job.completed = True
        self.service.free(job)
        self.service.release_input(job)
        self.engine.log(
            decode_queue.LOG_SOURCE, f"DECODE DONE {job.label} (gap half)"
        )
        sibling_weight = None
        if result is not None:
            sibling_weight = result.gap_half_weight
        key = job.gap_sibling_for
        joined = self.gap_joins.sibling_done(key, sibling_weight)
        if joined is not None:
            held_job, held_result = joined
            self._finish_weak(held_job, held_result)
        self.dispatcher.run()

    def _strong_decode_done(
        self, job: decoding_records.DecodeJob, result
    ) -> None:
        self.service.release_input(job)
        now = self.engine.now
        deliveries = self.strong_requests.deliveries_for(job, result, now)
        job.completed = True
        self.service.free(job)
        members = self.strong_requests.members_of(job)
        self.service.release_inputs(members)
        self.outcomes.conclude_strong(job, result, deliveries)
        self.dispatcher.run()

    def _external_decode_done(self, job: decoding_records.DecodeJob) -> None:
        job.completed = True
        self.service.free(job)
        # An external job that carried syndrome payloads through the
        # transport holds stored input like any other request, so its
        # credits come back before its callback runs and before the
        # same-tick drain; a self-contained job holds none and this is a
        # no-op.
        self.service.release_input(job)
        pool_tag = decode_queue.pool_tag_of(job.pool)
        free_now = self.service.free_unit_count(job.pool)
        self.engine.log(
            decode_queue.LOG_SOURCE,
            f"DECODE DONE {job.label} ({pool_tag}units free now {free_now})",
        )
        job.on_done()
        self.dispatcher.run()

    def _weak_decode_done(
        self, job: decoding_records.DecodeJob, result
    ) -> None:
        job.completed = True
        self.service.free(job)
        if self.gap_joins.has_join(job):
            # the unit is free either way; the OUTCOME waits at the join
            self.service.release_input(job)
        joined_result = self.gap_joins.take_weak_result(job, result)
        if joined_result is None:
            self.dispatcher.run()
            return
        self._finish_weak(job, joined_result)

    def _finish_weak(self, job: decoding_records.DecodeJob, result) -> None:
        """The weak outcome concluded; the memory and the queue move on."""
        self.outcomes.conclude_weak(job, result)
        self.service.release_input(job)
        self.dispatcher.run()

    # ------------------------------------------------- cancelling strong

    def _cancel_staged_strong(
        self, live: strong_requests_module.LiveStrongRequest
    ) -> None:
        """Cancelled in its slot before its decode started."""
        job = live.service_job
        job.cancelled = True
        self.service.evict(job)
        self.service.release_input(live.request_job)
        self.outcomes.report_request(
            live.request_job,
            None,
            decoding_records.RequestProcessingOutcome.STRONG_CANCELLED_WHILE_STAGED,
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
        self.outcomes.report_request(
            live.request_job,
            None,
            decoding_records.RequestProcessingOutcome.STRONG_CANCELLED_BEFORE_DISPATCH,
            None,
        )

    def _cancel_running_strong(
        self, live: strong_requests_module.LiveStrongRequest
    ) -> None:
        job = live.service_job
        job.service_cancelled_request_keys.add(live.request_job.request_key)
        if self.strong_requests.has_survivors(job):
            self.service.release_input(live.request_job)
            self.outcomes.report_request(
                live.request_job,
                None,
                decoding_records.RequestProcessingOutcome.STRONG_CANCELLED_MEMBER_SERVICE_CONTINUED,
                None,
            )
            return
        job.cancelled = True
        self.service.abort(job)
        self.service.release_input(live.request_job)
        self.outcomes.report_service(job)
        self.dispatcher.run()
        self.outcomes.report_request(
            live.request_job,
            None,
            decoding_records.RequestProcessingOutcome.STRONG_CANCELLED_DURING_SERVICE,
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


def _refuse_spent_job(job: decoding_records.DecodeJob) -> None:
    """A DecodeJob is submitted once; a live or terminal one is refused."""
    spent = _spent_state(job)
    if spent is None:
        return
    raise RuntimeError(
        f"decode job {job.label!r} for window "
        f"({job.operation_id}, {job.window_id}) has already been {spent}: a "
        "DecodeJob is submitted once, build a new one"
    )


def _spent_state(job: decoding_records.DecodeJob) -> Optional[str]:
    if job.cancelled:
        return "cancelled"
    if job.completed:
        return "completed"
    if job.submitted:
        return "admitted"
    return None


def _is_boundary_owed(job: decoding_records.DecodeJob) -> bool:
    if job.gate is None:
        return False
    return not job.gate.may_start(job)


def _is_live_weak_window_job(
    job: decoding_records.DecodeJob, window_key: tuple
) -> bool:
    key = (job.operation_id, job.window_id)
    if key != window_key:
        return False
    if job.strong_decode_for is not None:
        return False
    if job.on_done is not None:
        return False
    return not job.cancelled


def _unsettled_text(unsettled: dict) -> str:
    parts = []
    for state, keys in unsettled.items():
        parts.append(f"{state}: {keys}")
    return "; ".join(parts)
