"""The decoder manager: the facade that gives ready windows a decoder unit.

A job waits in the WaitingJobs, the DecodeDispatcher places
it on the DecoderUnit the DecoderPool offers, the DecodeService stages
its input into that unit's memory and starts the manager's decoder once
the input landed and the window owes no boundary, the StrongRequests say
which destination waits for which strong result, and the DecodeOutcomes
deliver a finished decode through the job's on_decoded and close the
request when the window side answers. The manager schedules, says when
a job's input moves, and returns results; it executes no send, holds no
join and holds no result for a third party. The facade implements the
DecodeQueue port (admits, cancels, withdraws, releases, awaits and
accepts a strong selection, resolves a weak request, settles) and wires
the five, the shape of gem5's cache (BaseCache owns its MSHR queue,
write buffer and tags, each one job, and implements the ports:
src/mem/cache/base.hh). One job reads as enqueue, dispatcher.run,
service.dispatch_to, service.begin, decode_completed,
outcomes.deliver_weak, job.on_decoded.

A run has one manager per side: the chip's over the default pool and,
when windows may escalate, the host's over the strong pool, two
instances of this class in the decoders part (decsim/build/decoders.py),
which is LATTE's shape
(2509.03954 lines 24-25 and 705-720: the local decoder on the control
FPGA has no scheduler, the host's Global Dynamic Scheduler owns the
decode queue and the thread pool). The StrongRequests ledger is one
component both bind, since a strong request is opened by the chip side
and served by the host side.
"""

import functools
from collections.abc import Callable
from typing import Optional

import decsim.config as config
import decsim.decoders.decode_dispatch as decode_dispatch
import decsim.decoders.decode_outcomes as decode_outcomes
import decsim.decoders.decode_queue as decode_queue
import decsim.decoders.decode_service as decode_service
import decsim.decoders.decoder_memory_transfer as staging_module
import decsim.decoders.decoder_pool as decoder_pool_module
import decsim.decoders.strong_requests as strong_requests_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.log_sources as log_sources
import decsim.records.seeds as seed_records
import decsim.records.windows as window_records


class DecoderManager:
    """Admits, cancels, withdraws, releases and settles every decode.

    Five attributes: four of the one-job components the module
    docstring names, and the engine it logs and reads the clock on. The
    decoder, the ledger and the escalation policy are ports the root
    binds, and the parts read them through the manager, as a gem5
    cache's packet queue holds the cache that owns it
    (gem5 src/mem/cache/base.hh:185-189).
    """

    # the one decoder this side's units run, bound straight to it as a
    # gem5 cache's port is bound to its one peer
    # (configs/learning_gem5/part1/caches.py:86-88); None on a run that
    # plans no windows and names no decoder
    decoder = ports.Port(ports.Decoder, optional=True)
    # one ledger for both sides: the chip's side opens a strong request
    # and the host's side serves it
    strong_requests = ports.Port(strong_requests_module.StrongRequests)
    # the switching policy, which learns from a strong result; None on a
    # run with no switching
    escalation_policy = ports.Port(ports.EscalationPolicy, optional=True)

    def __init__(
        self,
        engine,
        *,
        scheduler,
        pool_settings: decoder_pool_module.PoolSettings,
        bulk_strong: bool = False,
        clock: Optional[config.Clock] = None,
        dispatch_cycles: int = 0,
    ):
        self.engine = engine
        pool = decoder_pool_module.DecoderPool(self, pool_settings)
        is_strong_pool = pool_settings.name == decode_queue.STRONG_POOL
        merges_strong = bulk_strong and is_strong_pool
        self.queue = decode_queue.WaitingJobs(
            engine, scheduler, self, merges_strong
        )
        transport = staging_module.CancellableDecoderMemoryTransfer(engine)
        staging = staging_module.DecoderInputStaging(
            transport,
            engine,
            pool_settings.copies_input,
            pool_settings.formation,
        )
        self.service = decode_service.DecodeService(
            engine,
            pool,
            staging,
            self,
            clock=clock,
            dispatch_cycles=dispatch_cycles,
        )
        self.dispatcher = decode_dispatch.DecodeDispatcher(
            self.queue, pool, self.service
        )
        self.outcomes = decode_outcomes.DecodeOutcomes(engine, self)

    def start(self) -> None:
        """Hear every row of the decoder say a model pins no class.

        The decoder is a port, so its rows are known once the root has
        bound it, which is gem5's init (src/sim/sim_object.hh:186-194).
        """
        decoders = (self.decoder,)
        for row in decoder_pool_module.decoder_rows(decoders):
            row.forced_solve_unavailable.connect(self.report_unpinnable_model)

    def report_unpinnable_model(self, model, reason: str) -> None:
        """Say once per model that its windows can pin no logical class.

        Compiling the model is where that is known. A confidence built
        from forced-class solves reads no gap on such a model, so
        without this line the run shows one unexplained escalation per
        window of it. The manager narrates it because the manager owns
        the rows that report it: the line is in the run's log whether
        or not anything is observing.
        """
        detector_count = len(model.detector_ids)
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"NO FORCED SOLVE on a {detector_count}-detector window model: "
            f"{reason}",
        )

    @property
    def pool(self) -> decoder_pool_module.DecoderPool:
        """The pool the dispatcher and the service share."""
        return self.service.pool

    def copy_sources(self) -> list:
        """The copy_made sources of the manager's own hops, in hop order."""
        return [self.service.staging.trace.copy_made]

    def reference_sources(self) -> list:
        """The sources of the inputs this run reads instead of copying.

        A tier whose input is read in place reports a reference where a
        copying tier reports a copy (<tier>.input, decoders/settings.py).
        """
        return [self.service.staging.trace.hold_registered]

    def input_fold(self):
        """The input a window's gate hands its boundary mask to.

        The manager owns the unit memory the fold writes, so the window
        side asks the manager for the end that performs it rather than
        writing that memory itself (decisions.md D11).
        """
        return self.service.staging

    def run_seed_children(self) -> tuple:
        """The manager's own scheduler, whose rule may draw from the seed."""
        path = (seed_records.RunSeedPathSegment("field", "scheduler"),)
        scheduler = self.queue.scheduler
        child = seed_records.RunSeedChild(path, scheduler)
        return (child,)

    def input_transport(self):
        """The transport that moves an input into a unit's memory.

        The manager owns the hop, so it names it; a caller that needs to
        seed or observe the transport asks the manager rather than
        reaching through its service and its staging.
        """
        return self.service.staging.transport

    # ---------------------------------------------------------- admission

    def enqueue(
        self, job: decoding_records.DecodeJob, send_input=None, on_decoded=None
    ) -> None:
        """Admit once and queue; the rounds stay in the weak syndrome buffer.

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
        is_strong = job.kind is decoding_records.DecodeJobKind.STRONG_REDECODE
        if is_strong and not self.strong_requests.is_live_request(job):
            # cancelled across the link; its credits may admit another
            self.service.release_input(job)
            self.dispatcher.run()
            return
        self.queue.add(job)
        pool_tag = decode_queue.pool_tag_of(self.pool.name)
        queue_length = len(self.queue.waiting)
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"{job.label} READY -> enqueue "
            f"({pool_tag}ready-queue length = {queue_length})",
        )
        self.dispatcher.run()

    def enqueue_without_input(
        self,
        round_count: int,
        on_done: Callable[[], None],
        label: str = "external",
        code: Optional[str] = None,
        spatial_nodes: Optional[int] = None,
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
            kind=decoding_records.DecodeJobKind.SELF_CONTAINED,
        )
        self.queue.add(job)
        self.dispatcher.run()

    def dispatch(self) -> None:
        """Place every waiting job that fits; the service's trigger."""
        self.dispatcher.run()

    def release_parked(self, window_key: tuple) -> None:
        """The window's last boundary arrived: start its parked decode.

        A job still in transfer passes the gate at its own landing instead.
        """
        self.service.mark_startable(window_key)
        for job in self.service.parked_jobs():
            key = (job.operation_id, job.window_id)
            if key != window_key:
                continue
            if decode_service.is_boundary_owed(job):
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
        jobs = self._find_window_jobs(window_key)
        if not jobs:
            raise RuntimeError(
                f"no withdrawable decode for window {window_key}"
            )
        for job in jobs:
            self._withdraw_job(job)
        self.strong_requests.resolve_weak(window_key)
        self.dispatcher.run()

    def _withdraw_job(self, job: decoding_records.DecodeJob) -> None:
        """Take one unstarted job out of its queue or its slot."""
        if job.service_started or job.completed:
            raise RuntimeError(
                f"{job.label} cannot be withdrawn: its decode already started"
            )
        job.cancelled = True
        was_queued = self.queue.remove(job)
        if was_queued:
            # the withdrawn request stops holding its rounds; the window
            # keeps its own claim on them for the job that replaces it
            self.service.cancel_input(job)
        elif job.unit is not None:
            self.service.evict(job)
        else:
            raise RuntimeError(
                f"{job.label} is neither queued nor resident; nothing to "
                "withdraw"
            )
        self.outcomes.report_request(
            job,
            None,
            decoding_records.RequestProcessingOutcome.WEAK_WITHDRAWN_FOR_STRONG_WINDOW,
            None,
        )
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"WITHDRAW {job.label} (invalidated before start)",
        )

    # ------------------------------------------------- the strong requests

    def await_strong_result(
        self, window_key: tuple, request_key: window_records.DecoderRequestKey
    ) -> None:
        """The window asked for this request's strong result.

        Its selection is on the weak-to-strong link; a result that
        finishes first waits in the unit that produced it until the
        selection lands.
        """
        self.strong_requests.begin_selection(window_key, request_key)

    def accept_selection(
        self, window_key: tuple, request_key: window_records.DecoderRequestKey
    ) -> None:
        """The selection landed: the request's result may reach the window."""
        if not self.strong_requests.select(window_key, request_key):
            return
        completion = self.service.take_strong_output(window_key)
        if completion is None:
            return
        self.outcomes.complete_strong(completion)

    def charge_soft_output(
        self, job: decoding_records.DecodeJob, ticks: int
    ) -> None:
        """Charge the confidence's own computation on the job's unit.

        Decision D8: the walk over a decode's growth reads the evidence
        that decode left behind, and the evidence and its reader are the
        same hardware (Toshio 2510.25222 lines 152-160), so the time is
        the unit's. The unit gives its compute back when the confidence
        it fed is done, and the window side waits for the same ticks
        before its answer moves on.
        """
        if ticks <= 0:
            return
        job.soft_output_ticks = ticks
        unit_name = job.decoding_unit_name
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"CONFIDENCE {job.label} on unit {unit_name}: {ticks} ticks",
        )

    def resolve_weak_request(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        verdict: decoding_records.Verdict,
    ) -> None:
        """The window side decided this weak request; close its attempt."""
        self.outcomes.resolve_weak_request(job, result, verdict)

    def close_companion_request(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """A forced-class solve whose window is answered by the other one."""
        self.outcomes.close_companion_request(job, result)
        self.read_result(job)

    def read_result(self, job: decoding_records.DecodeJob) -> None:
        """The window side has the result in hand: a held unit is free now.

        <tier>.result_blocks_unit true is a unit with no output buffer,
        stalled by back-pressure until its output is taken (Bascones et
        al. 2605.01035 lines 607-609 size FIFOs to avoid that stall); a
        tier that does not block gave the unit back at the decode's end,
        or when the walk charged on it ended, and has nothing to give back
        here. A result is read once: a forced-class solve the confidence
        join held was read then, and its later close or commit finds its
        unit already given back.
        """
        if not self.pool.blocks_unit:
            return
        if job.unit is None:
            return
        self.service.free(job)
        self.dispatcher.run()

    def cancel_strong(self, key: tuple) -> None:
        """Cancel an unneeded strong re-decode wherever it is.

        Queued, crossing the link, running, or finished and waiting in
        its unit's output slot; the window side asks when a weak result
        is kept (Toshio 2510.25222 lines 606-614, the ongoing strong
        computation halted), and nothing happens when no request is
        live or done. A cancel ends one request; it passes no verdict on
        the destination. A destination that keeps its weak result stops
        being a consumer because its weak decode has resolved, and a
        destination still waiting keeps its demand, so the cancelled
        request can be replaced in either position.
        """
        completion = self.service.take_strong_output(key)
        if completion is not None:
            self.outcomes.report_request(
                completion.request_job,
                completion.result,
                decoding_records.RequestProcessingOutcome.STRONG_COMPLETED_DISCARDED,
                completion.decode_output_ticks,
            )
        live = self.strong_requests.take_live(key)
        if live is None:
            if completion is not None:
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
        unsettled = self.strong_requests.unsettled()
        unclaimed = self.service.windows_holding_output()
        if unclaimed:
            unsettled["holding an unclaimed strong result"] = unclaimed
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
        """One decode finished: free the unit and settle the outcome.

        Which settlement is the job's kind, looked up, not asked for
        field by field.
        """
        if job.cancelled:
            self.service.release_input(job)
            self.dispatcher.run()
            return
        settle = SETTLE_BY_JOB_KIND[job.kind]
        settle(self, job, result)

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

    def _self_contained_decode_done(
        self, job: decoding_records.DecodeJob, result
    ) -> None:
        del result  # a self-contained decode reports no correction
        job.completed = True
        self.service.free(job)
        # An external job that carried syndrome payloads through the
        # transport holds stored input like any other request, so its
        # credits come back before its callback runs and before the
        # same-tick drain; a self-contained job holds none and this is a
        # no-op.
        self.service.release_input(job)
        pool_tag = decode_queue.pool_tag_of(job.pool)
        free_now = self.service.free_unit_count()
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"DECODE DONE {job.label} ({pool_tag}units free now {free_now})",
        )
        job.on_done()
        self.dispatcher.run()

    def _window_decode_done(
        self, job: decoding_records.DecodeJob, result
    ) -> None:
        """The decode ended; its result goes to the destination that asked.

        The unit's compute goes back once the confidence its evidence
        feeds is computed, since that walk runs on the unit (decision
        D8, charge_soft_output), unless the tier blocks on the result,
        in which case it goes back when the window side has read it
        (<tier>.result_blocks_unit). The delivery is what charges the
        walk, so it comes first.
        """
        job.completed = True
        self.outcomes.deliver_weak(job, result)
        if not self.pool.blocks_unit:
            self._free_after_the_walk(job)
        self.service.release_input(job)
        self.dispatcher.run()

    def _free_after_the_walk(self, job: decoding_records.DecodeJob) -> None:
        """Give the unit back now, or when the walk charged on it ends."""
        ticks = job.soft_output_ticks
        if ticks <= 0:
            self.service.free(job)
            return
        free = functools.partial(self._free_and_dispatch, job)
        label = f"confidence_walk_done({job.label})"
        self.engine.schedule(ticks, free, label=label)

    def _free_and_dispatch(self, job: decoding_records.DecodeJob) -> None:
        self.service.free(job)
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
        self.dispatcher.run()
        self.outcomes.report_request(
            live.request_job,
            None,
            decoding_records.RequestProcessingOutcome.STRONG_CANCELLED_DURING_SERVICE,
            None,
        )

    def _find_window_jobs(self, window_key: tuple) -> list:
        """Every live weak job of the window: one attempt, one or two jobs."""
        candidates = list(self.queue.waiting)
        residents = self.service.resident_jobs()
        candidates.extend(residents)
        found = []
        for job in candidates:
            if _is_live_weak_window_job(job, window_key):
                found.append(job)
        return found


# how each job kind is settled when its decode ends; the manager reads
# the job's declared kind instead of asking after four fields in turn
_JOB_KINDS = decoding_records.DecodeJobKind
SETTLE_BY_JOB_KIND = {
    _JOB_KINDS.WINDOW: DecoderManager._window_decode_done,
    _JOB_KINDS.STRONG_REDECODE: DecoderManager._strong_decode_done,
    _JOB_KINDS.STRONG_BATCH: DecoderManager._strong_decode_done,
    _JOB_KINDS.SELF_CONTAINED: DecoderManager._self_contained_decode_done,
}
# a new job kind says how it settles here, at the table, and not with a
# KeyError inside the completion of the first decode that has it
assert set(SETTLE_BY_JOB_KIND) == set(_JOB_KINDS), (
    "every decode job kind says how it is settled"
)


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


def _is_live_weak_window_job(
    job: decoding_records.DecodeJob, window_key: tuple
) -> bool:
    key = (job.operation_id, job.window_id)
    if key != window_key:
        return False
    if job.kind is not decoding_records.DecodeJobKind.WINDOW:
        return False
    return not job.cancelled


def _unsettled_text(unsettled: dict) -> str:
    parts = []
    for state, keys in unsettled.items():
        parts.append(f"{state}: {keys}")
    return "; ".join(parts)
