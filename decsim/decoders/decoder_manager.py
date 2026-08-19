"""Assigns decoder units to ready windows: a queue per pool, unit
assignment, input staged into that unit's memory, service, completion. The
strong tier's request bookkeeping is the StrongRequestLedger's; the terminal
records at the bottom are optional capture for the switching study."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from ..message import (DecodeJob, DecodeOutcome, DecodeResult,
                      DecoderRequestKey, DecoderServiceKey, SoftOutput)
from ..message import Directive
from .decoder_memory import DecoderMemory, DecoderMemoryConfig
from .decoder_memory_transfer import DecoderInputStaging, FixedLatencyDecoderMemoryTransfer
from .strong_escalation import HeldStrongCompletion, StrongRequestLedger
from ..config import fmt


class RequestProcessingOutcome(Enum):
    WEAK_FORWARDED_FOR_DELIVERY = "weak_forwarded_for_delivery"
    WEAK_AWAITED_STRONG = "weak_awaited_strong"
    STRONG_FORWARDED_FOR_DELIVERY = "strong_forwarded_for_delivery"
    STRONG_COMPLETED_DISCARDED = "strong_completed_discarded"
    STRONG_CANCELLED_BEFORE_DISPATCH = "strong_cancelled_before_dispatch"
    STRONG_CANCELLED_DURING_SERVICE = "strong_cancelled_during_service"
    STRONG_CANCELLED_MEMBER_SERVICE_CONTINUED = (
        "strong_cancelled_member_service_continued")


@dataclass(frozen=True)
class TerminalRequestRecord:
    request_key: DecoderRequestKey
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
    service_key: Optional[DecoderServiceKey]
    soft_output: Optional[SoftOutput]
    terminal_processing_outcome: RequestProcessingOutcome


@dataclass(frozen=True)
class TerminalServiceRecord:
    service_key: DecoderServiceKey
    pool: str
    original_request_keys: tuple[DecoderRequestKey, ...]
    completed_request_keys: tuple[DecoderRequestKey, ...]
    cancelled_request_keys: tuple[DecoderRequestKey, ...]
    input_round_count: int
    dispatch_ticks: int
    terminal_ticks: int
    service_ticks: int


class DecoderManager:
    def __init__(self, engine, *, router, scheduler,
                 unit_pools: Optional[dict] = None, num_units: int = 1,
                 bulk_strong: bool = False,
                 lane_policy=None, log_name: str = "DecoderCluster",
                 capture_enabled: bool = False,
                 decoder_memory_transfer=None,
                 decoder_memory: Optional[DecoderMemoryConfig] = None,
                 escalation_policy, services, on_window_decoded: Callable,
                 on_strong_window_decoded: Callable):
        self.engine = engine
        self.router = router
        self.decoder_memory_transfer = (
            FixedLatencyDecoderMemoryTransfer(engine)
            if decoder_memory_transfer is None else decoder_memory_transfer
        )
        transport_engine = getattr(self.decoder_memory_transfer, "engine", engine)
        if transport_engine is not engine:
            raise ValueError("decoder_memory_transfer uses a different engine")
        self.staging = DecoderInputStaging(self.decoder_memory_transfer)
        self.scheduler = scheduler
        self.lane_policy = lane_policy
        self.bulk_strong = bulk_strong
        self.log_name = log_name
        self._terminal_request_records = [] if capture_enabled else None
        self._terminal_service_records = [] if capture_enabled else None

        self.escalation_policy = escalation_policy
        self.services = services                     # the EscalationServices seam (the window manager)
        self.on_window_decoded = on_window_decoded
        self.on_strong_window_decoded = on_strong_window_decoded

        if unit_pools is None:
            unit_pools = {"default": num_units}
        if "default" not in unit_pools:
            raise ValueError(f'unit_pools must include a "default" pool '
                             f'(got {sorted(unit_pools)})')
        for pool_name, units in unit_pools.items():
            if units < 1:
                raise ValueError(
                    f"pool {pool_name!r} needs at least 1 unit (got {units})")

        self.unit_totals = dict(unit_pools)
        self.pool_free = dict(unit_pools)
        # Every unit is a numbered engine with its own input memory.
        self._free_units = {pool: list(range(n)) for pool, n in unit_pools.items()}
        self.decoder_memories = {
            (pool, unit): DecoderMemory(
                pool, unit, None if decoder_memory is None else decoder_memory.capacity_for(pool))
            for pool, n in unit_pools.items() for unit in range(n)}
        self._dispatching = False
        self.num_units = self.unit_totals["default"]
        self.ready: list[DecodeJob] = []
        self.pool_ready: dict[str, list] = {
            p: [] for p in self.unit_totals if p != "default"}
        self.queue_log: list[tuple[int, int]] = []

        self.strong = StrongRequestLedger()

    @property
    def strong_needed(self) -> int:
        return self.strong.strong_needed

    @property
    def strong_cancelled(self) -> int:
        return self.strong.strong_cancelled

    @property
    def free_units(self) -> int:
        return self.pool_free["default"]

    def pool_for(self, job: DecodeJob) -> str:
        if job.hint in self.unit_totals:
            return job.hint
        if self.lane_policy is not None:
            lane = self.lane_policy.pool_for(job)
            if lane in self.unit_totals:
                return lane
        return "default"

    def queue_for(self, pool: str) -> list:
        return self.ready if pool == "default" else self.pool_ready[pool]

    def queued_total(self) -> int:
        return len(self.ready) + sum(len(q) for q in self.pool_ready.values())

    @staticmethod
    def pool_tag(pool: str) -> str:
        return "" if pool == "default" else f"{pool} "

    def enqueue(self, job: DecodeJob, reserve_transfer=None) -> None:
        """Admit once and queue the request; its rounds stay in Buffer 0.

        ``reserve_transfer`` is called at dispatch, after a unit is assigned,
        to reserve the input link and return the transfer delay in ticks (the
        accelerator pattern: invoke the unit, then DMA its input into that
        unit's memory, then compute; Aladdin aladdin_sys_connection.h and
        dma_interface.h). ``None`` means the job carries no syndrome data.
        """
        self._reject_spent_job(job)
        if job.strong_decode_for is not None:
            self.strong.admit_strong(job, self.engine.now)
        elif job.on_done is None:
            self.strong.admit_weak(job, self.engine.now)
        job.submitted = True
        job.reserve_transfer = reserve_transfer
        self._enqueue_now(job)

    @staticmethod
    def _reject_spent_job(job: DecodeJob) -> None:
        """Reject resubmission of a live or terminal job."""
        if job.cancelled or job.completed or job.submitted:
            spent = ("cancelled" if job.cancelled else
                     "completed" if job.completed else "admitted")
            raise RuntimeError(
                f"decode job {job.label!r} for window "
                f"({job.op_id}, {job.window_id}) has already been {spent}: a "
                f"DecodeJob is submitted once, build a new one")

    def _enqueue_now(self, job: DecodeJob) -> None:
        """Put one admitted job in its pool's ready queue; its rounds stay upstream."""
        if job.strong_decode_for is not None:
            live = self.strong.live(job.strong_decode_for)
            if live is None or live.request_job is not job:
                self.staging.release(job)
                self.try_dispatch()                # returned credits may admit
                return                             # cancelled across the link
        job.ready_time = self.engine.now
        pool = self.pool_for(job)
        queue = self.queue_for(pool)
        queue.append(job)
        self.engine.log(self.log_name,
                        f"{job.label} READY -> enqueue "
                        f"({self.pool_tag(pool)}ready-queue length = {len(queue)})")
        self.queue_log.append((self.engine.now, self.queued_total()))
        self.try_dispatch()

    def submit_decode(self, round_count: int, on_done: Callable[[], None],
                      label: str = "external", code: Optional[str] = None,
                      spatial_nodes: Optional[int] = None,
                      hint: Optional[str] = None) -> None:
        """Submit a self-contained external decode job (factory corrections,
        separate idle decodes).

        The job carries no syndrome data, so it skips input transport and
        storage entirely: its decoder-input round demand is zero and it charges
        no round credits.
        """
        job = DecodeJob(op_id=-1, window_id=0, n_rounds=round_count,
                        ready_time=self.engine.now,
                        on_done=on_done, label=label, code=code,
                        spatial_nodes=spatial_nodes, hint=hint)
        self.queue_for(self.pool_for(job)).append(job)
        self.queue_log.append((self.engine.now, self.queued_total()))
        self.try_dispatch()

    def check_strong_route(self, weak_job: DecodeJob,
                           strong_job: DecodeJob) -> None:
        """Fail early when a strong job would route back to the weak decoder."""
        if self.router.route(strong_job) is self.router.route(weak_job):
            raise RuntimeError(
                "Strong job routes to the same decoder as the weak job; pass a "
                "router (e.g. SwitchingRouter) that sends hint='strong' to a "
                "distinct decoder.")

    def cancel_strong(self, key: tuple) -> None:
        """Cancel an unneeded strong re-decode if queued, crossing the link,
        running, or held.

        A cancel ends one request; it passes no verdict on the destination. A
        destination that keeps its weak result stops being a consumer because
        its weak decode has resolved, and a destination still waiting keeps its
        demand, so the cancelled request can be replaced in either position.
        """
        held = self.strong.take_held(key)
        if held is not None:
            self._record_request(
                held.request_job, held.completion.result,
                RequestProcessingOutcome.STRONG_COMPLETED_DISCARDED,
                held.decode_output_ticks)
        live = self.strong.take_live(key)
        if live is None:
            if held is not None:
                self.strong.strong_cancelled += 1
            return
        job = live.service_job
        if job.pool is None:
            self.staging.cancel(job)
            for pool in self.unit_totals:
                queue = self.queue_for(pool)
                if job in queue:
                    queue.remove(job)
                    break
            # Credits belong to the original request, never to a batch service
            # job, and the request is its own service job before dispatch.
            self.staging.release(live.request_job)
            self._record_request(
                live.request_job, None,
                RequestProcessingOutcome.STRONG_CANCELLED_BEFORE_DISPATCH,
                None)
        else:
            job.service_cancelled_request_keys.add(live.request_job.request_key)
            if self.strong.has_survivors(job):
                outcome = RequestProcessingOutcome.STRONG_CANCELLED_MEMBER_SERVICE_CONTINUED
                self.staging.release(live.request_job)
            else:
                outcome = RequestProcessingOutcome.STRONG_CANCELLED_DURING_SERVICE
                job.cancelled = True
                cancel = getattr(self.router.route(job), "cancel", None)
                if cancel is not None:               # a staged decoder stops its stages
                    cancel(job)
                self.staging.cancel(job)
                self.staging.release(live.request_job)
                self._free_unit(job)
                self._record_service(job)
                self.try_dispatch()
            self._record_request(live.request_job, None, outcome, None)
        self.strong.strong_cancelled += 1
        self.try_dispatch()           # returned credits may admit a waiting head

    def try_dispatch(self) -> None:
        """Drain admissible decoder inputs, then dispatch every pool.

        The loop is not reentrant: an admission continuation or a credit return
        that happens inside it returns immediately and the outer loop keeps
        running while the stager reports drainable work. So no event boundary
        leaves a pool holding both free round credits and a waiting request
        that fits, and no drain needs its own engine event.
        """
        if self._dispatching:
            return
        self._dispatching = True
        try:
            for pool in self.unit_totals:
                self._dispatch_pool(pool)
        finally:
            self._dispatching = False

    def _free_unit(self, job: DecodeJob) -> None:
        """Return the job's unit to its pool."""
        self.pool_free[job.pool] += 1
        self._free_units[job.pool].append(job.unit)
        job.unit = None

    def _dispatch_pool(self, pool: str) -> None:
        queue = self.queue_for(pool)
        while self.pool_free[pool] > 0 and queue:
            job = self._next_job(pool, queue)
            self._start_job(pool, job)

    def _next_job(self, pool: str, queue: list) -> DecodeJob:
        if self.bulk_strong and pool != "default":
            job = self._merge_strong_batch(queue)
            return job
        return self.scheduler.pop(queue)

    def _merge_strong_batch(self, queue: list) -> DecodeJob:
        """Batch queued strong jobs (timing-only) into one decode."""
        jobs = [
            self.scheduler.pop(queue)
            for _ in range(len(queue))
        ]
        if len(jobs) > 1:
            for job in jobs:
                has_model = job.dem is not None
                has_bits = any(p.bits is not None for p in job.payloads)
                if has_model or has_bits:
                    raise RuntimeError(
                        "bulk_strong only merges timing-only strong re-decodes; "
                        "disable it for accuracy-coupled switching.")
        window_keys = [j.strong_decode_for for j in jobs
                       if j.strong_decode_for is not None]
        request_keys = tuple(j.request_key for j in jobs)
        if len(jobs) == 1:
            jobs[0].service_original_request_keys = request_keys
            return jobs[0]
        total = sum(j.n_rounds for j in jobs)
        batch = DecodeJob(op_id=-1, window_id=0, n_rounds=total,
                          ready_time=min(j.ready_time for j in jobs),
                          on_done=lambda: None,
                          label=f"strong-batch x{len(jobs)} ({total}r)",
                          hint="strong", spatial_nodes=jobs[0].spatial_nodes,
                          strong_decode_for=window_keys[0] if window_keys
                          else None)
        batch.service_original_request_keys = request_keys
        self.strong.register_batch(window_keys, jobs, batch)
        return batch

    def _start_job(self, pool: str, job: DecodeJob) -> None:
        job.pool = pool
        members = job.service_original_request_keys
        if not members and job.request_key is not None:
            members = (job.request_key,)
            job.service_original_request_keys = members
        if members:
            job.service_key = DecoderServiceKey(
                min(key.run_sequence for key in members))
            job.service_dispatch_ticks = self.engine.now
            for member in self.strong.members_of(job):
                member.service_key = job.service_key
                member.service_dispatch_ticks = self.engine.now
        self.pool_free[pool] -= 1
        job.unit = self._free_units[pool].pop(0)
        if job.window is not None:
            job.window.t_dispatch = self.engine.now
        waited_ticks = self.engine.now - job.ready_time
        self.engine.log(self.log_name,
                        f"ASSIGN UNIT {job.label} "
                        f"(waited {fmt(waited_ticks).strip()} in queue, "
                        f"{self.pool_tag(pool)}units free now "
                        f"{self.pool_free[pool]})")
        self.queue_log.append((self.engine.now, self.queued_total()))
        # Unit assigned: move every member request's rounds from Buffer 0 into
        # this unit's decoder memory (one transfer per request, a merged strong
        # batch has several), then start the decode when all have landed.
        members = [member for member in self.strong.members_of(job) if member is not job]
        members = members or [job]
        pending = {"count": len(members)}
        memory = self.decoder_memories[(pool, job.unit)]

        def landed(_member: DecodeJob) -> None:
            pending["count"] -= 1
            if pending["count"] == 0:
                self._begin_service(job)

        for member in members:
            self.staging.stage(member, memory, landed)

    def _begin_service(self, job: DecodeJob) -> None:
        """The unit's memory holds the input: start the decode."""
        if job.cancelled:                        # cancelled while its input was in flight
            return
        decoder = self.router.route(job)
        self.engine.log(self.log_name, f"START DECODE {job.label}")
        run = getattr(decoder, "run", None)
        if run is not None:                 # staged decoder reads memory itself
            run(job, self.engine,
                lambda result, j=job: self._on_decode_done(j, result))
            return
        if job.decoder_input is not None:   # a plain decoder reads its memory now
            job.payloads = [
                fragment
                for round_input in job.decoder_input.rounds
                for fragment in round_input.fragments
            ]

        def decode_now(j=job):              # the algorithm's result is ready when its time ends
            result = (None if j.cancelled or j.on_done is not None
                      else decoder.decode(j))
            self._on_decode_done(j, result)

        self.engine.schedule(decoder.latency(job), decode_now,
                             label=f"decode_done({job.label})")

    def _on_decode_done(self, job: DecodeJob, result) -> None:
        """One decode finished: free the unit, ask the escalation_policy, commit or await strong."""
        if job.cancelled:
            self.staging.release(job)
            self.try_dispatch()
            return
        if job.strong_decode_for is not None:
            self._validate_logical_observables(job, result)
            self.staging.release(job)
            strong_result_deliveries = self.strong.deliveries_for(job, result, self.engine.now)
            job.completed = True
            self._free_unit(job)
            self.staging.release_service_members(self.strong.members_of(job))
            self.strong.finish_service(job)
            self.escalation_policy.on_decode_outcome(DecodeOutcome(job, result),
                                            self.services)   # FINALIZE_STRONG
            for held in strong_result_deliveries:
                self._complete_strong_result(held)
            self._record_service(job)
            self.try_dispatch()
            return

        if job.on_done is not None:                            # external job
            job.completed = True
            self._free_unit(job)
            # An external job that carried syndrome payloads through the
            # transport holds stored input like any other request, so its
            # credits come back before its callback runs and before the
            # same-tick drain; a self-contained submit_decode job holds none
            # and this is a no-op.
            self.staging.release(job)
            self.engine.log(self.log_name,
                            f"DECODE DONE {job.label} "
                            f"({self.pool_tag(job.pool)}units free now "
                            f"{self.pool_free[job.pool]})")
            job.on_done()
            self.try_dispatch()
            return

        self._validate_logical_observables(job, result)
        job.completed = True
        self._free_unit(job)
        key = (job.op_id, job.window_id)
        directive = self.escalation_policy.on_decode_outcome(DecodeOutcome(job, result),
                                                    self.services)
        self.strong.resolve_weak(key)
        awaiting = directive.directive is Directive.AWAIT_STRONG
        if directive.directive is Directive.FINALIZE:
            self.cancel_strong(key)                # no-op unless one is live/held
        if awaiting:
            serial_job = None if directive.extra is None else directive.extra.job
            strong_request_key = directive.strong_request_key
            deferred = serial_job is None and strong_request_key is not None
            if serial_job is None and not deferred:
                (carrier,) = self.strong.carriers_for(key)
                strong_request_key = carrier.request_job.request_key
            selection_delay = self.services.prepare_strong_selection(
                job, strong_request_key, serial_job, deferred=deferred,
            )
            self.strong.begin_selection(key, strong_request_key)
            self.engine.schedule(
                selection_delay,
                lambda destination=key, request_key=strong_request_key:
                    self._select_strong_result(destination, request_key),
                label=f"select strong result {key}",
            )
        job.awaiting_strong_result = awaiting      # BEFORE the commit callback
        self.on_window_decoded(job, result)
        self._record_request(
            job, result,
            (RequestProcessingOutcome.WEAK_AWAITED_STRONG if awaiting else
             RequestProcessingOutcome.WEAK_FORWARDED_FOR_DELIVERY),
            self.engine.now)
        self._record_service(job)
        self.staging.release(job)
        self.try_dispatch()

    def _select_strong_result(self, key: tuple,
                              request_key: DecoderRequestKey) -> None:
        """Make one strong completion eligible only after WSD delivery."""
        held = self.strong.select(key, request_key)
        if held is not None:
            self._complete_strong_result(held)

    @staticmethod
    def _validate_logical_observables(
        job: DecodeJob,
        result: DecodeResult,
    ) -> None:
        logical_observables = result.logical_observables
        if logical_observables is None:
            return
        for observable_index, bit in enumerate(logical_observables):
            if bit not in (0, 1):
                raise ValueError(
                    f"job ({job.op_id}, {job.window_id}) "
                    f"logical_observables index {observable_index} must be "
                    f"0 or 1, got {bit}")

    def admitted_strong_work_snapshot(self) -> tuple:
        """Each physical strong job once in its phase (running, queued, in_transit)."""
        queue_memberships = {}
        for pool in self.unit_totals:
            for queued_job in self.queue_for(pool):
                queue_memberships.setdefault(id(queued_job), []).append(queued_job)
        return self.strong.snapshot(queue_memberships)

    def _complete_strong_result(self, held: HeldStrongCompletion) -> None:
        """Deliver a strong result to the destination that waits for it; the
        ledger holds it when the demand is still on its way."""
        if self.strong.complete(held):
            self.on_strong_window_decoded(held.completion)
            self._record_request(
                held.request_job, held.completion.result,
                RequestProcessingOutcome.STRONG_FORWARDED_FOR_DELIVERY,
                held.decode_output_ticks)

    def _record_request(self, job: DecodeJob, result: Optional[DecodeResult],
                        outcome: RequestProcessingOutcome,
                        decode_output_ticks: Optional[int]) -> None:
        if self._terminal_request_records is None:
            return
        window = job.window
        local_fragments = tuple(
            fragment
            for round_input in (() if job.decoder_input is None
                                 else job.decoder_input.rounds)
            for fragment in round_input.fragments
        )
        bits_known = bool(local_fragments) and all(
            payload.bits is not None for payload in local_fragments)
        bit_count = (sum(len(payload.bits) for payload in local_fragments)
                     if bits_known else None)
        weight = (sum(sum(payload.bits) for payload in local_fragments)
                  if bits_known else None)
        self._terminal_request_records.append(TerminalRequestRecord(
            job.request_key, window.start_round, window.buffer_hi, job.n_rounds,
            bit_count, weight, job.request_created_ticks,
            job.request_admitted_ticks, job.ready_time,
            job.service_dispatch_ticks, decode_output_ticks,
            job.service_key, None if result is None else result.soft_output,
            outcome))

    def _record_service(self, job: DecodeJob) -> None:
        if self._terminal_service_records is None or job.service_key is None:
            return
        original = job.service_original_request_keys
        cancelled = tuple(key for key in original
                          if key in job.service_cancelled_request_keys)
        completed = tuple(key for key in original if key not in cancelled)
        dispatch = job.service_dispatch_ticks
        self._terminal_service_records.append(TerminalServiceRecord(
            job.service_key, job.pool, original, completed, cancelled,
            job.n_rounds, dispatch, self.engine.now, self.engine.now - dispatch))

    def terminal_request_records_snapshot(self) -> tuple:
        return tuple(self._terminal_request_records)

    def terminal_service_records_snapshot(self) -> tuple:
        return tuple(self._terminal_service_records)

    def check_decode_work_settled(self) -> None:
        """Require every admitted decode to reach a final result.

        Decoder-input storage settles with it, configured or not: nothing may
        still hold rounds, wait for round credits, or hold returned credits
        nobody drained. Every stored input is released unconditionally, so a
        leak on either path is a real defect rather than a tolerated one.
        """
        unsettled = self.strong.unsettled()
        held = [f"{m.pool}#{m.unit}" for m in self.decoder_memories.values()
                if m.occupied_rounds]
        if held:
            unsettled["decoder memory still holding rounds"] = held
        if unsettled:
            detail = "; ".join(f"{state}: {keys}"
                               for state, keys in unsettled.items())
            raise RuntimeError(
                f"the run ended with decode work unsettled ({detail}): every "
                f"window is final once the simulation is quiescent")
