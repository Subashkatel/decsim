"""The decoder manager: gives ready windows a decoder unit.

One ready queue per pool; a unit assigned at dispatch; the input staged
into that unit's memory; the decode started once the input has landed and
the window owes no boundary; the result delivered to the window manager.
The strong tier's request bookkeeping is the StrongRequestLedger's
(strong_escalation.py); the terminal records at the bottom are optional
capture for the switching study.

Every unit is a depth-1 decoupled access-execute machine (Smith 1982; TI
EDMA ping-pong, SPRAAN4A Example D; gem5-Aladdin ready bits at
whole-buffer granularity, Shao et al. MICRO 2016 Sec IV-B-2): two input
slots, so the next window's DMA overlaps the current compute. Both inputs
are resident in the unit's DecoderMemory, so a unit needs capacity for
two windows or the run stops loudly. Compute is claimed separately from
the slots (Tomasulo's rule: an instruction whose operands are not ready
waits in its reservation station, never on the functional unit). A landed
job whose window still owes a boundary parks in its slot and releases its
compute claim, so a dependent that fills early never deadlocks the unit
against its own predecessor.
"""

import dataclasses
import enum
import math
from typing import Callable, Optional

import numpy

import decsim.confidence.complementary as complementary
import decsim.config as config
import decsim.decoders.decoder_memory as decoder_memory_module
import decsim.decoders.decoder_memory_transfer as staging_module
import decsim.decoders.strong_escalation as strong_escalation
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


@dataclasses.dataclass
class GapJoinState:
    """One window's split-gap rendezvous.

    The weak outcome is processed only when both forced-class halves have
    reported: an AND of two completions, the same shape as the landed
    input join in the staging (both DMAs must land before the decode
    starts). The weak unit itself is freed at its own solve end; only the
    outcome waits.
    """

    sibling_weight: Optional[float] = None
    sibling_reported: bool = False
    held_weak_job: Optional[message.DecodeJob] = None
    held_weak_result: Optional[message.DecodeResult] = None

    def hold_weak(self, job: message.DecodeJob, result) -> None:
        """Keep the primary decode's outcome until the sibling reports."""
        self.held_weak_job = job
        self.held_weak_result = result

    def report_sibling(self, weight) -> None:
        """The sibling half's forced-class weight is in."""
        self.sibling_reported = True
        self.sibling_weight = weight


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
    """Queues, places, stages, starts, settles and records every decode.

    Wide state (36 attributes) recorded for the structural slice that
    splits it into queue, pool, unit, dispatcher, service, strong
    requests, gap joins and outcomes (slice 6 design note, section 3).
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
        service_gate=None,
        apply_service_boundary=None,
        stage_admission=None,
        lane_policy=None,
        log_name: str = "DecoderCluster",
        capture_enabled: bool = False,
        decoder_memory: Optional[
            decoder_memory_module.DecoderMemoryConfig
        ] = None,
        escalation_policy,
        services,
        on_window_decoded: Callable,
        on_strong_window_decoded: Callable,
    ):
        self.engine = engine
        self.router = router
        self.decoder_memory_transfer = (
            staging_module.CancellableDecoderMemoryTransfer(engine)
        )
        self.staging = staging_module.DecoderInputStaging(
            self.decoder_memory_transfer, engine
        )
        self.scheduler = scheduler
        self.lane_policy = lane_policy
        self.bulk_strong = bulk_strong
        self.service_gate = service_gate
        self.apply_service_boundary = apply_service_boundary
        self.stage_admission = stage_admission
        # request_key -> job parked in its slot, a boundary still owed
        self._parked_service: dict = {}
        # (pool, unit) -> jobs whose input occupies or reserves a slot
        # (in transfer or landed), in dispatch order; at most two
        self._unit_residents: dict = {}
        # (pool, unit) -> the job holding or reserving the unit's compute
        # (from assignment through decode end), None when compute is free
        self._computing: dict = {}
        # Pipelined units only (a routed decoder declaring an initiation
        # interval): (pool, unit) -> decodes started and not yet finished,
        # and (pool, unit) -> (owner, depth) when the pipeline is full and
        # the intake stays claimed until the next completion. Both stay
        # empty under non-pipelined decoders, whose occupancy IS latency.
        self._pipeline_flights: dict = {}
        self._pipeline_stalled: dict = {}
        # A pipelined unit's intake remains unavailable until the declared
        # initiation interval ends, even when the response finishes first.
        # This is distinct from an in-flight result and from a depth stall.
        self._pipeline_intake_busy: dict = {}
        # (pool, unit) -> the tick its compute is expected to free, from the
        # running decode's declared latency (or initiation interval); the
        # staging choice among busy units reads it. A decoder measured on
        # the host clock declares none.
        self._compute_free_ticks: dict = {}
        self.log_name = log_name
        self._terminal_request_records = None
        self._terminal_service_records = None
        if capture_enabled:
            self._terminal_request_records = []
            self._terminal_service_records = []
        self.escalation_policy = escalation_policy
        # the EscalationServices seam (the window manager)
        self.services = services
        self.on_window_decoded = on_window_decoded
        self.on_strong_window_decoded = on_strong_window_decoded
        if unit_pools is None:
            unit_pools = {"default": num_units}
        _check_unit_pools(unit_pools)
        self.unit_totals = dict(unit_pools)
        self.pool_free = dict(unit_pools)
        # Every unit is a numbered engine with its own input memory.
        self._free_units = {}
        self.decoder_memories = {}
        for pool, unit_count in unit_pools.items():
            self._free_units[pool] = list(range(unit_count))
            capacity = None
            if decoder_memory is not None:
                capacity = decoder_memory.capacity_for(pool)
            for unit in range(unit_count):
                memory = decoder_memory_module.DecoderMemory(
                    pool, unit, capacity
                )
                self.decoder_memories[(pool, unit)] = memory
        self._dispatching = False
        # Split-pair gap joins: a router with a gap route turns them on.
        # (op_id, window_id) -> GapJoinState, created when the sibling
        # is spawned and removed when the join concludes the window.
        gap_route = getattr(router, "gap", None)
        self.gap_split_enabled = gap_route is not None
        self._gap_joins: dict[tuple, GapJoinState] = {}
        self.num_units = self.unit_totals["default"]
        self.ready: list[message.DecodeJob] = []
        self.pool_ready: dict[str, list] = {}
        for pool in self.unit_totals:
            if pool != "default":
                self.pool_ready[pool] = []
        self.queue_log: list[tuple[int, int]] = []
        self.strong = strong_escalation.StrongRequestLedger()

    @property
    def strong_needed(self) -> int:
        """Strong results the destinations asked for."""
        return self.strong.strong_needed

    @property
    def strong_cancelled(self) -> int:
        """Strong requests cancelled before anyone consumed them."""
        return self.strong.strong_cancelled

    @property
    def free_units(self) -> int:
        """Units of the default pool with free compute."""
        return self.pool_free["default"]

    # ---------------------------------------------------------- admission

    def enqueue(self, job: message.DecodeJob, send_input=None) -> None:
        """Admit once and queue the request; its rounds stay in Buffer 0.

        ``send_input(on_landed)`` is called at dispatch, after a unit is
        assigned, to send the input over its link; it calls ``on_landed``
        at the delivery and returns the delay the link expects (the
        accelerator pattern: invoke the unit, then DMA its input into that
        unit's memory, then compute; Aladdin aladdin_sys_connection.h and
        dma_interface.h). ``None`` means the job carries no syndrome data.
        """
        _refuse_spent_job(job)
        if job.strong_decode_for is not None:
            self.strong.admit_strong(job, self.engine.now)
        elif job.on_done is None and job.gap_sibling_for is None:
            # a gap half is neither tier: it feeds the join, not the ledger
            self.strong.admit_weak(job, self.engine.now)
        job.submitted = True
        job.send_input = send_input
        self._enqueue_now(job)

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
        pool = self.pool_for(job)
        queue = self.queue_for(pool)
        queue.append(job)
        self._sample_queue_depth()
        self.try_dispatch()

    def try_dispatch(self) -> None:
        """Drain admissible decoder inputs, then dispatch every pool.

        The loop is not reentrant: an admission continuation or a credit
        return that happens inside it returns immediately and the outer
        loop keeps running while the stager reports drainable work. So no
        event boundary leaves a pool holding both free round credits and a
        waiting request that fits, and no drain needs its own engine event.
        """
        if self._dispatching:
            return
        self._dispatching = True
        try:
            for pool in self.unit_totals:
                self._dispatch_pool(pool)
        finally:
            self._dispatching = False

    def release_parked(self, window_key: tuple) -> None:
        """The window's last boundary arrived: start its parked decode.

        A job still in transfer passes the gate at its own landing instead.
        """
        parked = dict(self._parked_service)
        for request_key, job in parked.items():
            key = (job.op_id, job.window_id)
            if key != window_key:
                continue
            if self._is_boundary_owed(job):
                continue  # another dependency still owed
            del self._parked_service[request_key]
            self._restart_parked(job)
        self.try_dispatch()  # a started decode may admit blocked stages

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
        pool = self.pool_for(job)
        queue = self.queue_for(pool)
        if job in queue:
            queue.remove(job)
            self.staging.release(job)
        elif job.unit is not None:
            self._evict_resident(job)
        else:
            raise RuntimeError(
                f"{job.label} is neither queued nor resident; nothing to "
                "withdraw"
            )
        self.strong.resolve_weak(window_key)
        self._record_request(
            job,
            None,
            RequestProcessingOutcome.WEAK_WITHDRAWN_FOR_STRONG_WINDOW,
            None,
        )
        self.engine.log(
            self.log_name, f"WITHDRAW {job.label} (invalidated before start)"
        )
        self.try_dispatch()

    # ------------------------------------------------- the strong requests

    def check_strong_route(
        self, weak_job: message.DecodeJob, strong_job: message.DecodeJob
    ) -> None:
        """Refuse a strong job that would route back to the weak decoder."""
        strong_decoder = self.router.route(strong_job)
        weak_decoder = self.router.route(weak_job)
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
        held = self.strong.take_held(key)
        if held is not None:
            self._record_request(
                held.request_job,
                held.completion.result,
                RequestProcessingOutcome.STRONG_COMPLETED_DISCARDED,
                held.decode_output_ticks,
            )
        live = self.strong.take_live(key)
        if live is None:
            if held is not None:
                self.strong.strong_cancelled += 1
            return
        job = live.service_job
        if job.unit is not None and not job.service_started:
            self._cancel_staged_strong(live)
        elif job.pool is None:
            self._cancel_queued_strong(live)
        else:
            self._cancel_running_strong(live)
        self.strong.strong_cancelled += 1
        self.try_dispatch()  # returned credits may admit a waiting head

    def admitted_strong_work_snapshot(self) -> tuple:
        """Each physical strong job once in its phase."""
        queue_memberships = {}
        for pool in self.unit_totals:
            for queued_job in self.queue_for(pool):
                identity = id(queued_job)
                memberships = queue_memberships.setdefault(identity, [])
                memberships.append(queued_job)
        return self.strong.snapshot(queue_memberships)

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
        if self._parked_service:
            parked = self._parked_labels()
            raise RuntimeError(
                f"run ended with parked decodes never released: {parked}"
            )
        if self._gap_joins:
            unresolved = sorted(self._gap_joins)
            raise RuntimeError(
                f"run ended with split-gap joins unresolved: {unresolved}"
            )
        leftover_flights = self._in_flight_labels()
        if leftover_flights:
            raise RuntimeError(
                "run ended with pipelined decodes still in flight: "
                f"{leftover_flights}"
            )
        if self._pipeline_intake_busy:
            busy = sorted(self._pipeline_intake_busy, key=repr)
            raise RuntimeError(
                "run ended with pipelined decoder intake intervals still "
                f"active: {busy}"
            )
        unsettled = self.strong.unsettled()
        held = self._units_holding_rounds()
        if held:
            unsettled["decoder memory still holding rounds"] = held
        if unsettled:
            detail = _unsettled_text(unsettled)
            raise RuntimeError(
                f"the run ended with decode work unsettled ({detail}): "
                "every window is final once the simulation is quiescent"
            )

    # ------------------------------------------------- pools and queues

    def pool_for(self, job: message.DecodeJob) -> str:
        """The pool a job queues in: its hint, its lane, else default."""
        if job.hint in self.unit_totals:
            return job.hint
        if self.lane_policy is not None:
            lane = self.lane_policy.pool_for(job)
            if lane in self.unit_totals:
                return lane
        return "default"

    def queue_for(self, pool: str) -> list:
        """One pool's ready queue."""
        if pool == "default":
            return self.ready
        return self.pool_ready[pool]

    def queued_total(self) -> int:
        """Jobs waiting in every pool's ready queue."""
        total = len(self.ready)
        for queue in self.pool_ready.values():
            total += len(queue)
        return total

    @staticmethod
    def pool_tag(pool: str) -> str:
        """The pool's name as a log prefix; the default pool has none."""
        if pool == "default":
            return ""
        return f"{pool} "

    # ------------------------------------------------- admission, private

    def _enqueue_now(self, job: message.DecodeJob) -> None:
        """Put one admitted job in its pool's ready queue."""
        if job.strong_decode_for is not None:
            live = self.strong.live(job.strong_decode_for)
            if live is None or live.request_job is not job:
                # cancelled across the link; its credits may admit another
                self.staging.release(job)
                self.try_dispatch()
                return
        job.ready_time = self.engine.now
        pool = self.pool_for(job)
        queue = self.queue_for(pool)
        queue.append(job)
        pool_tag = self.pool_tag(pool)
        queue_length = len(queue)
        self.engine.log(
            self.log_name,
            f"{job.label} READY -> enqueue "
            f"({pool_tag}ready-queue length = {queue_length})",
        )
        self._sample_queue_depth()
        self.try_dispatch()

    def _sample_queue_depth(self) -> None:
        depth = self.queued_total()
        self.queue_log.append((self.engine.now, depth))

    # ------------------------------------------------- dispatch

    def _dispatch_pool(self, pool: str) -> None:
        """Dispatch startable work first, in scheduler order.

        The scan passes over jobs with no eligible unit; boundary-blocked
        work is placed only when no startable job can be (gem5 O3 issues
        from its ready set oldest-first: non-ready work never displaces
        ready work).
        """
        queue = self.queue_for(pool)
        while queue:
            ordered = self._drain_in_scheduler_order(pool, queue)
            selection = self._select_placement(pool, ordered)
            if selection is None:
                queue.extend(ordered)
                return
            index, job, unit, claim_compute = selection
            del ordered[index]
            queue.extend(ordered)
            self._start_job(pool, job, unit=unit, claim_compute=claim_compute)

    def _drain_in_scheduler_order(self, pool: str, queue: list) -> list:
        ordered = []
        while queue:
            job = self._next_job(pool, queue)
            ordered.append(job)
        return ordered

    def _select_placement(self, pool: str, ordered: list):
        """(index, job, unit, claim_compute) of the first placeable job.

        Startable jobs are tried before boundary-blocked ones.
        """
        startable = self._first_placement(pool, ordered, True)
        if startable is not None:
            return startable
        return self._first_placement(pool, ordered, False)

    def _first_placement(self, pool: str, ordered: list, startable: bool):
        for index, job in enumerate(ordered):
            is_startable = self._startable(job)
            if is_startable is not startable:
                continue
            placement = self._eligible_unit(pool, job)
            if placement is None:
                continue
            unit, claim_compute = placement
            return index, job, unit, claim_compute
        return None

    def _next_job(self, pool: str, queue: list) -> message.DecodeJob:
        if self.bulk_strong and pool != "default":
            return self._merge_strong_batch(queue)
        return self.scheduler.pop(queue)

    def _merge_strong_batch(self, queue: list) -> message.DecodeJob:
        """Batch every queued strong job (timing-only) into one decode.

        A batch that found no unit last time waits in the queue like any
        job; it is opened back into its member requests here so the new
        batch serves every request exactly once.
        """
        jobs = self._open_queued_strong_jobs(queue)
        if len(jobs) > 1:
            for job in jobs:
                _refuse_bits_in_bulk_strong(job)
        window_keys = []
        request_keys = []
        for job in jobs:
            if job.strong_decode_for is not None:
                window_keys.append(job.strong_decode_for)
            request_keys.append(job.request_key)
        request_keys = tuple(request_keys)
        if len(jobs) == 1:
            jobs[0].service_original_request_keys = request_keys
            return jobs[0]
        batch = _batch_job(jobs, window_keys)
        batch.service_original_request_keys = request_keys
        self.strong.register_batch(window_keys, jobs, batch)
        return batch

    def _open_queued_strong_jobs(self, queue: list) -> list:
        jobs = []
        for _ in range(len(queue)):
            queued = self.scheduler.pop(queue)
            if self._is_merged_batch(queued):
                members = self.strong.members_of(queued)
                jobs.extend(members)
            else:
                jobs.append(queued)
        return jobs

    @staticmethod
    def _is_merged_batch(job: message.DecodeJob) -> bool:
        """A batch serves several strong requests and has no request itself."""
        if job.request_key is not None:
            return False
        return job.strong_decode_for is not None

    @staticmethod
    def _startable(job: message.DecodeJob) -> bool:
        """A job whose window owes no boundary may hold compute.

        Anything windowless (external, strong context, merged batch)
        always may.
        """
        window = job.window
        if window is None:
            return True
        return window.deps_remaining <= 0

    # ------------------------------------------------- placing on a unit

    def _eligible_unit(self, pool: str, job: message.DecodeJob):
        """(unit, claim_compute) for this job, or None.

        A startable job takes any unit with a free input slot and claims
        compute when that unit's compute is free. A boundary-blocked job
        takes an input slot only (its DMA overlaps other work, Tomasulo's
        reservation station), and only once the window manager's
        stage_admission says its release is already resolving, so parked
        work can never squat a slot against the decode that must free it.

        When every unit is computing, a job with input to move is staged
        on the unit whose compute frees earliest among those with a free
        slot: least work left, which starts each job when a central FIFO
        queue over the pool would (Harchol-Balter, Performance Modeling
        and Design of Computer Systems, 2013, Ch. 24). A job with no input
        has nothing to prefetch and waits in the queue for free compute.
        """
        startable = self._startable(job)
        if not startable and self._is_staging_refused(job):
            return None
        resident_capacity = self._resident_capacity(job)
        for unit in self._free_units[pool]:
            slot = (pool, unit)
            if self._has_room(slot, job, resident_capacity):
                return unit, startable
        if not self._carries_input(job):
            return None
        busy_with_room = self._busy_units_with_room(
            pool, job, resident_capacity
        )
        if not busy_with_room:
            return None
        unit = self._earliest_freeing(pool, busy_with_room)
        return unit, False

    def _is_staging_refused(self, job: message.DecodeJob) -> bool:
        if self.stage_admission is None:
            return False
        return not self.stage_admission(job)

    def _has_room(
        self, slot: tuple, job: message.DecodeJob, resident_capacity: int
    ) -> bool:
        residents = self._residents(slot)
        if len(residents) >= resident_capacity:
            return False
        return self._slot_memory_ok(slot, job)

    def _busy_units_with_room(
        self, pool: str, job: message.DecodeJob, resident_capacity: int
    ) -> list:
        free = set(self._free_units[pool])
        units = []
        for unit in range(self.unit_totals[pool]):
            if unit in free:
                continue
            slot = (pool, unit)
            if self._has_room(slot, job, resident_capacity):
                units.append(unit)
        return units

    def _earliest_freeing(self, pool: str, units: list) -> int:
        """The unit whose compute frees first: least work left."""
        earliest_unit = units[0]
        earliest_free = self._expected_compute_free(pool, earliest_unit)
        for unit in units[1:]:
            free_ticks = self._expected_compute_free(pool, unit)
            if free_ticks < earliest_free:
                earliest_free = free_ticks
                earliest_unit = unit
        return earliest_unit

    def _expected_compute_free(self, pool: str, unit: int):
        slot = (pool, unit)
        return self._compute_free_ticks.get(slot, math.inf)

    def _carries_input(self, job: message.DecodeJob) -> bool:
        """The job moves syndrome data into a unit's memory."""
        if job.send_input is not None:
            return True
        demand = self._memory_demand(job)
        return demand > 0

    def _resident_capacity(self, job: message.DecodeJob) -> int:
        """Residents a unit holds for this job's route.

        Two (the depth-1 access-execute machine) unless the routed decoder
        pipelines: then every in-flight decode stays resident (its input
        lives in the unit's memory until its result emerges) plus one
        landing next. Memory admission still gates every resident, so a
        deep pipeline pays its SRAM price visibly or refuses loudly.
        """
        decoder = self.router.route(job)
        interval_of = getattr(decoder, "initiation_interval", None)
        if interval_of is None:
            return 2
        interval = interval_of(job)
        depth = getattr(decoder, "pipeline_depth", None)
        if depth is None:
            latency_ticks = decoder.latency(job)
            depth = _full_pipeline_depth(latency_ticks, interval)
        return depth + 1

    def _slot_memory_ok(self, slot: tuple, job: message.DecodeJob) -> bool:
        """Whether the unit's memory holds this job beside its residents.

        A second resident joins only if the unit's memory holds both
        inputs; a unit sized for one window keeps serial residency (the
        doubled-SRAM price of overlap is paid explicitly, never assumed).
        A first resident is always admitted, so a genuinely oversized
        window still stops loudly at its deposit.
        """
        live = self._live_residents(slot)
        if not live:
            return True
        capacity = self.decoder_memories[slot].capacity_rounds
        if capacity is None:
            return True
        demand = 0
        for resident in live:
            demand += self._memory_demand(resident)
        demand += self._memory_demand(job)
        return demand <= capacity

    def _live_residents(self, slot: tuple) -> list:
        live = []
        for resident in self._residents(slot):
            if resident.cancelled or resident.completed:
                continue
            live.append(resident)
        return live

    def _memory_demand(self, job: message.DecodeJob) -> int:
        """The rounds a job's input occupies in unit memory.

        The distinct rounds of its payloads, the same count the deposit
        charges. An external job carries no syndrome data and stores
        nothing; a merged batch stores every member's input.
        """
        if job.on_done is not None:
            return 0
        if self._is_merged_batch(job):
            demand = 0
            for member in self.strong.members_of(job):
                demand += (
                    decoder_memory_module.count_decoder_input_round_demand(
                        member.payloads
                    )
                )
            return demand
        return decoder_memory_module.count_decoder_input_round_demand(
            job.payloads
        )

    def _residents(self, slot: tuple) -> list:
        return self._unit_residents.setdefault(slot, [])

    # ------------------------------------------------- the unit's compute

    def _claim_compute(self, slot: tuple, job: message.DecodeJob) -> None:
        pool, unit = slot
        self.pool_free[pool] -= 1
        self._free_units[pool].remove(unit)
        self._computing[slot] = job

    def _release_compute_claim(self, job: message.DecodeJob) -> None:
        """Tomasulo's rule at the boundary hazard.

        A parked job keeps its input slot but never the unit's compute.
        """
        slot = (job.pool, job.unit)
        holder = self._computing.get(slot)
        if holder is job:
            self._computing[slot] = None
            self._offer_compute(slot)
            self.try_dispatch()

    def _offer_compute(self, slot: tuple) -> None:
        """Free compute goes to the oldest startable resident.

        Or it stays reserved for the oldest one still in flight, or it
        returns to the pool. gem5 O3's scheduleReadyInsts is the reference
        rule: only a ready instruction acquires a functional unit
        (fu_pool->getUnit at issue), and blocked work waits in the queue,
        never on the unit.
        """
        pool, unit = slot
        holder = self._computing.get(slot)
        if holder is not None:
            return
        ready = self._oldest_landed_resident_ready_to_start(slot)
        if ready is not None:
            self._computing[slot] = ready
            self._begin_service(ready)
            return
        landing = self._oldest_landing_resident_that_may_start(slot)
        if landing is not None:
            self._computing[slot] = landing  # starts at its landing
            self._predict_compute_free(landing)
            return
        if unit in self._free_units[pool]:
            # a pipelined unit's intake went back to the pool at the end of
            # its initiation interval; the decode's completion offers again
            return
        self.pool_free[pool] += 1
        self._free_units[pool].append(unit)

    def _oldest_landed_resident_ready_to_start(self, slot: tuple):
        for resident in self._residents(slot):
            if _is_past_start(resident):
                continue
            if not resident.input_landed:
                continue
            if self._is_parked(resident):
                continue
            return resident
        return None

    def _oldest_landing_resident_that_may_start(self, slot: tuple):
        for resident in self._residents(slot):
            if _is_past_start(resident):
                continue
            if resident.input_landed:
                continue
            if not self._startable(resident):
                continue
            return resident
        return None

    def _is_parked(self, job: message.DecodeJob) -> bool:
        if job.request_key is None:
            return False
        parked = self._parked_service.get(job.request_key)
        return parked is job

    def _free_unit(self, job: message.DecodeJob) -> None:
        """Compute finished: drop the job from its slot and offer the compute.

        The ping-pong swap at compute end.
        """
        self._end_flight(job, offer_now=False)
        slot = (job.pool, job.unit)
        job.unit = None
        residents = self._residents(slot)
        if job in residents:
            residents.remove(job)
        intake_owner = self._pipeline_intake_busy.get(slot)
        holder = self._computing.get(slot)
        if holder is job and intake_owner is not job:
            self._computing[slot] = None
        pool, unit = slot
        self.engine.log_io(
            f"unit {pool}#{unit} SRAM",
            lambda: self._emitted_description(job, slot),
        )
        self._offer_compute(slot)

    def _evict_resident(self, job: message.DecodeJob) -> None:
        """Drop a job from its slot before its decode started.

        Compute it held or reserved passes onward.
        """
        slot = (job.pool, job.unit)
        residents = self._residents(slot)
        if job in residents:
            residents.remove(job)
        if job.request_key is not None:
            self._parked_service.pop(job.request_key, None)
        self.staging.cancel(job)
        holder = self._computing.get(slot)
        if holder is job:
            self._computing[slot] = None
            self._offer_compute(slot)
        job.unit = None

    def _emitted_description(self, job: message.DecodeJob, slot: tuple) -> str:
        holds = self._sram_description(slot)
        return f"emitted {job.label} result; holds {holds}"

    def _sram_description(self, slot: tuple) -> str:
        """One compact line of a unit's residents and their phase."""
        residents = self._residents(slot)
        if not residents:
            return "empty"
        parts = []
        for resident in residents:
            phase = self._resident_phase(slot, resident)
            parts.append(
                f"{resident.label} {phase}, {resident.n_rounds} rounds"
            )
        return "; ".join(parts)

    def _resident_phase(self, slot: tuple, resident: message.DecodeJob) -> str:
        holder = self._computing.get(slot)
        if holder is resident and resident.service_started:
            return "computing"
        if self._is_parked(resident):
            return "parked"
        if resident.input_landed:
            return "ready"
        return "capturing"

    def _predict_compute_free(self, job: message.DecodeJob) -> None:
        """Record when the compute this job holds frees.

        The decode starts once its input has landed and runs for the
        declared latency (the initiation interval on a pipelined unit). A
        decoder measured on the host clock declares no latency, so its
        unit stays unpredicted.
        """
        slot = (job.pool, job.unit)
        decoder = self.router.route(job)
        measures_wall_clock = getattr(decoder, "measures_wall_clock", False)
        if measures_wall_clock:
            self._compute_free_ticks.pop(slot, None)
            return
        interval_of = getattr(decoder, "initiation_interval", None)
        if interval_of is None:
            occupancy = decoder.latency(job)
        else:
            occupancy = interval_of(job)
        start = self.engine.now
        landing = job.input_landing_ticks
        if landing is not None:
            start = max(start, landing)
        self._compute_free_ticks[slot] = start + occupancy

    # ------------------------------------------------- the pipelined unit

    def _initiation_interval_ticks(
        self, decoder, job: message.DecodeJob, latency_ticks: int
    ):
        """(interval, depth) for a pipelined start, (None, None) otherwise.

        The pipelined model serves plain window and external decodes
        only; the strong tier, gap siblings, and merged batches keep
        occupancy == latency until they get their own design pass, and a
        pipelined route there refuses loudly rather than silently
        serializing.
        """
        interval_of = getattr(decoder, "initiation_interval", None)
        if interval_of is None:
            return None, None
        if _is_outside_pipelined_model(job):
            raise RuntimeError(
                f"decode job {job.label!r}: a pipelined decoder serves plain "
                "window or external decodes only; the strong tier, gap "
                "siblings, and merged batches are not pipelined yet"
            )
        interval = interval_of(job)
        depth = getattr(decoder, "pipeline_depth", None)
        if depth is None:
            depth = _full_pipeline_depth(latency_ticks, interval)
        return interval, depth

    def _track_pipelined_start(
        self,
        job: message.DecodeJob,
        latency_ticks: int,
        interval_ticks: int,
        depth: int,
    ) -> None:
        slot = (job.pool, job.unit)
        flights = self._pipeline_flights.setdefault(slot, [])
        # in-order completion: a hardware pipeline retires in issue
        # order, so every in-flight decode on one unit must declare
        # the same latency; mixed latencies refuse loudly
        mixed = _flights_with_other_latency(flights, latency_ticks)
        if mixed:
            other = mixed[0]
            raise RuntimeError(
                f"decode job {job.label!r} declares latency "
                f"{latency_ticks} ticks while {other.label!r} is in "
                "flight with a different latency: a pipelined unit "
                "completes in order and takes one latency per unit"
            )
        flights.append((job, latency_ticks))
        if slot in self._pipeline_intake_busy:
            raise RuntimeError(
                f"pipelined unit {slot!r} started {job.label!r} before "
                "its prior initiation interval completed"
            )
        self._pipeline_intake_busy[slot] = job
        self.engine.schedule(
            interval_ticks,
            lambda: self._initiation_complete(slot, job, depth),
            label=f"initiation_complete({job.label})",
        )

    def _initiation_complete(
        self, slot: tuple, job: message.DecodeJob, depth: int
    ) -> None:
        """The pipelined unit's intake is free again.

        Release the compute claim so the next start may begin, unless the
        pipeline is full; a full pipeline keeps the claim until the next
        completion.
        """
        intake_owner = self._pipeline_intake_busy.get(slot)
        if intake_owner is not job:
            return
        del self._pipeline_intake_busy[slot]
        holder = self._computing.get(slot)
        if holder is not job:
            return
        if job.cancelled or job.completed:
            self._computing[slot] = None
            self._offer_compute(slot)
            self.try_dispatch()
            return
        flights = self._pipeline_flights.get(slot, ())
        if len(flights) >= depth:
            self._pipeline_stalled[slot] = (job, depth)
            return
        self._computing[slot] = None
        self._offer_compute(slot)
        self.try_dispatch()

    def _end_flight(self, job: message.DecodeJob, offer_now: bool) -> None:
        """A pipelined decode left the unit (done or cancelled).

        Retire its flight and lift a full-pipeline stall. The caller's
        normal flow performs the compute offer unless offer_now says
        otherwise.
        """
        for slot, flights in self._pipeline_flights.items():
            flight = _flight_of(flights, job)
            if flight is None:
                continue
            flights.remove(flight)
            self._lift_pipeline_stall(slot, flights, offer_now)
            return

    def _lift_pipeline_stall(
        self, slot: tuple, flights: list, offer_now: bool
    ) -> None:
        stalled = self._pipeline_stalled.get(slot)
        if stalled is None:
            return
        owner, depth = stalled
        if len(flights) >= depth:
            return
        del self._pipeline_stalled[slot]
        holder = self._computing.get(slot)
        if holder is not owner:
            return
        if owner.cancelled or owner.completed:
            return
        self._computing[slot] = None
        if offer_now:
            self._offer_compute(slot)
            self.try_dispatch()

    # ------------------------------------------------- staging and service

    def _start_job(
        self,
        pool: str,
        job: message.DecodeJob,
        *,
        unit: int,
        claim_compute: bool,
    ) -> None:
        job.pool = pool
        self._assign_service_key(job)
        # the unit is assigned at DMA start (gem5-Aladdin's invocation
        # model: invoke the unit, then DMA its input); compute is claimed
        # only when this unit's compute is actually free
        job.unit = unit
        slot = (pool, unit)
        residents = self._residents(slot)
        residents.append(job)
        if claim_compute:
            self._claim_compute(slot, job)
        if job.window is not None:
            job.window.t_dispatch = self.engine.now
        self._log_assignment(pool, job, claim_compute)
        self._sample_queue_depth()
        self._stage_input(job, slot)
        if claim_compute:
            self._predict_compute_free(job)

    def _assign_service_key(self, job: message.DecodeJob) -> None:
        """One service key per decode, shared by every request it serves."""
        members = job.service_original_request_keys
        if not members and job.request_key is not None:
            members = (job.request_key,)
            job.service_original_request_keys = members
        if not members:
            return
        run_sequences = []
        for key in members:
            run_sequences.append(key.run_sequence)
        first_sequence = min(run_sequences)
        job.service_key = message.DecoderServiceKey(first_sequence)
        job.service_dispatch_ticks = self.engine.now
        for member in self.strong.members_of(job):
            member.service_key = job.service_key
            member.service_dispatch_ticks = self.engine.now

    def _log_assignment(
        self, pool: str, job: message.DecodeJob, claim_compute: bool
    ) -> None:
        waited_ticks = self.engine.now - job.ready_time
        waited = config.format_ticks(waited_ticks)
        waited = waited.strip()
        slot_note = ""
        if not claim_compute:
            slot_note = "staged, "
        pool_tag = self.pool_tag(pool)
        free_now = self.pool_free[pool]
        self.engine.log(
            self.log_name,
            f"ASSIGN UNIT {job.label} "
            f"({slot_note}waited {waited} in queue, "
            f"{pool_tag}units free now {free_now})",
        )

    def _stage_input(self, job: message.DecodeJob, slot: tuple) -> None:
        """Move every member request's rounds into this unit's memory.

        One transfer per request (a merged strong batch has several); the
        decode starts when all have landed.
        """
        members = self._transfer_members(job)
        memory = self.decoder_memories[slot]
        pool, unit = slot
        source = f"unit {pool}#{unit} SRAM"
        remaining = len(members)

        def landed(_member: message.DecodeJob) -> None:
            nonlocal remaining
            remaining -= 1
            if remaining > 0:
                return
            self._input_landed(job, slot)

        for member in members:
            self._log_input_receiving(source, member)
            self.staging.stage(member, memory, landed)

    def _transfer_members(self, job: message.DecodeJob) -> list:
        members = []
        for member in self.strong.members_of(job):
            if member is not job:
                members.append(member)
        if not members:
            members = [job]
        return members

    def _log_input_receiving(
        self, source: str, member: message.DecodeJob
    ) -> None:
        self.engine.log_io(source, lambda: _receiving_text(member))

    def _input_landed(self, job: message.DecodeJob, slot: tuple) -> None:
        """Every transfer landed: start now, or wait for the unit's compute."""
        job.input_landed = True
        pool, unit = slot
        self.engine.log_io(
            f"unit {pool}#{unit} SRAM",
            lambda: self._landed_description(job, slot),
        )
        holder = self._computing.get(slot)
        if holder is job:
            # this job holds or was reserved the unit's compute
            self._begin_service(job)
            return
        if holder is None and unit in self._free_units[pool]:
            # compute went back to the pool (a resident parked and
            # released its claim); take it now
            self._claim_compute(slot, job)
            self._begin_service(job)
        # otherwise the compute is busy: _offer_compute picks this
        # job up at the next compute end

    def _landed_description(self, job: message.DecodeJob, slot: tuple) -> str:
        defects = _job_defects_text(job)
        holds = self._sram_description(slot)
        return f"{job.label} input landed; {defects}; holds {holds}"

    def _begin_service(
        self, job: message.DecodeJob, gated: bool = True
    ) -> None:
        """The unit's memory holds the input: start the decode."""
        if job.cancelled:  # cancelled while its input was in flight
            return
        if gated and self._is_boundary_owed(job):
            self._park(job)
            return
        if self.apply_service_boundary is not None:
            # the seam mask is XORed into the landed input exactly once,
            # at the moment the decode actually starts
            self.apply_service_boundary(job)
        job.service_started = True
        if job.window is not None:
            job.window.service_began = True
        self._spawn_gap_sibling(job)
        decoder = self.router.route(job)
        self.engine.log(self.log_name, f"START DECODE {job.label}")
        self._predict_compute_free(job)
        run = getattr(decoder, "run", None)
        if run is not None:  # a staged decoder reads memory itself
            run(
                job,
                self.engine,
                lambda result: self._on_decode_done(job, result),
            )
            return
        if job.decoder_input is not None:
            # a plain decoder reads its memory now
            job.payloads = _fragments_in(job.decoder_input)
        self._schedule_decode(decoder, job)

    def _is_boundary_owed(self, job: message.DecodeJob) -> bool:
        if self.service_gate is None:
            return False
        return not self.service_gate(job)

    def _park(self, job: message.DecodeJob) -> None:
        self._parked_service[job.request_key] = job
        self.engine.log(
            self.log_name, f"PARK DECODE {job.label} (boundary pending)"
        )
        self._release_compute_claim(job)

    def _restart_parked(self, job: message.DecodeJob) -> None:
        """A released job takes the unit's compute if it can have it now."""
        slot = (job.pool, job.unit)
        holder = self._computing.get(slot)
        if holder is None and job.unit in self._free_units[job.pool]:
            self._claim_compute(slot, job)
            self._begin_service(job, gated=False)
            return
        if holder is None:
            return
        if holder.service_started:
            # the unit is decoding: no longer parked, so _offer_compute
            # starts this job at the next compute end
            return
        if holder.input_landed:
            return
        # steal a reservation held for an input still in flight:
        # ready work issues first; the in-flight job re-competes
        # at its own landing
        self._computing[slot] = job
        self._begin_service(job, gated=False)

    def _schedule_decode(self, decoder, job: message.DecodeJob) -> None:
        """The algorithm's result is ready when its latency ends."""
        latency_ticks = decoder.latency(job)
        interval_ticks, depth = self._initiation_interval_ticks(
            decoder, job, latency_ticks
        )
        self.engine.schedule(
            latency_ticks,
            lambda: self._decode_now(decoder, job),
            label=f"decode_done({job.label})",
        )
        if interval_ticks is None:
            return
        self._track_pipelined_start(job, latency_ticks, interval_ticks, depth)

    def _decode_now(self, decoder, job: message.DecodeJob) -> None:
        result = None
        if not job.cancelled and job.on_done is None:
            result = decoder.decode(job)
        self._on_decode_done(job, result)

    # ------------------------------------------------- the split gap

    def _spawn_gap_sibling(self, job: message.DecodeJob) -> None:
        """Submit the other forced-class solve to the gap pool.

        Fired at the primary weak decode's service start: the boundary
        mask is applied by then, so the sibling reads the same masked
        rounds the primary decodes (both units receive the same adjusted
        stream, the CSB dual-write pattern on the strong side). The
        sibling carries its own copy of the rounds, pays its own transfer
        to the weak decoder and queues for its own unit. The copy is taken
        here instead of holding Buffer 0 for a second read, so the
        sibling's transfer is priced but never blocks on round retention.
        """
        if not self._wants_gap_sibling(job):
            return
        key = (job.op_id, job.window_id)
        if key in self._gap_joins:
            return
        sibling = self._sibling_job(job, key)
        self._gap_joins[key] = GapJoinState()
        self.engine.log(
            self.log_name,
            f"SPLIT GAP {job.label}: sibling submitted to the gap pool",
        )
        self.enqueue(
            sibling,
            lambda on_landed: self._send_sibling_input(on_landed, sibling, job),
        )

    def _sibling_job(
        self, job: message.DecodeJob, key: tuple
    ) -> message.DecodeJob:
        """The other forced-class solve over the primary's landed rounds."""
        masked_fragments = _fragments_in(job.decoder_input)
        round_count = len(job.decoder_input.rounds)
        return message.DecodeJob(
            op_id=job.op_id,
            window_id=job.window_id,
            n_rounds=round_count,
            dem=job.dem,
            payloads=masked_fragments,
            ready_time=self.engine.now,
            label=f"gap({job.label})",
            hint="gap",
            spatial_nodes=job.spatial_nodes,
            code=job.code,
            gap_sibling_for=key,
        )

    def _wants_gap_sibling(self, job: message.DecodeJob) -> bool:
        if not self.gap_split_enabled:
            return False
        if job.strong_decode_for is not None:
            return False
        if job.on_done is not None:
            return False
        if job.gap_sibling_for is not None:
            return False
        if job.window is None:
            return False
        if job.dem is None:
            return False
        return job.decoder_input is not None

    def _send_sibling_input(
        self,
        on_landed: Callable[[], None],
        sibling: message.DecodeJob,
        primary: message.DecodeJob,
    ) -> int:
        window_manager = getattr(self.services, "wm", None)
        if window_manager is None:
            on_landed()
            return 0
        payload_bits = window_manager._job_payload_bits(sibling)
        # the link attribution is the window's, so the transfer rides
        # the primary job's window identity; the landing is the
        # sibling's own
        return window_manager._send_job_transfer(
            message.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER,
            primary,
            payload_bits=payload_bits,
            on_delivered=on_landed,
        )

    def _gap_sibling_done(self, key: tuple, sibling_weight) -> None:
        """One gap half landed; conclude the window if the other is in."""
        join = self._gap_joins.get(key)
        if join is None:
            raise RuntimeError(
                f"gap sibling finished for window {key} with no join entry"
            )
        join.report_sibling(sibling_weight)
        if join.held_weak_job is None:
            return  # the primary decode is still running
        del self._gap_joins[key]
        self._attach_joined_gap(join.held_weak_result, sibling_weight)
        self._conclude_weak(join.held_weak_job, join.held_weak_result)

    @staticmethod
    def _attach_joined_gap(
        result: message.DecodeResult, sibling_weight
    ) -> None:
        """Build the SoftOutput from the two forced-class weights.

        Either half missing leaves soft_output None, and the policy then
        escalates (the same behavior a metric-less serial decode has).
        """
        primary_weight = result.gap_half_weight
        if primary_weight is None or sibling_weight is None:
            return
        w_min = min(primary_weight, sibling_weight)
        w_comp = max(primary_weight, sibling_weight)
        gap = w_comp - w_min
        result.soft_output = message.SoftOutput(
            gap=gap,
            source=complementary.COMPLEMENTARY_GAP_SOURCE,
            w_min=w_min,
            w_comp=w_comp,
        )

    # ------------------------------------------------- outcomes

    def _on_decode_done(self, job: message.DecodeJob, result) -> None:
        """One decode finished: free the unit and settle the outcome."""
        if job.cancelled:
            self._end_flight(job, offer_now=True)
            self.staging.release(job)
            self.try_dispatch()
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
        self._free_unit(job)
        self.staging.release(job)
        self.engine.log(self.log_name, f"DECODE DONE {job.label} (gap half)")
        sibling_weight = None
        if result is not None:
            sibling_weight = result.gap_half_weight
        self._gap_sibling_done(job.gap_sibling_for, sibling_weight)
        self.try_dispatch()

    def _strong_decode_done(self, job: message.DecodeJob, result) -> None:
        self._validate_logical_observables(job, result)
        self.staging.release(job)
        deliveries = self.strong.deliveries_for(job, result, self.engine.now)
        job.completed = True
        self._free_unit(job)
        members = self.strong.members_of(job)
        self.staging.release_service_members(members)
        self.strong.finish_service(job)
        outcome = message.DecodeOutcome(job, result)
        # FINALIZE_STRONG
        self.escalation_policy.on_decode_outcome(outcome, self.services)
        for held in deliveries:
            self._complete_strong_result(held)
        self._record_service(job)
        self.try_dispatch()

    def _external_decode_done(self, job: message.DecodeJob) -> None:
        job.completed = True
        self._free_unit(job)
        # An external job that carried syndrome payloads through the
        # transport holds stored input like any other request, so its
        # credits come back before its callback runs and before the
        # same-tick drain; a self-contained submit_decode job holds none
        # and this is a no-op.
        self.staging.release(job)
        pool_tag = self.pool_tag(job.pool)
        free_now = self.pool_free[job.pool]
        self.engine.log(
            self.log_name,
            f"DECODE DONE {job.label} ({pool_tag}units free now {free_now})",
        )
        job.on_done()
        self.try_dispatch()

    def _weak_decode_done(self, job: message.DecodeJob, result) -> None:
        self._validate_logical_observables(job, result)
        job.completed = True
        self._free_unit(job)
        key = (job.op_id, job.window_id)
        join = self._gap_joins.get(key)
        if join is not None:
            # the unit is free either way; the OUTCOME waits at the join
            self.staging.release(job)
            if not join.sibling_reported:
                join.hold_weak(job, result)
                self.engine.log(
                    self.log_name,
                    f"GAP JOIN {job.label}: holding for the sibling half",
                )
                self.try_dispatch()
                return
            del self._gap_joins[key]
            self._attach_joined_gap(result, join.sibling_weight)
        self._conclude_weak(job, result)

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
        self.strong.resolve_weak(key)
        awaiting = directive.directive is message.Directive.AWAIT_STRONG
        if directive.directive is message.Directive.FINALIZE:
            self.cancel_strong(key)  # no-op unless one is live/held
        if awaiting:
            self._await_strong_result(job, key, directive)
        job.awaiting_strong_result = awaiting  # BEFORE the commit callback
        self.on_window_decoded(job, result)
        processing = RequestProcessingOutcome.PRIMARY_FORWARDED_FOR_DELIVERY
        if awaiting:
            processing = RequestProcessingOutcome.WEAK_AWAITED_STRONG
        self._record_request(job, result, processing, self.engine.now)
        self._record_service(job)
        self.staging.release(job)
        self.try_dispatch()

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
            (carrier,) = self.strong.carriers_for(key)
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
        self.strong.begin_selection(key, strong_request_key)

    def _select_strong_result(
        self, key: tuple, request_key: message.DecoderRequestKey
    ) -> None:
        """Make one strong completion eligible only after WSD delivery."""
        held = self.strong.select(key, request_key)
        if held is not None:
            self._complete_strong_result(held)

    def _complete_strong_result(
        self, held: strong_escalation.HeldStrongCompletion
    ) -> None:
        """Deliver a strong result to the destination that waits for it.

        The ledger holds it when the demand is still on its way.
        """
        if not self.strong.complete(held):
            return
        self.on_strong_window_decoded(held.completion)
        self._record_request(
            held.request_job,
            held.completion.result,
            RequestProcessingOutcome.STRONG_FORWARDED_FOR_DELIVERY,
            held.decode_output_ticks,
        )

    @staticmethod
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

    # ------------------------------------------------- cancelling strong

    def _cancel_staged_strong(
        self, live: strong_escalation.LiveStrongRequest
    ) -> None:
        """Cancelled in its slot before its decode started."""
        job = live.service_job
        job.cancelled = True
        self._evict_resident(job)
        self.staging.release(live.request_job)
        self._record_request(
            live.request_job,
            None,
            RequestProcessingOutcome.STRONG_CANCELLED_WHILE_STAGED,
            None,
        )

    def _cancel_queued_strong(
        self, live: strong_escalation.LiveStrongRequest
    ) -> None:
        job = live.service_job
        self.staging.cancel(job)
        self._remove_from_queues(job)
        # Credits belong to the original request, never to a batch service
        # job, and the request is its own service job before dispatch.
        self.staging.release(live.request_job)
        self._record_request(
            live.request_job,
            None,
            RequestProcessingOutcome.STRONG_CANCELLED_BEFORE_DISPATCH,
            None,
        )

    def _cancel_running_strong(
        self, live: strong_escalation.LiveStrongRequest
    ) -> None:
        job = live.service_job
        job.service_cancelled_request_keys.add(live.request_job.request_key)
        if self.strong.has_survivors(job):
            self.staging.release(live.request_job)
            self._record_request(
                live.request_job,
                None,
                RequestProcessingOutcome.STRONG_CANCELLED_MEMBER_SERVICE_CONTINUED,
                None,
            )
            return
        job.cancelled = True
        decoder = self.router.route(job)
        cancel = getattr(decoder, "cancel", None)
        if cancel is not None:  # a staged decoder stops its stages
            cancel(job)
        self.staging.cancel(job)
        self.staging.release(live.request_job)
        self._free_unit(job)
        self._record_service(job)
        self.try_dispatch()
        self._record_request(
            live.request_job,
            None,
            RequestProcessingOutcome.STRONG_CANCELLED_DURING_SERVICE,
            None,
        )

    def _remove_from_queues(self, job: message.DecodeJob) -> None:
        for pool in self.unit_totals:
            queue = self.queue_for(pool)
            if job in queue:
                queue.remove(job)
                return

    def _find_window_job(self, window_key: tuple):
        candidates = []
        for pool in self.unit_totals:
            queue = self.queue_for(pool)
            candidates.extend(queue)
        for residents in self._unit_residents.values():
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
            fragments = _fragments_in(job.decoder_input)
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

    def _parked_labels(self) -> list:
        labels = []
        for job in self._parked_service.values():
            labels.append(job.label)
        return sorted(labels)

    def _in_flight_labels(self) -> list:
        labels = []
        for flights in self._pipeline_flights.values():
            for flight_job, _latency_ticks in flights:
                labels.append(flight_job.label)
        return sorted(labels)

    def _units_holding_rounds(self) -> list:
        held = []
        for memory in self.decoder_memories.values():
            if memory.occupied_rounds:
                held.append(f"{memory.pool}#{memory.unit}")
        return held


def _check_unit_pools(unit_pools: dict) -> None:
    """A pool map names the default pool and gives every pool a unit."""
    if "default" not in unit_pools:
        pools = sorted(unit_pools)
        raise ValueError(
            f'unit_pools must include a "default" pool (got {pools})'
        )
    for pool_name, units in unit_pools.items():
        if units < 1:
            raise ValueError(
                f"pool {pool_name!r} needs at least 1 unit (got {units})"
            )


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


def _refuse_bits_in_bulk_strong(job: message.DecodeJob) -> None:
    """bulk_strong merges timing-only strong re-decodes; bits would be lost."""
    has_model = job.dem is not None
    has_bits = False
    for payload in job.payloads:
        if payload.bits is not None:
            has_bits = True
    if has_model or has_bits:
        raise RuntimeError(
            "bulk_strong only merges timing-only strong re-decodes; "
            "disable it for accuracy-coupled switching."
        )


def _batch_job(jobs: list, window_keys: list) -> message.DecodeJob:
    """One timing-only decode serving every member request.

    The batch is decoded like any strong job: its timing-only result is
    split into one empty completion per member request at decode end.
    """
    total_rounds = 0
    earliest_ready_time = jobs[0].ready_time
    for job in jobs:
        total_rounds += job.n_rounds
        earliest_ready_time = min(earliest_ready_time, job.ready_time)
    first_window_key = None
    if window_keys:
        first_window_key = window_keys[0]
    batch_size = len(jobs)
    return message.DecodeJob(
        op_id=-1,
        window_id=0,
        n_rounds=total_rounds,
        ready_time=earliest_ready_time,
        label=f"strong-batch x{batch_size} ({total_rounds}r)",
        hint="strong",
        spatial_nodes=jobs[0].spatial_nodes,
        strong_decode_for=first_window_key,
    )


def _is_past_start(job: message.DecodeJob) -> bool:
    """A job that never starts again; an in-flight decode stays resident."""
    if job.cancelled:
        return True
    if job.completed:
        return True
    return job.service_started


def _is_outside_pipelined_model(job: message.DecodeJob) -> bool:
    if job.strong_decode_for is not None:
        return True
    if job.gap_sibling_for is not None:
        return True
    return len(job.service_original_request_keys) > 1


def _is_live_weak_window_job(job: message.DecodeJob, window_key: tuple) -> bool:
    key = (job.op_id, job.window_id)
    if key != window_key:
        return False
    if job.strong_decode_for is not None:
        return False
    if job.on_done is not None:
        return False
    return not job.cancelled


def _full_pipeline_depth(latency_ticks: int, interval_ticks: int) -> int:
    """The decodes in flight when a start every interval fills the latency."""
    decodes_per_latency = latency_ticks / interval_ticks
    full_pipeline = math.ceil(decodes_per_latency)
    return max(1, full_pipeline)


def _flight_of(flights: list, job: message.DecodeJob):
    for flight in flights:
        if flight[0] is job:
            return flight
    return None


def _flights_with_other_latency(flights: list, latency_ticks: int) -> list:
    others = []
    for flight_job, flight_latency in flights:
        if flight_latency != latency_ticks:
            others.append(flight_job)
    return others


def _fragments_in(decoder_input) -> list:
    """The landed fragments of a unit's input, round by round."""
    fragments = []
    for round_input in decoder_input.rounds:
        fragments.extend(round_input.fragments)
    return fragments


def _receiving_text(member: message.DecodeJob) -> str:
    return (
        f"receiving {member.label} input "
        f"({member.n_rounds} rounds from the round store)"
    )


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


def _job_defects_text(job: message.DecodeJob) -> str:
    """The landed window input's cargo, sparse, for the I/O trace.

    The set detection-event indices of the rounds now in this unit's
    memory (the algorithm stage reads the same fragments).
    """
    if job.decoder_input is not None:
        fragments = _fragments_in(job.decoder_input)
    else:
        fragments = job.payloads
        if fragments is None:
            fragments = []
    bit_arrays = []
    for fragment in fragments:
        if fragment.bits is None:
            continue
        bits = numpy.asarray(fragment.bits, dtype=numpy.uint8)
        bit_arrays.append(bits)
    if not bit_arrays:
        return "no payload bits"
    all_bits = numpy.concatenate(bit_arrays)
    defects = numpy.flatnonzero(all_bits)
    if defects.size == 0:
        return "no defects"
    defect_texts = []
    for defect in defects.tolist():
        defect_texts.append(str(defect))
    listed = ", ".join(defect_texts)
    return f"defects {{{listed}}}"
