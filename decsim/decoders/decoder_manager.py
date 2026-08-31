"""Assigns decoder units to ready windows: a queue per pool, unit
assignment, input staged into that unit's memory, service, completion. The
strong tier's request bookkeeping is the StrongRequestLedger's; the terminal
records at the bottom are optional capture for the switching study."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
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
    STRONG_CANCELLED_WHILE_STAGED = "strong_cancelled_while_staged"
    STRONG_CANCELLED_DURING_SERVICE = "strong_cancelled_during_service"
    STRONG_CANCELLED_MEMBER_SERVICE_CONTINUED = (
        "strong_cancelled_member_service_continued")
    WEAK_WITHDRAWN_FOR_STRONG_WINDOW = "weak_withdrawn_for_strong_window"


@dataclass
class GapJoinState:
    """One window's split-gap rendezvous: the weak outcome is processed
    only when both forced-class halves have reported.

    The join is an AND of two completions, the same shape as the
    landed-input join in ``_start_job`` (both DMAs must land before the
    decode starts); here both SOLVES must land before the escalation
    decision runs. The weak unit itself is freed at its own solve end;
    only the outcome waits.
    """

    sibling_weight: Optional[float] = None
    sibling_reported: bool = False
    held_weak_job: Optional[DecodeJob] = None
    held_weak_result: Optional["DecodeResult"] = None

    def hold_weak(self, job: DecodeJob, result) -> None:
        self.held_weak_job = job
        self.held_weak_result = result

    def report_sibling(self, weight) -> None:
        self.sibling_reported = True
        self.sibling_weight = weight


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
                 service_gate=None, apply_service_boundary=None,
                 stage_admission=None,
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
        # Every unit is a depth-1 decoupled access-execute machine (Smith
        # 1982; TI EDMA ping-pong, SPRAAN4A Example D; gem5-Aladdin ready
        # bits at whole-buffer granularity, Shao et al. MICRO 2016 Sec
        # IV-B-2): two input slots, so the next window's DMA overlaps the
        # current compute. The hardware price is visible, not hidden: both
        # inputs are resident in the unit's DecoderMemory, so a unit needs
        # capacity for two windows or the run stops loudly.
        #
        # Compute is claimed separately from the slots (Tomasulo's rule:
        # an instruction whose operands are not ready waits in its
        # reservation station, never on the functional unit). A landed job
        # whose window still owes a boundary parks in its slot and
        # releases its compute claim, so a dependent that fills early can
        # never deadlock the unit against its own predecessor.
        self.service_gate = service_gate
        self.apply_service_boundary = apply_service_boundary
        self.stage_admission = stage_admission
        self._parked_service: dict = {}  # request_key -> job in its slot, boundary owed
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
        # Split-pair gap joins: a router with a gap route turns them on.
        # (op_id, window_id) -> GapJoinState, created when the sibling
        # is spawned and removed when the join concludes the window.
        self.gap_split_enabled = getattr(router, "gap", None) is not None
        self._gap_joins: dict[tuple, GapJoinState] = {}
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
        elif job.on_done is None and job.gap_sibling_for is None:
            # a gap half is neither tier: it feeds the join, not the ledger
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
        slot = None if job.unit is None else (job.pool, job.unit)
        if slot is not None and not job.service_started:
            # cancelled in its slot before its decode started: drop the
            # resident and pass any compute claim onward
            residents = self._residents(slot)
            if job in residents:
                residents.remove(job)
            if job.request_key is not None:
                self._parked_service.pop(job.request_key, None)
            job.cancelled = True
            self.staging.cancel(job)
            self.staging.release(live.request_job)
            if self._computing.get(slot) is job:
                self._computing[slot] = None
                self._offer_compute(slot)
            job.unit = None
            self._record_request(
                live.request_job, None,
                RequestProcessingOutcome.STRONG_CANCELLED_WHILE_STAGED,
                None)
        elif job.pool is None:
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

    def _claim_compute(self, slot: tuple, job: DecodeJob) -> None:
        pool, unit = slot
        self.pool_free[pool] -= 1
        self._free_units[pool].remove(unit)
        self._computing[slot] = job

    def _is_parked(self, job: DecodeJob) -> bool:
        return (job.request_key is not None
                and self._parked_service.get(job.request_key) is job)

    def _free_unit(self, job: DecodeJob) -> None:
        """Compute finished: drop the job from its slot and offer the
        compute onward (the ping-pong swap at compute end)."""
        self._end_flight(job, offer_now=False)
        pool, unit = job.pool, job.unit
        job.unit = None
        slot = (pool, unit)
        residents = self._residents(slot)
        if job in residents:
            residents.remove(job)
        if self._computing.get(slot) is job:
            self._computing[slot] = None
        self.engine.log_io(
            f"unit {pool}#{unit} SRAM",
            lambda: f"emitted {job.label} result; holds "
                    f"{self._sram_description(slot)}")
        self._offer_compute(slot)

    def _sram_description(self, slot: tuple) -> str:
        """One compact line of a unit's residents and their phase, for the
        I/O trace."""
        residents = self._residents(slot)
        if not residents:
            return "empty"
        parts = []
        for resident in residents:
            if self._computing.get(slot) is resident and resident.service_started:
                phase = "computing"
            elif self._is_parked(resident):
                phase = "parked"
            elif resident.input_landed:
                phase = "ready"
            else:
                phase = "capturing"
            parts.append(f"{resident.label} {phase}, {resident.n_rounds} rounds")
        return "; ".join(parts)

    def _offer_compute(self, slot: tuple) -> None:
        """Free compute goes to the oldest startable resident, stays
        reserved for the oldest one still in flight, or returns to the
        pool. gem5 O3's scheduleReadyInsts is the reference rule: only a
        ready instruction acquires a functional unit (fu_pool->getUnit at
        issue), and blocked work waits in the queue, never on the unit."""
        pool, unit = slot
        if self._computing.get(slot) is not None:
            return
        for resident in self._residents(slot):
            if resident.cancelled or resident.completed or resident.service_started:
                continue                 # an in-flight pipelined decode stays resident
            if resident.input_landed and not self._is_parked(resident):
                self._computing[slot] = resident
                self._begin_service(resident)
                return
        for resident in self._residents(slot):
            if resident.cancelled or resident.completed or resident.service_started:
                continue
            if not resident.input_landed and self._startable(resident):
                self._computing[slot] = resident   # starts at its landing
                return
        self.pool_free[pool] += 1
        self._free_units[pool].append(unit)

    def _release_compute_claim(self, job: DecodeJob) -> None:
        """Tomasulo's rule at the boundary hazard: a parked job keeps its
        input slot but never the unit's compute."""
        slot = (job.pool, job.unit)
        if self._computing.get(slot) is job:
            self._computing[slot] = None
            self._offer_compute(slot)
            self.try_dispatch()

    def _dispatch_pool(self, pool: str) -> None:
        """Dispatch startable work first, in scheduler order, scanning past
        jobs with no eligible unit; boundary-blocked work is placed only
        when no startable job can be (gem5 O3 issues from its ready set
        oldest-first: non-ready work never displaces ready work)."""
        queue = self.queue_for(pool)
        while queue:
            ordered = []
            while queue:
                ordered.append(self._next_job(pool, queue))
            selection = None
            for prefer_startable in (True, False):
                for index, job in enumerate(ordered):
                    if self._startable(job) is not prefer_startable:
                        continue
                    placement = self._eligible_unit(pool, job)
                    if placement is not None:
                        selection = (index, job, placement)
                        break
                if selection is not None:
                    break
            if selection is None:
                queue.extend(ordered)
                return
            index, job, (unit, claim_compute) = selection
            queue.extend(other for position, other in enumerate(ordered)
                         if position != index)
            self._start_job(pool, job, unit=unit, claim_compute=claim_compute)

    def _residents(self, slot: tuple) -> list:
        return self._unit_residents.setdefault(slot, [])

    @staticmethod
    def _startable(job: DecodeJob) -> bool:
        """A job whose window owes no boundary may hold compute; anything
        windowless (external, strong context, merged batch) always may."""
        window = job.window
        return window is None or window.deps_remaining <= 0

    def _eligible_unit(self, pool: str, job: DecodeJob):
        """(unit, claim_compute) for this job, or None.

        A startable job takes any unit with a free input slot and claims
        compute when that unit's compute is free. A boundary-blocked job
        takes an input slot only (its DMA overlaps other work, Tomasulo's
        reservation station), and only once the window manager's
        stage_admission says its release is already resolving, so parked
        work can never squat a slot against the decode that must free it."""
        startable = self._startable(job)
        if not startable and (self.stage_admission is not None
                              and not self.stage_admission(job)):
            return None
        resident_capacity = self._resident_capacity(job)
        free = set(self._free_units[pool])
        for unit in self._free_units[pool]:
            slot = (pool, unit)
            if len(self._residents(slot)) < resident_capacity \
                    and self._slot_memory_ok(slot, job):
                return unit, startable
        for unit in range(self.unit_totals[pool]):
            slot = (pool, unit)
            if unit in free or len(self._residents(slot)) >= resident_capacity:
                continue
            if self._slot_memory_ok(slot, job):
                return unit, False
        return None

    def _resident_capacity(self, job: DecodeJob) -> int:
        """Two residents per unit (the depth-1 access-execute machine)
        unless the routed decoder pipelines: then every in-flight decode
        stays resident (its input lives in the unit's memory until its
        result emerges) plus one landing next. Memory admission still
        gates every resident, so a deep pipeline pays its SRAM price
        visibly or refuses loudly."""
        decoder = self.router.route(job)
        if getattr(decoder, "initiation_interval", None) is None:
            return 2
        interval = decoder.initiation_interval(job)
        depth = getattr(decoder, "pipeline_depth", None)
        if depth is None:
            depth = max(1, math.ceil(decoder.latency(job) / interval))
        return depth + 1

    def _initiation_interval_ticks(self, decoder, job: DecodeJob,
                                   latency_ticks: int):
        """(interval, depth) for a pipelined start, (None, None) otherwise.

        The pipelined model serves plain window and external decodes
        only; the strong tier, gap siblings, and merged batches keep
        occupancy == latency until they get their own design pass, and a
        pipelined route there refuses loudly rather than silently
        serializing."""
        interval_of = getattr(decoder, "initiation_interval", None)
        if interval_of is None:
            return None, None
        members = job.service_original_request_keys or ()
        if (job.strong_decode_for is not None
                or job.gap_sibling_for is not None
                or len(members) > 1):
            raise RuntimeError(
                f"decode job {job.label!r}: a pipelined decoder serves plain "
                f"window or external decodes only; the strong tier, gap "
                f"siblings, and merged batches are not pipelined yet")
        interval = interval_of(job)
        depth = getattr(decoder, "pipeline_depth", None)
        if depth is None:
            depth = max(1, math.ceil(latency_ticks / interval))
        return interval, depth

    def _initiation_complete(self, slot: tuple, job: DecodeJob,
                             depth: int) -> None:
        """The pipelined unit's intake is free again: release the compute
        claim so the next start may begin, unless the pipeline is full;
        a full pipeline keeps the claim until the next completion."""
        if job.cancelled or job.completed:
            return
        if self._computing.get(slot) is not job:
            return
        if len(self._pipeline_flights.get(slot, ())) >= depth:
            self._pipeline_stalled[slot] = (job, depth)
            return
        self._computing[slot] = None
        self._offer_compute(slot)
        self.try_dispatch()

    def _end_flight(self, job: DecodeJob, offer_now: bool) -> None:
        """A pipelined decode left the unit (done or cancelled): retire
        its flight and lift a full-pipeline stall. The caller's normal
        flow performs the compute offer unless offer_now says otherwise."""
        for slot, flights in self._pipeline_flights.items():
            entry = next((f for f in flights if f[0] is job), None)
            if entry is None:
                continue
            flights.remove(entry)
            stalled = self._pipeline_stalled.get(slot)
            if stalled is not None and len(flights) < stalled[1]:
                owner = stalled[0]
                del self._pipeline_stalled[slot]
                if (self._computing.get(slot) is owner
                        and not owner.cancelled and not owner.completed):
                    self._computing[slot] = None
                    if offer_now:
                        self._offer_compute(slot)
                        self.try_dispatch()
            return

    def _slot_memory_ok(self, slot: tuple, job: DecodeJob) -> bool:
        """A second resident joins only if the unit's memory holds both
        inputs; a unit sized for one window keeps serial residency (the
        doubled-SRAM price of overlap is paid explicitly, never assumed).
        A first resident is always admitted, so a genuinely oversized
        window still stops loudly at its deposit."""
        live = [resident for resident in self._residents(slot)
                if not resident.cancelled and not resident.completed]
        if not live:
            return True
        capacity = self.decoder_memories[slot].capacity_rounds
        if capacity is None:
            return True
        demand = sum(self._memory_demand(resident) for resident in live)
        return demand + self._memory_demand(job) <= capacity

    @staticmethod
    def _memory_demand(job: DecodeJob) -> int:
        # an external job carries no syndrome data and stores nothing
        return 0 if job.on_done is not None else job.n_rounds

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

    def _start_job(self, pool: str, job: DecodeJob, *,
                   unit: int, claim_compute: bool) -> None:
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
        # the unit is assigned at DMA start (gem5-Aladdin's invocation
        # model: invoke the unit, then DMA its input); compute is claimed
        # only when this unit's compute is actually free
        job.unit = unit
        slot = (pool, unit)
        self._residents(slot).append(job)
        if claim_compute:
            self._claim_compute(slot, job)
        if job.window is not None:
            job.window.t_dispatch = self.engine.now
        waited_ticks = self.engine.now - job.ready_time
        slot_note = "" if claim_compute else "staged, "
        self.engine.log(self.log_name,
                        f"ASSIGN UNIT {job.label} "
                        f"({slot_note}waited {fmt(waited_ticks).strip()} in queue, "
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
            if pending["count"] > 0:
                return
            job.input_landed = True
            self.engine.log_io(
                f"unit {pool}#{unit} SRAM",
                lambda: f"{job.label} input landed; {_job_defects_text(job)}; "
                        f"holds {self._sram_description(slot)}")
            if self._computing.get(slot) is job:
                # this job holds or was reserved the unit's compute
                self._begin_service(job)
            elif (self._computing.get(slot) is None
                    and unit in self._free_units[pool]):
                # compute went back to the pool (a resident parked and
                # released its claim); take it now
                self._claim_compute(slot, job)
                self._begin_service(job)
            # otherwise the compute is busy: _offer_compute picks this
            # job up at the next compute end

        for member in members:
            self.engine.log_io(
                f"unit {pool}#{unit} SRAM",
                lambda staged=member: f"receiving {staged.label} input "
                                      f"({staged.n_rounds} rounds from the "
                                      f"round store)")
            self.staging.stage(member, memory, landed)

    def _begin_service(self, job: DecodeJob, gated: bool = True) -> None:
        """The unit's memory holds the input: start the decode."""
        if job.cancelled:                        # cancelled while its input was in flight
            return
        if (gated and self.service_gate is not None
                and not self.service_gate(job)):
            self._parked_service[job.request_key] = job
            self.engine.log(self.log_name,
                            f"PARK DECODE {job.label} (boundary pending)")
            self._release_compute_claim(job)
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

        latency_ticks = decoder.latency(job)
        interval_ticks, pipeline_depth = self._initiation_interval_ticks(
            decoder, job, latency_ticks)
        self.engine.schedule(latency_ticks, decode_now,
                             label=f"decode_done({job.label})")
        if interval_ticks is not None:
            slot = (job.pool, job.unit)
            flights = self._pipeline_flights.setdefault(slot, [])
            # in-order completion: a hardware pipeline retires in issue
            # order, so every in-flight decode on one unit must declare
            # the same latency; mixed latencies refuse loudly
            mixed = [f for f, ticks in flights if ticks != latency_ticks]
            if mixed:
                raise RuntimeError(
                    f"decode job {job.label!r} declares latency "
                    f"{latency_ticks} ticks while {mixed[0].label!r} is in "
                    f"flight with a different latency: a pipelined unit "
                    f"completes in order and takes one latency per unit")
            flights.append((job, latency_ticks))
            self.engine.schedule(
                interval_ticks,
                lambda s=slot, j=job, d=pipeline_depth:
                    self._initiation_complete(s, j, d),
                label=f"initiation_complete({job.label})")

    def _spawn_gap_sibling(self, job: DecodeJob) -> None:
        """Submit the other forced-class solve to the gap pool.

        Fired at the primary weak decode's service start: the boundary
        mask is applied by then, so the sibling reads the same masked
        rounds the primary decodes (both units receive the same adjusted
        stream, the CSB dual-write pattern on the strong side). The
        sibling carries its own copy of the rounds, pays its own WBD
        transfer and queues for its own unit. The copy is taken here
        instead of holding Buffer 0 for a second read, so the sibling's
        transfer is priced but never blocks on round retention.
        """
        if not self.gap_split_enabled:
            return
        if job.strong_decode_for is not None or job.on_done is not None:
            return
        if job.gap_sibling_for is not None:
            return
        if job.window is None or job.dem is None or job.decoder_input is None:
            return
        key = (job.op_id, job.window_id)
        if key in self._gap_joins:
            return
        masked_fragments = [
            fragment
            for round_input in job.decoder_input.rounds
            for fragment in round_input.fragments
        ]
        sibling = DecodeJob(
            op_id=job.op_id, window_id=job.window_id,
            n_rounds=len(job.decoder_input.rounds),
            dem=job.dem, payloads=masked_fragments,
            ready_time=self.engine.now,
            label=f"gap({job.label})", hint="gap",
            spatial_nodes=job.spatial_nodes, code=job.code,
            gap_sibling_for=key)
        self._gap_joins[key] = GapJoinState()
        window_manager = getattr(self.services, "wm", None)

        def reserve_transfer(sibling=sibling, primary=job):
            if window_manager is None:
                return 0
            from ..links.links import LinkPath
            payload_bits = window_manager._job_payload_bits(sibling)
            # the link attribution is the window's, so the reservation
            # rides the primary job's window identity; the delay applies
            # to the sibling's own delivery
            arrival = window_manager._link_arrival(
                LinkPath.WBD, primary, payload_bits=payload_bits)
            return arrival - self.engine.now

        self.engine.log(self.log_name,
                        f"SPLIT GAP {job.label}: sibling submitted to the "
                        f"gap pool")
        self.enqueue(sibling, reserve_transfer)

    def _gap_sibling_done(self, key: tuple, sibling_weight) -> None:
        """One gap half landed; conclude the window if the other is in."""
        join = self._gap_joins.get(key)
        if join is None:
            raise RuntimeError(
                f"gap sibling finished for window {key} with no join entry")
        join.report_sibling(sibling_weight)
        if join.held_weak_job is None:
            return                    # the primary decode is still running
        del self._gap_joins[key]
        self._attach_joined_gap(join.held_weak_result, sibling_weight)
        self._conclude_weak(join.held_weak_job, join.held_weak_result)

    @staticmethod
    def _attach_joined_gap(result: DecodeResult, sibling_weight) -> None:
        """Build the SoftOutput from the two forced-class weights.

        Either half missing leaves soft_output None, and the policy then
        escalates (the same behavior a metric-less serial decode has).
        """
        primary_weight = result.gap_half_weight
        if primary_weight is None or sibling_weight is None:
            return
        from ..confidence.complementary import COMPLEMENTARY_GAP_SOURCE
        w_min = min(primary_weight, sibling_weight)
        w_comp = max(primary_weight, sibling_weight)
        result.soft_output = SoftOutput(
            gap=w_comp - w_min,
            source=COMPLEMENTARY_GAP_SOURCE,
            w_min=w_min,
            w_comp=w_comp)

    def withdraw_window(self, window_key: tuple) -> None:
        """Take back one window's submitted, not-yet-started weak decode:
        the window is being rewritten (a strong window absorbs it), so its
        raw input is superseded. The attempt closes in the ledger and the
        caller resubmits a fresh job if the window is rebuilt."""
        job = self._find_window_job(window_key)
        if job is None:
            raise RuntimeError(
                f"no withdrawable decode for window {window_key}")
        if job.service_started or job.completed:
            raise RuntimeError(
                f"{job.label} cannot be withdrawn: its decode already started")
        job.cancelled = True
        pool = self.pool_for(job)
        queue = self.queue_for(pool)
        if job in queue:
            queue.remove(job)
            self.staging.release(job)
        elif job.unit is not None:
            slot = (job.pool, job.unit)
            residents = self._residents(slot)
            if job in residents:
                residents.remove(job)
            if job.request_key is not None:
                self._parked_service.pop(job.request_key, None)
            self.staging.cancel(job)
            if self._computing.get(slot) is job:
                self._computing[slot] = None
                self._offer_compute(slot)
            job.unit = None
        else:
            raise RuntimeError(
                f"{job.label} is neither queued nor resident; nothing to withdraw")
        self.strong.resolve_weak(window_key)
        self._record_request(
            job, None,
            RequestProcessingOutcome.WEAK_WITHDRAWN_FOR_STRONG_WINDOW, None)
        self.engine.log(self.log_name,
                        f"WITHDRAW {job.label} (invalidated before start)")
        self.try_dispatch()

    def _find_window_job(self, window_key: tuple):
        candidates = []
        for pool in self.unit_totals:
            candidates.extend(self.queue_for(pool))
        for residents in self._unit_residents.values():
            candidates.extend(residents)
        for job in candidates:
            if ((job.op_id, job.window_id) == window_key
                    and job.strong_decode_for is None
                    and job.on_done is None and not job.cancelled):
                return job
        return None

    def release_parked(self, window_key: tuple) -> None:
        """The window's last boundary arrived: start its parked decode, if
        its input has landed (a job still in transfer passes the gate at
        its own landing instead)."""
        for request_key, job in list(self._parked_service.items()):
            if (job.op_id, job.window_id) != window_key:
                continue
            if self.service_gate is not None and not self.service_gate(job):
                continue                     # another dependency still owed
            del self._parked_service[request_key]
            slot = (job.pool, job.unit)
            holder = self._computing.get(slot)
            if holder is None and job.unit in self._free_units[job.pool]:
                self._claim_compute(slot, job)
                self._begin_service(job, gated=False)
            elif (holder is not None and not holder.service_started
                    and not holder.input_landed):
                # steal a reservation held for an input still in flight:
                # ready work issues first; the in-flight job re-competes
                # at its own landing
                self._computing[slot] = job
                self._begin_service(job, gated=False)
            # else the unit is decoding: no longer parked, so
            # _offer_compute starts this job at the next compute end
        self.try_dispatch()   # a started decode may admit blocked stages

    def _on_decode_done(self, job: DecodeJob, result) -> None:
        """One decode finished: free the unit, ask the escalation_policy, commit or await strong."""
        if job.cancelled:
            self._end_flight(job, offer_now=True)
            self.staging.release(job)
            self.try_dispatch()
            return
        if job.gap_sibling_for is not None:
            job.completed = True
            self._free_unit(job)
            self.staging.release(job)
            self.engine.log(self.log_name,
                            f"DECODE DONE {job.label} (gap half)")
            sibling_weight = None if result is None else result.gap_half_weight
            self._gap_sibling_done(job.gap_sibling_for, sibling_weight)
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
        join = self._gap_joins.get(key)
        if join is not None:
            # the unit is free either way; the OUTCOME waits at the join
            self.staging.release(job)
            if not join.sibling_reported:
                join.hold_weak(job, result)
                self.engine.log(self.log_name,
                                f"GAP JOIN {job.label}: holding for the "
                                f"sibling half")
                self.try_dispatch()
                return
            del self._gap_joins[key]
            self._attach_joined_gap(result, join.sibling_weight)
        self._conclude_weak(job, result)

    def _conclude_weak(self, job: DecodeJob, result) -> None:
        """The weak outcome's decision and delivery; in split-pair mode
        this runs at the gap join, otherwise straight from decode end."""
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
        if self._parked_service:
            parked = sorted(job.label for job in self._parked_service.values())
            raise RuntimeError(
                f"run ended with parked decodes never released: {parked}")
        if self._gap_joins:
            raise RuntimeError(
                f"run ended with split-gap joins unresolved: "
                f"{sorted(self._gap_joins)}")
        leftover_flights = sorted(
            flight_job.label for flights in self._pipeline_flights.values()
            for flight_job, _ in flights)
        if leftover_flights:
            raise RuntimeError(
                f"run ended with pipelined decodes still in flight: "
                f"{leftover_flights}")
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


def _job_defects_text(job: DecodeJob) -> str:
    """The landed window input's cargo: set detection-event indices of the
    rounds now in this unit's memory (the algorithm stage reads the same
    fragments), sparse for the I/O trace."""
    import numpy as np
    if job.decoder_input is not None:
        fragments = [fragment for round_input in job.decoder_input.rounds
                     for fragment in round_input.fragments]
    else:
        fragments = list(job.payloads or [])
    bit_arrays = [np.asarray(fragment.bits, dtype=np.uint8)
                  for fragment in fragments if fragment.bits is not None]
    if not bit_arrays:
        return "no payload bits"
    defects = np.flatnonzero(np.concatenate(bit_arrays))
    if defects.size == 0:
        return "no defects"
    return f"defects {{{', '.join(map(str, defects.tolist()))}}}"
