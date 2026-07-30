"""Decoder queues, dispatch, switching ownership, and terminal records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from .message import (DecodeJob, DecodeOutcome, DecodeResult,
                      DecoderRequestKey, DecoderServiceKey,
                      SoftOutput, StrongDecodeCompletion,
                      stable_identity_order_key)
from .protocols import Directive
from .config import fmt


@dataclass(frozen=True)
class _LiveStrongRequest:
    request_job: DecodeJob
    service_job: DecodeJob


@dataclass(frozen=True)
class _HeldStrongCompletion:
    request_job: DecodeJob
    completion: StrongDecodeCompletion
    decode_output_ticks: int


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
    scheduler_priority_ticks: int
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
    """Own decoder queues, pools, routing, dispatch, and completion pipeline."""

    def __init__(self, engine, *, router, scheduler,
                 unit_pools: Optional[dict] = None, num_units: int = 1,
                 bulk_strong: bool = False,
                 lane_policy=None, log_name: str = "DecoderCluster",
                 capture_enabled: bool = False):
        self.engine = engine
        self.router = router
        self.scheduler = scheduler
        self.lane_policy = lane_policy
        self.bulk_strong = bulk_strong
        self.log_name = log_name
        self._terminal_request_records = [] if capture_enabled else None
        self._terminal_service_records = [] if capture_enabled else None

        self.strategy = None
        self.services = None
        self.on_window_decoded: Optional[Callable] = None
        self.on_strong_window_decoded: Optional[Callable] = None

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
        self.num_units = self.unit_totals["default"]
        self.ready: list[DecodeJob] = []
        self.pool_ready: dict[str, list] = {
            p: [] for p in self.unit_totals if p != "default"}
        self.queue_log: list[tuple[int, int]] = []

        self.strong_needed = 0
        self.strong_cancelled = 0
        self._running_strong_decodes: dict[tuple, _LiveStrongRequest] = {}
        self._windows_waiting_for_strong_selection: dict[tuple, DecoderRequestKey] = {}
        self._windows_waiting_for_strong_result: dict[tuple, DecoderRequestKey] = {}
        self._completed_strong_results: dict[tuple, _HeldStrongCompletion] = {}
        self._unresolved_weak_decodes: set[tuple] = set()


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

    def decoder_for(self, job: DecodeJob):
        return self.router.route(job)

    def enqueue(self, job: DecodeJob, delay_ticks: int = 0) -> None:
        """Entry point for the window_manager's strategy Submissions."""
        self._reject_spent_job(job)
        if job.strong_decode_for is not None:
            self._admit_strong_request(job)
        elif job.on_done is None:
            self._admit_weak_decode(job)
        job.submitted = True
        if delay_ticks <= 0:
            self._enqueue_now(job)
            return
        self.engine.log(self.log_name,
                        f"weak decoder unsure about window "
                        f"({job.op_id},{job.window_id}) -> hand {job.n_rounds} "
                        f"rounds to the strong decoder (after the weak->strong "
                        f"link)")
        self.engine.schedule(delay_ticks, lambda: self._enqueue_now(job),
                             label=f"weak->strong handoff {job.label}")

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

    def _admit_strong_request(self, job: DecodeJob) -> None:
        """Give one destination's next strong result to this request."""
        key = job.strong_decode_for
        if job.request_key is None:
            raise RuntimeError("built-in strong decode requires a request key")
        if (key in self._running_strong_decodes
                or key in self._completed_strong_results):
            raise RuntimeError(
                f"duplicate strong decode for window {key}: a destination "
                f"window has at most one unconsumed strong result")
        self._running_strong_decodes[key] = _LiveStrongRequest(job, job)

    def _admit_weak_decode(self, job: DecodeJob) -> None:
        """Open one destination window's decode attempt."""
        key = (job.op_id, job.window_id)
        if job.request_key is None:
            raise RuntimeError("built-in weak decode requires a request key")
        job.request_admitted_ticks = self.engine.now
        if key in self._unresolved_weak_decodes:
            raise RuntimeError(
                f"second weak decode for window {key} while the first is "
                f"unresolved: a destination window decodes once at a time, so "
                f"that its strong result reaches the attempt that asked")
        self._unresolved_weak_decodes.add(key)

    def _destination_may_consume_strong(self, key: tuple) -> bool:
        """Whether a strong result for this destination still has a consumer:
        the destination is waiting for one now, or its weak decode is open and
        its directive may still ask for one."""
        return (key in self._windows_waiting_for_strong_result
                or key in self._windows_waiting_for_strong_selection
                or key in self._unresolved_weak_decodes)

    def _enqueue_now(self, job: DecodeJob) -> None:
        if job.strong_decode_for is not None:
            live = self._running_strong_decodes.get(job.strong_decode_for)
            if live is None or live.request_job is not job:
                return                             # cancelled across the link
            job.ready_time = self.engine.now
            job.request_admitted_ticks = self.engine.now
        pool = self.pool_for(job)
        queue = self.queue_for(pool)
        self.scheduler.insert(queue, job)
        self.engine.log(self.log_name,
                        f"{job.label} READY -> enqueue "
                        f"({self.pool_tag(pool)}ready-queue length = {len(queue)})")
        self.queue_log.append((self.engine.now, self.queued_total()))
        self.try_dispatch()

    def submit_decode(self, round_count: int, on_done: Callable[[], None],
                      label: str = "external", deadline: Optional[int] = None,
                      code: Optional[str] = None,
                      spatial_nodes: Optional[int] = None,
                      hint: Optional[str] = None) -> None:
        """Submit a self-contained external decode job (factory corrections,
        separate idle decodes)."""
        job = DecodeJob(op_id=-1, window_id=0, n_rounds=round_count,
                        ready_time=self.engine.now,
                        deadline=self.engine.now if deadline is None else deadline,
                        on_done=on_done, label=label, code=code,
                        spatial_nodes=spatial_nodes, hint=hint)
        self.scheduler.insert(self.queue_for(self.pool_for(job)), job)
        self.queue_log.append((self.engine.now, self.queued_total()))
        self.try_dispatch()

    def check_strong_route(self, weak_job: DecodeJob,
                           strong_job: DecodeJob) -> None:
        """Fail early when a strong job would route back to the weak decoder."""
        if self.decoder_for(strong_job) is self.decoder_for(weak_job):
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
        held = self._completed_strong_results.pop(key, None)
        if held is not None:
            self._record_request(
                held.request_job, held.completion.result,
                RequestProcessingOutcome.STRONG_COMPLETED_DISCARDED,
                held.decode_output_ticks)
        live = self._running_strong_decodes.pop(key, None)
        if live is None:
            return
        job = live.service_job
        if job.pool is None:
            for pool in self.unit_totals:
                queue = self.queue_for(pool)
                if job in queue:
                    queue.remove(job)
                    break
            self._record_request(
                live.request_job, None,
                RequestProcessingOutcome.STRONG_CANCELLED_BEFORE_DISPATCH,
                None)
        else:
            job.service_cancelled_request_keys.add(live.request_job.request_key)
            survivors = any(
                survivor.service_job is job
                for survivor in self._running_strong_decodes.values())
            if survivors:
                outcome = RequestProcessingOutcome.STRONG_CANCELLED_MEMBER_SERVICE_CONTINUED
            else:
                outcome = RequestProcessingOutcome.STRONG_CANCELLED_DURING_SERVICE
                job.cancelled = True
                self.pool_free[job.pool] += 1
                self._record_service(job)
                self.try_dispatch()
            self._record_request(live.request_job, None, outcome, None)
        self.strong_cancelled += 1

    def try_dispatch(self) -> None:
        for pool in self.unit_totals:
            self._dispatch_pool(pool)

    def _dispatch_pool(self, pool: str) -> None:
        queue = self.queue_for(pool)
        while self.pool_free[pool] > 0 and queue:
            job = self._next_job(pool, queue)
            self._start_job(pool, job)

    def _next_job(self, pool: str, queue: list) -> DecodeJob:
        if self.bulk_strong and pool != "default":
            job = self._merge_strong_batch(queue)
            return job
        return self.scheduler.pop(queue, self.engine.now)

    def _merge_strong_batch(self, queue: list) -> DecodeJob:
        """Batch queued strong jobs (timing-only) into one decode."""
        jobs = [
            self.scheduler.pop(queue, self.engine.now)
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
                          deadline=min(j.deadline for j in jobs),
                          on_done=lambda: None,
                          label=f"strong-batch x{len(jobs)} ({total}r)",
                          hint="strong", spatial_nodes=jobs[0].spatial_nodes,
                          strong_decode_for=window_keys[0] if window_keys
                          else None)
        batch.service_original_request_keys = request_keys
        for window_key, request_job in zip(window_keys, jobs):
            self._running_strong_decodes[window_key] = _LiveStrongRequest(
                request_job, batch)
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
            for live in self._running_strong_decodes.values():
                if live.service_job is job:
                    live.request_job.service_key = job.service_key
                    live.request_job.service_dispatch_ticks = self.engine.now
        self.pool_free[pool] -= 1
        if job.window is not None:
            job.window.t_dispatch = self.engine.now
        latency_ticks = self.decoder_for(job).latency(job)
        waited_ticks = self.engine.now - job.ready_time
        self.engine.log(self.log_name,
                        f"START DECODE {job.label} "
                        f"(waited {fmt(waited_ticks).strip()} in queue, "
                        f"{self.pool_tag(pool)}units free now "
                        f"{self.pool_free[pool]})")
        self.queue_log.append((self.engine.now, self.queued_total()))
        self.engine.schedule(
            latency_ticks, lambda j=job: self._on_decode_done(j),
            label=f"decode_done({job.label})")

    def _on_decode_done(self, job: DecodeJob) -> None:
        """Contract 2b pipeline with the strategy seam in the switching slots."""
        if job.cancelled:
            return
        if job.completed:
            raise RuntimeError(
                f"duplicate decoder completion for job "
                f"({job.op_id}, {job.window_id})")

        if job.strong_decode_for is not None:
            result = self._decode_and_validate_result(job)
            strong_result_deliveries = self._prepare_strong_result_deliveries(
                job, result)
            job.completed = True
            self.pool_free[job.pool] += 1
            self._finish_strong_bookkeeping(job)
            self.strategy.on_decode_outcome(DecodeOutcome(job, result),
                                            self.services)   # FINALIZE_STRONG
            self._handle_strong_decode_result(strong_result_deliveries)
            self._record_service(job)
            self.try_dispatch()
            return

        if job.on_done is not None:                            # external job
            job.completed = True
            self.pool_free[job.pool] += 1
            self.engine.log(self.log_name,
                            f"DECODE DONE {job.label} "
                            f"({self.pool_tag(job.pool)}units free now "
                            f"{self.pool_free[job.pool]})")
            job.on_done()
            self.try_dispatch()
            return

        result = self._decode_and_validate_result(job)
        job.completed = True
        self.pool_free[job.pool] += 1
        key = (job.op_id, job.window_id)
        directive = self.strategy.on_decode_outcome(DecodeOutcome(job, result),
                                                    self.services)
        self._resolve_weak_decode(key)
        awaiting = directive.directive is Directive.AWAIT_STRONG
        if not awaiting and (directive.extra is not None
                             or directive.strong_request_key is not None):
            name = directive.directive.name.lower()
            raise RuntimeError(f"{name} directive cannot carry strong identity")
        if directive.directive is Directive.FINALIZE:
            self.cancel_strong(key)                # no-op unless one is live/held
        if awaiting:
            serial_job = None if directive.extra is None else directive.extra.job
            strong_request_key = directive.strong_request_key
            carriers = tuple(filter(None, (
                self._running_strong_decodes.get(key),
                self._completed_strong_results.get(key),
            )))
            deferred = serial_job is None and strong_request_key is not None
            if serial_job is not None:
                if (strong_request_key is None or carriers
                        or serial_job.request_key != strong_request_key):
                    raise RuntimeError("serial directive request key mismatch")
            elif deferred:
                if carriers:
                    raise RuntimeError(
                        "parallel directive cannot provide an explicit key")
            else:
                if len(carriers) != 1:
                    raise RuntimeError(
                        "parallel strong selection needs exactly one carrier")
                carrier = carriers[0]
                strong_request_key = carrier.request_job.request_key
            selection_delay = self.services.prepare_strong_selection(
                job, strong_request_key, serial_job, deferred=deferred,
            )
            self.strong_needed += 1
            self._windows_waiting_for_strong_selection[key] = strong_request_key
            self.engine.schedule(
                selection_delay,
                lambda destination=key, request_key=strong_request_key:
                    self._select_strong_result(destination, request_key),
                label=f"select strong result {key}",
            )
        job.awaiting_strong_result = awaiting      # BEFORE the commit callback
        if self.on_window_decoded is None:
            raise RuntimeError("DecoderManager has no window completion callback")
        self.on_window_decoded(job, result)
        self._record_request(
            job, result,
            (RequestProcessingOutcome.WEAK_AWAITED_STRONG if awaiting else
             RequestProcessingOutcome.WEAK_FORWARDED_FOR_DELIVERY),
            self.engine.now)
        self._record_service(job)
        self.try_dispatch()

    def _select_strong_result(self, key: tuple,
                              request_key: DecoderRequestKey) -> None:
        """Make one strong completion eligible only after WSD delivery."""
        if self._windows_waiting_for_strong_selection.get(key) != request_key:
            return
        del self._windows_waiting_for_strong_selection[key]
        self._windows_waiting_for_strong_result[key] = request_key
        self._apply_held_strong_result(key, request_key)

    def _resolve_weak_decode(self, key: tuple) -> None:
        """This destination's weak decode has produced its directive, so it
        stops being a reason to keep a strong result for the window and the
        destination is free to be decoded again. The reservation covers
        on_decode_outcome, which is where the directive is chosen, and ends
        before that directive is applied. Every job reaching here was admitted
        by _admit_weak_decode, so the key is present."""
        self._unresolved_weak_decodes.remove(key)

    def _decode_and_validate_result(self, job: DecodeJob) -> DecodeResult:
        """Decode one job and reject output for any other operation or window."""
        result = self.decoder_for(job).decode(job)
        if not isinstance(result, DecodeResult):
            raise TypeError(
                f"decoder for job ({job.op_id}, {job.window_id}) must return "
                f"DecodeResult, got {type(result).__name__}")
        expected = (job.op_id, job.window_id)
        actual = (result.op_id, result.window_id)
        if actual != expected:
            raise RuntimeError(
                f"decoder result identity {actual} does not match job "
                f"identity {expected}")
        self._validate_logical_observables(job, result)
        return result

    @staticmethod
    def _validate_logical_observables(
        job: DecodeJob,
        result: DecodeResult,
    ) -> None:
        logical_observables = result.logical_observables
        if logical_observables is None:
            return
        if type(logical_observables) is not tuple:
            raise TypeError(
                f"job ({job.op_id}, {job.window_id}) logical_observables "
                f"must be an exact tuple, got "
                f"{type(logical_observables).__name__}")
        for observable_index, bit in enumerate(logical_observables):
            if type(bit) is not int:
                raise TypeError(
                    f"job ({job.op_id}, {job.window_id}) "
                    f"logical_observables index {observable_index} must be "
                    f"an exact int bit, got {type(bit).__name__}")
            if bit not in (0, 1):
                raise ValueError(
                    f"job ({job.op_id}, {job.window_id}) "
                    f"logical_observables index {observable_index} must be "
                    f"0 or 1, got {bit}")

    def _finish_strong_bookkeeping(self, job: DecodeJob) -> None:
        if job.strong_decode_for is not None:
            for key, live in tuple(self._running_strong_decodes.items()):
                if live.service_job is job:
                    self._running_strong_decodes.pop(key)

    def admitted_strong_work_snapshot(self) -> tuple:
        """Snapshot each physical strong job once in its authoritative phase."""
        queue_memberships = {}
        for pool in self.unit_totals:
            for queued_job in self.queue_for(pool):
                queue_memberships.setdefault(id(queued_job), []).append(queued_job)

        jobs_by_identity = {}
        keys_by_identity = {}
        for destination_key, live in self._running_strong_decodes.items():
            job = live.service_job
            identity = id(job)
            jobs_by_identity[identity] = job
            keys_by_identity.setdefault(identity, []).append(destination_key)

        records = []
        for identity, job in jobs_by_identity.items():
            queued_matches = queue_memberships.get(identity, ())
            if any(candidate is not job for candidate in queued_matches):
                raise RuntimeError("strong-work identity collision in ready queues")
            if len(queued_matches) > 1:
                raise RuntimeError("one strong job appears in multiple ready queues")
            if job.pool is not None:
                if queued_matches:
                    raise RuntimeError("running strong job also remains queued")
                phase = "running"
            elif queued_matches:
                phase = "queued"
            else:
                phase = "in_transit"
            destination_keys = tuple(sorted(
                keys_by_identity[identity], key=stable_identity_order_key
            ))
            records.append((destination_keys, phase, job.n_rounds))
        return tuple(sorted(
            records,
            key=lambda record: (
                tuple(stable_identity_order_key(key) for key in record[0]),
                record[1],
                record[2],
            ),
        ))

    def _prepare_strong_result_deliveries(
        self, job: DecodeJob, result: DecodeResult,
    ) -> tuple:
        """Validate batch provenance and return per-window result deliveries."""
        requests = tuple(
            live.request_job
            for live in self._running_strong_decodes.values()
            if live.service_job is job
        )
        keys = tuple(request.strong_decode_for for request in requests)
        result_identity = (result.op_id, result.window_id)
        is_merged_delivery = (
            len(keys) > 1
            or any(key != result_identity for key in keys)
        )
        if not is_merged_delivery:
            completion = StrongDecodeCompletion(requests[0].request_key, result)
            return ((_HeldStrongCompletion(
                requests[0], completion, self.engine.now)),)

        accuracy_field_names = (
            "correction",
            "logical_observables",
            "soft_output",
            "boundary_defects",
            "boundary_data",
        )
        populated_field_names = [
            field_name
            for field_name in accuracy_field_names
            if getattr(result, field_name) is not None
        ]
        if populated_field_names:
            populated_fields = ", ".join(populated_field_names)
            raise RuntimeError(
                "merged strong decode returned accuracy-bearing fields "
                f"({populated_fields}); disable bulk_strong for accuracy-coupled "
                "switching")

        return tuple(_HeldStrongCompletion(
            request,
            StrongDecodeCompletion(
                request.request_key,
                DecodeResult(op_id=key[0], window_id=key[1])),
            self.engine.now,
        ) for key, request in zip(keys, requests))

    def _handle_strong_decode_result(
        self, strong_result_deliveries: tuple,
    ) -> None:
        for held in strong_result_deliveries:
            self._complete_strong_result(held)

    def _complete_strong_result(self, held: _HeldStrongCompletion) -> None:
        """Apply a strong result if demanded, otherwise hold it briefly."""
        completion = held.completion
        key = (completion.request_key.operation_id, completion.request_key.window_id)
        if self._windows_waiting_for_strong_result.get(key) == completion.request_key:
            del self._windows_waiting_for_strong_result[key]
            if self.on_strong_window_decoded is None:
                raise RuntimeError(
                    "DecoderManager has no strong completion callback")
            self.on_strong_window_decoded(completion)
            self._record_request(
                held.request_job, completion.result,
                RequestProcessingOutcome.STRONG_FORWARDED_FOR_DELIVERY,
                held.decode_output_ticks)
            return
        if key in self._running_strong_decodes:
            raise RuntimeError(
                f"strong result for window {key} arrived after a newer strong "
                f"request took the destination's next result: nothing would "
                f"consume this one")
        if not self._destination_may_consume_strong(key):
            raise RuntimeError(
                f"strong result for window {key} has no destination waiting "
                f"for it: the destination registered no strong demand and its "
                f"decode attempt has resolved")
        self._completed_strong_results[key] = held

    def _apply_held_strong_result(self, key: tuple,
                                  request_key: DecoderRequestKey) -> None:
        """Deliver a completion held from before this destination's demand."""
        if key in self._completed_strong_results:
            held = self._completed_strong_results[key]
            if held.completion.request_key != request_key:
                return
            self._completed_strong_results.pop(key)
            self._complete_strong_result(held)

    def _record_request(self, job: DecodeJob, result: Optional[DecodeResult],
                        outcome: RequestProcessingOutcome,
                        decode_output_ticks: Optional[int]) -> None:
        if self._terminal_request_records is None:
            return
        if job.request_key is None or job.request_created_ticks is None:
            raise RuntimeError("terminal built-in request has no identity")
        window = job.window
        if window is None:
            raise RuntimeError("terminal built-in request has no window")
        bits_known = bool(job.payloads) and all(
            payload.bits is not None for payload in job.payloads)
        bit_count = (sum(len(payload.bits) for payload in job.payloads)
                     if bits_known else None)
        weight = (sum(sum(payload.bits) for payload in job.payloads)
                  if bits_known else None)
        self._terminal_request_records.append(TerminalRequestRecord(
            job.request_key, window.start_round, window.buffer_hi, job.n_rounds,
            bit_count, weight, job.request_created_ticks,
            job.request_admitted_ticks, job.ready_time,
            job.service_dispatch_ticks, decode_output_ticks, job.deadline,
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
        if dispatch is None or job.pool is None:
            raise RuntimeError("terminal decoder service has no dispatch")
        self._terminal_service_records.append(TerminalServiceRecord(
            job.service_key, job.pool, original, completed, cancelled,
            job.n_rounds, dispatch, self.engine.now, self.engine.now - dispatch))

    def terminal_request_records_snapshot(self) -> tuple:
        if self._terminal_request_records is None:
            raise RuntimeError("switching record capture is disabled")
        return tuple(self._terminal_request_records)

    def terminal_service_records_snapshot(self) -> tuple:
        if self._terminal_service_records is None:
            raise RuntimeError("switching record capture is disabled")
        return tuple(self._terminal_service_records)

    def check_decode_work_settled(self) -> None:
        """Require every admitted decode to reach a final result."""
        unsettled = {
            state: sorted(keys) for state, keys in (
                ("waiting for a strong result",
                 self._windows_waiting_for_strong_result),
                ("waiting for strong selection",
                 self._windows_waiting_for_strong_selection),
                ("holding an unclaimed strong result",
                 self._completed_strong_results),
                ("still holding a strong request", self._running_strong_decodes),
                ("decoding with no outcome", self._unresolved_weak_decodes),
            ) if keys
        }
        if unsettled:
            detail = "; ".join(f"{state}: {keys}"
                               for state, keys in unsettled.items())
            raise RuntimeError(
                f"the run ended with decode work unsettled ({detail}): every "
                f"window is final once the simulation is quiescent")


class StrategyServicesImpl:
    """The StrategyServices seam handed to DecodingStrategy hooks."""

    def __init__(self, engine, window_manager, pool):
        self._engine = engine
        self._runtime = window_manager
        self._pool = pool

    @property
    def now(self) -> int:
        return self._engine.now

    def make_strong_job(self, weak_job: DecodeJob, n_rounds: int,
                        label: str) -> DecodeJob:
        strong = self._runtime.make_strong_decode_job(weak_job, n_rounds, label)
        self._pool.check_strong_route(weak_job, strong)   # fail at build time
        return strong

    def defer_strong_escalation(self, weak_job: DecodeJob) -> DecoderRequestKey:
        """Faithful double window: the runtime submits the strong job once
        the far-side weak boundary is determined (arXiv:2510.25222 III C)."""
        return self._runtime.defer_strong_escalation(weak_job)

    def check_strong_route(self, weak_job: DecodeJob,
                           strong_job: DecodeJob) -> None:
        self._pool.check_strong_route(weak_job, strong_job)

    def cancel_strong(self, key: tuple) -> None:
        self._pool.cancel_strong(key)

    def prepare_strong_selection(self, weak_job: DecodeJob,
                                 strong_request_key: DecoderRequestKey,
                                 serial_strong_job: Optional[DecodeJob], *,
                                 deferred: bool) -> int:
        return self._runtime.prepare_strong_selection(
            weak_job, strong_request_key, serial_strong_job,
            deferred=deferred,
        )
