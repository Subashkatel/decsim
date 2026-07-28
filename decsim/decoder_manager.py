"""Decoder unit pools: queues, dispatch, strategy-hook completion pipeline.

Part module (port 12): faithful port of decoder_manager.py with the switching
branches replaced by the DecodingStrategy seam (Contract 2b/2c). The pool owns
what the core never sees: unit occupancy, per-pool queues, strong-job
bookkeeping (hold-or-deliver, cancellation), external jobs, bulk batching.

Parity anchors:
  - completion order (Contract 2b): cancelled/duplicate guard -> decode and
    validate identity -> free unit -> strong bookkeeping/strategy, or strategy
    directive (one transition: the demand is registered, the replacement the
    directive carries is enqueued, and awaiting is set, all BEFORE the commit
    callback, so no reader reached from that callback sees a destination that
    asked for a strong result and is not yet recorded as waiting) ->
    window_manager commit -> apply held early strong same tick -> try_dispatch.
    External jobs have no decoder result and free their unit before their
    callback.
  - only strong jobs re-stamp ready/deadline at enqueue (dm:139-140);
    weak jobs keep their DeadlinePolicy deadline.
  - cancel: queued -> remove silently (never dispatched); executing -> mark
    cancelled + free the unit immediately (dm:178-192); a held early strong
    result is discarded on keep-weak (dm:180).
  - weak and strong must route to distinct decoders (dm:169-176).
"""

from __future__ import annotations

from typing import Callable, Optional

from .message import DecodeJob, DecodeOutcome, DecodeResult
from .protocols import Directive
from .config import fmt


class DecoderManager:
    """Own decoder queues, pools, routing, dispatch, and completion pipeline."""

    def __init__(self, engine, *, router, scheduler,
                 unit_pools: Optional[dict] = None, num_units: int = 1,
                 ws_delay_ticks: int = 0, bulk_strong: bool = False,
                 lane_policy=None, log_name: str = "DecoderCluster"):
        self.engine = engine
        self.router = router
        self.scheduler = scheduler
        self.lane_policy = lane_policy
        self.ws_delay_ticks = ws_delay_ticks
        self.bulk_strong = bulk_strong
        self.log_name = log_name

        # Wired post-construction by the composition root:
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
        self.strong_running_rounds = 0
        # One destination window owns at most one unconsumed strong result at
        # a time: a live request, or a completion held for a demand that has
        # not registered yet. Whether a result still has a consumer is decided
        # when it completes, not when it is requested: between the two the
        # runtime submits the rest of the attempt's work and the strategy
        # returns its directive.
        self._running_strong_decodes: dict[tuple, DecodeJob] = {}
        self._windows_waiting_for_strong_result: set[tuple] = set()
        self._completed_strong_results: dict[tuple, DecodeResult] = {}
        # Destinations whose weak decode is admitted and has not yet produced
        # an outcome directive: while one is open the destination may still ask
        # for a strong result. One per destination, because every structure
        # above is keyed by destination alone.
        self._unresolved_weak_decodes: set[tuple] = set()

    # -------------------------------------------------------------- queues

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

    # ------------------------------------------------------------- enqueue

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
        """A DecodeJob is submitted once: it carries the unit it occupies and
        the cancellation handle its destination holds, so a second submission
        of the same object hands out both twice. Admitted covers the whole
        span a queue slot or a unit is held, from the moment enqueue accepts
        the job through crossing the weak->strong link, queueing and
        execution; cancelled and completed name the two ways that span ends.
        """
        if job.cancelled or job.completed or job.submitted:
            spent = ("cancelled" if job.cancelled else
                     "completed" if job.completed else "admitted")
            raise RuntimeError(
                f"decode job {job.label!r} for window "
                f"({job.op_id}, {job.window_id}) has already been {spent}: a "
                f"DecodeJob is submitted once, build a new one")

    def _admit_strong_request(self, job: DecodeJob) -> None:
        """Give one destination window's next strong result to this job.

        Admission decides only what it can decide now: a destination that
        still owns an unconsumed strong result, live or held, cannot hand its
        next result to a second request without clobbering the first's
        cancellation handle or replacing a held completion. Whether anything
        will consume the result is settled at completion instead, because
        between submission and completion the runtime enqueues the rest of the
        attempt's work and the strategy returns its directive. Admission runs
        at submission, ahead of the handoff log and of every mutation, so a
        refused job leaves no state and no trace behind, and a request still
        crossing the weak->strong link is cancellable.
        """
        key = job.strong_decode_for
        if (key in self._running_strong_decodes
                or key in self._completed_strong_results):
            raise RuntimeError(
                f"duplicate strong decode for window {key}: a destination "
                f"window has at most one unconsumed strong result")
        self._running_strong_decodes[key] = job

    def _admit_weak_decode(self, job: DecodeJob) -> None:
        """Open one destination window's decode attempt.

        Every strong structure is keyed by destination alone, so two of a
        destination's weak decodes open at once are indistinguishable: either
        decode's directive consumes whichever result the destination owns, and
        the other attempt commits awaiting a result nothing will deliver. One
        open decode per destination is what makes the destination key
        sufficient, so a second is refused here rather than left to strand a
        window. Admission runs ahead of every mutation, so a refused job leaves
        no state and no trace behind.
        """
        key = (job.op_id, job.window_id)
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
                or key in self._unresolved_weak_decodes)

    def _enqueue_now(self, job: DecodeJob) -> None:
        if job.strong_decode_for is not None:
            if self._running_strong_decodes.get(job.strong_decode_for) is not job:
                return                             # cancelled across the link
            job.ready_time = self.engine.now       # re-stamp strong only
            job.deadline = self.engine.now
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
        self._completed_strong_results.pop(key, None)
        job = self._running_strong_decodes.pop(key, None)
        if job is None:
            return
        if job.pool is None:
            # scan every queue rather than recomputing pool_for(job): a lane
            # policy may be reconfigured after this job was queued, and a
            # queued job keeps its enqueue-time placement
            for pool in self.unit_totals:
                queue = self.queue_for(pool)
                if job in queue:
                    queue.remove(job)
                    break
        else:
            merged = getattr(job, "merged_keys", None) or []
            survivors = [k for k in merged
                         if k in self._running_strong_decodes]
            if survivors:
                # a merged sibling still needs this running decode:
                # keep the batch, drop only this key from delivery
                # (cancelling the whole batch would silently lose the
                # siblings' results and hang their windows)
                job.merged_keys = survivors
            else:
                job.cancelled = True
                self.pool_free[job.pool] += 1
                self.strong_running_rounds = max(
                    0, self.strong_running_rounds - job.n_rounds)
                self.try_dispatch()
        self.strong_cancelled += 1

    # ------------------------------------------------------------- dispatch

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
            self.strong_running_rounds = job.n_rounds
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
        if len(jobs) == 1:
            jobs[0].merged_keys = window_keys
            return jobs[0]
        total = sum(j.n_rounds for j in jobs)
        batch = DecodeJob(op_id=-1, window_id=0, n_rounds=total,
                          ready_time=min(j.ready_time for j in jobs),
                          deadline=self.engine.now, on_done=lambda: None,
                          label=f"strong-batch x{len(jobs)} ({total}r)",
                          hint="strong", spatial_nodes=jobs[0].spatial_nodes,
                          strong_decode_for=window_keys[0] if window_keys
                          else None)
        batch.merged_keys = window_keys
        for window_key in window_keys:
            self._running_strong_decodes[window_key] = batch
        return batch

    def _start_job(self, pool: str, job: DecodeJob) -> None:
        job.pool = pool
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

    # ----------------------------------------------------------- completion

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
        if directive.directive is Directive.FINALIZE:
            self.cancel_strong(key)                # no-op unless one is live/held
        if awaiting:
            self.strong_needed += 1
            # applying the directive is one transition: the demand is
            # registered, then the replacement the directive carries is
            # enqueued, and no reader runs between the two
            self._windows_waiting_for_strong_result.add(key)
            if directive.extra is not None:        # serial redo, after ws
                self.enqueue(directive.extra.job, directive.extra.delay_ticks)
        job.awaiting_strong_result = awaiting      # BEFORE the commit callback
        if self.on_window_decoded is None:
            raise RuntimeError("DecoderManager has no window completion callback")
        self.on_window_decoded(job, result)
        if awaiting:
            self._apply_held_strong_result(key)    # early strong, same tick
        self.try_dispatch()

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
            for key in getattr(job, "merged_keys", None) or [job.strong_decode_for]:
                self._running_strong_decodes.pop(key, None)
            self.strong_running_rounds = max(
                0, self.strong_running_rounds - job.n_rounds)

    def _prepare_strong_result_deliveries(
        self, job: DecodeJob, result: DecodeResult,
    ) -> tuple:
        """Validate batch provenance and return per-window result deliveries."""
        keys = tuple(
            getattr(job, "merged_keys", None) or [job.strong_decode_for])
        result_identity = (result.op_id, result.window_id)
        is_merged_delivery = (
            len(keys) > 1
            or any(key != result_identity for key in keys)
        )
        if not is_merged_delivery:
            return ((keys[0], result),)

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

        return tuple(
            (key, DecodeResult(op_id=key[0], window_id=key[1]))
            for key in keys
        )

    def _handle_strong_decode_result(
        self, strong_result_deliveries: tuple,
    ) -> None:
        for key, result in strong_result_deliveries:
            self._complete_strong_result(key, result)

    def _complete_strong_result(self, key: tuple, result: DecodeResult) -> None:
        """Apply a strong result if the weak already committed, else hold it.

        A hold is the early-strong case: it lasts only until the destination's
        demand arrives, so it is legitimate only while that destination's weak
        decode is still open to raise one. A destination whose
        decode attempt has resolved without asking will never ask, and one
        whose next result already belongs to another live request would have
        this one replaced unseen: both refuse rather than park a value in a map
        nothing consumes.
        """
        if key in self._windows_waiting_for_strong_result:
            self._windows_waiting_for_strong_result.remove(key)
            if self.on_strong_window_decoded is None:
                raise RuntimeError(
                    "DecoderManager has no strong completion callback")
            self.on_strong_window_decoded(key, result)
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
        self._completed_strong_results[key] = result

    def _apply_held_strong_result(self, key: tuple) -> None:
        """Deliver a completion held from before this destination's demand."""
        if key in self._completed_strong_results:
            result = self._completed_strong_results.pop(key)
            self._complete_strong_result(key, result)

    def check_decode_work_settled(self) -> None:
        """Every window the run decoded reached a final result.

        Each state below is legitimate while the run continues, so the pool
        cannot judge one as it happens; at quiescence none is, because no
        decode is running and no queue holds work. A destination still recorded
        here never became final, and no metric or view reports any of these
        structures, so the run would otherwise return a logical accounting with
        that window's value silently missing (arXiv:2510.25222 Sec. III A Step
        4: an unconfident window's final estimate *is* the strong result).
        """
        unsettled = {
            state: sorted(keys) for state, keys in (
                ("waiting for a strong result",
                 self._windows_waiting_for_strong_result),
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

    def defer_strong_escalation(self, weak_job: DecodeJob, n_rounds: int,
                                label: str) -> None:
        """Faithful double window: the runtime submits the strong job once
        the far-side weak boundary is determined (arXiv:2510.25222 III C)."""
        self._runtime.defer_strong_escalation(weak_job, n_rounds, label)

    def check_strong_route(self, weak_job: DecodeJob,
                           strong_job: DecodeJob) -> None:
        self._pool.check_strong_route(weak_job, strong_job)

    def cancel_strong(self, key: tuple) -> None:
        self._pool.cancel_strong(key)

    def ws_delay(self) -> int:
        return self._pool.ws_delay_ticks
