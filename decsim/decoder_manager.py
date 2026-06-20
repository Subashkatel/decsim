"""Decoder queue and unit-pool runtime.

The manager receives ready jobs, runs them on decoder pools, and reports window
results back to the window manager. It does not own syndrome buffers or commits.
"""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

from .config import fmt
from .decoders import CodeRouter
from .links import LinkModel
from .message import DecodeJob, DecodeResult

if TYPE_CHECKING:
    from .engine import Engine
    from .protocols import Decoder, DecoderRouter, Scheduler
    from .switching import Switching


class DecoderManager:
    """Own decoder queues, decoder pools, routing, dispatch, and switching side jobs."""

    def __init__(self, engine: "Engine", decoder: "Decoder", scheduler: "Scheduler", *,
                 decoders: Optional[dict] = None,
                 router: Optional["DecoderRouter"] = None,
                 links: Optional[LinkModel] = None,
                 num_units: int = 1,
                 unit_pools: Optional[dict] = None,
                 switching: Optional["Switching"] = None,
                 on_window_decoded: Optional[Callable[[DecodeJob, DecodeResult], None]] = None,
                 log_name: str = "DecoderCluster"):
        self.engine = engine
        self.decoder = decoder
        self.decoders = dict(decoders) if decoders else {}
        self.router = router if router is not None \
            else CodeRouter(default=decoder, by_code=self.decoders)
        self.scheduler = scheduler
        self.links = links if links is not None else LinkModel()
        self.switching = switching
        self.on_window_decoded = on_window_decoded
        self.log_name = log_name

        if unit_pools is None:
            unit_pools = {"default": num_units}
        if "default" not in unit_pools:
            raise ValueError(f'unit_pools must include a "default" pool '
                             f'(got {sorted(unit_pools)})')
        for pool_name, units in unit_pools.items():
            if units < 1:
                raise ValueError(f"pool {pool_name!r} needs at least 1 unit (got {units})")

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
        self._running_strong_decodes: dict[tuple, DecodeJob] = {}

    @property
    def free_units(self) -> int:
        """Free units in the default pool."""
        return self.pool_free["default"]

    def pool_for(self, job: DecodeJob) -> str:
        """The unit pool a job runs on: its hint when that names a pool, else default."""
        return job.hint if job.hint in self.unit_totals else "default"

    def queue_for(self, pool: str) -> list:
        """A pool's ready queue."""
        return self.ready if pool == "default" else self.pool_ready[pool]

    def queued_total(self) -> int:
        """Jobs waiting across all pools."""
        return len(self.ready) + sum(len(q) for q in self.pool_ready.values())

    @staticmethod
    def pool_tag(pool: str) -> str:
        """Log prefix naming a non-default pool."""
        return "" if pool == "default" else f"{pool} "

    def decoder_for(self, job: DecodeJob) -> "Decoder":
        """Pick the decoder for a job via the configured router."""
        return self.router.route(job)

    def submit_window(self, job: DecodeJob) -> None:
        """Queue a ready operation-window decode job."""
        pool = self.pool_for(job)
        queue = self.queue_for(pool)
        self.scheduler.insert(queue, job)
        self.engine.log(self.log_name,
                        f"{job.label} READY -> enqueue "
                        f"({self.pool_tag(pool)}ready-queue length = {len(queue)})")
        self.queue_log.append((self.engine.now, self.queued_total()))
        if self.switching is not None and self.switching.run_both_at_once:
            self._start_strong_decode(job, in_parallel=True)
        self.try_dispatch()

    def submit_decode(self, round_count: int, on_done: Callable[[], None],
                      label: str = "external", deadline: Optional[int] = None,
                      code: Optional[str] = None,
                      spatial_nodes: Optional[int] = None,
                      hint: Optional[str] = None) -> None:
        """Submit a self-contained external decode job."""
        job = DecodeJob(op_id=-1, window_id=0, n_rounds=round_count,
                        ready_time=self.engine.now,
                        deadline=self.engine.now if deadline is None else deadline,
                        on_done=on_done, label=label, code=code,
                        spatial_nodes=spatial_nodes, hint=hint)
        self.scheduler.insert(self.queue_for(self.pool_for(job)), job)
        self.queue_log.append((self.engine.now, self.queued_total()))
        self.try_dispatch()

    def _start_strong_decode(self, job: DecodeJob, in_parallel: bool = False) -> None:
        """Queue a strong re-decode side job for a weak window decode."""
        if self.switching is None or job.window is None:
            return
        window_key = (job.op_id, job.window_id)
        round_count = self.switching.calculate_strong_redo_rounds(job.window)
        strong_label = getattr(job, "strong_label", f"strong({job.label})")

        def queue_strong_decode():
            strong = DecodeJob(op_id=-1, window_id=0, n_rounds=round_count,
                               ready_time=self.engine.now, deadline=self.engine.now,
                               on_done=lambda: None, label=strong_label, hint="strong",
                               spatial_nodes=job.spatial_nodes, strong_decode_for=window_key)
            self._running_strong_decodes[window_key] = strong
            self.scheduler.insert(self.queue_for(self.pool_for(strong)), strong)
            self.queue_log.append((self.engine.now, self.queued_total()))
            self.try_dispatch()

        if in_parallel:
            queue_strong_decode()
        else:
            self.engine.log(self.log_name,
                            f"weak decoder unsure about {job.label} -> hand "
                            f"{round_count} rounds to the strong decoder "
                            f"(after the weak->strong link)")
            self.engine.schedule(self.links.ws.cost(), queue_strong_decode,
                                 label=f"weak->strong handoff {strong_label}")

    def _cancel_strong_decode(self, key: tuple) -> None:
        """Cancel an unneeded strong re-decode if it is queued or running."""
        job = self._running_strong_decodes.pop(key, None)
        if job is None:
            return
        if job.pool is None:
            queue = self.queue_for(self.pool_for(job))
            if job in queue:
                queue.remove(job)
        else:
            job.cancelled = True
            self.pool_free[job.pool] += 1
            self.try_dispatch()
        self.strong_cancelled += 1

    def _merge_strong_batch(self, queue: list) -> DecodeJob:
        """Batch queued strong jobs when bulk strong decoding is enabled."""
        jobs = [self.scheduler.pop(queue) for _ in range(len(queue))]
        window_keys = [j.strong_decode_for for j in jobs
                       if j.strong_decode_for is not None]
        if len(jobs) == 1:
            jobs[0].merged_keys = window_keys
            return jobs[0]
        total = sum(j.n_rounds for j in jobs)
        batch = DecodeJob(op_id=-1, window_id=0, n_rounds=total,
                          ready_time=min(j.ready_time for j in jobs),
                          deadline=self.engine.now,
                          on_done=lambda: None,
                          label=f"strong-batch x{len(jobs)} ({total}r)",
                          hint="strong", spatial_nodes=jobs[0].spatial_nodes,
                          strong_decode_for=window_keys[0] if window_keys else None)
        batch.merged_keys = window_keys
        for window_key in window_keys:
            self._running_strong_decodes[window_key] = batch
        return batch

    def try_dispatch(self) -> None:
        """Dispatch ready jobs while their target pools have free decoder units."""
        bulk_strong = (
            self.switching is not None
            and getattr(self.switching, "bulk_strong", False)
        )
        for pool in self.unit_totals:
            self._dispatch_pool(pool, bulk_strong)

    def _dispatch_pool(self, pool: str, bulk_strong: bool) -> None:
        """Dispatch as many jobs as this pool can run now."""
        queue = self.queue_for(pool)
        while self.pool_free[pool] > 0 and queue:
            job = self._next_job(pool, queue, bulk_strong)
            self._start_job(pool, job)

    def _next_job(self, pool: str, queue: list, bulk_strong: bool) -> DecodeJob:
        """Pop the next job for a pool."""
        if bulk_strong and pool != "default":
            job = self._merge_strong_batch(queue)
            self.strong_running_rounds = job.n_rounds
            return job
        return self.scheduler.pop(queue)

    def _start_job(self, pool: str, job: DecodeJob) -> None:
        """Occupy one unit and schedule this job's completion event."""
        job.pool = pool
        self.pool_free[pool] -= 1
        if job.window is not None:
            job.window.t_dispatch = self.engine.now
        latency_ticks = self.decoder_for(job).latency(job)
        waited_ticks = self.engine.now - job.ready_time
        self.engine.log(self.log_name,
                        f"START DECODE {job.label} "
                        f"(waited {fmt(waited_ticks).strip()} in queue, "
                        f"{self.pool_tag(pool)}units free now {self.pool_free[pool]})")
        self.queue_log.append((self.engine.now, self.queued_total()))
        self.engine.schedule(latency_ticks, lambda j=job: self._on_decode_done(j),
                             label=f"decode_done({job.label})")

    def _on_decode_done(self, job: DecodeJob) -> None:
        """Finish a dispatched job and route the result to the owning layer."""
        if job.cancelled:
            return
        self._release_job_unit(job)
        self._finish_strong_bookkeeping(job)
        if self._finish_external_job(job):
            return

        result = self.decoder_for(job).decode(job)
        self._handle_switching_result(job, result)
        if self.on_window_decoded is None:
            raise RuntimeError("DecoderManager has no window completion callback")
        self.on_window_decoded(job, result)
        self.try_dispatch()

    def _release_job_unit(self, job: DecodeJob) -> None:
        """Return the decoder unit used by a finished job."""
        self.pool_free[job.pool] += 1

    def _finish_strong_bookkeeping(self, job: DecodeJob) -> None:
        """Remove finished strong re-decodes from the cancellation table."""
        if job.strong_decode_for is not None:
            for key in getattr(job, "merged_keys", None) or [job.strong_decode_for]:
                self._running_strong_decodes.pop(key, None)
            self.strong_running_rounds = max(0, self.strong_running_rounds - job.n_rounds)

    def _finish_external_job(self, job: DecodeJob) -> bool:
        """Run the external callback when this is not a window job."""
        if job.on_done is not None:
            self.engine.log(self.log_name,
                            f"DECODE DONE {job.label} "
                            f"({self.pool_tag(job.pool)}units free now "
                            f"{self.pool_free[job.pool]})")
            job.on_done()
            self.try_dispatch()
            return True
        return False

    def _handle_switching_result(self, job: DecodeJob, result: DecodeResult) -> None:
        """Apply the weak result to the switching policy, if one is configured."""
        if self.switching is not None and job.attempt == 0:
            window_key = (job.op_id, job.window_id)
            if self.switching.keep_weak_result(result):
                self._cancel_strong_decode(window_key)
            else:
                self.strong_needed += 1
                if not self.switching.run_both_at_once:
                    self._start_strong_decode(job)
