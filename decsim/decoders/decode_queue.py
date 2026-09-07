"""The jobs waiting for a decoder unit, per pool, in scheduler order.

gem5's instruction queue holds ready work in priority order and hands
it out through one scheduling rule (src/cpu/o3/inst_queue.hh:160-178,
scheduleReadyInsts); here the Scheduler port (schedulers.py) is that
rule, per pool. A job queues in the pool its hint names, else the
default pool. Under bulk_strong the strong pool's queue is served as one
merged batch: every queued strong job, timing-only, becomes one decode
serving every member request. The queue depth is sampled on every
change for the switching study.
"""

from typing import Optional

import decsim.decoders.strong_requests as strong_requests_module
import decsim.observe.trace_source as trace_source
import decsim.records.decoding as decoding_records

# The decoder manager component's name in the narrator (docs/
# architecture.md's component table). It lives here because every part of
# the manager imports this module, and the manager's facade imports them.
LOG_SOURCE = "Decoder manager"
DEFAULT_POOL = "default"


class WaitingJobs:
    """One ready queue per pool, and the depth reported at every change.

    Trace sources: job_enqueued(job) as a job joins its queue, the job
    naming the rounds it references in the store; depth_changed(tick,
    depth) whenever the jobs waiting over every pool change.
    """

    def __init__(
        self,
        engine,
        scheduler,
        pools,
        strong_requests: strong_requests_module.StrongRequests,
        is_bulk_strong: bool = False,
    ) -> None:
        self.engine = engine
        self.scheduler = scheduler
        self.waiting_by_pool: dict[str, list] = {}
        for pool in pools:
            self.waiting_by_pool[pool] = []
        self.strong_requests = strong_requests
        self.is_bulk_strong = is_bulk_strong
        self.job_enqueued = trace_source.TraceSource()
        self.depth_changed = trace_source.TraceSource()

    def pool_of(self, job: decoding_records.DecodeJob) -> str:
        """The pool a job queues in: its hint when a pool has that name."""
        if job.hint in self.waiting_by_pool:
            return job.hint
        return DEFAULT_POOL

    def add(self, job: decoding_records.DecodeJob) -> None:
        """Put one admitted job at the back of its pool's queue, logged."""
        job.ready_time = self.engine.now
        pool = self.pool_of(job)
        queue = self.waiting_by_pool[pool]
        queue.append(job)
        pool_tag = pool_tag_of(pool)
        queue_length = len(queue)
        self.engine.log(
            LOG_SOURCE,
            f"{job.label} READY -> enqueue "
            f"({pool_tag}ready-queue length = {queue_length})",
        )
        self.job_enqueued.fire(job)
        self.sample_depth()

    def add_quietly(self, job: decoding_records.DecodeJob) -> None:
        """Put a job carrying no input in its queue; no READY line."""
        pool = self.pool_of(job)
        queue = self.waiting_by_pool[pool]
        queue.append(job)
        self.job_enqueued.fire(job)
        self.sample_depth()

    def remove(self, job: decoding_records.DecodeJob) -> bool:
        """Take the job out of whichever queue holds it; whether one did."""
        for queue in self.waiting_by_pool.values():
            if job in queue:
                queue.remove(job)
                return True
        return False

    def has_jobs(self, pool: str) -> bool:
        """Whether the pool's queue holds anything."""
        queue = self.waiting_by_pool[pool]
        return bool(queue)

    def jobs(self) -> list:
        """Every waiting job, pool by pool, in queue order."""
        every = []
        for queue in self.waiting_by_pool.values():
            every.extend(queue)
        return every

    def total(self) -> int:
        """Jobs waiting over every pool."""
        total = 0
        for queue in self.waiting_by_pool.values():
            total += len(queue)
        return total

    def sample_depth(self) -> None:
        """Report the total depth now."""
        depth = self.total()
        self.depth_changed.fire(self.engine.now, depth)

    def drain_in_scheduler_order(self, pool: str) -> list:
        """Empty the pool's queue into a list, next job first."""
        queue = self.waiting_by_pool[pool]
        ordered = []
        while queue:
            job = self.next(pool)
            ordered.append(job)
        return ordered

    def restore(self, pool: str, ordered: list) -> None:
        """Put drained jobs back at the queue's end, in the given order."""
        queue = self.waiting_by_pool[pool]
        queue.extend(ordered)

    def next(self, pool: str) -> decoding_records.DecodeJob:
        """Remove and return the pool's next job by the scheduler's rule."""
        queue = self.waiting_by_pool[pool]
        if self.is_bulk_strong and pool != DEFAULT_POOL:
            return self._merge_strong_batch(queue)
        return self.scheduler.pop(queue)

    def _merge_strong_batch(self, queue: list) -> decoding_records.DecodeJob:
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
        self.strong_requests.register_batch(window_keys, jobs, batch)
        return batch

    def _open_queued_strong_jobs(self, queue: list) -> list:
        jobs = []
        for _ in range(len(queue)):
            queued = self.scheduler.pop(queue)
            if strong_requests_module.is_merged_batch(queued):
                members = self.strong_requests.members_of(queued)
                jobs.extend(members)
            else:
                jobs.append(queued)
        return jobs


def pool_tag_of(pool: str) -> str:
    """The pool's name as a log prefix; the default pool has none."""
    if pool == DEFAULT_POOL:
        return ""
    return f"{pool} "


def _refuse_bits_in_bulk_strong(job: decoding_records.DecodeJob) -> None:
    """bulk_strong merges timing-only strong re-decodes; bits would be lost."""
    has_model = job.detector_error_model is not None
    has_bits = False
    for payload in job.payloads:
        if payload.bits is not None:
            has_bits = True
    if has_model or has_bits:
        raise RuntimeError(
            "bulk_strong only merges timing-only strong re-decodes; "
            "disable it for accuracy-coupled switching."
        )


def _batch_job(jobs: list, window_keys: list) -> decoding_records.DecodeJob:
    """One timing-only decode serving every member request.

    The batch is decoded like any strong job: its timing-only result is
    split into one empty completion per member request at decode end.
    """
    total_rounds = 0
    earliest_ready_time = jobs[0].ready_time
    for job in jobs:
        total_rounds += job.round_count
        earliest_ready_time = min(earliest_ready_time, job.ready_time)
    first_window_key: Optional[tuple] = None
    if window_keys:
        first_window_key = window_keys[0]
    batch_size = len(jobs)
    return decoding_records.DecodeJob(
        operation_id=-1,
        window_id=0,
        round_count=total_rounds,
        ready_time=earliest_ready_time,
        label=f"strong-batch x{batch_size} ({total_rounds}r)",
        hint="strong",
        spatial_nodes=jobs[0].spatial_nodes,
        strong_decode_for=first_window_key,
    )
