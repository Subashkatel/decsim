"""The jobs waiting for a unit of the manager's pool, in scheduler order.

As gem5's instruction queue hands out ready work through one scheduling
rule (src/cpu/o3/inst_queue.hh:160-178, scheduleReadyInsts), the
scheduler's pop is that rule here (schedulers.py). Under
DecoderManagerSettings.bulk_strong the strong pool alone is served as
one merged batch: every queued strong job, timing-only, becomes one
decode serving every member request. Toshio et al. 2510.25222 lines
1253-1264 ask for bulk decoding of one escalation's own region; merging
several escalated windows goes past that, as the cheapest strong tier a
run can be priced against, the floor of the tuning the paper leaves open
(lines 1259-1262).
"""

import dataclasses
from typing import TYPE_CHECKING, Optional

import decsim.decoders.decoder_unit as decoder_unit_module
import decsim.decoders.schedulers as schedulers
import decsim.decoders.strong_requests as strong_requests_module
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.trace_source as trace_source

if TYPE_CHECKING:
    import decsim.decoders.decoder_manager as decoder_manager_module

# The names the build gives the two pools (decoder_pool.PoolSettings).
DEFAULT_POOL = "default"
STRONG_POOL = "strong"


class WaitingJobs:
    """The pool's ready queue.

    Trace sources: job_enqueued(job) as a job joins the queue;
    job_withdrawn(job) as one leaves it unserved; depth_changed(tick,
    depth) whenever the jobs waiting change.
    """

    def __init__(
        self,
        engine: engine_module.Engine,
        scheduler: schedulers.Scheduler,
        manager: "decoder_manager_module.DecoderManager",
        merges_strong: bool = False,
    ) -> None:
        self.engine = engine
        self.scheduler = scheduler
        self.waiting: list = []
        self.manager = manager
        # decoder_manager.bulk_strong on the strong pool's queue
        self.merges_strong = merges_strong
        self.trace = _TraceSources()

    def add(self, job: decoding_records.DecodeJob) -> None:
        """Put one admitted job at the back of the queue."""
        job.ready_time = self.engine.now
        self.waiting.append(job)
        self.trace.job_enqueued.fire(job)
        self.sample_depth()

    def remove(self, job: decoding_records.DecodeJob) -> bool:
        """Take the job out of the queue; whether it was there."""
        if job not in self.waiting:
            return False
        self.waiting.remove(job)
        self.trace.job_withdrawn.fire(job)
        self.sample_depth()
        return True

    def sample_depth(self) -> None:
        """Report the depth now."""
        now = self.engine.now
        depth = len(self.waiting)
        self.trace.depth_changed.fire(now, depth)

    def drain_in_scheduler_order(self) -> list:
        """Empty the queue into a list, next job first."""
        if self.merges_strong:
            return self._merged_strong_jobs()
        ordered = []
        while self.waiting:
            job = self.scheduler.pop(self.waiting)
            ordered.append(job)
        return ordered

    def restore(self, ordered: list) -> None:
        """Put drained jobs back at the queue's end, in the given order."""
        self.waiting.extend(ordered)

    def _merged_strong_jobs(self) -> list:
        """The startable strong jobs as one batch, then each blocked job.

        A job whose window owes a boundary stays out of the batch, so a
        batch takes a unit only when every member may start, and the
        blocked job waits in the queue on its own.
        """
        jobs = self._open_queued_strong_jobs(self.waiting)
        startable = []
        blocked = []
        for job in jobs:
            if decoder_unit_module.is_startable(job):
                startable.append(job)
            else:
                blocked.append(job)
        if not startable:
            return blocked
        batch = self._merge_strong_batch(startable)
        return [batch, *blocked]

    def _merge_strong_batch(self, jobs: list) -> decoding_records.DecodeJob:
        """Batch these strong jobs (timing-only) into one decode.

        A batch that found no unit last time is opened back into its
        member requests before this, so the new batch serves every
        request once.
        """
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
        self.manager.strong_requests.register_batch(window_keys, jobs, batch)
        return batch

    def _open_queued_strong_jobs(self, queue: list) -> list:
        jobs = []
        for _ in range(len(queue)):
            queued = self.scheduler.pop(queue)
            if strong_requests_module.is_merged_batch(queued):
                members = self.manager.strong_requests.members_of(queued)
                jobs.extend(members)
            else:
                jobs.append(queued)
        return jobs


def pool_tag_of(pool: str) -> str:
    """The pool's name as a log prefix; the default pool has none."""
    if pool == DEFAULT_POOL:
        return ""
    return f"{pool} "


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
        kind=decoding_records.DecodeJobKind.STRONG_BATCH,
        spatial_nodes=jobs[0].spatial_nodes,
        strong_decode_for=first_window_key,
    )


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the waiting jobs reports, as one member."""

    job_enqueued: trace_source.TraceSource = trace_source.new_source()
    job_withdrawn: trace_source.TraceSource = trace_source.new_source()
    depth_changed: trace_source.TraceSource = trace_source.new_source()
