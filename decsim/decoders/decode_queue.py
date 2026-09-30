"""The jobs waiting for a unit of the manager's pool, in scheduler order.

gem5's instruction queue holds ready work in priority order and hands
it out through one scheduling rule (src/cpu/o3/inst_queue.hh:160-178,
scheduleReadyInsts); here the Scheduler port (schedulers.py) is that
rule. Under the yaml's decoder_manager.bulk_strong the STRONG POOL alone
is served as one merged batch: every queued strong job, timing-only,
becomes one decode serving every member request. Toshio et al. 2510.25222
lines 1253-1264 ask for bulk decoding of one escalation's own
contiguous region once both its boundaries are determined; merging
several independent escalated windows is decsim's own step past that,
and it exists because it is the cheapest strong tier the study can
price a run against, the floor of the tuning the paper explicitly
leaves open at 1259-1262. The queue depth is sampled on every change
for the switching study.
"""

import dataclasses
from typing import TYPE_CHECKING, Optional

import decsim.decoders.strong_requests as strong_requests_module
import decsim.records.decoding as decoding_records
import decsim.trace_source as trace_source

if TYPE_CHECKING:
    import decsim.decoders.decoder_manager as decoder_manager_module

# The names the build gives the two pools (decoder_pool.PoolSettings).
DEFAULT_POOL = "default"
STRONG_POOL = "strong"


class WaitingJobs:
    """The pool's ready queue, and the depth reported at every change.

    Trace sources: job_enqueued(job) as a job joins the queue, the job
    naming the rounds it references in the store; job_withdrawn(job) as
    one leaves it unserved; depth_changed(tick, depth) whenever the jobs
    waiting change.
    """

    def __init__(
        self,
        engine,
        scheduler,
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
        return True

    def sample_depth(self) -> None:
        """Report the depth now."""
        now = self.engine.now
        depth = len(self.waiting)
        self.trace.depth_changed.fire(now, depth)

    def drain_in_scheduler_order(self) -> list:
        """Empty the queue into a list, next job first."""
        ordered = []
        while self.waiting:
            job = self.next()
            ordered.append(job)
        return ordered

    def restore(self, ordered: list) -> None:
        """Put drained jobs back at the queue's end, in the given order."""
        self.waiting.extend(ordered)

    def next(self) -> decoding_records.DecodeJob:
        """Remove and return the next job by the scheduler's rule."""
        if self.merges_strong:
            return self._merge_strong_batch(self.waiting)
        return self.scheduler.pop(self.waiting)

    def _merge_strong_batch(self, queue: list) -> decoding_records.DecodeJob:
        """Batch every queued strong job (timing-only) into one decode.

        A batch that found no unit last time waits in the queue like any
        job; it is opened back into its member requests here so the new
        batch serves every request exactly once.
        """
        jobs = self._open_queued_strong_jobs(queue)
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
    """Every event the waiting jobs reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (gem5 src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    job_enqueued: trace_source.TraceSource = trace_source.new_source()
    job_withdrawn: trace_source.TraceSource = trace_source.new_source()
    depth_changed: trace_source.TraceSource = trace_source.new_source()
