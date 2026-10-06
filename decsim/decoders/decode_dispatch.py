"""Dispatch: jobs that may start, in scheduler order, onto an offered unit.

A job takes a unit only when its window owes no boundary. A blocked job
stays in the queue with no slot and no input copy, and when its boundary
arrives the next dispatch puts it on any free unit. gem5 O3 picks a
functional unit at issue, for ready instructions only
(src/cpu/o3/inst_queue.cc:920, fu_pool.cc:165-190), and CUDA-Q's host
dispatcher picks an idle worker when data is present, then copies the
input to it (host_side_dispatcher_design.md lines 36-42); a job bound
early waits for its one unit while another is idle (lines 22 and 118).
The scan passes over jobs with no eligible unit. The loop is not
reentrant: a call from inside it returns at once and the outer loop runs
while the queue has work, so no event leaves a pool holding both free
compute and a waiting request that fits.
"""

from typing import Optional

import decsim.decoders.decode_queue as decode_queue_module
import decsim.decoders.decode_service as decode_service_module
import decsim.decoders.decoder_pool as decoder_pool_module
import decsim.decoders.decoder_unit as decoder_unit_module
import decsim.records.decoding as decoding_records


class DecodeDispatcher:
    """Places waiting jobs on units of the manager's pool."""

    def __init__(
        self,
        queue: decode_queue_module.WaitingJobs,
        pool: decoder_pool_module.DecoderPool,
        service: decode_service_module.DecodeService,
    ) -> None:
        self.queue = queue
        self.pool = pool
        self.service = service
        self.is_dispatching = False

    def run(self) -> None:
        """Dispatch the queue; a call from inside the loop returns at once."""
        if self.is_dispatching:
            return
        self.is_dispatching = True
        try:
            self._dispatch()
        finally:
            self.is_dispatching = False

    def _dispatch(self) -> None:
        while self.queue.waiting:
            ordered = self.queue.drain_in_scheduler_order()
            selection = self._first_placement(ordered)
            if selection is None:
                self.queue.restore(ordered)
                return
            index, job, unit, claim_compute = selection
            del ordered[index]
            self.queue.restore(ordered)
            self.queue.sample_depth()
            self.service.dispatch_to(job, unit, claim_compute)

    def _first_placement(self, ordered: list) -> Optional[tuple]:
        """(index, job, unit, claim_compute) of the first placeable job.

        A job whose window owes a boundary is passed over.
        """
        for index, job in enumerate(ordered):
            if not decoder_unit_module.is_startable(job):
                continue
            placement = self._eligible_unit(job)
            if placement is None:
                continue
            unit, claim_compute = placement
            return index, job, unit, claim_compute
        return None

    def _eligible_unit(
        self, job: decoding_records.DecodeJob
    ) -> Optional[tuple]:
        """(unit, claim_compute) for this job, or None.

        The job takes the unit the pool offers (decoder_pool.py) and
        claims its compute when that compute is free.
        """
        carries_input = self.service.carries_input(job)
        return self.pool.offer(
            job,
            now=self.service.engine.now,
            carries_input=carries_input,
            resident_capacity=decoder_unit_module.INPUT_SLOT_COUNT,
            memory_demand_of=self.service.memory_demand,
            input_is_on_the_unit=self.service.input_is_on_the_unit,
        )
