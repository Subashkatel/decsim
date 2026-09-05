"""Dispatch: startable jobs first, in scheduler order, onto an offered unit.

gem5 O3 issues from its ready set oldest-first and non-ready work never
displaces ready work (src/cpu/o3/inst_queue.hh scheduleReadyInsts): the
scan passes over jobs with no eligible unit, and a boundary-blocked job
is placed only when no startable job can be. The loop is not reentrant:
an admission continuation or a compute return that happens inside it
returns immediately and the outer loop keeps running while the queue
has work, so no event boundary leaves a pool holding both free compute
and a waiting request that fits, and no drain needs its own engine
event.
"""

from typing import Optional

import decsim.decoders.decode_queue as decode_queue_module
import decsim.decoders.decode_service as decode_service_module
import decsim.decoders.decoder_pool as decoder_pool_module
import decsim.decoders.decoder_unit as decoder_unit_module
import decsim.message as message


class DecodeDispatcher:
    """Places waiting jobs on units, one pass over every pool."""

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
        """Dispatch every pool; a call from inside the loop returns at once."""
        if self.is_dispatching:
            return
        self.is_dispatching = True
        try:
            for pool in self.queue.waiting_by_pool:
                self._dispatch_pool(pool)
        finally:
            self.is_dispatching = False

    def _dispatch_pool(self, pool: str) -> None:
        while self.queue.has_jobs(pool):
            ordered = self.queue.drain_in_scheduler_order(pool)
            selection = self._select_placement(pool, ordered)
            if selection is None:
                self.queue.restore(pool, ordered)
                return
            index, job, unit, claim_compute = selection
            del ordered[index]
            self.queue.restore(pool, ordered)
            self.queue.sample_depth()
            self.service.dispatch_to(pool, job, unit, claim_compute)

    def _select_placement(self, pool: str, ordered: list) -> Optional[tuple]:
        """(index, job, unit, claim_compute) of the first placeable job.

        Startable jobs are tried before boundary-blocked ones.
        """
        startable = self._first_placement(pool, ordered, True)
        if startable is not None:
            return startable
        return self._first_placement(pool, ordered, False)

    def _first_placement(
        self, pool: str, ordered: list, startable: bool
    ) -> Optional[tuple]:
        for index, job in enumerate(ordered):
            is_startable = decoder_unit_module.is_startable(job)
            if is_startable is not startable:
                continue
            placement = self._eligible_unit(pool, job)
            if placement is None:
                continue
            unit, claim_compute = placement
            return index, job, unit, claim_compute
        return None

    def _eligible_unit(
        self, pool: str, job: message.DecodeJob
    ) -> Optional[tuple]:
        """(unit, claim_compute) for this job, or None.

        A startable job takes any unit with a free input slot and claims
        compute when that unit's compute is free. A boundary-blocked job
        takes an input slot only (its DMA overlaps other work, Tomasulo's
        reservation station), and only once its gate
        (WindowInputGate.may_stage) says its release is already
        resolving, so parked work can never squat a slot against the
        decode that must free it. The pool offers a free unit first, else
        the busy unit with room that frees earliest (decoder_pool.py).
        """
        startable = decoder_unit_module.is_startable(job)
        if not startable and _is_staging_refused(job):
            return None
        resident_capacity = self.service.resident_capacity(job)
        carries_input = self.service.carries_input(job)
        placement = self.pool.offer(
            pool,
            job,
            carries_input=carries_input,
            resident_capacity=resident_capacity,
            memory_demand_of=self.service.memory_demand,
        )
        if placement is None:
            return None
        unit, has_free_compute = placement
        claim_compute = has_free_compute and startable
        return unit, claim_compute


def _is_staging_refused(job: message.DecodeJob) -> bool:
    if job.gate is None:
        return False
    return not job.gate.may_stage(job)
