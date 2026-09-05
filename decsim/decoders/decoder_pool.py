"""The decoder pools: their units, the free ones, the unit a job is offered.

gem5's FUPool (src/cpu/o3/fu_pool.hh:64-75): the pool keeps the units and
knows which are free; the issue logic decides what runs. The router
names the algorithm each job runs (a Decoder row, ports.py): by code, or
by tier under switching (decoders.py). A job is offered a free unit
with a free slot first. When every unit computes, a
job with input to move is staged on the busy unit with room whose
compute frees earliest: least work left (Harchol-Balter, Performance
Modeling and Design of Computer Systems, 2013, Ch. 24; with known
deterministic work, immediate dispatch by least work left starts every
job when a central FIFO queue over the pool would; rowD2). A job with no
input has nothing to prefetch and waits in the queue for free compute.
"""

from typing import Callable, Optional

import decsim.decoders.decoder_memory as decoder_memory_module
import decsim.decoders.decoder_unit as decoder_unit_module
import decsim.message as message

DEFAULT_POOL = "default"


class DecoderPool:
    """The units of every pool and the free ones, by pool name."""

    def __init__(
        self,
        router,
        unit_pools: dict,
        decoder_memory: Optional[
            decoder_memory_module.DecoderMemoryConfig
        ] = None,
    ) -> None:
        _check_unit_pools(unit_pools)
        self.router = router
        self.units_by_pool: dict[str, list] = {}
        # the units whose compute is back in the pool, in return order
        self.free_by_pool: dict[str, list] = {}
        for pool, unit_count in unit_pools.items():
            capacity = None
            if decoder_memory is not None:
                capacity = decoder_memory.capacity_for(pool)
            units = []
            for index in range(unit_count):
                memory = decoder_memory_module.DecoderMemory(
                    pool, index, capacity
                )
                unit = decoder_unit_module.DecoderUnit(pool, index, memory)
                units.append(unit)
            self.units_by_pool[pool] = units
            self.free_by_pool[pool] = list(units)

    def decoder_for(self, job: message.DecodeJob):
        """The decoder the job runs on, by the router's rule."""
        return self.router.route(job)

    def units(self) -> list:
        """Every unit of every pool, pool by pool."""
        every = []
        for units in self.units_by_pool.values():
            every.extend(units)
        return every

    def free_count(self, pool: str) -> int:
        """Units of the pool with free compute."""
        free = self.free_by_pool[pool]
        return len(free)

    def is_free(self, unit: decoder_unit_module.DecoderUnit) -> bool:
        """Whether the unit's compute is back in its pool."""
        free = self.free_by_pool[unit.pool]
        return unit in free

    def offer(
        self,
        pool: str,
        job: message.DecodeJob,
        *,
        carries_input: bool,
        resident_capacity: int,
        memory_demand_of: Callable[[message.DecodeJob], int],
    ) -> Optional[tuple]:
        """(unit, has free compute) for this job, or None.

        A free unit with room comes first. Otherwise a job carrying
        input is staged on the busy unit with room that frees earliest.
        """
        for unit in self.free_by_pool[pool]:
            if unit.has_room(job, resident_capacity, memory_demand_of):
                return unit, True
        if not carries_input:
            return None
        busy_with_room = []
        for unit in self.units_by_pool[pool]:
            if self.is_free(unit):
                continue
            if unit.has_room(job, resident_capacity, memory_demand_of):
                busy_with_room.append(unit)
        if not busy_with_room:
            return None
        unit = _earliest_freeing(busy_with_room)
        return unit, False

    def claim(
        self, unit: decoder_unit_module.DecoderUnit, job: message.DecodeJob
    ) -> None:
        """The job takes the unit's compute out of the pool."""
        free = self.free_by_pool[unit.pool]
        free.remove(unit)
        unit.claim_compute(job)

    def release(self, unit: decoder_unit_module.DecoderUnit) -> None:
        """The unit's compute goes back to its pool."""
        free = self.free_by_pool[unit.pool]
        free.append(unit)


def _earliest_freeing(units: list) -> decoder_unit_module.DecoderUnit:
    """The unit whose compute frees first: least work left."""
    earliest_unit = units[0]
    earliest_free = earliest_unit.compute.expected_free_ticks
    for unit in units[1:]:
        free_ticks = unit.compute.expected_free_ticks
        if free_ticks < earliest_free:
            earliest_free = free_ticks
            earliest_unit = unit
    return earliest_unit


def _check_unit_pools(unit_pools: dict) -> None:
    """A pool map names the default pool and gives every pool a unit."""
    if DEFAULT_POOL not in unit_pools:
        pools = sorted(unit_pools)
        raise ValueError(
            f'unit_pools must include a "default" pool (got {pools})'
        )
    for pool_name, units in unit_pools.items():
        if units < 1:
            raise ValueError(
                f"pool {pool_name!r} needs at least 1 unit (got {units})"
            )
