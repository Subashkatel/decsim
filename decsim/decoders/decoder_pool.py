"""The decoder pools: their units, the free ones, the unit a job is offered.

gem5's FUPool (src/cpu/o3/fu_pool.hh:64-75): the pool keeps the units and
knows which are free; the issue logic decides what runs. The router
names the algorithm each job runs (a Decoder row, ports.py): by code, or
by tier under switching (decoders.py).

A job that may start is offered a free unit with a free slot first.
Among those, a unit that already holds this job's rounds, or is
receiving them for another reader, comes first: there the job moves
nothing and starts at the landing it shares (the staging's rule,
decoder_memory_transfer.py). Where its rounds are nowhere yet, every
free unit starts the job at the same tick, so the choice decides only
whom the job delays, and the unit the fewest jobs already wait on is
taken: a count of jobs, not of their work. gem5's pool takes any unit
that is not busy (src/cpu/o3/fu_pool.cc:165-190) because its functional
units hold no staged input; a unit here does.

A job that cannot take compute now, because every unit computes or
because its window still owes a boundary, is staged with its input on
the unit with room that has the least work left: the rest of the running
decode and the declared cost of every job already staged there. A free
unit a staged job waits on is not an empty one, so two jobs that become
startable together are staged on two units, and the two forced-class
solves of one window overlap whenever two units have room. Least work
left is the task-assignment rule that sends each job to the server whose
outstanding work ends soonest. Where no cost is declared the work is
unbounded, and the unit holding the fewest live jobs is taken, then the
one that has the rounds.

The rule ranks compute only. When a parked job is released is not
known, so its decode is counted whole, which bounds the newcomer's
worst start and can cost it the rest of a running decode when the
parked job is released much later. What an input copy costs is the
link's to say, not the pool's, so a second unit is taken even where a
bounded input link makes the second copy slower than waiting for the
first unit would have been.

When the work is known and deterministic, as a declared decode cost is,
immediate dispatch of startable jobs by least work left starts every
job at the tick a central FIFO queue over the pool would, so staging
costs them nothing in start time and buys the input move. That equality
is checked here, not taken on authority:
validation/component_matrix/rowD2_access_execute/compare_overlap_laws.py
runs law_lwl_pool against the pool. The
textbook treatment is Harchol-Balter, Performance Modeling and Design
of Computer Systems, Cambridge 2013, which is not on disk under the
sandbox, so no chapter or page is claimed. A job with no input has
nothing to prefetch and waits in the queue for free compute.
"""

import dataclasses
import functools
import math
from typing import Callable, Optional

import decsim.decoders.decoder_memory as decoder_memory_module
import decsim.decoders.decoder_unit as decoder_unit_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.seeding as seeding
import decsim.trace_source as trace_source

DEFAULT_POOL = "default"

# whether a job's rounds are in that unit's memory or on their way there
InputIsOnTheUnit = Callable[
    [decoding_records.DecodeJob, decoder_unit_module.DecoderUnit], bool
]


class DecoderPool:
    """The units of every pool and the free ones, by pool name.

    Trace sources: unit_busy(unit) when a job takes a unit's compute out
    of the free list, unit_freed(unit) when the compute goes back, so a
    listener integrates the busy units without sampling the list.
    """

    def __init__(
        self,
        router,
        unit_pools: dict,
        decoder_memory: Optional[
            decoder_memory_module.DecoderMemoryConfig
        ] = None,
        blocks_unit_by_pool: Optional[dict] = None,
    ) -> None:
        _check_unit_pools(unit_pools)
        self.router = router
        # pool name -> whether a finished decode holds its unit until
        # the window side reads the result (<tier>.result_blocks_unit)
        self.blocks_unit_by_pool = blocks_unit_by_pool or {}
        self.units_by_pool: dict[str, list] = {}
        # the units whose compute is back in the pool, in return order
        self.free_by_pool: dict[str, list] = {}
        self.trace = _TraceSources()
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

    def blocks_unit(self, job: decoding_records.DecodeJob) -> bool:
        """Whether this job's tier holds its unit until the result is read."""
        return self.blocks_unit_by_pool.get(job.pool, False)

    def decoder_for(self, job: decoding_records.DecodeJob):
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
        job: decoding_records.DecodeJob,
        *,
        now: int,
        carries_input: bool,
        resident_capacity: int,
        memory_demand_of: Callable[[decoding_records.DecodeJob], int],
        input_is_on_the_unit: InputIsOnTheUnit,
    ) -> Optional[tuple]:
        """(unit, has free compute) for this job, or None.

        A job that may start takes a free unit with room: one that
        already has its rounds, else the one the fewest jobs are
        already waiting on. A job carrying input that finds none, or
        that may not start yet, is staged on the unit with room that
        has the least work left. now is the current tick; a unit's
        work left is counted from it.
        """
        takes_free_compute = decoder_unit_module.is_startable(job)
        if takes_free_compute or not carries_input:
            free = self.free_by_pool[pool]
            free_with_room = _with_room(
                free, job, resident_capacity, memory_demand_of
            )
            unit = _free_unit_for(free_with_room, job, input_is_on_the_unit)
            if unit is not None:
                return unit, True
        if not carries_input:
            return None
        units = self.units_by_pool[pool]
        with_room = _with_room(units, job, resident_capacity, memory_demand_of)
        if not with_room:
            return None
        staging_rank = functools.partial(
            self._staging_rank, job, now, input_is_on_the_unit
        )
        unit = min(with_room, key=staging_rank)
        has_free_compute = self.is_free(unit)
        return unit, has_free_compute

    def claim(
        self,
        unit: decoder_unit_module.DecoderUnit,
        job: decoding_records.DecodeJob,
    ) -> None:
        """The job takes the unit's compute out of the pool."""
        free = self.free_by_pool[unit.pool]
        free.remove(unit)
        unit.claim_compute(job)
        self.trace.unit_busy.fire(unit)

    def release(self, unit: decoder_unit_module.DecoderUnit) -> None:
        """The unit's compute goes back to its pool."""
        free = self.free_by_pool[unit.pool]
        free.append(unit)
        self.trace.unit_freed.fire(unit)

    def _staging_rank(
        self,
        job: decoding_records.DecodeJob,
        now: int,
        input_is_on_the_unit: InputIsOnTheUnit,
        unit: decoder_unit_module.DecoderUnit,
    ) -> tuple:
        """The order a job that must wait picks its unit by, smallest first.

        Least work left, so the job's worst wait is shortest; then the
        fewest live jobs; then a unit that already holds its rounds, so
        nothing is copied. The job stays on the unit it picks.
        """
        work_left = unit.work_left_ticks(now, self._occupancy_ticks)
        live_residents = unit.live_residents()
        moves_input = not input_is_on_the_unit(job, unit)
        return work_left, len(live_residents), moves_input

    def _occupancy_ticks(self, job: decoding_records.DecodeJob) -> float:
        """Ticks the job's decode holds a unit; unbounded when undeclared.

        The staging rank totals these to find a unit's work left.
        """
        decoder = self.decoder_for(job)
        occupancy = decoder.occupancy(job)
        if occupancy is None:
            return math.inf
        return occupancy


def routed_decoders(router) -> list:
    """Every decoder row the router can reach, in the order the walk finds.

    A row is anything that answers the runtime-checkable Decoder port,
    which every row of DECODERS does and a row written outside decsim
    does too without inheriting decsim's base class; the port declares
    the sources a row reports on, so the walk asks nothing further about
    what a row has. The recursion asks the seeding protocol whether a
    value names children of its own: the routers name the tiers and the
    per-code rows, and a row that wraps another (the confidence, staged
    and check wrappers) names its inner decoder the same way.
    """
    found = []
    seen = set()
    pending = [router]
    while pending:
        decoder = pending.pop()
        identity = id(decoder)
        if decoder is None or identity in seen:
            continue
        seen.add(identity)
        if isinstance(decoder, ports.Decoder):
            found.append(decoder)
        if not isinstance(decoder, seeding.RunSeedComposite):
            continue
        children = decoder.run_seed_children()
        for child in children:
            pending.append(child.child)
    return found


def _with_room(
    units: list,
    job: decoding_records.DecodeJob,
    resident_capacity: int,
    memory_demand_of: Callable[[decoding_records.DecodeJob], int],
) -> list:
    """The units of that list whose slots and memory hold this job."""
    with_room = []
    for unit in units:
        if unit.has_room(job, resident_capacity, memory_demand_of):
            with_room.append(unit)
    return with_room


def _unit_holding_input(
    units: list,
    job: decoding_records.DecodeJob,
    input_is_on_the_unit: InputIsOnTheUnit,
) -> Optional[decoder_unit_module.DecoderUnit]:
    """The first unit this job's rounds are on or on their way to, or None."""
    for unit in units:
        if input_is_on_the_unit(job, unit):
            return unit
    return None


def _free_unit_for(
    free_with_room: list,
    job: decoding_records.DecodeJob,
    input_is_on_the_unit: InputIsOnTheUnit,
) -> Optional[decoder_unit_module.DecoderUnit]:
    """The free unit a job that may start takes, or None when none has room.

    The unit that already holds the job's rounds comes first, so no
    input moves; else the unit the fewest jobs wait on.
    """
    if not free_with_room:
        return None
    unit_holding_input = _unit_holding_input(
        free_with_room, job, input_is_on_the_unit
    )
    if unit_holding_input is not None:
        return unit_holding_input
    return _fewest_awaiting_compute(free_with_room)


def _fewest_awaiting_compute(units: list) -> decoder_unit_module.DecoderUnit:
    chosen = units[0]
    fewest_count = chosen.residents_awaiting_compute_count()
    for unit in units[1:]:
        awaiting_count = unit.residents_awaiting_compute_count()
        if awaiting_count < fewest_count:
            fewest_count = awaiting_count
            chosen = unit
    return chosen


def _check_unit_pools(unit_pools: dict) -> None:
    """A pool map names at least one pool and gives every pool a unit.

    Which pools a manager holds is the assembly's: the chip's manager
    holds the default pool, the host's the strong one.
    """
    if not unit_pools:
        raise ValueError("unit_pools names no pool")
    for pool_name, units in unit_pools.items():
        if units < 1:
            raise ValueError(
                f"pool {pool_name!r} needs at least 1 unit (got {units})"
            )


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the decoder pool reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    unit_busy: trace_source.TraceSource = trace_source.new_source()
    unit_freed: trace_source.TraceSource = trace_source.new_source()
