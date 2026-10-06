"""A manager's pool: its units, the free ones, the unit a job is offered.

As gem5's FUPool (src/cpu/o3/fu_pool.hh:64-75), the pool keeps the units
and knows which are free; the issue logic decides what runs. The chip's
manager holds the units of the tier that decodes the plan's windows, the
host's those of a switching run's strong tier, and every job of a pool
runs the one algorithm of its manager's decoder port.

A job reaches the pool only when it may start: its window owes no
boundary (decode_dispatch.py). It takes a free unit with a free slot:
first one that already holds or is receiving its rounds, where it moves
nothing (decoder_memory_transfer.py), else the one longest free.
gem5's pool takes any unit that is not busy (src/cpu/o3/fu_pool.cc
165-190) because its units hold no staged input; a unit here does.

When every unit computes, the job is staged with its input on a unit
with room among those with the pool's least work left: the rest of the
running decode and the declared cost of every job staged there. So two
jobs that become startable together go to two units, and the two
forced-class solves of one window overlap whenever two units have room.
The job waits in the queue while every such unit is full. With no
declared cost the work is unbounded, and the unit holding the fewest
live jobs is taken, then the one that has the rounds.

The rule ranks compute only; an input copy's cost is the link's, not
the pool's. With known, deterministic costs, least-work-left dispatch
starts every job at the tick a central FIFO queue over the pool would
(Kiefer and Wolfowitz 1955, c identical servers; Harchol-Balter,
Performance Modeling and Design of Computer Systems, 2013), so staging
costs no start time and buys the input move.
"""

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import decsim.decoders.decoder_memory as decoder_memory_module
import decsim.decoders.decoder_unit as decoder_unit_module
import decsim.decoders.detection_events as detection_events_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.seeding as seeding
import decsim.trace_source as trace_source

if TYPE_CHECKING:
    import decsim.decoders.decoder_manager as decoder_manager_module

# whether a job's rounds are in that unit's memory or on their way there
InputIsOnTheUnit = Callable[
    [decoding_records.DecodeJob, decoder_unit_module.DecoderUnit], bool
]


@dataclasses.dataclass(frozen=True)
class PoolSettings:
    """One pool as the root derives it from its tier's settings.

    As a gem5 FUPool takes its units as one parameter
    (src/cpu/o3/FUPool.py:49). name is "default" for the tier that
    decodes the plan's windows and "strong" for a switching run's strong
    tier; unit_count is <tier>.units; capacity_bits is one unit's memory
    (None unbounded); copies_input is <tier>.input; blocks_unit is
    result_blocks_unit on the default pool and false on the strong one;
    formation is the tier's event-detection logic when
    detection_events.formed_at seats it at the tier's decoder, else
    None.
    """

    name: str
    unit_count: int
    capacity_bits: Optional[int] = None
    copies_input: bool = True
    blocks_unit: bool = False
    formation: Optional[detection_events_module.TierFormation] = None


class DecoderPool:
    """The pool of one tier's decoder units.

    It tracks which are free.

    Trace sources: unit_busy(unit) when a job takes a unit's compute,
    unit_freed(unit) when it goes back.
    """

    def __init__(
        self,
        manager: "decoder_manager_module.DecoderManager",
        settings: PoolSettings,
    ) -> None:
        self.manager = manager
        self.name = settings.name
        self.blocks_unit = settings.blocks_unit
        self.units = []
        for index in range(settings.unit_count):
            memory = decoder_memory_module.DecoderMemory(
                settings.name, index, settings.capacity_bits
            )
            unit = decoder_unit_module.DecoderUnit(settings.name, index, memory)
            self.units.append(unit)
        # the units whose compute is back in the pool, in return order
        self.free = list(self.units)
        self.trace = _TraceSources()

    def offer(
        self,
        job: decoding_records.DecodeJob,
        *,
        now: int,
        carries_input: bool,
        resident_capacity: int,
        memory_demand_of: Callable[[decoding_records.DecodeJob], Optional[int]],
        input_is_on_the_unit: InputIsOnTheUnit,
    ) -> Optional[tuple]:
        """(unit, has free compute) for this job, which may start, or None.

        now is the current tick; a unit's work left is counted from it.
        The ranking is the module's.
        """
        free_with_room = _with_room(
            self.free, job, resident_capacity, memory_demand_of
        )
        unit = _free_unit_for(free_with_room, job, input_is_on_the_unit)
        if unit is not None:
            return unit, True
        if not carries_input:
            return None
        return self._staging_offer(
            job,
            now,
            resident_capacity,
            memory_demand_of,
            input_is_on_the_unit,
        )

    def claim(
        self,
        unit: decoder_unit_module.DecoderUnit,
        job: decoding_records.DecodeJob,
    ) -> None:
        """The job takes the unit's compute out of the pool."""
        self.free.remove(unit)
        unit.claim_compute(job)
        self.trace.unit_busy.fire(unit)

    def release(self, unit: decoder_unit_module.DecoderUnit) -> None:
        """The unit's compute goes back to the pool."""
        self.free.append(unit)
        self.trace.unit_freed.fire(unit)

    def _staging_offer(
        self,
        job: decoding_records.DecodeJob,
        now: int,
        resident_capacity: int,
        memory_demand_of: Callable[[decoding_records.DecodeJob], Optional[int]],
        input_is_on_the_unit: InputIsOnTheUnit,
    ) -> Optional[tuple]:
        """(unit, False) to stage the job's input on, or None.

        Every unit with free compute and room was offered first, so the
        unit chosen here computes.
        """
        with_room = _with_room(
            self.units, job, resident_capacity, memory_demand_of
        )
        with_room = self._freeing_first(with_room, now)
        if not with_room:
            return None
        staging_rank = functools.partial(
            self._staging_rank, job, now, input_is_on_the_unit
        )
        unit = min(with_room, key=staging_rank)
        return unit, False

    def _freeing_first(self, with_room: list, now: int) -> list:
        """The units with room among those with the pool's least work left.

        Staged on a unit with more work left, the job would start later
        than a central FIFO queue would start it, so it waits in the
        queue while these are full.
        """
        work_left_by_unit = {}
        for unit in self.units:
            work_left = unit.work_left_ticks(now, self._occupancy_ticks)
            work_left_by_unit[unit] = work_left
        work_left_values = work_left_by_unit.values()
        least_work_left = min(work_left_values)
        freeing_first = []
        for unit in with_room:
            if work_left_by_unit[unit] == least_work_left:
                freeing_first.append(unit)
        return freeing_first

    def _staging_rank(
        self,
        job: decoding_records.DecodeJob,
        now: int,
        input_is_on_the_unit: InputIsOnTheUnit,
        unit: decoder_unit_module.DecoderUnit,
    ) -> tuple:
        """The order a job that must wait picks its unit by, smallest first.

        Least work left, so the worst wait is shortest; then the fewest
        live jobs; then a unit that holds the rounds, so nothing is
        copied.
        """
        work_left = unit.work_left_ticks(now, self._occupancy_ticks)
        live_residents = unit.live_residents()
        moves_input = not input_is_on_the_unit(job, unit)
        return work_left, len(live_residents), moves_input

    def _occupancy_ticks(self, job: decoding_records.DecodeJob) -> float:
        """Ticks the job's decode holds a unit; unbounded when undeclared."""
        decoder = self.manager.decoder
        occupancy = decoder.occupancy(job)
        if occupancy is None:
            return math.inf
        return occupancy


def decoder_rows(decoders: tuple) -> list:
    """Every decoder row inside these decoders, in the order the walk finds.

    A row is anything that answers the runtime-checkable Decoder port,
    inherited from decsim's base class or not. A row that wraps another
    (the confidence, staged and check wrappers) names its inner decoder
    through the seeding protocol, which the walk follows.
    """
    found = []
    seen = set()
    pending = list(decoders)
    while pending:
        decoder = pending.pop()
        identity = id(decoder)
        if decoder is None or identity in seen:
            continue
        seen.add(identity)
        if isinstance(decoder, ports.Decoder):
            found.append(decoder)
        inner_decoders = _inner_decoders(decoder)
        pending.extend(inner_decoders)
    return found


def _inner_decoders(decoder) -> list:
    """The decoders a wrapping row names through the seeding protocol."""
    if not isinstance(decoder, seeding.RunSeedComposite):
        return []
    children = decoder.run_seed_children()
    inner_decoders = []
    for child in children:
        inner_decoders.append(child.child)
    return inner_decoders


def _with_room(
    units: list,
    job: decoding_records.DecodeJob,
    resident_capacity: int,
    memory_demand_of: Callable[[decoding_records.DecodeJob], Optional[int]],
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
    """The free unit a job that may start takes, or None when none has room."""
    if not free_with_room:
        return None
    unit_holding_input = _unit_holding_input(
        free_with_room, job, input_is_on_the_unit
    )
    if unit_holding_input is not None:
        return unit_holding_input
    return free_with_room[0]


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the decoder pool reports, as one member."""

    unit_busy: trace_source.TraceSource = trace_source.new_source()
    unit_freed: trace_source.TraceSource = trace_source.new_source()
