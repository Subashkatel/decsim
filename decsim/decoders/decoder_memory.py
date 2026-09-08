"""The memory inside one decoder unit.

Each unit holds the input of the jobs it is decoding: the manager assigns
a unit, the window's rounds move from Buffer 0 into that unit's memory
as one immutable DecoderInput, the engine reads them, and the memory is
freed when the decode completes. Capacity is rounds per unit; a window
larger than the unit's memory cannot be decoded by that unit and stops
the run. There is no shared store, no credits and no waiting: a job
waits in Buffer 0 for a unit, never for memory. Precedent: XQsim's error
decode unit holds one syndrome input at a time in its own registers.

An input is held per input, not per job, with its readers recorded, so
two jobs that read the same rounds on one unit are one copy and one
transfer: gem5's MSHR keeps every target of a single fill
(src/mem/cache/mshr.hh), and OpenMP's shared clause says every task
reads the storage of the original item (openmp_spec_5_2.txt:4315-4317).
The rule the data-movement study rests on is one copy per unit that
reads the window, never one copy per job and never a copy taken from
another unit.
"""

import dataclasses
import types
from collections.abc import Mapping
from typing import Any, Optional

import decsim.observe.trace_source as trace_source
import decsim.records.decoding as decoding_records
import decsim.records.identity as identity_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records


class DecoderMemoryCapacityError(RuntimeError):
    """A window does not fit the memory of the unit assigned to it."""

    def __init__(
        self,
        *,
        pool: str,
        unit: int,
        requested_rounds: int,
        capacity_rounds: int,
    ) -> None:
        text = (
            f"decoder unit {pool!r}#{unit} holds {capacity_rounds} rounds; "
            f"the window needs {requested_rounds}"
        )
        RuntimeError.__init__(self, text)
        self.pool = pool
        self.unit = unit
        self.requested_rounds = requested_rounds
        self.capacity_rounds = capacity_rounds


@dataclasses.dataclass(frozen=True)
class DecoderMemoryConfig:
    """Rounds of memory per decoder unit, by pool.

    A pool absent from the map is unbounded; no config at all leaves
    every unit unbounded.
    """

    capacity_rounds_by_pool: Mapping[str, int]

    def __post_init__(self) -> None:
        copied = dict(self.capacity_rounds_by_pool)
        for pool, capacity in copied.items():
            if capacity < 1:
                raise ValueError(
                    f"pool {pool!r} needs a positive round capacity, "
                    f"got {capacity}"
                )
        frozen = types.MappingProxyType(copied)
        object.__setattr__(self, "capacity_rounds_by_pool", frozen)

    def capacity_for(self, pool: str) -> Optional[int]:
        """The pool's rounds per unit; None when unbounded."""
        return self.capacity_rounds_by_pool.get(pool)


@dataclasses.dataclass(frozen=True)
class MaterializedSyndromeRound:
    """One immutable syndrome round owned by the decoder side."""

    operation_id: Any
    round_index: int
    fragments: tuple[round_records.RetainedSyndromeFragment, ...]


@dataclasses.dataclass(frozen=True)
class DecoderInput:
    """Immutable local input for one decoder request.

    Rounds are ordered by operation identity and round index.
    """

    operation_id: int
    window_id: int
    request_key: Optional[window_records.DecoderRequestKey]
    rounds: tuple[MaterializedSyndromeRound, ...]

    def fragments(self) -> list:
        """The landed fragments, round by round."""
        fragments = []
        for round_input in self.rounds:
            fragments.extend(round_input.fragments)
        return fragments


@dataclasses.dataclass
class ResidentInput:
    """One landed input of a unit and the jobs still reading it."""

    decoder_input: DecoderInput
    readers: list


@dataclasses.dataclass(frozen=True)
class DecoderMemorySnapshot:
    """Immutable observation of one unit's memory."""

    pool: str
    unit: int
    capacity_rounds: Optional[int]
    occupied_rounds: int
    peak_occupied_rounds: int
    admissions: int


def materialize_decoder_input(job: decoding_records.DecodeJob) -> DecoderInput:
    """Build one immutable decoder memory input from a job's fragments."""
    fragments_by_round: dict = {}
    for payload in job.payloads:
        identity = (payload.operation_id, payload.round_index)
        fragments = fragments_by_round.setdefault(identity, [])
        fragments.append(payload)
    rounds_with_fragments = fragments_by_round.items()
    ordered = sorted(rounds_with_fragments, key=_round_order_key)
    rounds = []
    for identity, fragments in ordered:
        operation_id, round_index = identity
        round_input = MaterializedSyndromeRound(
            operation_id=operation_id,
            round_index=round_index,
            fragments=tuple(fragments),
        )
        rounds.append(round_input)
    rounds = tuple(rounds)
    _check_detector_row_layout(job, rounds)
    return DecoderInput(
        operation_id=job.operation_id,
        window_id=job.window_id,
        request_key=job.request_key,
        rounds=rounds,
    )


class DecoderMemory:
    """The input memory of one decoder unit.

    Trace sources: deposited(job, decoder_input) when a job's rounds land
    here, taken(job, decoder_input) when they are freed; a residence in
    this memory runs between the two (data_path.md hop 6), and both
    carry the rounds so a listener counts what is held.
    """

    def __init__(
        self, pool: str, unit: int, capacity_rounds: Optional[int]
    ) -> None:
        self.pool = pool
        self.unit = unit
        self.capacity_rounds = capacity_rounds
        self._inputs: dict = {}  # input key -> ResidentInput
        self.statistics = _MemoryStatistics()
        self.trace = _TraceSources()

    @property
    def name(self) -> str:
        """The unit's memory as the trace names it."""
        return f"unit {self.pool}#{self.unit} memory"

    @property
    def occupied_rounds(self) -> int:
        """Rounds held right now, over every input."""
        occupied = 0
        for resident in self._inputs.values():
            occupied += len(resident.decoder_input.rounds)
        return occupied

    def landing_key(self, job: decoding_records.DecodeJob) -> tuple:
        """The identity of the rounds this job reads, in this unit."""
        key = _memory_key(job)
        return (self.pool, self.unit, key)

    def holds(self, job: decoding_records.DecodeJob) -> bool:
        """Whether the rounds this job reads are already in this memory."""
        key = _memory_key(job)
        return key in self._inputs

    def deposit(self, job: decoding_records.DecodeJob) -> DecoderInput:
        """Materialize one job's rounds into this unit's memory."""
        key = _memory_key(job)
        if key in self._inputs:
            raise RuntimeError(
                f"unit {self.pool!r}#{self.unit} already holds {job.label!r}"
            )
        decoder_input = materialize_decoder_input(job)
        needed = self.occupied_rounds + len(decoder_input.rounds)
        if self.capacity_rounds is not None and needed > self.capacity_rounds:
            raise DecoderMemoryCapacityError(
                pool=self.pool,
                unit=self.unit,
                requested_rounds=needed,
                capacity_rounds=self.capacity_rounds,
            )
        self._inputs[key] = ResidentInput(decoder_input, [job])
        self.statistics.peak_occupied_rounds = max(
            self.statistics.peak_occupied_rounds, needed
        )
        self.statistics.admissions += 1
        self.trace.deposited.fire(job, decoder_input)
        return decoder_input

    def add_reader(self, job: decoding_records.DecodeJob) -> DecoderInput:
        """One more job reads the rounds already here; no second copy."""
        key = _memory_key(job)
        resident = self._inputs[key]
        resident.readers.append(job)
        return resident.decoder_input

    def take(self, job: decoding_records.DecodeJob) -> None:
        """Drop the job's read; the rounds go when the last reader does."""
        key = _memory_key(job)
        resident = self._inputs.get(key)
        if resident is None:
            return
        _drop_reader(resident, job)
        if resident.readers:
            return
        del self._inputs[key]
        self.trace.taken.fire(job, resident.decoder_input)

    def snapshot(self) -> DecoderMemorySnapshot:
        """The memory's counters as one immutable record."""
        return DecoderMemorySnapshot(
            self.pool,
            self.unit,
            self.capacity_rounds,
            self.occupied_rounds,
            self.statistics.peak_occupied_rounds,
            self.statistics.admissions,
        )


def _memory_key(job: decoding_records.DecodeJob):
    """The identity of the rounds a job reads.

    The request whose transfer brought them when the job shares another
    request's landed input, its own request key otherwise, and the job
    itself when it carries no request.
    """
    if job.input_key is not None:
        return job.input_key
    if job.request_key is not None:
        return job.request_key
    return id(job)


def _drop_reader(
    resident: ResidentInput, job: decoding_records.DecodeJob
) -> None:
    """Take one job out of an input's readers; an unknown job is ignored."""
    for reader in resident.readers:
        if reader is job:
            resident.readers.remove(reader)
            return


def _round_order_key(item: tuple) -> tuple:
    identity, _fragments = item
    operation_id, round_index = identity
    order = identity_records.stable_identity_order_key(operation_id)
    return order, round_index


def _check_detector_row_layout(
    job: decoding_records.DecodeJob,
    rounds: tuple[MaterializedSyndromeRound, ...],
) -> None:
    """A model-backed input lies in the model's rows, or the run stops.

    The operation, the round order and the dense positions in each round
    must match; a wrong layout would decode the wrong syndrome silently.
    Every model that reaches a job is a WindowErrorModel.
    """
    model = job.detector_error_model
    if model is None:
        return
    input_rows = _input_row_identities(job, rounds)
    model_rows = _model_row_identities(
        job.operation_id, model.detector_ids, model.defect_positions
    )
    if input_rows != model_rows:
        raise RuntimeError(
            f"{job.label}: canonical decoder-input row layout "
            f"{input_rows!r} does not match the window error model's row "
            f"layout {model_rows!r}"
        )


def _input_row_identities(
    job: decoding_records.DecodeJob,
    rounds: tuple[MaterializedSyndromeRound, ...],
) -> tuple:
    """(operation, round, position) of every bit the input carries."""
    identities = []
    for round_input in rounds:
        is_same_operation = identity_records.same_stable_identity(
            round_input.operation_id, job.operation_id
        )
        if not is_same_operation:
            raise RuntimeError(
                f"{job.label}: model-backed decoder-input round operation "
                f"{round_input.operation_id!r} does not match job operation "
                f"{job.operation_id!r}"
            )
        round_rows = _round_row_identities(round_input)
        identities.extend(round_rows)
    return tuple(identities)


def _round_row_identities(round_input: MaterializedSyndromeRound) -> list:
    identities = []
    position_in_round = 0
    for fragment in round_input.fragments:
        if fragment.bits is None:
            continue
        for bit_offset in range(len(fragment.bits)):
            position = position_in_round + bit_offset
            identities.append(
                (round_input.operation_id, round_input.round_index, position)
            )
        position_in_round += len(fragment.bits)
    return identities


def _model_row_identities(
    operation_id, detector_ids, defect_positions
) -> tuple:
    """(operation, round, position) of every detector row of the model."""
    identities = []
    for detector_id in detector_ids:
        round_index, position_in_round = defect_positions[detector_id]
        identities.append((operation_id, round_index, position_in_round))
    return tuple(identities)


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the decoder memory reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    deposited: trace_source.TraceSource = trace_source.new_source()
    taken: trace_source.TraceSource = trace_source.new_source()


@dataclasses.dataclass
class _MemoryStatistics:
    """What one unit's memory has held, over the run.

    peak_occupied_rounds is the high-water mark a study sizes the SRAM
    by; admissions counts the inputs that landed. gem5 keeps a
    component's counters in one Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92).
    """

    peak_occupied_rounds: int = 0
    admissions: int = 0
