"""The memory inside one decoder unit.

Each unit holds the input of the jobs it is decoding: the manager assigns
a unit, the window's rounds move from Buffer 0 into that unit's memory
as one immutable DecoderInput, the engine reads them, and the memory is
freed when the decode completes. Capacity is rounds per unit; a window
larger than the unit's memory cannot be decoded by that unit and stops
the run. There is no shared store, no credits and no waiting: a job
waits in Buffer 0 for a unit, never for memory. Precedent: XQsim's error
decode unit holds one syndrome input at a time in its own registers.
"""

import dataclasses
import types
from collections.abc import Mapping
from typing import Any, Optional

import decsim.message as message


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
    fragments: tuple[message.RetainedSyndromeFragment, ...]


@dataclasses.dataclass(frozen=True)
class DecoderInput:
    """Immutable local input for one decoder request.

    Rounds are ordered by operation identity and round index.
    """

    op_id: int
    window_id: int
    request_key: Optional[message.DecoderRequestKey]
    rounds: tuple[MaterializedSyndromeRound, ...]

    def fragments(self) -> list:
        """The landed fragments, round by round."""
        fragments = []
        for round_input in self.rounds:
            fragments.extend(round_input.fragments)
        return fragments


@dataclasses.dataclass(frozen=True)
class DecoderMemorySnapshot:
    """Immutable observation of one unit's memory."""

    pool: str
    unit: int
    capacity_rounds: Optional[int]
    occupied_rounds: int
    peak_occupied_rounds: int
    admissions: int


def materialize_decoder_input(job: message.DecodeJob) -> DecoderInput:
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
        op_id=job.op_id,
        window_id=job.window_id,
        request_key=job.request_key,
        rounds=rounds,
    )


class DecoderMemory:
    """The input memory of one decoder unit."""

    def __init__(
        self, pool: str, unit: int, capacity_rounds: Optional[int]
    ) -> None:
        self.pool = pool
        self.unit = unit
        self.capacity_rounds = capacity_rounds
        self._inputs: dict = {}  # request key -> DecoderInput
        self.peak_occupied_rounds = 0
        self.admissions = 0

    @property
    def occupied_rounds(self) -> int:
        """Rounds held right now, over every input."""
        occupied = 0
        for decoder_input in self._inputs.values():
            occupied += len(decoder_input.rounds)
        return occupied

    def deposit(self, job: message.DecodeJob) -> DecoderInput:
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
        self._inputs[key] = decoder_input
        self.peak_occupied_rounds = max(self.peak_occupied_rounds, needed)
        self.admissions += 1
        return decoder_input

    def take(self, job: message.DecodeJob) -> None:
        """Free the job's rounds; a job this unit never held is ignored."""
        key = _memory_key(job)
        self._inputs.pop(key, None)

    def snapshot(self) -> DecoderMemorySnapshot:
        """The memory's counters as one immutable record."""
        return DecoderMemorySnapshot(
            self.pool,
            self.unit,
            self.capacity_rounds,
            self.occupied_rounds,
            self.peak_occupied_rounds,
            self.admissions,
        )


def _memory_key(job: message.DecodeJob):
    """A window job is held under its request key, any other under itself."""
    if job.request_key is not None:
        return job.request_key
    return id(job)


def _round_order_key(item: tuple) -> tuple:
    identity, _fragments = item
    operation_id, round_index = identity
    order = message.stable_identity_order_key(operation_id)
    return order, round_index


def _check_detector_row_layout(
    job: message.DecodeJob, rounds: tuple[MaterializedSyndromeRound, ...]
) -> None:
    """A model-backed input lies in the model's rows, or the run stops.

    The operation, the round order and the dense positions in each round
    must match; a wrong layout would decode the wrong syndrome silently.
    Every model that reaches a job is a WindowErrorModel.
    """
    model = job.dem
    if model is None:
        return
    input_rows = _input_row_identities(job, rounds)
    model_rows = _model_row_identities(
        job.op_id, model.detector_ids, model.defect_positions
    )
    if input_rows != model_rows:
        raise RuntimeError(
            f"{job.label}: canonical decoder-input row layout "
            f"{input_rows!r} does not match the window error model's row "
            f"layout {model_rows!r}"
        )


def _input_row_identities(
    job: message.DecodeJob, rounds: tuple[MaterializedSyndromeRound, ...]
) -> tuple:
    """(operation, round, position) of every bit the input carries."""
    identities = []
    for round_input in rounds:
        is_same_operation = message.same_stable_identity(
            round_input.operation_id, job.op_id
        )
        if not is_same_operation:
            raise RuntimeError(
                f"{job.label}: model-backed decoder-input round operation "
                f"{round_input.operation_id!r} does not match job operation "
                f"{job.op_id!r}"
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


def _model_row_identities(op_id, detector_ids, defect_positions) -> tuple:
    """(operation, round, position) of every detector row of the model."""
    identities = []
    for detector_id in detector_ids:
        round_index, position_in_round = defect_positions[detector_id]
        identities.append((op_id, round_index, position_in_round))
    return tuple(identities)
