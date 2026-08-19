"""The memory inside one decoder unit.

Each unit holds the input of the one job it is decoding: the manager assigns
a unit, the window's rounds move from Buffer 0 into that unit's memory as one
immutable ``DecoderInput``, the engine reads them, and the memory is freed when
the decode completes. Capacity is rounds per unit; a window larger than the
unit's memory cannot be decoded by that unit and stops the run. There is no
shared store, no credits and no waiting: a job waits in Buffer 0 for a unit,
never for memory. Precedent: XQsim's error decode unit holds one syndrome
input at a time in its own registers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional

from ..message import (
    DecodeJob,
    DecoderRequestKey,
    RetainedSyndromeFragment,
    same_stable_identity,
    stable_identity_order_key,
)


class DecoderMemoryCapacityExhaustion(RuntimeError):
    """A window does not fit the memory of the unit assigned to it."""

    status = "decoder_memory_capacity_exhaustion"

    def __init__(self, *, pool: str, unit: int, requested_rounds: int,
                 capacity_rounds: int) -> None:
        super().__init__(
            f"decoder unit {pool!r}#{unit} holds {capacity_rounds} rounds; the "
            f"window needs {requested_rounds}")
        self.pool = pool
        self.unit = unit
        self.requested_rounds = requested_rounds
        self.capacity_rounds = capacity_rounds


@dataclass(frozen=True)
class DecoderMemoryConfig:
    """Rounds of memory per decoder unit, by pool; a pool absent from the map
    is unbounded. ``RunSpec.decoder_memory`` unset leaves every unit unbounded."""

    capacity_rounds_by_pool: Mapping[str, int]

    def __post_init__(self) -> None:
        copied = {}
        for pool, capacity in dict(self.capacity_rounds_by_pool).items():
            if capacity < 1:
                raise ValueError(f"pool {pool!r} needs a positive int round capacity")
            copied[pool] = capacity
        object.__setattr__(self, "capacity_rounds_by_pool", MappingProxyType(copied))

    def capacity_for(self, pool: str) -> Optional[int]:
        return self.capacity_rounds_by_pool.get(pool)


@dataclass(frozen=True)
class MaterializedSyndromeRound:
    """One immutable syndrome round owned by the decoder side."""

    operation_id: Any
    round_index: int
    fragments: tuple[RetainedSyndromeFragment, ...]


@dataclass(frozen=True)
class DecoderInput:
    """Immutable local input for one decoder request.

    Rounds are ordered by operation identity and round index.
    """

    op_id: int
    window_id: int
    request_key: Optional[DecoderRequestKey]
    rounds: tuple[MaterializedSyndromeRound, ...]


def _check_detector_row_layout(
    job: DecodeJob,
    rounds: tuple[MaterializedSyndromeRound, ...],
) -> None:
    """Check model-backed operation/round order and dense row positions."""
    model = getattr(job, "dem", None)
    if model is None:
        return
    missing_layout_member = object()
    detector_ids = getattr(model, "detector_ids", missing_layout_member)
    defect_positions = getattr(
        model, "defect_positions", missing_layout_member
    )
    if (
        detector_ids is missing_layout_member
        or defect_positions is missing_layout_member
    ):
        return

    input_row_identities = []
    for round_input in rounds:
        if not same_stable_identity(round_input.operation_id, job.op_id):
            raise ValueError(
                f"{getattr(job, 'label', '')}: model-backed decoder-input "
                f"round operation {round_input.operation_id!r} does not match "
                f"job operation {job.op_id!r}"
            )
        position_in_round = 0
        for fragment in round_input.fragments:
            if fragment.bits is None:
                continue
            input_row_identities.extend(
                (
                    round_input.operation_id,
                    round_input.round_index,
                    position_in_round + bit_offset,
                )
                for bit_offset in range(len(fragment.bits))
            )
            position_in_round += len(fragment.bits)
    input_row_identities = tuple(input_row_identities)

    model_row_identities = []
    for detector_id in detector_ids:
        round_index, position_in_round = defect_positions[detector_id]
        model_row_identities.append(
            (job.op_id, round_index, position_in_round)
        )
    model_row_identities = tuple(model_row_identities)
    if input_row_identities != model_row_identities:
        raise ValueError(
            f"{getattr(job, 'label', '')}: canonical decoder-input row layout "
            f"{input_row_identities!r} does not match the window error model's "
            f"row layout {model_row_identities!r}"
        )

def materialize_decoder_input(job: DecodeJob) -> DecoderInput:
    """Build one immutable decoder memory input from a job's fragments."""
    fragments_by_round: dict[tuple, list[RetainedSyndromeFragment]] = {}
    for payload in job.payloads:
        identity = (payload.operation_id, payload.round_index)
        fragments_by_round.setdefault(identity, []).append(payload)
    ordered = sorted(
        fragments_by_round.items(),
        key=lambda item: (
            stable_identity_order_key(item[0][0]), item[0][1]
        ),
    )
    rounds = tuple(
        MaterializedSyndromeRound(
            operation_id=identity[0],
            round_index=identity[1],
            fragments=tuple(fragments),
        )
        for identity, fragments in ordered
    )
    _check_detector_row_layout(job, rounds)
    return DecoderInput(
        op_id=job.op_id,
        window_id=job.window_id,
        request_key=job.request_key,
        rounds=rounds,
    )



@dataclass(frozen=True)
class DecoderMemorySnapshot:
    """Immutable observation of one unit's memory."""

    pool: str
    unit: int
    capacity_rounds: Optional[int]
    occupied_rounds: int
    peak_occupied_rounds: int
    admissions: int


def count_decoder_input_round_demand(job: DecodeJob) -> int:
    """Count the distinct syndrome rounds one job will store.

    The demand is the number of distinct ``(operation_id, round_index)``
    identities in the job's payloads, which is exactly the number of rounds
    ``materialize_decoder_input`` groups them into. It is counted here so that
    capacity is charged in actual stored rounds without building a second
    immutable input while a request waits, and without reading ``n_rounds``,
    which is service extent rather than stored data. Payload type admission
    stays at materialization, after credits fit.
    """
    round_identities = {
        (payload.operation_id, payload.round_index) for payload in job.payloads
    }
    return len(round_identities)



class DecoderMemory:
    """The input memory of one decoder unit."""

    def __init__(self, pool: str, unit: int, capacity_rounds: Optional[int]) -> None:
        self.pool = pool
        self.unit = unit
        self.capacity_rounds = capacity_rounds
        self._inputs: dict = {}            # request key -> DecoderInput
        self.peak_occupied_rounds = 0
        self.admissions = 0

    @property
    def occupied_rounds(self) -> int:
        return sum(len(decoder_input.rounds) for decoder_input in self._inputs.values())

    def deposit(self, job: DecodeJob) -> DecoderInput:
        """Materialize one job's rounds into this unit's memory."""
        key = job.request_key if job.request_key is not None else id(job)
        if key in self._inputs:
            raise RuntimeError(f"unit {self.pool!r}#{self.unit} already holds {job.label!r}")
        decoder_input = materialize_decoder_input(job)
        needed = self.occupied_rounds + len(decoder_input.rounds)
        if self.capacity_rounds is not None and needed > self.capacity_rounds:
            raise DecoderMemoryCapacityExhaustion(
                pool=self.pool, unit=self.unit, requested_rounds=needed,
                capacity_rounds=self.capacity_rounds)
        self._inputs[key] = decoder_input
        self.peak_occupied_rounds = max(self.peak_occupied_rounds, needed)
        self.admissions += 1
        return decoder_input

    def take(self, job: DecodeJob) -> None:
        """Free the job's rounds; a job this unit never held is ignored."""
        key = job.request_key if job.request_key is not None else id(job)
        self._inputs.pop(key, None)

    def snapshot(self) -> DecoderMemorySnapshot:
        return DecoderMemorySnapshot(self.pool, self.unit, self.capacity_rounds,
                                     self.occupied_rounds, self.peak_occupied_rounds,
                                     self.admissions)
