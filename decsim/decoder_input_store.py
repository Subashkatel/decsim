"""Own the immutable input used by a decoder request.

A transfer reserves one local slot, deposits a materialized input, and releases
the slot after decoding or cancellation. This module does not queue or schedule
decoder work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .message import (
    DecodeJob,
    DecoderRequestKey,
    RetainedSyndromeFragment,
    same_stable_identity,
    stable_identity_order_key,
)


class DecoderInputStoreCapacityExhaustion(RuntimeError):
    """A reserve was refused because every decoder-input store slot is in use."""


@dataclass(frozen=True)
class MaterializedSyndromeRound:
    """One immutable syndrome round owned by the decoder side."""

    operation_id: Any
    round_index: int
    fragments: tuple[RetainedSyndromeFragment, ...]

    def __post_init__(self) -> None:
        if type(self.round_index) is not int:
            raise TypeError("round_index must be an exact built-in int")
        for fragment in self.fragments:
            if not same_stable_identity(
                    fragment.operation_id, self.operation_id):
                raise ValueError(
                    "materialized fragments must share operation identity")


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
    """Build one immutable decoder-input store input from a job's fragments."""
    fragments_by_round: dict[tuple, list[RetainedSyndromeFragment]] = {}
    for payload in job.payloads:
        if type(payload) is not RetainedSyndromeFragment:
            raise TypeError(
                "every job payload must be a RetainedSyndromeFragment")
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


class _SlotState(Enum):
    RESERVED = "reserved"
    DEPOSITED = "deposited"


class DecoderInputStore:
    """Manage local input slots through reserve, deposit, and release.

    One key owns one slot. ``capacity=None`` means unbounded. Invalid state
    changes raise ``RuntimeError`` without changing the memory.
    """

    def __init__(self, capacity: Optional[int] = None) -> None:
        if capacity is not None and (type(capacity) is not int or capacity < 1):
            raise TypeError(
                "capacity must be None (unbounded) or a positive built-in int")
        self.capacity = capacity
        self._slots: dict[Any, tuple[_SlotState, Optional[DecoderInput]]] = {}

    @property
    def slots_in_use(self) -> int:
        return len(self._slots)

    def reserve(self, key: Any) -> None:
        """Claim one empty decoder-input store slot for a future deposit."""
        if key in self._slots:
            raise RuntimeError(
                f"decoder-input store slot {key!r} is already "
                f"{self._slots[key][0].value}")
        if self.capacity is not None and len(self._slots) >= self.capacity:
            raise DecoderInputStoreCapacityExhaustion(
                f"all {self.capacity} decoder-input store slots are in use")
        self._slots[key] = (_SlotState.RESERVED, None)

    def deposit(self, key: Any, job: DecodeJob) -> DecoderInput:
        """Materialize one job's input into a reserved slot and return it."""
        state = self._slots[key]
        if state[0] is not _SlotState.RESERVED:
            raise RuntimeError(
                f"decoder-input store slot {key!r} already holds a deposit")
        decoder_input = materialize_decoder_input(job)
        self._slots[key] = (_SlotState.DEPOSITED, decoder_input)
        return decoder_input

    def take(self, key: Any) -> DecoderInput:
        """Consume one deposited input and free its slot."""
        state = self._slots[key]
        if state[0] is not _SlotState.DEPOSITED:
            raise RuntimeError(
                f"take from decoder-input store slot {key!r} before any deposit")
        del self._slots[key]
        return state[1]

    def discard(self, key: Any) -> None:
        """Free one live slot (reserved or deposited) without reading it."""
        if key not in self._slots:
            raise RuntimeError(
                f"discard of unknown decoder-input store slot {key!r}")
        del self._slots[key]
