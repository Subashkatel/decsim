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
)


class DecoderLocalCapacityExhaustion(RuntimeError):
    """A reserve was refused because every decoder-local slot is in use."""


@dataclass(frozen=True)
class MaterializedSyndromeRound:
    """One immutable syndrome round owned by the decoder side."""

    operation_id: Any
    round_index: int
    fragments: tuple[RetainedSyndromeFragment, ...]

    def __post_init__(self) -> None:
        if type(self.round_index) is not int:
            raise TypeError("round_index must be an exact built-in int")
        if (
            type(self.fragments) is not tuple
            or not self.fragments
            or any(type(fragment) is not RetainedSyndromeFragment
                   for fragment in self.fragments)
        ):
            raise TypeError(
                "materialized round fragments must be a nonempty tuple of "
                "retained fragments")
        for fragment in self.fragments:
            if not same_stable_identity(
                    fragment.operation_id, self.operation_id):
                raise ValueError(
                    "materialized fragments must share operation identity")
            if fragment.round_index != self.round_index:
                raise ValueError(
                    "materialized fragments must share round_index")


@dataclass(frozen=True)
class DecoderInput:
    """Immutable local input for one decoder request.

    Rounds are ordered by operation identity and round index. Fragment values
    are newly created; immutable bit tuples may be reused as values.
    """

    op_id: int
    window_id: int
    label: str
    request_key: Optional[DecoderRequestKey]
    rounds: tuple[MaterializedSyndromeRound, ...]

    def __post_init__(self) -> None:
        if type(self.op_id) is not int:
            raise TypeError("op_id must be an exact built-in int")
        if type(self.window_id) is not int:
            raise TypeError("window_id must be an exact built-in int")
        if type(self.label) is not str:
            raise TypeError("label must be a built-in string")
        if self.request_key is not None and (
                type(self.request_key) is not DecoderRequestKey):
            raise TypeError(
                "request_key must be a DecoderRequestKey when supplied")
        if type(self.rounds) is not tuple or any(
                type(entry) is not MaterializedSyndromeRound
                for entry in self.rounds):
            raise TypeError(
                "rounds must be a tuple of materialized syndrome rounds")
        identities = [
            (entry.operation_id, entry.round_index) for entry in self.rounds
        ]
        from .message import stable_identity_order_key
        expected = sorted(
            identities,
            key=lambda identity: (
                stable_identity_order_key(identity[0]), identity[1]
            ),
        )
        if identities != expected or len(set(identities)) != len(identities):
            raise ValueError(
                "rounds must be uniquely ordered by operation and round")

    @property
    def round_count(self) -> int:
        return len(self.rounds)


def _rematerialized_fragment(
        fragment: RetainedSyndromeFragment) -> RetainedSyndromeFragment:
    """Build a fresh decoder-local fragment value; leaf bit tuple is shared."""
    return RetainedSyndromeFragment(
        operation_id=fragment.operation_id,
        patch_id=fragment.patch_id,
        round_index=fragment.round_index,
        bits=fragment.bits,
        code=fragment.code,
        size_bits=fragment.size_bits,
        fragment_index=fragment.fragment_index,
    )


def materialize_decoder_input(job: DecodeJob) -> DecoderInput:
    """Build one immutable decoder-local input from a job's fragments."""
    if type(job) is not DecodeJob:
        raise TypeError("materialization requires a DecodeJob")
    for payload in job.payloads:
        if type(payload) is not RetainedSyndromeFragment:
            raise TypeError(
                "every job payload must be a RetainedSyndromeFragment")
    fragments_by_round: dict[tuple, list[RetainedSyndromeFragment]] = {}
    for payload in job.payloads:
        identity = (payload.operation_id, payload.round_index)
        fragments_by_round.setdefault(identity, []).append(
            _rematerialized_fragment(payload))
    from .message import stable_identity_order_key
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
    return DecoderInput(
        op_id=job.op_id,
        window_id=job.window_id,
        label=job.label,
        request_key=job.request_key,
        rounds=rounds,
    )


class _SlotState(Enum):
    RESERVED = "reserved"
    DEPOSITED = "deposited"


class DecoderLocalMemory:
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
        """Claim one empty decoder-local slot for a future deposit."""
        if key in self._slots:
            raise RuntimeError(
                f"decoder-local slot {key!r} is already "
                f"{self._slots[key][0].value}")
        if self.capacity is not None and len(self._slots) >= self.capacity:
            raise DecoderLocalCapacityExhaustion(
                f"all {self.capacity} decoder-local slots are in use")
        self._slots[key] = (_SlotState.RESERVED, None)

    def deposit(self, key: Any, job: DecodeJob) -> DecoderInput:
        """Materialize one job's input into a reserved slot and return it."""
        state = self._slots.get(key)
        if state is None:
            raise RuntimeError(
                f"deposit into unreserved decoder-local slot {key!r}")
        if state[0] is not _SlotState.RESERVED:
            raise RuntimeError(
                f"decoder-local slot {key!r} already holds a deposit")
        decoder_input = materialize_decoder_input(job)
        self._slots[key] = (_SlotState.DEPOSITED, decoder_input)
        return decoder_input

    def take(self, key: Any) -> DecoderInput:
        """Consume one deposited input and free its slot."""
        state = self._slots.get(key)
        if state is None:
            raise RuntimeError(
                f"take from unknown decoder-local slot {key!r}")
        if state[0] is not _SlotState.DEPOSITED:
            raise RuntimeError(
                f"take from decoder-local slot {key!r} before any deposit")
        del self._slots[key]
        return state[1]

    def discard(self, key: Any) -> None:
        """Free one live slot (reserved or deposited) without reading it."""
        if key not in self._slots:
            raise RuntimeError(
                f"discard of unknown decoder-local slot {key!r}")
        del self._slots[key]
