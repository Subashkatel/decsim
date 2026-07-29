"""The single owner of how long raw syndrome rounds stay in memory.

Retention is refcounted through leases. A lease is a named set of
(op_id, round) keys; a round's payload is freed when the last lease
holding it drops. Weak-window reads and strong-redo reads hold separate
leases so the strong lease can already be released at weak commit — safe
because a strong redo job copies its payloads before the weak commit
callback runs. Also tracks payloads_held / peak_payloads for the memory
accounting metrics.
"""

from __future__ import annotations

from typing import Optional

from .message import SyndromeRoundPacket


class PayloadStore:
    """Refcounted retained syndrome rounds, keyed (op_id, round_index).

    memory_model (port 18) observes the physical storage inside the lease
    lifetime: store() on every retained fragment, evict() when a round frees.
    The default (None) is an unbounded in-memory dict."""

    def __init__(self, memory_model=None) -> None:
        # op_id -> round_index -> one complete immutable packet
        self._payloads: dict = {}
        self._round_complete_ticks: dict[tuple, int] = {}
        self._round_refs: dict[tuple, int] = {}
        self._leases: dict = {}          # lease_id -> list[(op_id, round)]
        self.memory_model = memory_model
        self.payloads_held = 0
        self.peak_payloads = 0

    # ------------------------------------------------------------- op scope

    def register_op(self, op_id) -> None:
        """Open an op's payload storage; rounds arrive via store_round()."""
        self._payloads.setdefault(op_id, {})

    def has_op(self, op_id) -> bool:
        """False once free_op ran (late arrivals must error, parity :494-500)."""
        return op_id in self._payloads

    def free_op(self, op_id) -> None:
        """Free an op's entire payload RAM at op finish (parity :967-969)."""
        freed = self._payloads.pop(op_id, None)
        if freed:
            self.payloads_held -= sum(
                len(packet.fragments) for packet in freed.values()
            )
            if self.memory_model is not None:
                for round_index, packet in freed.items():
                    for fragment in packet.fragments:
                        self.memory_model.evict((op_id, round_index,
                                                 fragment.patch_id))
            for round_index in freed:
                self._round_complete_ticks.pop((op_id, round_index), None)

    # ------------------------------------------------------------- storage

    def store_round(
        self,
        packet: SyndromeRoundPacket,
        completion_tick: int,
    ) -> None:
        """Retain one complete immutable round and its decoder-arrival tick."""
        if type(packet) is not SyndromeRoundPacket:
            raise TypeError("store_round requires an exact SyndromeRoundPacket")
        if type(completion_tick) is not int:
            raise TypeError("completion_tick must be an exact built-in int")
        if completion_tick < 0:
            raise ValueError("completion_tick must be nonnegative")
        if packet.operation_id not in self._payloads:
            raise RuntimeError(
                f"operation {packet.operation_id!r} payload storage is not open"
            )
        rounds = self._payloads[packet.operation_id]
        if packet.round_index in rounds:
            raise ValueError(
                f"syndrome round {(packet.operation_id, packet.round_index)!r} "
                "is already retained"
            )
        rounds[packet.round_index] = packet
        self._round_complete_ticks[
            (packet.operation_id, packet.round_index)
        ] = completion_tick
        self.payloads_held += len(packet.fragments)
        if self.memory_model is not None:
            for fragment in packet.fragments:
                self.memory_model.store(
                    (
                        packet.operation_id,
                        packet.round_index,
                        fragment.patch_id,
                    ),
                    fragment,
                )
        self.peak_payloads = max(self.peak_payloads, self.payloads_held)

    def fragments(self, op_id, round_index: int) -> Optional[tuple]:
        """The retained fragments of one round (None if absent/freed)."""
        packet = self._payloads.get(op_id, {}).get(round_index)
        return None if packet is None else packet.fragments

    def round_complete_tick(self, op_id, round_index: int) -> Optional[int]:
        """The exact decoder-arrival tick for a retained complete round."""
        return self._round_complete_ticks.get((op_id, round_index))

    def replay(self, round_keys) -> list:
        """Assemble retained payloads for a task rebuild, in round order.

        Returns [(op_id, round_index, fragments), ...]; missing rounds
        are skipped (parity with _assemble_payloads' .get chain)."""
        out = []
        for op_id, round_index in round_keys:
            frags = self.fragments(op_id, round_index)
            if frags is not None:
                out.append((op_id, round_index, frags))
        return out

    # -------------------------------------------------------------- leases

    def lease_round_keys(self, lease_id) -> tuple:
        """Return one lease's exact ordered keys without exposing mutation."""
        if lease_id not in self._leases:
            raise KeyError(lease_id)
        return tuple(self._leases[lease_id])

    def lease(self, lease_id, round_keys) -> None:
        """Register a named lease over round keys, bumping each ref."""
        if lease_id in self._leases:
            raise ValueError(f"duplicate lease {lease_id!r}")
        keys = list(round_keys)
        self._leases[lease_id] = keys
        for key in keys:
            self._round_refs[key] = self._round_refs.get(key, 0) + 1

    def replace(self, lease_id, round_keys) -> None:
        """Replace a lease's round set.

        Acquire-before-release: a round retained by both the old and new
        sets must never transiently hit refcount zero. Releasing
        first would transiently zero shared rounds and free their arrived
        payloads (unrecoverable — store() only happens at arrival), so new
        refs are added before old refs drop."""
        keys = list(round_keys)
        old = self._leases.pop(lease_id, ())
        for key in keys:
            self._round_refs[key] = self._round_refs.get(key, 0) + 1
        self._leases[lease_id] = keys
        for key in old:
            self._drop_round_ref(key)

    def release(self, lease_id) -> None:
        """Drop a lease; idempotent (releasing an unknown lease is a no-op)."""
        for key in self._leases.pop(lease_id, ()):
            self._drop_round_ref(key)

    def _drop_round_ref(self, round_key: tuple) -> None:
        """Drop one reference; free the round's payloads if it was the last
       ."""
        self._round_refs[round_key] = self._round_refs.get(round_key, 0) - 1
        if self._round_refs[round_key] > 0:
            return
        round_op, round_no = round_key
        packet = self._payloads.get(round_op, {}).pop(round_no, None)
        if packet is not None:
            self.payloads_held -= len(packet.fragments)
            if self.memory_model is not None:
                for fragment in packet.fragments:
                    self.memory_model.evict(
                        (round_op, round_no, fragment.patch_id)
                    )
            self._round_complete_ticks.pop((round_op, round_no), None)
        self._round_refs.pop(round_key, None)
