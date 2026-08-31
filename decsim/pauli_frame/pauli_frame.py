# Data-core semantics adapted from PECOS PauliFrameAccumulator and ObsMask.
# Source: https://github.com/PECOS-packages/PECOS
# Commit: 7c679509ec7e87410f99445c2ec5442eb91016fd
# Files: crates/pecos-decoder-core/src/pauli_frame.rs:53-108; crates/pecos-decoder-core/src/obs_mask.rs:132-141
# Copyright 2026 The PECOS Developers.
# License: Apache-2.0. Full text: tmp/references/code/pecos/LICENSE.
# NOTICE: Copyright 2018 The PECOS Developers. The copyright for the code in PECOS is held by the contributors (or their
# NOTICE: employers) for the code they contributed and is licensed under Apache-2.0. See the revision history in source control
# NOTICE: for the list of contributors.
# NOTICE: 
# NOTICE: Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# NOTICE: DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
# Modified for decsim: Python tuple/None semantics, stream and window keys, idempotent async commit transaction, immutable records, and simulated commit latency.

"""Minimal final-weak Pauli-frame sink with explicitly priced writes.

The frame records corrections that are final for their window; an
escalation-pending weak result is withheld by design. The write cost is
charged before the commit body, so it also delays boundary handoff and the
potential-strong hold release; there is no outstanding-write queue.
"""
from dataclasses import dataclass
import math
from typing import Any, Callable, Optional

from ..config import us
from ..message import stable_identity_order_key


@dataclass(frozen=True)
class PauliFrameConfig:
    """Card selecting the minimal Pauli frame and pricing one frame write.

    A dropped duplicate performs no frame write and therefore charges no write
    cost; the drop is decided synchronously on arrival.
    """

    commit_us: float
    zero_commit_cost_justification: Optional[str] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.commit_us) or self.commit_us < 0:
            raise ValueError("commit_us must be a finite nonnegative number")
        if self.commit_us > 0 and us(self.commit_us) == 0:
            raise ValueError("commit_us is positive but rounds to zero ticks")
        if self.commit_us == 0 and not self.zero_commit_cost_justification:
            raise ValueError(
                "zero commit_us requires zero_commit_cost_justification"
            )
        if (
            self.commit_us > 0
            and self.zero_commit_cost_justification is not None
        ):
            raise ValueError(
                "zero_commit_cost_justification is only valid for zero commit_us"
            )

    def commit_ticks(self) -> int:
        """Return the configured write cost in integer ticks."""
        return us(self.commit_us)

    def resolve(self, engine) -> "PauliFrame":
        """Build this card's runtime frame on the run engine."""
        return PauliFrame(engine, commit_ticks=self.commit_ticks())


@dataclass(frozen=True)
class PauliFrameCommitRecord:
    """One accepted write, retained in acceptance order."""

    window_key: tuple
    tier: str
    run_sequence: int
    accepted_ticks: int
    committed_ticks: int
    logical_observables: Optional[tuple[int, ...]]


@dataclass(frozen=True)
class PauliFrameDuplicateDrop:
    """One defensively dropped arrival for an already accepted window.

    This cannot occur in the supported weak-final scope and is not evidence
    about strong-duplicate drops.
    """

    window_key: tuple
    tier: str
    run_sequence: int
    arrived_ticks: int


@dataclass(frozen=True)
class PauliFrameSnapshot:
    """Immutable report of the frame at one instant."""

    configured_commit_ticks: int
    commit_count: int
    duplicate_drop_count: int
    pending_write_count: int
    charged_ticks: int
    first_commit_ticks: Optional[int]
    last_commit_ticks: Optional[int]
    frames: tuple
    records: tuple
    duplicate_drops: tuple


@dataclass(frozen=True)
class _AcceptedWrite:
    """One reserved write awaiting installation."""

    record: PauliFrameCommitRecord
    on_committed: Callable[[], None]


class PauliFrame:
    """Accumulate committed weak corrections, charging every accepted write."""

    def __init__(self, engine, *, commit_ticks: int) -> None:
        self.engine = engine
        self.commit_ticks = commit_ticks
        self._accepted_window_keys: set = set()
        self._pending_by_window_key: dict[tuple, _AcceptedWrite] = {}
        self._entry_by_window_key: dict[tuple, PauliFrameCommitRecord] = {}
        self._window_keys_by_stream: dict[Any, list] = {}
        self._records: list[PauliFrameCommitRecord] = []
        self._duplicate_drops: list[PauliFrameDuplicateDrop] = []

    def commit_weak_correction(
        self,
        *,
        window_key,
        logical_observables,
        request_key,
        on_committed,
    ) -> None:
        """Accept one window correction once, charge its write, then continue."""
        if window_key in self._accepted_window_keys:
            self._duplicate_drops.append(
                PauliFrameDuplicateDrop(
                    window_key=window_key,
                    tier=request_key.tier.value,
                    run_sequence=request_key.run_sequence,
                    arrived_ticks=self.engine.now,
                )
            )
            return

        self._accepted_window_keys.add(window_key)
        self.engine.log_io(
            "PauliFrame",
            lambda: f"received {request_key.tier.value} correction for "
                    f"window {window_key}; logical observables "
                    f"{None if logical_observables is None else tuple(logical_observables)}")
        accepted_ticks = self.engine.now
        record = PauliFrameCommitRecord(
            window_key=window_key,
            tier=request_key.tier.value,
            run_sequence=request_key.run_sequence,
            accepted_ticks=accepted_ticks,
            committed_ticks=accepted_ticks + self.commit_ticks,
            logical_observables=(
                None
                if logical_observables is None
                else tuple(logical_observables)
            ),
        )
        accepted_write = _AcceptedWrite(record, on_committed)
        self._pending_by_window_key[window_key] = accepted_write
        self._records.append(record)

        if self.commit_ticks == 0:
            self._install_accepted_write(window_key)
            return
        self.engine.schedule(
            self.commit_ticks,
            lambda: self._install_accepted_write(window_key),
            label=f"pauli frame commit {window_key}",
        )

    def _install_accepted_write(self, window_key) -> None:
        accepted_write = self._pending_by_window_key.pop(window_key)
        record = accepted_write.record
        self._entry_by_window_key[window_key] = record
        stream_id = window_key[0]
        self._window_keys_by_stream.setdefault(stream_id, []).append(window_key)
        self.engine.log_io(
            "PauliFrame",
            lambda: f"committed window {window_key}; logical observables "
                    f"{record.logical_observables}; holds "
                    f"{len(self._entry_by_window_key)} window corrections")
        accepted_write.on_committed()

    # Adapted data core: non-destructive, zero-seeded, width-independent XOR.
    def frame_for(self, stream_id) -> Optional[tuple[int, ...]]:
        """Return one stream's non-destructive observable-frame fold."""
        window_keys = self._window_keys_by_stream.get(stream_id, ())
        records = tuple(
            self._entry_by_window_key[window_key]
            for window_key in window_keys
        )
        if not records:
            return ()
        if any(record.logical_observables is None for record in records):
            return None

        observable_arity = len(records[0].logical_observables)
        frame = [0] * observable_arity
        for record in records:
            logical_observables = record.logical_observables
            if len(logical_observables) != observable_arity:
                raise RuntimeError(
                    f"Pauli frame stream {stream_id!r} changed observable "
                    "arity during aggregation"
                )
            for observable_index, bit in enumerate(logical_observables):
                frame[observable_index] ^= bit
        return tuple(frame)

    def snapshot(self) -> PauliFrameSnapshot:
        """Return a frozen, non-destructive report of current frame state."""
        stream_ids = sorted(
            self._window_keys_by_stream,
            key=stable_identity_order_key,
        )
        frames = tuple(
            (stream_id, self.frame_for(stream_id))
            for stream_id in stream_ids
        )
        commit_ticks = tuple(record.committed_ticks for record in self._records)
        commit_count = len(self._accepted_window_keys)
        return PauliFrameSnapshot(
            configured_commit_ticks=self.commit_ticks,
            commit_count=commit_count,
            duplicate_drop_count=len(self._duplicate_drops),
            pending_write_count=len(self._pending_by_window_key),
            charged_ticks=commit_count * self.commit_ticks,
            first_commit_ticks=commit_ticks[0] if commit_ticks else None,
            last_commit_ticks=commit_ticks[-1] if commit_ticks else None,
            frames=frames,
            records=tuple(self._records),
            duplicate_drops=tuple(self._duplicate_drops),
        )
