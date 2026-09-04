"""The Pauli frame remembers the corrections the decoders have made.

Each window gets one correction: a few bits, one per logical observable.
The frame stores that correction and refuses a second one for the same
window. A stream's total correction is all of its window corrections
XORed together.

Every write costs a fixed number of ticks. The caller is called back only
after that time has passed, so anything waiting on the write waits too.

The XOR fold follows PECOS's Pauli frame accumulator. Applying a correction
exactly once follows Riesebos (DAC 2017). One write costs one clock cycle,
4 ns at 250 MHz (Yang et al. 2605.04892).
"""

import dataclasses
import math
from collections.abc import Mapping
from typing import Any, Callable, Optional

import decsim.config as config
import decsim.message as message

ObservableBits = tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class PauliFrameConfig:
    """The frame's settings: what one write costs.

    A write is one XOR into a register, one clock cycle of the frame unit.
    Yang et al. (2605.04892, Fig. 1) measure 4 ns per frame update inside a
    550 ns loop, one cycle at 250 MHz. Writes to different windows are
    charged in parallel, never queued behind each other.
    """

    commit_microseconds: float
    zero_commit_cost_justification: Optional[str] = None

    def __post_init__(self) -> None:
        cost = self.commit_microseconds
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(
                "commit_microseconds must be finite and not negative"
            )
        ticks = config.microseconds_to_ticks(cost)
        if cost > 0 and ticks == 0:
            raise ValueError(
                "commit_microseconds is positive but rounds to zero ticks"
            )
        is_free = cost == 0
        has_justification = bool(self.zero_commit_cost_justification)
        if is_free and not has_justification:
            raise ValueError(
                "a free write needs zero_commit_cost_justification"
            )
        if not is_free and has_justification:
            raise ValueError(
                "zero_commit_cost_justification needs a free write"
            )

    @classmethod
    def from_yaml(
        cls, section: Mapping, clocks: config.ClockSettings
    ) -> "PauliFrameConfig":
        """The `pauli_frame` section: write_cycles on its clock."""
        clock = section["clock"]
        write_cycles = section["write_cycles"]
        commit_microseconds = clocks.microseconds(write_cycles, clock)
        return cls(commit_microseconds=commit_microseconds)

    def commit_ticks(self) -> int:
        """The write cost in ticks."""
        return config.microseconds_to_ticks(self.commit_microseconds)

    def resolve(self, engine) -> "PauliFrame":
        """Build the frame these settings describe, on the run's engine."""
        commit_ticks = self.commit_ticks()
        return PauliFrame(engine, commit_ticks=commit_ticks)


@dataclasses.dataclass(frozen=True)
class PauliFrameCommitRecord:
    """One accepted correction, kept in the order the frame accepted it."""

    window_key: tuple
    tier: str
    run_sequence: int
    accepted_ticks: int
    committed_ticks: int
    logical_observables: Optional[ObservableBits]


@dataclasses.dataclass(frozen=True)
class PauliFrameSnapshot:
    """What the frame holds at one instant."""

    configured_commit_ticks: int
    commit_count: int
    pending_write_count: int
    charged_ticks: int
    first_commit_ticks: Optional[int]
    last_commit_ticks: Optional[int]
    frames: tuple
    records: tuple


class PauliFrame:
    """Keeps every committed correction and charges each write once."""

    def __init__(self, engine, *, commit_ticks: int) -> None:
        self.engine = engine
        self.commit_ticks = commit_ticks
        self._records: list[PauliFrameCommitRecord] = []
        self._pending_by_window: dict[tuple, _PendingWrite] = {}
        self._committed_by_window: dict[tuple, PauliFrameCommitRecord] = {}
        # A stream id is whatever the front end chose; the frame never
        # looks inside it.
        self._windows_by_stream: dict[Any, list[tuple]] = {}

    def commit_correction(
        self, *, window_key, logical_observables, request_key, on_committed
    ) -> None:
        """Accept a window's correction, charge the write, then call back."""
        if self._has_accepted(window_key):
            earlier_tier = self._tier_of(window_key)
            raise RuntimeError(
                f"window {window_key} already has a {earlier_tier} "
                f"correction; the {request_key.tier.value} correction "
                f"(request {request_key.run_sequence}) is a second write"
            )
        observables = _as_observable_bits(logical_observables)
        tier = request_key.tier.value
        self._log_received(tier, window_key, observables)
        accepted_ticks = self.engine.now
        committed_ticks = accepted_ticks + self.commit_ticks
        record = PauliFrameCommitRecord(
            window_key=window_key,
            tier=tier,
            run_sequence=request_key.run_sequence,
            accepted_ticks=accepted_ticks,
            committed_ticks=committed_ticks,
            logical_observables=observables,
        )
        self._records.append(record)
        pending = _PendingWrite(record, on_committed)
        self._pending_by_window[window_key] = pending
        self._charge_write(window_key)

    def frame_for_stream(self, stream_id) -> Optional[ObservableBits]:
        """The XOR of every committed correction on one stream.

        Empty when the stream has no corrections yet. None when any of its
        corrections carries no observables, since a fold over an unknown
        value is unknown.
        """
        window_keys = self._windows_by_stream.get(stream_id, ())
        records = [self._committed_by_window[key] for key in window_keys]
        if not records:
            return ()
        observables_per_record = [
            record.logical_observables for record in records
        ]
        has_unknown = any(
            observables is None for observables in observables_per_record
        )
        if has_unknown:
            return None
        return _fold_by_xor(observables_per_record, stream_id)

    def snapshot(self) -> PauliFrameSnapshot:
        """A frozen copy of what the frame holds now."""
        stream_ids = sorted(
            self._windows_by_stream, key=message.stable_identity_order_key
        )
        frames = []
        for stream_id in stream_ids:
            frame = self.frame_for_stream(stream_id)
            frames.append((stream_id, frame))
        commit_ticks = [record.committed_ticks for record in self._records]
        commit_count = len(self._records)
        first_commit_ticks = None
        last_commit_ticks = None
        if commit_ticks:
            first_commit_ticks = commit_ticks[0]
            last_commit_ticks = commit_ticks[-1]
        charged_ticks = commit_count * self.commit_ticks
        return PauliFrameSnapshot(
            configured_commit_ticks=self.commit_ticks,
            commit_count=commit_count,
            pending_write_count=len(self._pending_by_window),
            charged_ticks=charged_ticks,
            first_commit_ticks=first_commit_ticks,
            last_commit_ticks=last_commit_ticks,
            frames=tuple(frames),
            records=tuple(self._records),
        )

    def _log_received(self, tier, window_key, observables) -> None:
        self.engine.log_io(
            "PauliFrame",
            lambda: (
                f"received {tier} correction for window {window_key}; "
                f"logical observables {observables}"
            ),
        )

    def _charge_write(self, window_key) -> None:
        if self.commit_ticks == 0:
            self._finish_write(window_key)
            return
        self.engine.schedule(
            self.commit_ticks,
            lambda: self._finish_write(window_key),
            label=f"pauli frame commit {window_key}",
        )

    def _has_accepted(self, window_key) -> bool:
        is_pending = window_key in self._pending_by_window
        is_committed = window_key in self._committed_by_window
        return is_pending or is_committed

    def _tier_of(self, window_key) -> str:
        pending = self._pending_by_window.get(window_key)
        if pending is not None:
            return pending.record.tier
        record = self._committed_by_window[window_key]
        return record.tier

    def _finish_write(self, window_key) -> None:
        pending = self._pending_by_window.pop(window_key)
        record = pending.record
        self._committed_by_window[window_key] = record
        # A window key starts with the stream it belongs to.
        stream_id = window_key[0]
        windows_on_stream = self._windows_by_stream.setdefault(stream_id, [])
        windows_on_stream.append(window_key)
        held = len(self._committed_by_window)
        self.engine.log_io(
            "PauliFrame",
            lambda: (
                f"committed window {window_key}; logical observables "
                f"{record.logical_observables}; holds {held} window corrections"
            ),
        )
        pending.on_committed()


@dataclasses.dataclass(frozen=True)
class _PendingWrite:
    """A correction whose write cost is still being charged."""

    record: PauliFrameCommitRecord
    on_committed: Callable[[], None]


def _as_observable_bits(logical_observables) -> Optional[ObservableBits]:
    if logical_observables is None:
        return None
    return tuple(logical_observables)


def _fold_by_xor(
    observables_per_record: list[ObservableBits], stream_id
) -> ObservableBits:
    """XOR corrections bit by bit; every correction must have the same width."""
    width = len(observables_per_record[0])
    folded = [0] * width
    for observables in observables_per_record:
        if len(observables) != width:
            raise RuntimeError(
                f"Pauli frame stream {stream_id!r} changed its number of "
                f"observables mid-run"
            )
        for index, bit in enumerate(observables):
            folded[index] ^= bit
    return tuple(folded)
