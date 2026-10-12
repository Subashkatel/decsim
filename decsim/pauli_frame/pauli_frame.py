"""The Pauli frame remembers the corrections the decoders have made.

Each window gets one correction, one bit per logical observable, and a
second for the same window is refused. A stream's total correction is
its window corrections XORed together, as PECOS's frame accumulator
XORs each decode's observable mask
(crates/pecos-decoder-core/src/pauli_frame.rs:82). One correction per
window is the sliding-window commit: a window decides its commit region
once and hands on its effect on the logical operators (Skoric et al.
2209.08552 lines 102-105, 444-445). The frame never applies a correction
to a qubit, so the flush before a non-Clifford gate (Riesebos, TU Delft
MSc thesis CE-MS-2016, Sec. 3.2) has no counterpart here. A write's
caller is called back when the write lands on a clock edge.
"""

import dataclasses
import functools
from collections.abc import Callable
from typing import Any, Optional

import decsim.config as config
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.identity as identity_records
import decsim.records.log_sources as log_sources
import decsim.records.windows as window_records
import decsim.statistics.kinds as statistic_kinds
import decsim.statistics.registry as statistics_registry
import decsim.trace_source as trace_source

ObservableBits = tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class PauliFrameConfig:
    """The frame's settings: what one write costs.

    A write is one XOR into a register, one cycle of the frame unit: Yang
    et al. (2605.04892 Table I, line 1051) measure 4 ns per frame update
    inside a 550 ns loop, one cycle at 250 MHz. Writes to different windows
    are charged in parallel. clock None is the machine's clock.
    """

    write_cycles: int = 0
    clock: Optional[config.Clock] = None

    def __post_init__(self) -> None:
        config.check_cycles("pauli_frame.write_cycles", self.write_cycles)

    def build(self, engine: engine_module.Engine) -> "PauliFrame":
        """A fresh frame on these settings, on the run's engine."""
        return PauliFrame(
            engine, clock=self.clock, write_cycles=self.write_cycles
        )


@dataclasses.dataclass(frozen=True)
class PauliFrameSnapshot:
    """What the frame holds at one instant."""

    commit_count: int
    pending_write_count: int
    frames: tuple
    records: tuple


class PauliFrame:
    """Keeps every committed correction.

    It charges each write once. Its statistics are the corrections it
    accepted and each one's wait for its write, from acceptance to
    commit.
    """

    STATISTICS = (
        statistics_registry.Statistic(
            "commits",
            statistic_kinds.Count,
            "corrections accepted",
            "corrections",
        ),
        statistics_registry.Statistic(
            "write_wait",
            statistic_kinds.Distribution,
            "ticks from a correction's acceptance to its commit",
            "ticks",
        ),
    )
    statistics = statistics_registry.unrecorded(STATISTICS)

    def __init__(
        self,
        engine: engine_module.Engine,
        *,
        clock: Optional[config.Clock],
        write_cycles: int,
    ) -> None:
        self.engine = engine
        self.clock = clock
        self.write_cycles = write_cycles
        self._state = _FrameState()
        self.trace = _TraceSources()

    def commit_correction(
        self,
        *,
        window_key: tuple,
        logical_observables: Optional[ObservableBits],
        request_key: window_records.DecoderRequestKey,
        on_committed: Callable[[], None],
    ) -> None:
        """Accept a window's correction, charge the write, then call back."""
        self._refuse_let_go(window_key, request_key)
        if self._has_accepted(window_key):
            earlier_tier = self._tier_of(window_key)
            raise RuntimeError(
                f"window {window_key} already has a {earlier_tier} "
                f"correction; the {request_key.pool.word} correction "
                f"(request {request_key.run_sequence}) is a second write"
            )
        observables = _as_observable_bits(logical_observables)
        tier = request_key.pool.word
        self._log_received(tier, window_key, observables)
        accepted_ticks = self.engine.now
        committed_ticks = self._write_edge(accepted_ticks)
        record = decoding_records.PauliFrameCommitRecord(
            window_key=window_key,
            tier=tier,
            run_sequence=request_key.run_sequence,
            accepted_ticks=accepted_ticks,
            committed_ticks=committed_ticks,
            logical_observables=observables,
        )
        self._state.records.append(record)
        self.statistics.commits.add()
        write_wait = committed_ticks - accepted_ticks
        self.statistics.write_wait.add(write_wait)
        pending = _PendingWrite(record, on_committed)
        self._state.pending_by_window[window_key] = pending
        self.trace.correction_accepted.fire(record)
        self._charge_write(window_key)

    def frame_for_stream(
        self,
        stream_id: Any,  # an opaque identity
    ) -> Optional[ObservableBits]:
        """The XOR of every committed correction on one stream.

        None when any correction carries no observables, since a fold over an
        unknown value is unknown.
        """
        observables_per_record = self._observables_on(stream_id)
        if not observables_per_record:
            return ()
        has_unknown = any(
            observables is None for observables in observables_per_record
        )
        if has_unknown:
            return None
        return _fold_by_xor(observables_per_record, stream_id)

    def let_go_before(
        self,
        stream_id: Any,  # an opaque identity
        window_index: int,
    ) -> None:
        """Fold the stream's corrections before the window index into one.

        The windows are final (window_release.py): each took its one
        correction, and a write to one now is refused as a second write.
        The frame keeps their XOR only, as PECOS's accumulator keeps a
        running mask (crates/pecos-decoder-core/src/pauli_frame.rs:82);
        the snapshot's records are the kept ones.
        """
        is_let_go = functools.partial(_is_before, stream_id, window_index)
        observables_per_record = self._take_let_go(stream_id, is_let_go)
        earlier = self._state.let_go_by_stream.get(stream_id)
        if earlier is not None:
            observables_per_record.extend(earlier.folded)
        folded = _folded(observables_per_record, stream_id)
        let_go = _LetGo(window_index, folded)
        self._state.let_go_by_stream[stream_id] = let_go
        records = self._state.records
        self._state.records = _kept_records(records, is_let_go)

    def snapshot(self) -> PauliFrameSnapshot:
        """A frozen copy of what the frame holds now."""
        stream_ids = sorted(
            self._state.windows_by_stream,
            key=identity_records.stable_identity_bytes,
        )
        frames = []
        for stream_id in stream_ids:
            frame = self.frame_for_stream(stream_id)
            frames.append((stream_id, frame))
        return PauliFrameSnapshot(
            commit_count=len(self._state.records),
            pending_write_count=len(self._state.pending_by_window),
            frames=tuple(frames),
            records=tuple(self._state.records),
        )

    def _log_received(self, tier, window_key, observables) -> None:
        self.engine.log_io(
            log_sources.PAULI_FRAME,
            lambda: (
                f"received {tier} correction for window {window_key}; "
                f"logical observables {observables}"
            ),
        )

    def _charge_write(self, window_key) -> None:
        if self.write_cycles == 0:
            self._finish_write(window_key)
            return
        now = self.engine.now
        edge = self._write_edge(now)
        delay = edge - now
        self.engine.schedule(
            delay,
            lambda: self._finish_write(window_key),
            label=f"pauli frame commit {window_key}",
        )

    def _write_edge(self, now: int) -> int:
        """The clock edge the write started at `now` lands on."""
        if self.write_cycles == 0:
            return now
        return self.clock.edge(self.write_cycles, now)

    def _observables_on(self, stream_id) -> list:
        """Each kept correction's observables on the stream, and the fold."""
        window_keys = self._state.windows_by_stream.get(stream_id, ())
        observables_per_record = []
        for window_key in window_keys:
            record = self._state.committed_by_window[window_key]
            observables_per_record.append(record.logical_observables)
        let_go = self._state.let_go_by_stream.get(stream_id)
        if let_go is not None:
            observables_per_record.extend(let_go.folded)
        return observables_per_record

    def _take_let_go(self, stream_id, is_let_go) -> list:
        """The let-go windows' observables, their records taken out."""
        window_keys = self._state.windows_by_stream.get(stream_id, [])
        kept_keys = []
        observables_per_record = []
        for window_key in window_keys:
            if not is_let_go(window_key):
                kept_keys.append(window_key)
                continue
            record = self._state.committed_by_window.pop(window_key)
            observables_per_record.append(record.logical_observables)
        self._state.windows_by_stream[stream_id] = kept_keys
        return observables_per_record

    def _refuse_let_go(self, window_key, request_key) -> None:
        """A window the stream let go is final: a write to it is a second."""
        let_go = self._state.let_go_by_stream.get(window_key[0])
        if let_go is None or window_key[1] >= let_go.first_kept_index:
            return
        raise RuntimeError(
            f"window {window_key} is final and its correction folded; the "
            f"{request_key.pool.word} correction (request "
            f"{request_key.run_sequence}) is a second write"
        )

    def _has_accepted(self, window_key) -> bool:
        is_pending = window_key in self._state.pending_by_window
        is_committed = window_key in self._state.committed_by_window
        return is_pending or is_committed

    def _tier_of(self, window_key) -> str:
        pending = self._state.pending_by_window.get(window_key)
        if pending is not None:
            return pending.record.tier
        record = self._state.committed_by_window[window_key]
        return record.tier

    def _finish_write(self, window_key) -> None:
        pending = self._state.pending_by_window.pop(window_key)
        record = pending.record
        self._state.committed_by_window[window_key] = record
        # A window key starts with the stream it belongs to.
        stream_id = window_key[0]
        windows_on_stream = self._state.windows_by_stream.setdefault(
            stream_id, []
        )
        windows_on_stream.append(window_key)
        held = len(self._state.committed_by_window)
        self.engine.log_io(
            log_sources.PAULI_FRAME,
            lambda: (
                f"committed window {window_key}; logical observables "
                f"{record.logical_observables}; holds {held} window corrections"
            ),
        )
        self.trace.correction_committed.fire(record)
        pending.on_committed()


@dataclasses.dataclass(frozen=True)
class _PendingWrite:
    """A correction whose write cost is still being charged."""

    record: decoding_records.PauliFrameCommitRecord
    on_committed: Callable[[], None]


def _is_before(stream_id, window_index: int, window_key: tuple) -> bool:
    """Whether the window key is the stream's and before the index."""
    if window_key[0] != stream_id:
        return False
    return window_key[1] < window_index


def _kept_records(records: list, is_let_go) -> list:
    """The records of the windows not let go, in acceptance order."""
    return [record for record in records if not is_let_go(record.window_key)]


def _folded(observables_per_record: list, stream_id) -> tuple:
    """The corrections as one: their XOR, None when any is unknown.

    A tuple of that one, or empty when there was none to fold.
    """
    if not observables_per_record:
        return ()
    for observables in observables_per_record:
        if observables is None:
            return (None,)
    folded = _fold_by_xor(observables_per_record, stream_id)
    return (folded,)


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


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the pauli frame reports, as one member."""

    correction_accepted: trace_source.TraceSource = trace_source.new_source()
    correction_committed: trace_source.TraceSource = trace_source.new_source()


@dataclasses.dataclass(frozen=True)
class _LetGo:
    """A stream's corrections let go: where the kept windows start, the XOR.

    folded holds their XOR, or None when any was unknown, or is empty when
    none was folded.
    """

    first_kept_index: int
    folded: tuple


@dataclasses.dataclass
class _FrameState:
    """What the frame holds, as one member.

    records is every accepted correction in acceptance order, a write still
    landing included; pending_by_window the writes not yet landed;
    committed_by_window the windows already corrected; let_go_by_stream a
    stream's corrections it let go, folded (let_go_before). A stream id is
    opaque to the frame.
    """

    records: list = dataclasses.field(default_factory=list)
    pending_by_window: dict = dataclasses.field(default_factory=dict)
    committed_by_window: dict = dataclasses.field(default_factory=dict)
    windows_by_stream: dict = dataclasses.field(default_factory=dict)
    let_go_by_stream: dict = dataclasses.field(default_factory=dict)
