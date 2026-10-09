"""The round tracker: which rounds arrived, and each window's readiness.

Readiness is the arrivals read against the planner's geometry: the
tracker builds a WindowReadiness and asks the scheme, as qLDPC's decode
loop asks each planned window for its detectors
(qldpc/decoders/sinter.py, CompiledSequentialWindowDecoder). The strong
tier's readiness reads the strong syndrome buffer's stored-through
round, counted here, never the weak one's. A stream's length (its source
limit, sealed length and closed boundaries) lives here; its geometry is
the planner's.
"""

import functools
from collections.abc import Callable
from typing import Any, Optional

import decsim.ports as ports
import decsim.records.identity as identity_records
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.windows.schemes.window_data as window_data
import decsim.windows.window_planner as window_planner


class RoundTracker:
    """Each window's readiness, from the rounds that arrived.

    It counts the arrivals per operation.
    """

    scheme = ports.Port(ports.WindowingScheme)
    planner = ports.Port(window_planner.WindowPlanner)

    def __init__(self) -> None:
        self.operation_by_id: dict = {}
        self.arrivals_by_operation: dict = {}
        self.stream_by_id: dict = {}
        self.blocking_operation_ids: set = set()

    # ---- operations and streams

    def register_operation(self, operation: program_records.Operation) -> bool:
        """Track an operation's arrivals and feedback role; True if new."""
        is_new = operation.id not in self.operation_by_id
        if is_new:
            self.arrivals_by_operation[operation.id] = _Arrivals()
        self.operation_by_id[operation.id] = operation
        if operation.blocked_by is not None:
            self.blocking_operation_ids.add(operation.blocked_by)
        return is_new

    def register_stream(
        self,
        stream_operation: program_records.Operation,
        source_round_limit: Optional[int],
    ) -> None:
        """Track a stream's arrivals and what is known of its length."""
        stream_id = stream_operation.id
        self.operation_by_id[stream_id] = stream_operation
        if stream_id not in self.arrivals_by_operation:
            self.arrivals_by_operation[stream_id] = _Arrivals()
        self.stream_by_id[stream_id] = _StreamLength(source_round_limit)

    # ---- arrivals

    def note_arrival(
        self,
        operation_id: Any,  # an opaque identity
        round_index: int,
    ) -> int:
        """Advance the readiness counter; returns the rounds arrived now.

        The counter is the rounds arrived from round 1 without a gap. A
        store with several write ports completes a narrow round before a
        wider one written ahead of it, so a round past a gap waits here
        until the gap fills, as gem5's reorder buffer retires only its
        head once it is ready (src/cpu/o3/rob.cc isHeadReady).
        """
        arrivals = self.arrivals_by_operation[operation_id]
        arrivals.rounds_past_a_gap.add(round_index)
        next_round = arrivals.rounds + 1
        while next_round in arrivals.rounds_past_a_gap:
            arrivals.rounds_past_a_gap.remove(next_round)
            arrivals.rounds = next_round
            next_round = arrivals.rounds + 1
        return arrivals.rounds

    def note_memory_round(self, operation_id: Any) -> int:  # an opaque identity
        """Record one idle or memory round; returns the count so far."""
        arrivals = self.arrivals_by_operation[operation_id]
        arrivals.memory_rounds += 1
        return arrivals.memory_rounds

    def note_room_round(
        self,
        operation_id: Any,  # an opaque identity
        round_index: int,
    ) -> None:
        """The strong syndrome buffer stored a round of the operation."""
        if operation_id not in self.arrivals_by_operation:
            self.arrivals_by_operation[operation_id] = _Arrivals()
        arrivals = self.arrivals_by_operation[operation_id]
        arrivals.strong_rounds = max(arrivals.strong_rounds, round_index)

    def operation(
        self,
        operation_id: Any,  # an opaque identity
    ) -> program_records.Operation:
        """The operation record of that id."""
        return self.operation_by_id[operation_id]

    def rounds_arrived(self, operation_id: Any) -> int:  # an opaque identity
        """The rounds the arrival authority has published from round 1."""
        arrivals = self.arrivals_by_operation.get(operation_id)
        if arrivals is None:
            return 0
        return arrivals.rounds

    def memory_rounds(self, operation_id: Any) -> int:  # an opaque identity
        """The idle or memory rounds recorded for the operation."""
        return self.arrivals_by_operation[operation_id].memory_rounds

    def strong_rounds_arrived(
        self,
        operation_id: Any,  # an opaque identity
    ) -> int:
        """The operation's round stored through the strong syndrome buffer."""
        arrivals = self.arrivals_by_operation.get(operation_id)
        if arrivals is None:
            return 0
        return arrivals.strong_rounds

    # ---- a stream's length

    def is_sealed(self, operation_id: Any) -> bool:  # an opaque identity
        """True for non-streams and sealed streams."""
        stream = self.stream_by_id.get(operation_id)
        if stream is None:
            return True
        return stream.sealed_round_count is not None

    def has_unsealed_streams(self) -> bool:
        """True while any registered stream has not sealed."""
        for stream in self.stream_by_id.values():
            if stream.sealed_round_count is None:
                return True
        return False

    def source_round_limit(
        self,
        stream_id: Any,  # an opaque identity
    ) -> Optional[int]:
        """The rounds the stream's source can supply, None when unbounded."""
        return self.stream_by_id[stream_id].source_round_limit

    def reaches_source_limit(
        self,
        stream_id: Any,  # an opaque identity
    ) -> bool:
        """Whether every round the source can supply has arrived."""
        stream = self.stream_by_id[stream_id]
        if stream.source_round_limit is None:
            return False
        arrived = self.rounds_arrived(stream_id)
        return arrived >= stream.source_round_limit

    def seal(
        self,
        stream_id: Any,  # an opaque identity
        stream_round_count: int,
    ) -> None:
        """The stream's full length has arrived."""
        self.stream_by_id[stream_id].sealed_round_count = stream_round_count

    def arrival_round_limit(
        self,
        operation_id: Any,  # an opaque identity
    ) -> Optional[int]:
        """Maximum legal device round, or None for an open unbounded stream."""
        stream = self.stream_by_id.get(operation_id)
        if stream is None:
            return self.planner.round_count_of(operation_id)
        if stream.sealed_round_count is not None:
            return stream.sealed_round_count
        return stream.source_round_limit

    def close_boundary(
        self,
        stream_id: Any,  # an opaque identity
        stream_round_count: int,
    ) -> None:
        """Mark a live stream round as a measurement-closed boundary.

        A finite real-syndrome stream refuses one inside its registered
        circuit.
        """
        stream = self.stream_by_id[stream_id]
        stream.refuse_boundary_inside_finite_source(stream_round_count)
        stream.closed_boundaries.add(stream_round_count)

    # ---- round counts and readiness

    def round_count_for_window(
        self,
        operation_id: Any,  # an opaque identity
        window: Optional[window_records.Window] = None,
    ) -> int:
        """The round count to check or read one window against."""
        stream = self.stream_by_id.get(operation_id)
        if stream is None:
            return self.planner.round_count_of(operation_id)
        if stream.sealed_round_count is not None:
            return stream.sealed_round_count
        if stream.source_round_limit is not None:
            return stream.source_round_limit
        if window is not None:
            return window.buffer_hi
        return self.rounds_arrived(operation_id)

    def effective_round_count_for_window(
        self,
        operation_id: Any,  # an opaque identity
        window: Optional[window_records.Window],
    ) -> int:
        """The round count a window reads against, clipped at a closed tail."""
        round_count = self.round_count_for_window(operation_id, window)
        if window is None:
            return round_count
        closed = self.closed_boundary_round_for_window(window)
        if closed is None:
            return round_count
        return min(round_count, closed)

    def closed_boundary_round_for_window(
        self, window: window_records.Window
    ) -> Optional[int]:
        """The closed boundary in the window's trailing buffer, if any."""
        stream = self.stream_by_id.get(window.operation_id)
        if stream is not None:
            stream_boundary = stream.closed_boundary_for_window(window)
            if stream_boundary is not None:
                return stream_boundary
        operation = self.operation_by_id[window.operation_id]
        if operation.feedback_boundary_mode != "measurement_closed":
            return None
        if operation.id not in self.blocking_operation_ids:
            return None
        round_count = self.round_count_for_window(window.operation_id, window)
        if window.commit_hi <= round_count < window.buffer_hi:
            return round_count
        return None

    def readiness(
        self, window: window_records.Window
    ) -> window_records.WindowReadiness:
        """What the scheme sees when deciding whether a window has its data."""
        return self._readiness(window, self.rounds_arrived)

    def is_data_complete(self, window: window_records.Window) -> bool:
        """Whether the window has every round it reads, by the scheme's rule."""
        readiness = self.readiness(window)
        return self.scheme.data_complete(window, readiness=readiness)

    def is_read_out_whole(
        self,
        window: window_records.Window,
        rounds_read_out_by_operation: dict,
    ) -> bool:
        """Whether the window would have its data were its read-out rounds in.

        The scheme's own rule, on every operation's rounds read out at the
        QPU in place of those arrived in the store, its successors' too, so
        the readout path's transit and a round the controller holds for
        store room are not part of it. A strong region decodes an absorbed
        window's commit rounds, so its own buffer is no part of it.
        """
        read_out = rounds_read_out_by_operation.get(window.operation_id, 0)
        # a window reads its own commit rounds, and one whose data landed
        # was read out, so these answer most windows without a readiness,
        # which the backlog sampler would build after every action
        if window.commit_hi > read_out:
            return False
        if window.t_data_complete is not None:
            return True
        if window.is_absorbed:
            return True
        rounds_of = functools.partial(
            _rounds_read_out, rounds_read_out_by_operation
        )
        readiness = self._readiness(window, rounds_of)
        return self.scheme.data_complete(window, readiness=readiness)

    def is_buffer_filled_by_memory(self, window: window_records.Window) -> bool:
        """Whether memory rounds alone satisfy the buffer past the operation."""
        readiness = self.readiness(window)
        return window_data.buffer_filled_by_memory_only(window, readiness)

    def has_first_round(self, window: window_records.Window) -> bool:
        """Whether the window's first read round has arrived."""
        arrived = self.rounds_arrived(window.operation_id)
        return arrived >= window.start_round

    def _readiness(
        self,
        window: window_records.Window,
        rounds_of: Callable[[Any], int],
    ) -> window_records.WindowReadiness:
        """The readiness with each operation's rounds counted by rounds_of.

        Memory rounds keep their store count: the QPU reads them out under
        an idle identity of their own and they carry no syndrome, so a
        buffer they fill is whole once they land.
        """
        successor_ids = sorted(
            self.planner.successors_by_operation[window.operation_id],
            key=identity_records.stable_identity_bytes,
        )
        successors = []
        for successor_id in successor_ids:
            arrived = rounds_of(successor_id)
            round_count = self.round_count_for_window(successor_id)
            successor = window_records.SuccessorReadiness(
                successor_id, arrived, round_count
            )
            successors.append(successor)
        local_round_count = self.effective_round_count_for_window(
            window.operation_id, window
        )
        closed_boundary = self.closed_boundary_round_for_window(window)
        is_tail_closed = closed_boundary is not None
        local_rounds_arrived = rounds_of(window.operation_id)
        memory_rounds_arrived = self.memory_rounds(window.operation_id)
        return window_records.WindowReadiness(
            local_rounds_arrived=local_rounds_arrived,
            local_round_count=local_round_count,
            successors=tuple(successors),
            memory_rounds_arrived=memory_rounds_arrived,
            tail_closed=is_tail_closed,
        )


class _Arrivals:
    """One operation's arrival counters."""

    def __init__(self) -> None:
        self.rounds = 0
        # published rounds above the first round still missing
        self.rounds_past_a_gap: set = set()
        self.memory_rounds = 0
        self.strong_rounds = 0


class _StreamLength:
    """What is known of one stream's length."""

    def __init__(self, source_round_limit: Optional[int]) -> None:
        self.source_round_limit = source_round_limit
        self.sealed_round_count: Optional[int] = None
        self.closed_boundaries: set = set()

    def refuse_boundary_inside_finite_source(
        self, stream_round_count: int
    ) -> None:
        """Reject internal closed boundaries in finite real-syndrome streams."""
        if self.source_round_limit is None:
            return
        if stream_round_count >= self.source_round_limit:
            return
        raise RuntimeError(
            "measurement_closed live-stream boundaries inside a finite "
            "real-syndrome stream need a source circuit with that destructive "
            "boundary. A continuous stream model was registered for "
            f"{self.source_round_limit} rounds, but the feedback boundary "
            f"closes at round {stream_round_count}. Use timing-only payloads "
            "for this timing study, split the workload into finite operation "
            "circuits, or provide a boundary-aware syndrome source."
        )

    def closed_boundary_for_window(
        self, window: window_records.Window
    ) -> Optional[int]:
        """The earliest closed boundary in the window's trailing buffer."""
        covered = []
        for boundary in self.closed_boundaries:
            if window.commit_hi <= boundary < window.buffer_hi:
                covered.append(boundary)
        if not covered:
            return None
        return min(covered)


def _rounds_read_out(
    rounds_read_out_by_operation: dict,
    operation_id: Any,  # an opaque identity
) -> int:
    return rounds_read_out_by_operation.get(operation_id, 0)
