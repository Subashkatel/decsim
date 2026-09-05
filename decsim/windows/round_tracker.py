"""The round tracker: which rounds arrived, and each window's readiness.

Readiness is a function of the arrivals and the planner's geometry: the
tracker builds a WindowReadiness and asks the scheme, the way qLDPC's
decode loop asks each planned window for its detectors
(qldpc/decoders/sinter.py, CompiledSequentialWindowDecoder). A window's
buffer that overflows the operation's end is satisfied by a successor's
rounds, memory rounds or a closed tail (Skoric et al. 2209.08552, the
artificial defects at the commit edge carry into the next window). The
room-side store's stored-through round per operation is counted here
too: the strong tier's readiness reads it, never Buffer 0's counter.
A stream's length knowledge (its source limit, its sealed length, its
closed feedback boundaries) lives here; its geometry is the planner's.
"""

from typing import Optional

import decsim.message as message


class RoundTracker:
    """The arrivals per operation, and each window's readiness."""

    def __init__(self, scheme, planner) -> None:
        self.scheme = scheme
        self.planner = planner
        self.operation_by_id: dict = {}
        self.arrivals_by_operation: dict = {}
        self.stream_by_id: dict = {}
        self.blocking_operation_ids: set = set()

    # ---- operations and streams

    def register_operation(self, operation: message.Operation) -> bool:
        """Track an operation's arrivals and feedback role.

        True the first time the operation is seen.
        """
        is_new = operation.id not in self.operation_by_id
        if is_new:
            self.arrivals_by_operation[operation.id] = _Arrivals()
        self.operation_by_id[operation.id] = operation
        if operation.blocked_by is not None:
            self.blocking_operation_ids.add(operation.blocked_by)
        return is_new

    def register_stream(
        self, stream_operation: message.Operation, source_round_limit
    ) -> None:
        """Track a stream's arrivals and what is known of its length."""
        stream_id = stream_operation.id
        self.operation_by_id[stream_id] = stream_operation
        if stream_id not in self.arrivals_by_operation:
            self.arrivals_by_operation[stream_id] = _Arrivals()
        self.stream_by_id[stream_id] = _StreamLength(source_round_limit)

    # ---- arrivals

    def note_arrival(self, operation_id, round_index: int) -> int:
        """Advance the readiness counter; returns the rounds arrived now."""
        arrivals = self.arrivals_by_operation[operation_id]
        arrivals.rounds = max(arrivals.rounds, round_index)
        return arrivals.rounds

    def note_memory_round(self, operation_id) -> int:
        """Record one idle or memory round; returns the count so far."""
        arrivals = self.arrivals_by_operation[operation_id]
        arrivals.memory_rounds += 1
        return arrivals.memory_rounds

    def note_room_round(self, operation_id, round_index: int) -> None:
        """Syndrome buffer 1 stored a round of the operation."""
        if operation_id not in self.arrivals_by_operation:
            self.arrivals_by_operation[operation_id] = _Arrivals()
        arrivals = self.arrivals_by_operation[operation_id]
        arrivals.strong_rounds = max(arrivals.strong_rounds, round_index)

    def rounds_arrived(self, operation_id) -> int:
        """The highest round the arrival authority has published."""
        arrivals = self.arrivals_by_operation.get(operation_id)
        if arrivals is None:
            return 0
        return arrivals.rounds

    def memory_rounds(self, operation_id) -> int:
        """The idle or memory rounds recorded for the operation."""
        return self.arrivals_by_operation[operation_id].memory_rounds

    def strong_rounds_arrived(self, operation_id) -> int:
        """The room-side store's stored-through round of the operation."""
        arrivals = self.arrivals_by_operation.get(operation_id)
        if arrivals is None:
            return 0
        return arrivals.strong_rounds

    # ---- a stream's length

    def is_sealed(self, operation_id) -> bool:
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

    def source_round_limit(self, stream_id) -> Optional[int]:
        """The rounds the stream's source can supply, None when unbounded."""
        return self.stream_by_id[stream_id].source_round_limit

    def reaches_source_limit(self, stream_id) -> bool:
        """Whether every round the source can supply has arrived."""
        stream = self.stream_by_id[stream_id]
        if stream.source_round_limit is None:
            return False
        arrived = self.rounds_arrived(stream_id)
        return arrived >= stream.source_round_limit

    def check_stream_length(self, stream_id, stream_round_count: int) -> None:
        """Refuse a stream longer than its source can supply."""
        stream_operation = self.operation_by_id[stream_id]
        self.planner.models.check_stream_length(
            stream_operation, stream_round_count
        )

    def seal(self, stream_id, stream_round_count: int) -> None:
        """The stream's full length has arrived."""
        self.stream_by_id[stream_id].sealed_round_count = stream_round_count

    def arrival_round_limit(self, operation_id) -> Optional[int]:
        """Maximum legal device round, or None for an open unbounded stream."""
        stream = self.stream_by_id.get(operation_id)
        if stream is None:
            return self.planner.round_count_of(operation_id)
        if stream.sealed_round_count is not None:
            return stream.sealed_round_count
        return stream.source_round_limit

    def close_boundary(self, stream_id, stream_round_count: int) -> None:
        """Mark a live stream round as a measurement-closed boundary.

        A finite real-syndrome stream cannot close a boundary inside its
        registered circuit: that is a front call's mistake.
        """
        stream = self.stream_by_id[stream_id]
        stream.refuse_boundary_inside_finite_source(stream_round_count)
        stream.closed_boundaries.add(stream_round_count)

    # ---- round counts and readiness

    def round_count_for_window(
        self, operation_id, window: Optional[message.Window] = None
    ) -> int:
        """The round count to check or read one window against.

        A non-stream operation has its planned rounds; a sealed stream its
        sealed length; a capped stream its source limit; an open stream
        the window's own read end, or the rounds arrived so far.
        """
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
        self, operation_id, window: Optional[message.Window]
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
        self, window: message.Window
    ) -> Optional[int]:
        """The closed boundary in the window's trailing buffer, if any.

        A stream's closed feedback boundary first; else, in
        measurement_closed mode, a feedback source's own last round when
        it falls inside the buffer.
        """
        stream = self.stream_by_id.get(window.op_id)
        if stream is not None:
            stream_boundary = stream.closed_boundary_for_window(window)
            if stream_boundary is not None:
                return stream_boundary
        operation = self.operation_by_id[window.op_id]
        if operation.feedback_boundary_mode != "measurement_closed":
            return None
        if operation.id not in self.blocking_operation_ids:
            return None
        round_count = self.round_count_for_window(window.op_id, window)
        if window.commit_hi <= round_count < window.buffer_hi:
            return round_count
        return None

    def readiness(self, window: message.Window) -> message.WindowReadiness:
        """What the scheme sees when deciding whether a window has its data."""
        successor_ids = sorted(
            self.planner.successors_by_operation[window.op_id],
            key=message.stable_identity_order_key,
        )
        successors = []
        for successor_id in successor_ids:
            arrived = self.rounds_arrived(successor_id)
            round_count = self.round_count_for_window(successor_id)
            successor = message.SuccessorReadiness(
                successor_id, arrived, round_count
            )
            successors.append(successor)
        local_round_count = self.effective_round_count_for_window(
            window.op_id, window
        )
        closed_boundary = self.closed_boundary_round_for_window(window)
        is_tail_closed = closed_boundary is not None
        local_rounds_arrived = self.rounds_arrived(window.op_id)
        memory_rounds_arrived = self.memory_rounds(window.op_id)
        return message.WindowReadiness(
            local_rounds_arrived=local_rounds_arrived,
            local_round_count=local_round_count,
            successors=tuple(successors),
            memory_rounds_arrived=memory_rounds_arrived,
            tail_closed=is_tail_closed,
        )

    def is_data_complete(self, window: message.Window) -> bool:
        """Whether the window has every round it reads, by the scheme's rule."""
        readiness = self.readiness(window)
        return self.scheme.data_complete(window, readiness=readiness)

    def has_first_round(self, window: message.Window) -> bool:
        """Whether the window's first read round has arrived."""
        arrived = self.rounds_arrived(window.op_id)
        return arrived >= window.start_round


class _Arrivals:
    """One operation's arrival counters."""

    def __init__(self) -> None:
        self.rounds = 0
        self.memory_rounds = 0
        self.strong_rounds = 0


class _StreamLength:
    """What is known of one stream's length, and its closed boundaries."""

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
        self, window: message.Window
    ) -> Optional[int]:
        """The earliest closed boundary in the window's trailing buffer."""
        covered = []
        for boundary in self.closed_boundaries:
            if window.commit_hi <= boundary < window.buffer_hi:
                covered.append(boundary)
        if not covered:
            return None
        return min(covered)
