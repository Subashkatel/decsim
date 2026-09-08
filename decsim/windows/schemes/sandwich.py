"""The sandwich row: Tan et al.'s zero-seam sandwich decoder.

Tan et al. 2209.09219 (supplement S8): type-1 cores of width
w = s + 2b every s rounds, with a one-layer type-2 seam between each
adjacent pair. commit_round_count is the paper's step s and
buffer_round_count is the two-sided buffer b; the seam offset is t = 0.
The typed detector intervals are the vertex sets whose incident
correction edges form each core.
"""

import math

import decsim.records.windows as window_records
import decsim.windows.schemes.window_data as window_data


class TanSandwichScheme:
    """Tan et al.'s zero-seam sandwich decoder for graphlike memory DEMs."""

    scheme_label = (
        "Tan zero-seam sandwich (type-1 cores / type-2 seam reconciliation)"
    )

    def plan_operation(
        self,
        operation_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ) -> window_records.OperationWindowPlan:
        """Type-1 cores every s rounds with a type-2 seam between each pair."""
        step = commit_round_count
        buffer = buffer_round_count
        _check_step_and_buffer(step, buffer)
        width = step + 2 * buffer
        type_1_count = _type_1_count(round_count, step, width)
        cores = _TanCores(round_count, step, buffer, width, type_1_count)
        windows, edges = _cores_and_seams(cores)
        window_count = len(windows)
        entry_window_indices = tuple(range(0, window_count, 2))
        exit_window_indices = _seam_indices(window_count)
        return window_records.OperationWindowPlan(
            operation_id=operation_id,
            windows=windows,
            internal_dependencies=edges,
            entry_window_indices=entry_window_indices,
            exit_window_indices=exit_window_indices,
            windowed=True,
            batch_preceding_idle_rounds=False,
            protocol=window_records.WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE,
        )

    def data_complete(
        self,
        window: window_records.Window,
        *,
        readiness: window_records.WindowReadiness,
    ) -> bool:
        """Whether the window has every round it reads."""
        return window_data.sliding_data_complete(window, readiness)

    def validate_buffer(self, geometry) -> None:
        """Reject a step below 2 or a buffer below 1."""
        _check_step_and_buffer(
            geometry.commit_round_count, geometry.buffer_round_count
        )


class _TanCores:
    """The type-1 windows of one Tan schedule, by index."""

    def __init__(self, round_count, step, buffer, width, type_1_count):
        self.round_count = round_count
        self.step = step
        self.buffer = buffer
        self.width = width
        self.type_1_count = type_1_count

    def type_1_window(self, index: int) -> window_records.WindowGeometry:
        """Core index reads w rounds from 1 + index*s, commits the middle s."""
        read_lo = 1 + index * self.step
        commit_lo = read_lo + self.buffer
        if index == 0:
            commit_lo = 1
        commit_hi = self.buffer + self.step - 1 + index * self.step
        if index == self.type_1_count - 1:
            commit_hi = self.round_count
        read_end = read_lo + self.width - 1
        buffer_hi = min(self.round_count, read_end)
        return window_records.WindowGeometry(
            buffer_lo=read_lo,
            commit_lo=commit_lo,
            commit_hi=commit_hi,
            buffer_hi=buffer_hi,
        )

    def seam_window(self, left_index: int) -> window_records.WindowGeometry:
        """The one-layer type-2 seam to the right of core left_index."""
        seam_round = self.buffer + self.step + left_index * self.step
        return window_records.WindowGeometry(
            seam_round,
            seam_round,
            seam_round,
            seam_round,
            closed_temporal_boundaries=True,
        )


def _check_step_and_buffer(step: int, buffer: int) -> None:
    """The paper's schedule needs s >= 2 and overlapping windows, b >= 1."""
    if step < 2:
        raise ValueError("Tan sandwich decoding requires step size s >= 2")
    if buffer < 1:
        raise ValueError(
            "Tan sandwich decoding requires overlapping windows (b >= 1)"
        )


def _type_1_count(round_count: int, step: int, width: int) -> int:
    """How many cores of width w cover the operation, striding s."""
    span = round_count - width
    span_in_steps = span / step
    type_1_count = math.ceil(span_in_steps)
    type_1_count += 1
    return max(1, type_1_count)


def _cores_and_seams(cores: _TanCores) -> tuple:
    """The cores and the seams between them, and the edges each seam waits on.

    Windows alternate core, seam, core, and a seam depends on the core
    on each side of it.
    """
    windows = [cores.type_1_window(0)]
    edges = []
    seam_count = cores.type_1_count - 1
    for left_type_1 in range(seam_count):
        seam_index = len(windows)
        seam = cores.seam_window(left_type_1)
        windows.append(seam)
        right_type_1_index = len(windows)
        right_index = left_type_1 + 1
        right_type_1 = cores.type_1_window(right_index)
        windows.append(right_type_1)
        left_index = seam_index - 1
        edges.append((left_index, seam_index))
        edges.append((right_type_1_index, seam_index))
    return tuple(windows), tuple(edges)


def _seam_indices(window_count: int) -> tuple:
    """The seams are the exits; a schedule with one core exits on it."""
    seam_indices = tuple(range(1, window_count, 2))
    if not seam_indices:
        return (0,)
    return seam_indices
