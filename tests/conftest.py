"""Shared test helpers."""

from decsim.message import Operation


def trace_time(lines, needle):
    """Timestamp (us, float) of the first trace line containing `needle`."""
    for line in lines:
        if needle in line:
            return float(line.split("us]")[0].lstrip("["))
    raise AssertionError(f"no trace line contains {needle!r}")


# ---- continuous-stream workload builder ------------------------------------
# Builds the segment operations of a continuous syndrome stream. Only tests
# construct stream workloads this way, so it lives here rather than in the
# package (real workloads arrive via a frontend).

def _stream_operation(circuit, *, stream_id: int, patch: int, name: str) -> Operation:
    """Create the synthetic decode unit for a continuous stream."""
    return Operation(stream_id, name, (patch,), clifford=True, circuit=circuit)


def _segment_operation(circuit, *, segment_id: int, stream_id: int,
                       segment_index: int, segment_count: int, offset: int,
                       patch: int, name: str) -> Operation:
    """Create one scheduled segment of a continuous stream."""
    predecessor = (segment_id - 1,) if segment_index > 0 else ()
    return Operation(
        segment_id,
        f"{name}[{segment_index}]",
        (patch,),
        clifford=True,
        circuit=circuit,
        stream_id=stream_id,
        stream_offset=offset,
        predecessors=predecessor,
        has_successor=(segment_index < segment_count - 1))


def _segments_and_rounds(circuit, segment_rounds, *, stream_id: int,
                         patch: int, base_id: int, name: str) -> tuple:
    """Create scheduled segment operations and the per-operation round map."""
    total_rounds = sum(segment_rounds)
    segments: list = []
    rounds_by_operation: dict = {stream_id: total_rounds}
    offset = 0

    for segment_index, rounds in enumerate(segment_rounds):
        segment_id = base_id + 1 + segment_index
        segment = _segment_operation(
            circuit, segment_id=segment_id, stream_id=stream_id,
            segment_index=segment_index, segment_count=len(segment_rounds),
            offset=offset, patch=patch, name=name)
        segments.append(segment)
        rounds_by_operation[segment_id] = int(rounds)
        offset += int(rounds)

    return segments, rounds_by_operation


def continuous_stream(circuit, segment_rounds, *, patch: int = 0, base_id: int = 0,
                      name: str = "stream"):
    """Build scheduled segments plus one decode operation for a continuous stream."""
    if not segment_rounds:
        raise ValueError("segment_rounds must be non-empty")
    stream_id = base_id
    stream_op = _stream_operation(
        circuit, stream_id=stream_id, patch=patch, name=name)
    segments, rounds_by_operation = _segments_and_rounds(
        circuit, segment_rounds, stream_id=stream_id,
        patch=patch, base_id=base_id, name=name)
    return segments, stream_op, rounds_by_operation
