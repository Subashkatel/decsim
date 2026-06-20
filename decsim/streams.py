from __future__ import annotations

from typing import Any

from .message import Operation

# ==================================================================================
# CONTINUOUS STREAMS (3b / 5-real)
# A continuous patch is one logical qubit whose syndrome stream is UNBROKEN across several
# operations (no destructive measurement between them). It is decoded as ONE continuous record
# -- one circuit, one window sequence, one observable -- spanning the operation seams (the
# correctness anchor: a windowed decode of the continuous record equals the global decode,
# Skoric arXiv:2209.08552 App C; the interior seams are OPEN boundaries, Tan arXiv:2209.09219).
#
# decsim separates the DECODE unit from the SCHEDULING unit: the chip schedules the segment
# operations (each with its own rounds / place in the DAG), while the cluster windows and decodes
# the single STREAM. The device tags each segment's rounds to the stream at the right global
# round, so the cluster sees one continuous record with no special routing.
# ==================================================================================


def continuous_stream(circuit, segment_rounds, *, patch: int = 0, base_id: int = 0,
                      name: str = "stream"):
    """Build a continuous-patch workload from one continuous `circuit` and the per-segment round
    counts `segment_rounds` (summing to the circuit's total rounds). Pass ``circuit=None`` for a
    timing-only stream: the same decode windows/dependencies are planned, but payloads carry no
    syndrome bits and no window error models are built.

    Returns ``(segments, stream_op, rounds_by_op)``:
      - ``segments``     -- the SCHEDULING ops the chip runs (``build_and_run(ops=segments, ...)``);
                            each emits its own rounds, which the StimDevice tags to the stream.
      - ``stream_op``    -- the DECODE unit (``build_and_run(decode_ops=[stream_op], ...)``);
                            windowed over the whole continuous circuit/round stream.
      - ``rounds_by_op`` -- ``{op_id: rounds}`` for ``PerOpRounds`` (stream -> R_total, each
                            segment -> its own length).

    Segments run sequentially on one patch (segment i+1's predecessor is segment i); there is no
    destructive measurement between them, so the seams are open and the decode is continuous.
    """
    if not segment_rounds:
        raise ValueError("segment_rounds must be non-empty")
    R_total = sum(segment_rounds)
    stream_id = base_id                                    # the stream (decode) op's id
    stream_op = Operation(stream_id, name, (patch,), clifford=True, circuit=circuit)
    segments: list = []
    rounds_by_op: dict = {stream_id: R_total}
    offset = 0
    for i, r in enumerate(segment_rounds):
        seg_id = base_id + 1 + i
        seg = Operation(seg_id, f"{name}[{i}]", (patch,), clifford=True, circuit=circuit,
                        stream_id=stream_id, stream_offset=offset,
                        predecessors=((base_id + i,) if i > 0 else ()),
                        has_successor=(i < len(segment_rounds) - 1))
        segments.append(seg)
        rounds_by_op[seg_id] = int(r)
        offset += int(r)
    return segments, stream_op, rounds_by_op
