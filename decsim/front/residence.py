"""How long the data sat, and how long a move waited, per sweep point.

Every complete residence of a traced shot (a round in a store, a job's
input in a unit's memory, a round in the controller's packing workspace,
a correction in the frame) and every move's wait on the wire, gathered
into one row per structure and one row per link path.

These come from the traced shots only, and that is the design and not a
gap: a trace is one shot's whole data path, so a run writes a file for
the shots `observation.trace_shots` names and a point of two thousand
shots writes one (docs/how-to/read_a_trace.md, and reference.yaml's
observation section says what a trace costs). A point whose shots were
not traced writes no row here, the way a point that traced nothing draws
no timeline.

The trace is read with `front/trace_file.py`, the reader
`decsim trace follow` reads it with, so one parser reads every trace and
`args.tick` is the only clock either of them trusts.
"""

import statistics
from pathlib import Path

import decsim.config as config
import decsim.front.report as report
import decsim.front.trace_file as trace_file

# what a row counts: a structure's residences, or a link path's moves
RESIDENCE = "residence"
LINK_PATH = "link_path"
# the Chrome phase of a thing with a length; an instant has none
COMPLETE_PHASE = "X"


def rows_of(measurements: list) -> list:
    """One row per traced shot per structure, then per link path.

    A measurement whose shot was not traced contributes nothing.
    """
    rows = []
    for measurement in measurements:
        if measurement.trace_path is None:
            continue
        document = trace_file.load(measurement.trace_path)
        for row in _rows_of_one_shot(measurement, document):
            rows.append(row)
    return rows


def write_residence(rows: list, report_dir: Path) -> None:
    """residence.csv, left unwritten when no shot of the run was traced."""
    if not rows:
        return
    path = Path(report_dir) / "residence.csv"
    report.write_csv(rows, path)


def residence_ticks_by_structure(document) -> dict:
    """Each structure's residence lengths in ticks, in the written order.

    A residence is the span from the tick the thing took its slot to the
    tick it was freed, which is what the writer records as one complete
    event on that structure's lane (observe/trace_writer.py's
    _begin_residence and _end_residence).
    """
    held = {}
    for event in document.events:
        if event["ph"] != COMPLETE_PHASE:
            continue
        if RESIDENCE not in event["cat"]:
            continue
        spans = held.setdefault(event["thread"], [])
        span = _span_ticks(event)
        spans.append(span)
    return held


def queue_wait_ticks_by_link_path(document) -> dict:
    """Each link path's per-move queue waits in ticks, in send order.

    The wait is the move's own counter, the ticks it sat behind the
    transfers already on that wire before its serialization began.
    """
    waited = {}
    for event in document.events:
        if event["ph"] != COMPLETE_PHASE:
            continue
        if "link" not in event["cat"]:
            continue
        waits = waited.setdefault(event["thread"], [])
        waits.append(event["args"]["queue_wait_ticks"])
    return waited


def _rows_of_one_shot(measurement, document) -> list:
    """One traced shot's structures, then its link paths, by name."""
    point = report.measured_point(measurement)
    rows = []
    residences = residence_ticks_by_structure(document)
    for structure in sorted(residences):
        spans = residences[structure]
        row = _row(point, measurement, RESIDENCE, structure, spans)
        rows.append(row)
    waits = queue_wait_ticks_by_link_path(document)
    for path in sorted(waits):
        row = _row(point, measurement, LINK_PATH, path, waits[path])
        rows.append(row)
    return rows


def _row(
    point: tuple, measurement, counting: str, name: str, ticks: list
) -> dict:
    """One structure's residences or one path's waits, in microseconds."""
    row = report.point_columns(point)
    row["seed"] = measurement.seed
    row["counting"] = counting
    row["name"] = name
    row["samples"] = len(ticks)
    row["mean_us"] = _mean_microseconds(ticks)
    row["longest_us"] = _longest_microseconds(ticks)
    return row


def _span_ticks(event: dict) -> int:
    """One complete event's length, from its own tick to its last."""
    start = trace_file.tick_of(event)
    end = trace_file.end_tick_of(event)
    return end - start


def _mean_microseconds(ticks: list) -> float:
    """The mean of the ticks, in microseconds."""
    spans = _microseconds_each(ticks)
    return statistics.fmean(spans)


def _longest_microseconds(ticks: list) -> float:
    """The largest of the ticks, in microseconds."""
    spans = _microseconds_each(ticks)
    return max(spans)


def _microseconds_each(ticks: list) -> list:
    """Every tick count as a float of microseconds."""
    spans = []
    for count in ticks:
        span = count / config.TICKS_PER_MICROSECOND
        spans.append(span)
    return spans
