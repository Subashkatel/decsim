"""One shot's Chrome trace, read back from disk and indexed.

The file `decsim run --trace` writes (observe/trace_writer.py, the
Chrome Trace Event Format's JSON Array Format), each event beside the
thread it sat on. The exact integer tick is always `args.tick`, never
the viewer's `ts`, because `ts` is microseconds as a float
(trace_and_viewer.md section 2).
"""

import gzip
import json

import decsim.config as config

METADATA_PHASE = "M"
PROCESS_NAME = "process_name"
THREAD_NAME = "thread_name"


class TraceDocument:
    """The events of one traced shot, with the thread each one sat on."""

    def __init__(self, events: list, process_name: str) -> None:
        self.events = tuple(events)
        self.process_name = process_name


def load(path) -> TraceDocument:
    """Read one trace file, plain or gzipped, and index its events."""
    text = _read_text(path)
    document = json.loads(text)
    process_name = ""
    thread_by_id = {}
    events = []
    for row in document:
        if row["ph"] != METADATA_PHASE:
            events.append(row)
            continue
        if row["name"] == PROCESS_NAME:
            process_name = row["args"]["name"]
        if row["name"] == THREAD_NAME:
            thread_by_id[row["tid"]] = row["args"]["name"]
    for event in events:
        event["thread"] = thread_by_id.get(event["tid"], "")
    return TraceDocument(events, process_name)


def tick_of(event: dict) -> int:
    """The event's exact tick, the integer the writer recorded.

    Chrome's `ts` is microseconds as a float, so every reader takes the
    tick from args (trace_and_viewer.md section 2).
    """
    return event["args"]["tick"]


def end_tick_of(event: dict) -> int:
    """A complete event's last tick: its own tick plus its duration.

    `dur` is a tick count divided by TICKS_PER_MICROSECOND, so rounding
    the product back recovers the exact tick.
    """
    start = tick_of(event)
    microseconds = event["dur"]
    exact_ticks = microseconds * config.TICKS_PER_MICROSECOND
    span_ticks = round(exact_ticks)
    return start + span_ticks


def range_of(text: str) -> tuple:
    """A "lo..hi" argument as the two integers it names."""
    words = text.split("..")
    low = int(words[0])
    high = int(words[1])
    return low, high


def _read_text(path) -> str:
    name = str(path)
    if name.endswith(".gz"):
        with gzip.open(name, "rt") as handle:
            return handle.read()
    with open(name) as handle:
        return handle.read()
