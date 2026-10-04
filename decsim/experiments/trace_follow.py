"""`decsim trace follow`: one round's or one window's path, hop by hop.

Hops are ordered by args.tick, then by pipeline lane, and printed with
where the thing sat, how long, whether the hop copied or referenced the
bits, and how many bits crossed; the counts follow, sinter's one flat
row per thing counted (_data/_task_stats.py) applied per hop. Perfetto
draws the timeline; this is the path it would make the reader assemble
by clicking through flow arrows.
"""

import argparse
import dataclasses
from typing import Optional

import decsim.config as config
import decsim.experiments.refusal as refusal
import decsim.experiments.trace_file as trace_file

ROUND = "round"
WINDOW = "window"
# how many keys a refusal lists before it names the ends and the count
LISTED_KEYS = 12
# a hop is something that happened to the bits; a counter sample and a
# flow arrow describe the file, not the path
HOP_PHASES = ("X", "i")


@dataclasses.dataclass(frozen=True)
class Hop:
    """One event on the followed thing's path, as the table prints it."""

    tick: int
    where: str
    what: str
    duration_ticks: Optional[int]
    transfer: str
    bits: Optional[int]


@dataclasses.dataclass(frozen=True)
class Counts:
    """What the path did with the bits, counted over its hops."""

    copies: int
    job_references: int
    holds: int
    moves: int
    longest_residence: Optional[Hop]
    longest_queue_wait: Optional[Hop]


@dataclasses.dataclass(frozen=True)
class FollowedPath:
    """One round's or one window's hops and counts, in tick order."""

    kind: str
    key: str
    process_name: str
    hops: tuple
    counts: Counts


def follow(document, kind: str, key: str) -> FollowedPath:
    """Every hop of one round or window in one traced shot."""
    events = _events_of(document, kind, key)
    hops = []
    for event in events:
        hop = _hop_of(event)
        hops.append(hop)
    counts = _counts_of(events)
    return FollowedPath(
        kind=kind,
        key=key,
        process_name=document.process_name,
        hops=tuple(hops),
        counts=counts,
    )


def table_lines(hops) -> list:
    """One header line and one line per hop, columns aligned."""
    rows = [("tick (us)", "where", "what", "dur (us)", "transfer", "bits")]
    for hop in hops:
        cells = _cells_of(hop)
        rows.append(cells)
    widths = _widths_of(rows)
    lines = []
    for row in rows:
        line = _padded(row, widths)
        lines.append(line)
    return lines


def count_lines(counts: Counts) -> list:
    """The counts the note prints below the table."""
    references = f"{counts.job_references} job"
    if counts.job_references != 1:
        references = f"{counts.job_references} jobs"
    holds = f"{counts.holds} hold"
    if counts.holds != 1:
        holds = f"{counts.holds} holds"
    lines = [
        f"copies {counts.copies}, references {references} and {holds}, "
        f"moves {counts.moves}"
    ]
    longest = _longest_line("longest residence", counts.longest_residence)
    lines.append(longest)
    waited = _longest_line("longest queue wait", counts.longest_queue_wait)
    lines.append(waited)
    return lines


def main(argv: list) -> None:
    """`decsim trace follow <file> --round k:n | --window k:n`."""
    parser = argparse.ArgumentParser(prog="decsim trace")
    parser.add_argument("action", help="follow: one thing's path")
    parser.add_argument("file", help="the trace file one shot wrote")
    parser.add_argument("--round", default=None, help="the round, as op:index")
    parser.add_argument("--window", default=None, help="the window, as op:id")
    parsed = parser.parse_args(argv)
    _refuse_an_unknown_action(parsed.action)
    kind, key = _followed(parsed.round, parsed.window)
    document = trace_file.load(parsed.file)
    _refuse_a_key_the_trace_does_not_carry(document, kind, key)
    path = follow(document, kind, key)
    lines = _report_lines(path)
    text = "\n".join(lines)
    print(text)


def _report_lines(path: FollowedPath) -> list:
    """The table and the counts, as the command prints them."""
    lines = [f"{path.kind} {path.key} of {path.process_name}", ""]
    table = table_lines(path.hops)
    lines.extend(table)
    lines.append("")
    counted = count_lines(path.counts)
    lines.extend(counted)
    return lines


def _carried_keys(document, kind: str) -> tuple:
    """Every round or window key the trace's hops name, in key order."""
    keys = set()
    for event in document.events:
        if event["ph"] not in HOP_PHASES:
            continue
        key = event["args"].get(kind)
        if key is None:
            continue
        keys.add(key)
    return tuple(sorted(keys, key=_key_parts))


def _followed(round_key: Optional[str], window_key: Optional[str]) -> tuple:
    """Which thing the command line named, refusing anything but one."""
    if round_key is not None and window_key is not None:
        raise refusal.RefusalError(
            "decsim trace follow takes one of --round and --window, not both"
        )
    if round_key is not None:
        _refuse_a_key_that_is_not_two_parts(ROUND, round_key)
        return ROUND, round_key
    if window_key is not None:
        _refuse_a_key_that_is_not_two_parts(WINDOW, window_key)
        return WINDOW, window_key
    raise refusal.RefusalError(
        "decsim trace follow needs --round k:n or --window k:n, the keys "
        "the trace's args carry"
    )


def _refuse_an_unknown_action(action: str) -> None:
    """The one thing `decsim trace` does today."""
    if action == "follow":
        return
    raise refusal.RefusalError(
        f"decsim trace has no action {action}; it follows one round or "
        "window: decsim trace follow <file> --round 1:1"
    )


def _refuse_a_key_that_is_not_two_parts(kind: str, key: str) -> None:
    """A key is the operation and the index, as the trace's args write it."""
    words = key.split(":")
    if len(words) == 2 and words[1].isdigit():
        return
    raise refusal.RefusalError(
        f"--{kind} {key} is not a {kind} key; write the operation and the "
        f"index the trace carries, so --{kind} 1:0"
    )


def _refuse_a_key_the_trace_does_not_carry(
    document, kind: str, key: str
) -> None:
    """A key no hop names, with what this file does carry instead."""
    carried = _carried_keys(document, kind)
    if key in carried:
        return
    listed = _carried_text(carried)
    raise refusal.RefusalError(
        f"no hop of {document.process_name} carries {kind} {key}; its "
        f"{kind} keys are {listed}"
    )


def _carried_text(carried: tuple) -> str:
    """The carried keys, listed while they are few, else ends and count."""
    if not carried:
        return "none"
    if len(carried) <= LISTED_KEYS:
        return ", ".join(carried)
    return f"{carried[0]} to {carried[-1]}, {len(carried)} in all"


def _events_of(document, kind: str, key: str) -> list:
    """The events that carry the thing, by tick, then by lane."""
    landed = _input_landed_ticks(document)
    ordered = []
    for position, event in enumerate(document.events):
        if event["ph"] not in HOP_PHASES:
            continue
        if not _belongs(event, kind, key):
            continue
        if kind == ROUND and _is_after_the_decode(event, landed):
            continue
        tick = trace_file.tick_of(event)
        ordered.append((tick, event["tid"], position, event))
    ordered.sort()
    events = []
    for _tick, _tid, _position, event in ordered:
        events.append(event)
    return events


def _input_landed_ticks(document) -> dict:
    """When each window's input had landed in a decoder unit's memory.

    data_path.md's hop table ends a round's path there (hop 5): what
    moves afterwards on the window's own key is the correction, not the
    round's bits. A window decoded twice keeps the later landing.
    """
    landed = {}
    for event in document.events:
        if event["ph"] != "X":
            continue
        if "residence" not in event["cat"]:
            continue
        window = _decoder_input_window(event["args"])
        if window is None:
            continue
        tick = trace_file.tick_of(event)
        held = landed.get(window, tick)
        landed[window] = max(tick, held)
    return landed


def _decoder_input_window(args: dict) -> Optional[str]:
    """The window of a decoder input residence; None for another room."""
    window = args.get("window")
    if window is None:
        return None
    if args.get("rounds_by_operation") is None:
        return None
    if args.get("request") is None:
        return None
    return window


def _is_after_the_decode(event: dict, landed: dict) -> bool:
    """Whether a window's hop came after that window's input had landed."""
    window = event["args"].get("window")
    if window is None:
        return False
    landing = landed.get(window)
    if landing is None:
        return False
    tick = trace_file.tick_of(event)
    return tick > landing


def _belongs(event: dict, kind: str, key: str) -> bool:
    """Whether one event carries the followed round or window."""
    args = event["args"]
    if args.get(kind) == key:
        return True
    if kind == WINDOW:
        return _requests_the_window(args, key)
    return _rounds_cover(args, key)


def _requests_the_window(args: dict, key: str) -> bool:
    """A decode request names its window first: `op:window:tier:sequence`."""
    request = args.get("request")
    if request is None:
        return False
    return request.startswith(f"{key}:")


def _rounds_cover(args: dict, key: str) -> bool:
    """Whether an event's round range holds the followed round.

    A move, a hold, a window's readiness and a decoder input name a
    range per operation and not one round, so a round's own path runs
    through the window that reads it.
    """
    operation, index = _key_parts(key)
    ranges = args.get("rounds_by_operation", {})
    text = ranges.get(operation)
    if text is None:
        return False
    low, high = trace_file.range_of(text)
    if index < low:
        return False
    return index <= high


def _key_parts(key: str) -> tuple:
    """A key as its operation text and its integer index."""
    words = key.split(":")
    index = int(words[1])
    return words[0], index


def _hop_of(event: dict) -> Hop:
    """One event as the table's row."""
    args = event["args"]
    tick = trace_file.tick_of(event)
    duration = None
    if event["ph"] == "X":
        end = trace_file.end_tick_of(event)
        duration = end - tick
    what = _what_of(event)
    transfer = args.get("transfer", "")
    bits = args.get("bits")
    return Hop(
        tick=tick,
        where=event["thread"],
        what=what,
        duration_ticks=duration,
        transfer=transfer,
        bits=bits,
    )


def _what_of(event: dict) -> str:
    """What happened, in the words of the hop's own kind."""
    category = event["cat"]
    args = event["args"]
    if "residence" in category:
        return _residence_phrase(args)
    if "link" in category:
        return _move_phrase(args)
    if "queue" in category:
        return _queue_phrase(args)
    if "stage" in category:
        return f"stage {event['name']}"
    if "service" in category:
        return "decode service"
    return event["name"]


def _residence_phrase(args: dict) -> str:
    """Where the bits sat: the room, when they were readable, why they left."""
    room = _room_phrase(args)
    words = ["residence", room]
    ready = args.get("data_ready")
    if ready is not None:
        span = _microseconds(ready)
        words.append(f"data ready {span}")
    committed = args.get("committed")
    if committed is not None:
        span = _microseconds(committed)
        words.append(f"committed {span}")
    reason = args.get("freed_reason")
    if reason is not None:
        words.append(f"freed at {reason}")
    return ", ".join(words)


def _room_phrase(args: dict) -> str:
    """The room's size: a memory states bits, the workspace rounds."""
    capacity_bits = args.get("capacity_bits")
    if capacity_bits is not None:
        return f"of {capacity_bits} bits"
    capacity = args.get("capacity")
    if capacity is not None:
        return f"of {capacity} rounds"
    return "unbounded"


def _move_phrase(args: dict) -> str:
    """One link hop, and the window whose rounds it carried."""
    words = ["move"]
    window = args.get("window")
    rounds = args.get("rounds_by_operation")
    if window is not None and rounds is not None:
        index = _window_index(window)
        rounds_text = _rounds_text(rounds)
        words.append(f"with W{index} rounds {rounds_text}")
    waited = args.get("queue_wait_ticks")
    if waited:
        span = _microseconds(waited)
        words.append(f"after {span} on the wire")
    return ", ".join(words)


def _rounds_text(rounds_by_operation: dict) -> str:
    """Each operation's range as `operation:lo..hi`, the way keys read."""
    ranges = []
    for operation, text in rounds_by_operation.items():
        ranges.append(f"{operation}:{text}")
    return " and ".join(ranges)


def _queue_phrase(args: dict) -> str:
    """The wait in the ready queue, and the unit or reason that ended it."""
    words = ["queued"]
    unit = args.get("unit")
    if unit is not None:
        words.append(f"dispatched to {unit}")
    reason = args.get("freed_reason")
    if reason is not None:
        words.append(reason)
    return ", ".join(words)


def _counts_of(events) -> Counts:
    """The copies, references, moves and longest waits over the hops."""
    copies = 0
    moves = 0
    holds = 0
    requests = set()
    for event in events:
        category = event["cat"]
        if event["ph"] == "i" and "copy" in category:
            copies += 1
        if event["ph"] == "X" and "link" in category:
            moves += 1
        if event["name"] == "hold registered":
            holds += 1
        _note_request(requests, event)
    residence = _longest(events, "residence")
    queue_wait = _longest(events, "queue")
    return Counts(
        copies=copies,
        job_references=len(requests),
        holds=holds,
        moves=moves,
        longest_residence=residence,
        longest_queue_wait=queue_wait,
    )


def _note_request(requests: set, event: dict) -> None:
    """One decode request that read the thing, counted once."""
    request = event["args"].get("request")
    if request is None:
        return
    requests.add(request)


def _longest(events, category: str) -> Optional[Hop]:
    """The longest complete event of one category, None when there is none."""
    longest = None
    for event in events:
        if event["ph"] != "X":
            continue
        if category not in event["cat"]:
            continue
        hop = _hop_of(event)
        if longest is None:
            longest = hop
            continue
        if hop.duration_ticks > longest.duration_ticks:
            longest = hop
    return longest


def _longest_line(label: str, hop: Optional[Hop]) -> str:
    """One count line for a longest wait, or that there was none."""
    if hop is None:
        return f"{label}: none"
    span = _microseconds(hop.duration_ticks)
    return f"{label}: {span} us in {hop.where} ({hop.what})"


def _cells_of(hop: Hop) -> tuple:
    """One hop as the table's six cells."""
    duration = ""
    if hop.duration_ticks is not None:
        duration = _microseconds(hop.duration_ticks)
    bits = ""
    if hop.bits is not None:
        bits = str(hop.bits)
    return (
        _microseconds(hop.tick),
        hop.where,
        hop.what,
        duration,
        hop.transfer,
        bits,
    )


def _widths_of(rows) -> list:
    """The width of each column, over every row."""
    widths = []
    for cell in rows[0]:
        widths.append(len(cell))
    for row in rows:
        _widen(widths, row)
    return widths


def _widen(widths: list, row) -> None:
    """Grow each column to hold this row's cell."""
    for index, cell in enumerate(row):
        widths[index] = max(widths[index], len(cell))


def _padded(row, widths: list) -> str:
    """One table line, each cell padded to its column."""
    cells = []
    for index, cell in enumerate(row):
        width = widths[index]
        padded = cell.ljust(width)
        cells.append(padded)
    line = "  ".join(cells)
    return line.rstrip()


def _microseconds(ticks: int) -> str:
    """A tick count in microseconds, as the table writes it."""
    span = config.ticks_to_microseconds(ticks)
    return f"{span:.3f}"


def _window_index(window: str) -> str:
    """The window's own number, as the log and the trace name it: W3."""
    words = window.split(":")
    return words[1]
