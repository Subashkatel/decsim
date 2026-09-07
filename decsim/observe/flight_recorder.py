"""The flight recorder: one causal row per hardware transition of a run.

A listener over listeners. It holds the round event recorder, the window
ledger, the runtime stamps, the command events and the frame
corrections of one run, and on `events` walks them in pipeline order,
giving each transition its causal predecessor so a round or a window can
be followed from the QPU to the frame. The program's operations are
handed to it at build, because the dependency edge between two
operations is a fact of the workload and not of any listener. It reads
no component, records nothing itself and writes nothing back.
"""

import dataclasses
from typing import Optional

import decsim.records.identity as identity_records


class FlightRecorder:
    """The listeners of one run, walked into one causal event list."""

    def __init__(
        self,
        round_events,
        windows,
        runtime_stamps,
        command_events,
        corrections: "FrameCorrections",
        operations: tuple,
    ) -> None:
        self.round_events = round_events
        self.windows = windows
        self.runtime_stamps = runtime_stamps
        self.command_events = command_events
        self.corrections = corrections
        # the workload's operations, for the dependency edges
        self.operations = operations

    @property
    def ledger(self) -> "RunLedgerView":
        """Assemble the causal record of the run so far.

        The chains are added in pipeline order: every round's controller
        chain, the strong store's landings, every window's chain, the
        frame commits, the controller's output path, the runtime's
        releases, the QPU's command arrivals and the result returns.
        """
        rows = _LedgerRows()
        round_events = self.round_events
        rounds = _round_chains(rows, round_events.events)
        stored = _store_landings(rows, round_events.stored_rounds, rounds)
        frame_prev = _window_chains(rows, self.windows.windows, rounds, stored)
        committed_of_op = _frame_commits(rows, self.corrections, frame_prev)
        operations = _operations_by_id(self.operations)
        outputs = _output_path(
            rows, round_events.output_events, operations, committed_of_op
        )
        stamps = self.runtime_stamps
        released_of_op = _releases(
            rows, stamps.decode_release, operations, outputs, committed_of_op
        )
        _rewire_commands_through_releases(outputs, released_of_op)
        _qpu_commands(rows, self.command_events.events, outputs)
        _result_returns(rows, stamps.result_return, outputs)
        numbered = rows.numbered_events()
        return RunLedgerView(events=numbered)


class FrameCorrections:
    """The frame's corrections, as accepted and as landed.

    A listener on PauliFrame.correction_accepted and
    correction_committed; the flight recorder pairs the two for each
    window's write, and the frame keeps no reader of its own.
    """

    def __init__(self) -> None:
        self.accepted: list = []
        self.committed: list = []

    def correction_accepted(self, record) -> None:
        """One window's correction was taken and its write charged."""
        self.accepted.append(record)

    def correction_committed(self, record) -> None:
        """One window's write has landed in the frame."""
        self.committed.append(record)


@dataclasses.dataclass(frozen=True)
class LedgerEvent:
    """One row of the run's flight recorder.

    A hardware-significant transition with its causal predecessor.
    ``status`` is "terminal" on the last event of a syndrome round's
    controller chain and on the frame/release tail of a window chain,
    empty elsewhere.
    """

    event_id: int
    kind: str
    tick: int
    op: object
    round: Optional[int] = None
    window: Optional[int] = None
    patch: object = None
    route: str = ""
    prev_event_id: Optional[int] = None
    status: str = ""


@dataclasses.dataclass(frozen=True)
class RunLedgerView:
    """The assembled cycle/event ledger of one completed run.

    Sources are the owners' own records (packing round events, syndrome
    buffer 1's stored log, window stamps, frame records, release times);
    the ledger derives nothing a component did not record, so it can be
    used as evidence. ``check()`` proves the accounting: every emitted
    window-input round reaches exactly one terminal state and every
    chain is causally ordered.
    """

    events: tuple

    def chain(
        self, *, op, round: Optional[int] = None, window: Optional[int] = None
    ) -> tuple:
        """The events of one operation, one round or one window, in order."""
        rows = []
        for event in self.events:
            if event.op != op:
                continue
            if round is not None and event.round != round:
                continue
            if window is not None and event.window != window:
                continue
            rows.append(event)
        ordered = sorted(rows, key=_event_id)
        return tuple(ordered)

    def check(self) -> None:
        """Every effect follows its cause; every round ends exactly once."""
        problems = self._causes_before_effects()
        problems += self._one_terminal_per_round()
        if problems:
            listed = "\n  ".join(problems)
            raise RuntimeError(f"ledger check failed:\n  {listed}")

    def _causes_before_effects(self) -> list:
        by_id = {}
        for event in self.events:
            by_id[event.event_id] = event
        problems = []
        for event in self.events:
            if event.prev_event_id is None:
                continue
            prev = by_id[event.prev_event_id]
            if event.tick < prev.tick:
                problems.append(
                    f"{event.kind}@{event.tick} precedes its cause "
                    f"{prev.kind}@{prev.tick} (op {event.op})"
                )
        return problems

    def _one_terminal_per_round(self) -> list:
        round_terminals: dict = {}
        emitted_rounds = set()
        for event in self.events:
            if event.round is None or event.window is not None:
                continue
            key = (event.op, event.round)
            if event.kind == "EMITTED":
                emitted_rounds.add(key)
            if event.status == "terminal":
                terminals = round_terminals.setdefault(key, [])
                terminals.append(event.kind)
        problems = []
        for key in sorted(emitted_rounds, key=repr):
            terminals = round_terminals.get(key, [])
            if len(terminals) != 1:
                problems.append(
                    f"round {key} reached {len(terminals)} terminal states "
                    f"{terminals}: expected exactly one"
                )
        return problems


# ---- private


def _first_identity_order(item) -> tuple:
    return identity_records.stable_identity_order_key(item[0])


def _unit_name(unit) -> tuple:
    return (unit.pool, unit.index)


def _event_id(event: LedgerEvent) -> int:
    return event.event_id


_ROUND_TERMINALS = ("PUBLISHED", "DROPPED", "FEEDBACK_MEMORY_DELIVERED")


class _LedgerRows:
    """The rows in the order they were added, before they are numbered."""

    def __init__(self) -> None:
        self.rows: list = []

    def add(
        self,
        kind: str,
        tick: int,
        op,
        *,
        round=None,
        window=None,
        patch=None,
        route: str = "",
        prev=None,
        status: str = "",
    ) -> dict:
        """One row, whose cause is another row or None."""
        row = {
            "kind": kind,
            "tick": tick,
            "op": op,
            "round": round,
            "window": window,
            "patch": patch,
            "route": route,
            "prev": prev,
            "status": status,
        }
        self.rows.append(row)
        return row

    def numbered_events(self) -> tuple:
        """The rows as events, numbered by tick then by order of addition."""
        indexed = enumerate(self.rows)
        ordered = sorted(indexed, key=_tick_then_order)
        ids = {}
        for event_id, (_index, row) in enumerate(ordered):
            ids[id(row)] = event_id
        events = []
        for _index, row in ordered:
            prev_event_id = None
            if row["prev"] is not None:
                prev_event_id = ids[id(row["prev"])]
            row_identity = id(row)
            event = LedgerEvent(
                event_id=ids[row_identity],
                kind=row["kind"],
                tick=row["tick"],
                op=row["op"],
                round=row["round"],
                window=row["window"],
                patch=row["patch"],
                route=row["route"],
                prev_event_id=prev_event_id,
                status=row["status"],
            )
            events.append(event)
        return tuple(events)


def _tick_then_order(pair) -> tuple:
    index, row = pair
    return (row["tick"], index)


@dataclasses.dataclass
class _RoundChains:
    """What the controller chains leave for the chains that follow."""

    last_of_round: dict
    packed_of_round: dict
    published_rounds: set


def _round_chains(rows: _LedgerRows, events) -> _RoundChains:
    """The controller chain of every round, from the packing records."""
    chains = _RoundChains({}, {}, set())
    for event in events:
        key = (event.operation_id, event.round_index)
        status = ""
        if event.kind in _ROUND_TERMINALS:
            status = "terminal"
        if event.kind == "PUBLISHED":
            chains.published_rounds.add(key)
        prev = chains.last_of_round.get(key)
        row = rows.add(
            event.kind,
            event.tick,
            event.operation_id,
            round=event.round_index,
            patch=event.patch_id,
            route=event.route,
            prev=prev,
            status=status,
        )
        chains.last_of_round[key] = row
        if event.kind == "PACKED":
            chains.packed_of_round[key] = row
    return chains


def _store_landings(
    rows: _LedgerRows, stored_rounds, chains: _RoundChains
) -> dict:
    """The landing of every round in syndrome buffer 1, by round key.

    Its cause is the PACKED round, because the strong write leaves
    packing in parallel with the Buffer 0 publication and a fast
    crossing legitimately lands first. A round never published to
    Buffer 0 (the strong tier is primary) ends its journey here.
    """
    stored_of_round = {}
    for tick, operation_id, round_index in stored_rounds:
        key = (operation_id, round_index)
        status = "terminal"
        if key in chains.published_rounds:
            status = ""
        prev = chains.packed_of_round.get(key)
        stored_of_round[key] = rows.add(
            "STORED_SB1",
            tick,
            operation_id,
            round=round_index,
            prev=prev,
            status=status,
        )
    return stored_of_round


_WINDOW_STAMPS = (
    ("DECODE_QUEUED", "t_queued"),
    ("UNIT_ASSIGNED", "t_dispatch"),
    ("DECODE_DONE", "t_done"),
)


def _window_chains(
    rows: _LedgerRows, windows, chains: _RoundChains, stored_of_round: dict
) -> dict:
    """Every window's chain from its stamps; the last row by window key."""
    frame_prev = {}
    window_items = windows.items()
    for (operation_id, window_id), window in sorted(window_items, key=repr):
        if window.t_data_complete is None:
            continue
        input_key = (operation_id, window.buffer_hi)
        prev = chains.last_of_round.get(input_key)
        if prev is None:
            prev = stored_of_round.get(input_key)
        row = rows.add(
            "WINDOW_DATA_COMPLETE",
            window.t_data_complete,
            operation_id,
            window=window_id,
            prev=prev,
        )
        row = _window_stamp_rows(rows, window, operation_id, window_id, row)
        frame_prev[(operation_id, window_id)] = row
    return frame_prev


def _window_stamp_rows(
    rows: _LedgerRows, window, operation_id, window_id: int, row: dict
) -> dict:
    """The queued, assigned and done rows of one window; the last of them."""
    for kind, stamp in _WINDOW_STAMPS:
        tick = getattr(window, stamp)
        if tick is None:
            continue
        row = rows.add(kind, tick, operation_id, window=window_id, prev=row)
    return row


def _frame_commits(
    rows: _LedgerRows, corrections: "FrameCorrections", frame_prev: dict
) -> dict:
    """Every correction as accepted then committed; the last commit per op.

    A correction the frame accepted but whose write never landed has an
    accepted row and no committed row, so the ledger says what happened
    rather than what was expected to.
    """
    committed_of_op: dict = {}
    landed = _landed_identities(corrections.committed)
    for record in corrections.accepted:
        operation_id, window_id = record.window_key
        prev = frame_prev.get(record.window_key)
        accepted = rows.add(
            "FRAME_ACCEPTED",
            record.accepted_ticks,
            operation_id,
            window=window_id,
            route=record.tier,
            prev=prev,
        )
        identity = id(record)
        if identity not in landed:
            continue
        committed = rows.add(
            "FRAME_COMMITTED",
            record.committed_ticks,
            operation_id,
            window=window_id,
            route=record.tier,
            prev=accepted,
            status="terminal",
        )
        best = committed_of_op.get(operation_id)
        if best is None or committed["tick"] > best["tick"]:
            committed_of_op[operation_id] = committed
    return committed_of_op


def _landed_identities(records) -> set:
    """The identity of every correction whose write landed."""
    landed = set()
    for record in records:
        identity = id(record)
        landed.add(identity)
    return landed


def _operations_by_id(operations: tuple) -> dict:
    """The workload's operations, looked up by their identity."""
    by_id = {}
    for operation in operations:
        by_id[operation.id] = operation
    return by_id


@dataclasses.dataclass
class _OutputRows:
    """The controller's output rows, by operation and by payload identity."""

    decision_of_op: dict
    decision_issued_of_op: dict
    command_issued_by_identity: dict
    preloaded_by_identity: dict


def _output_path(
    rows: _LedgerRows, output_events, operations, committed_of_op: dict
) -> _OutputRows:
    """The controller's digital-to-QPU path from its own payload records.

    A release is available after the frame's decision reaches the
    controller; its real command then crosses controller output and
    the command link before it arrives and starts at the QPU.
    """
    outputs = _OutputRows({}, {}, {}, {})
    for event in output_events:
        _output_row(rows, event, operations, committed_of_op, outputs)
    return outputs


def _output_row(
    rows: _LedgerRows, event, operations, committed_of_op: dict, outputs
) -> None:
    """One output event as a row, filed by its kind."""
    operation_id = event.operation_id
    if event.kind == "DECISION_AVAILABLE":
        operation = operations.get(operation_id)
        blocking_op = getattr(operation, "blocked_by", None)
        prev = committed_of_op.get(blocking_op)
        row = rows.add(event.kind, event.tick, operation_id, prev=prev)
        outputs.decision_of_op[operation_id] = row
        return
    if event.kind == "CONTROL_PULSE_COMMAND_ISSUED":
        prev = outputs.decision_of_op.get(operation_id)
        row = rows.add(event.kind, event.tick, operation_id, prev=prev)
        outputs.command_issued_by_identity[id(event.payload)] = row
        return
    if event.kind == "PRELOADED_COMMAND":
        row = rows.add(event.kind, event.tick, operation_id)
        outputs.preloaded_by_identity[id(event.payload)] = row
        return
    if event.kind == "CONTROL_DECISION_ISSUED":
        prev = outputs.decision_of_op.get(operation_id)
        row = rows.add(event.kind, event.tick, operation_id, prev=prev)
        outputs.decision_issued_of_op[operation_id] = row


def _releases(
    rows: _LedgerRows,
    release_time: dict,
    operations,
    outputs: _OutputRows,
    committed_of_op: dict,
) -> dict:
    """The runtime's release of every blocked operation, by operation."""
    released_of_op = {}
    release_items = release_time.items()
    for operation_id, tick in sorted(release_items, key=repr):
        prev = outputs.decision_of_op.get(operation_id)
        if prev is None:
            operation = operations.get(operation_id)
            blocking_op = getattr(operation, "blocked_by", None)
            prev = committed_of_op.get(blocking_op)
        released_of_op[operation_id] = rows.add(
            "DECODE_RELEASED", tick, operation_id, prev=prev
        )
    return released_of_op


def _rewire_commands_through_releases(
    outputs: _OutputRows, released_of_op: dict
) -> None:
    """A dynamic command's issue follows the runtime's release at that tick."""
    for row in outputs.command_issued_by_identity.values():
        release = released_of_op.get(row["op"])
        if release is not None:
            row["prev"] = release


def _qpu_commands(rows: _LedgerRows, command_events, outputs: _OutputRows):
    """Every command's arrival and start at the QPU, from its own events."""
    arrived_by_identity = {}
    for event in command_events:
        command_identity = id(event.command)
        operation_id = event.command.operation.id
        if event.kind == "ARRIVED":
            prev = _issue_of_command(outputs, command_identity)
            row = rows.add(
                "QPU_COMMAND_ARRIVED", event.tick, operation_id, prev=prev
            )
            arrived_by_identity[command_identity] = row
            continue
        if event.kind == "STARTED":
            prev = arrived_by_identity.get(command_identity)
            rows.add(
                "QPU_COMMAND_STARTED",
                event.tick,
                operation_id,
                prev=prev,
                status="terminal",
            )


def _issue_of_command(outputs: _OutputRows, command_identity: int):
    """The row that issued a command: dynamic, else preloaded, else none."""
    issued = outputs.command_issued_by_identity.get(command_identity)
    if issued is not None:
        return issued
    return outputs.preloaded_by_identity.get(command_identity)


def _result_returns(
    rows: _LedgerRows, return_time: dict, outputs: _OutputRows
) -> None:
    """Every result returned to the QPU, caused by the decision it carried."""
    return_items = return_time.items()
    for operation_id, tick in sorted(return_items, key=repr):
        prev = outputs.decision_issued_of_op.get(operation_id)
        if prev is None:
            prev = outputs.decision_of_op.get(operation_id)
        rows.add(
            "RESULT_RETURNED_TO_QPU",
            tick,
            operation_id,
            prev=prev,
            status="terminal",
        )
