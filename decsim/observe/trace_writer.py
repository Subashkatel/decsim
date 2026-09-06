"""One shot's Chrome trace: where every round and window sat and moved.

A listener on the trace sources of slice 10, in the event model of
docs/rewrite/notes/trace_and_viewer.md section 2: one thread per
component and one per wired link path, a residence or a service as an
`X` complete event written at its end, a move as an `X` on its link's
thread from send to delivery, a decision or a copy as an `i` instant,
store and queue occupancy as a `C` counter, and one round's hops joined
by `s`, `t` and `f` flow events. The Chrome Trace Event Format is the
JSON Array Format of the spec (an array of event objects carrying name,
cat, ph, ts, pid, tid and args); Perfetto and chrome://tracing read it.

`ts` is the tick over TICKS_PER_MICROSECOND, so a viewer places the
event; `args.tick` keeps the exact integer, and every reader uses that.
The writer schedules nothing and calls no component, so a run with it
connected has the same ticks, the same log and the same results as one
without it; with no writer connected each source fires into an empty
list.
"""

import json
from typing import Optional

import decsim.config as config
import decsim.message as message

# The threads in the order the pipeline uses them, so a viewer's lanes
# read top to bottom as the data flows. A thread the run never uses gets
# no tid and no metadata event; one this list does not name (a decoder
# unit, a decoder lane) is allocated after these, on first use.
THREAD_ORDER = (
    "QPU",
    "qpu_to_controller",
    "Controller",
    "controller_to_weak_buffer",
    "Buffer 0",
    "controller_to_strong_buffer",
    "Buffer 1",
    "Window planner",
    "weak_buffer_to_weak_decoder",
    "strong_buffer_to_strong_decoder",
    "weak_decoder_to_strong_decoder",
    "decoder_to_decoder",
    "weak_decoder_to_frame",
    "strong_decoder_to_frame",
    "Frame",
    "frame_to_controller",
    "controller_to_qpu",
)

PROCESS_ID = 1


def round_text(round_key) -> str:
    """A round key as the flow id and the args carry it: `op:index`."""
    operation_id, round_index = round_key
    return f"{operation_id}:{round_index}"


def window_text(window_key) -> str:
    """A window key as the args carry it: `op:index`."""
    operation_id, window_id = window_key
    return f"{operation_id}:{window_id}"


def request_text(request_key) -> str:
    """A request key as the args carry it: `op:window:tier:sequence`."""
    if request_key is None:
        return ""
    return (
        f"{request_key.op_id}:{request_key.window_id}:"
        f"{request_key.tier.value}:{request_key.run_sequence}"
    )


class TraceWriter:
    """Every event of one shot, in the Chrome trace event format."""

    def __init__(self, engine, process_name: str) -> None:
        self.engine = engine
        self.process_name = process_name
        self.events: list[dict] = []
        self._tid_by_thread: dict[str, int] = {}
        # a residence or service that has begun and not yet ended, by
        # (thread, key); each writes one X at its end
        self._open_residence: dict[tuple, dict] = {}
        self._open_service: dict[tuple, dict] = {}
        # the rounds whose flow has started, so a step never precedes it
        self._flowing_rounds: set = set()
        # which unit is decoding each window, so a stage lands on its
        # lane: the stage record names the window, not the unit
        self._unit_thread_by_window: dict[tuple, str] = {}
        self._counter_value: dict[str, int] = {}

    # ---- the file

    def write(self, path: str) -> None:
        """Dump the metadata events and the event list as one JSON array."""
        document = self.document()
        opener = _open_for(path)
        with opener(path, "wt") as handle:
            json.dump(document, handle)

    def document(self) -> list:
        """The metadata events, then every event, in one list."""
        rows = [self._process_name_event()]
        for thread, tid in sorted(self._tid_by_thread.items(), key=_by_tid):
            rows.append(_thread_name_event(tid, thread))
        rows.extend(self.events)
        return rows

    # ---- the QPU and the controller

    def round_emitted(self, readout: message.QPUReadout) -> None:
        """One readout leaves the QPU."""
        round_key = (readout.operation_id, readout.round_index)
        self._instant(
            "QPU",
            f"emitted round {readout.round_index}",
            "round",
            {"round": round_text(round_key), "bits": readout.size_bits},
        )

    def copy_made(self, key, bits, source_name: str, target_name: str) -> None:
        """Bits duplicated into a structure the receiver owns."""
        thread = _copy_thread(target_name)
        self._instant(
            thread,
            f"{target_name} copy",
            "copy",
            {
                "key": _key_text(key),
                "bits": bits,
                "from": source_name,
                "to": target_name,
                "transfer": "copy",
            },
        )

    def command_event(self, event) -> None:
        """A command arrived at the QPU or started on a boundary."""
        self._instant(
            "QPU",
            f"command {event.kind.lower()}",
            "command",
            {"operation": event.command.operation.id},
            tick=event.tick,
        )

    # ---- the links

    def transfer_delivered(self, record: message.TransferRecord) -> None:
        """One move, from the wire's send to the receiver's delivery."""
        transfer = record.transfer
        thread = record.path.value
        attribution = record.attribution
        args = {
            "bits": transfer.payload_bits,
            "transfer": "move",
            "channel": record.channel,
            "queue_wait_ticks": transfer.queue_wait_ticks,
            "delivery_tick": transfer.delivery_ticks,
            "operation": attribution.operation_id,
        }
        if attribution.window_id is not None:
            args["window"] = window_text(
                (attribution.operation_id, attribution.window_id)
            )
        if attribution.first_round is not None:
            args["rounds"] = f"{attribution.first_round}..{attribution.last_round}"
        name = _move_name(attribution)
        self._complete(
            thread,
            name,
            "link",
            transfer.send_ticks,
            transfer.delivery_ticks - transfer.send_ticks,
            args,
        )
        self._flow_over_move(thread, attribution, transfer)

    # ---- the stores

    def round_stored(self, store_name: str, round_key) -> None:
        """A round takes a slot in the store."""
        self._begin_residence(
            store_name,
            round_key,
            f"round {round_key[1]}",
            "round,residence",
            {"round": round_text(round_key), "transfer": "copy"},
        )
        self._step_flow(store_name, round_key)
        self._count(store_name, f"{store_name} rounds", 1)

    def round_published(self, store_name: str, round_key, tick: int) -> None:
        """The round's bits are readable in the store."""
        open_row = self._open_residence.get((store_name, round_key))
        if open_row is None:
            return
        open_row["args"]["data_ready"] = tick

    def round_released(self, store_name: str, round_key) -> None:
        """The round's last holder let go; the slot is free."""
        self._end_residence(store_name, round_key, {"freed": self.engine.now})
        self._count(store_name, f"{store_name} rounds", -1)

    def hold_registered(self, store_name: str, holder, round_keys) -> None:
        """One consumer token keeps the listed rounds alive."""
        self._instant(
            store_name,
            "hold registered",
            "hold",
            {
                "holder": _holder_text(holder),
                "rounds": _rounds_text(round_keys),
                "transfer": "reference",
            },
        )

    def hold_transferred(self, store_name: str, old_holder, new_holder) -> None:
        """A live hold moves to a new token, freeing nothing."""
        self._instant(
            store_name,
            "hold transferred",
            "hold",
            {
                "holder": _holder_text(old_holder),
                "to_holder": _holder_text(new_holder),
                "transfer": "reference",
            },
        )

    def hold_released(self, store_name: str, holder) -> None:
        """A hold ends; rounds with no other holder are freed next."""
        self._instant(
            store_name,
            "hold released",
            "hold",
            {"holder": _holder_text(holder)},
        )

    # ---- the window side

    def window_planned(self, window: message.Window) -> None:
        """A stream laid one more window."""
        self._instant(
            "Window planner",
            f"W{window.k} planned",
            "window",
            {
                "window": window_text(window.key),
                "commit": f"{window.commit_lo}..{window.commit_hi}",
            },
        )

    def job_enqueued(self, job: message.DecodeJob) -> None:
        """A window's decode request joins its ready queue."""
        self._begin_residence(
            "Window planner",
            job.request_key,
            f"W{job.window_id} queued",
            "window,queue",
            {
                "window": window_text((job.op_id, job.window_id)),
                "request": request_text(job.request_key),
            },
        )

    def depth_changed(self, tick: int, depth: int) -> None:
        """The jobs waiting over every pool changed."""
        self._counter(
            "Window planner", "ready queue depth", {"jobs": depth}, tick
        )

    def verdict_given(self, window_key, request_key, verdict) -> None:
        """The policy answered one weak result."""
        self._instant(
            "Window planner",
            "verdict",
            "window",
            {
                "window": window_text(window_key),
                "request": request_text(request_key),
                "verdict": verdict.value,
            },
        )

    def window_committed(self, window: message.Window, contribution) -> None:
        """A window committed under the contribution that owns its rounds."""
        self._instant(
            "Window planner",
            f"W{window.k} committed",
            "window",
            {
                "window": window_text(window.key),
                "owner": window_text(contribution.owner_key),
                "commit": f"{contribution.commit_lo}..{contribution.commit_hi}",
                "ownership": contribution.ownership_kind,
            },
        )

    def window_absorbed(self, key, owner_key) -> None:
        """A strong window covers the window; the weak chain skips it."""
        self._instant(
            "Window planner",
            "window absorbed",
            "window",
            {"window": window_text(key), "into": window_text(owner_key)},
        )

    # ---- the decoder units

    def job_dispatched(self, job: message.DecodeJob, unit) -> None:
        """The job left the ready queue for a unit."""
        self._end_residence(
            "Window planner", job.request_key, {"unit": unit.name}
        )

    def input_landed(self, job: message.DecodeJob, unit) -> None:
        """The job's input is in the unit's own memory."""
        self._begin_residence(
            _unit_thread(unit.name),
            job.request_key,
            f"W{job.window_id} input in memory",
            "window,residence",
            {
                "window": window_text((job.op_id, job.window_id)),
                "request": request_text(job.request_key),
                "bits": job.payload_bits(),
                "transfer": "copy",
            },
        )
        self._end_flow(_unit_thread(unit.name), job)

    def job_started(self, job: message.DecodeJob, unit) -> None:
        """The unit began this job's physical decode."""
        thread = _unit_thread(unit.name)
        self._unit_thread_by_window[(job.op_id, job.window_id)] = thread
        key = ("service", job.request_key)
        self._open_service[(thread, key)] = {
            "start": self.engine.now,
            "args": {
                "request": request_text(job.request_key),
                "service": str(job.service_key),
            },
        }

    def job_finished(self, job: message.DecodeJob, unit) -> None:
        """The unit's physical decode ended."""
        thread = _unit_thread(unit.name)
        key = ("service", job.request_key)
        open_row = self._open_service.pop((thread, key), None)
        if open_row is None:
            return
        self._complete(
            thread,
            f"W{job.window_id} service",
            "window,service",
            open_row["start"],
            self.engine.now - open_row["start"],
            open_row["args"],
        )

    def stage_recorded(self, record) -> None:
        """One stage of one job on the lane of the unit that started it."""
        window_key = (record.op_id, record.window_id)
        thread = self._unit_thread_by_window.get(window_key)
        if thread is None:
            return
        self._complete(
            thread,
            record.stage,
            "stage",
            record.start_ticks,
            record.end_ticks - record.start_ticks,
            {
                "window": window_text((record.op_id, record.window_id)),
                "cycles": record.cycles,
            },
        )

    def memory_taken(self, unit_name: str, job: message.DecodeJob) -> None:
        """The unit's memory freed the job's rounds."""
        thread = _unit_thread(unit_name)
        self._end_residence(
            thread, job.request_key, {"freed": self.engine.now}
        )

    # ---- the frame

    def correction_accepted(self, record) -> None:
        """The frame took one window's correction."""
        self._begin_residence(
            "Frame",
            (record.window_key, record.run_sequence),
            f"{window_text(record.window_key)} correction",
            "window,residence",
            {
                "window": window_text(record.window_key),
                "tier": record.tier,
                "run_sequence": record.run_sequence,
                "accepted": record.accepted_ticks,
                "transfer": "copy",
            },
        )

    def correction_committed(self, record) -> None:
        """The frame's write for one window has landed."""
        self._instant(
            "Frame",
            f"{window_text(record.window_key)} committed",
            "window",
            {
                "window": window_text(record.window_key),
                "tier": record.tier,
                "observables": list(record.logical_observables),
            },
            tick=record.committed_ticks,
        )

    def close_open_residences(self) -> None:
        """End every residence still open at the end of the run."""
        for (thread, key) in list(self._open_residence):
            self._end_residence(thread, key, {"freed_reason": "end of run"})

    # ---- private

    def _process_name_event(self) -> dict:
        return {
            "ph": "M",
            "name": "process_name",
            "pid": PROCESS_ID,
            "tid": 0,
            "args": {"name": self.process_name},
        }

    def _tid(self, thread: str) -> int:
        known = self._tid_by_thread.get(thread)
        if known is not None:
            return known
        allocated = len(self._tid_by_thread) + 1
        self._tid_by_thread[thread] = allocated
        return allocated

    def _instant(
        self, thread: str, name: str, category: str, args: dict, tick=None
    ) -> None:
        exact = self.engine.now if tick is None else tick
        row = {
            "ph": "i",
            "s": "t",
            "name": name,
            "cat": category,
            "ts": _microseconds(exact),
            "pid": PROCESS_ID,
            "tid": self._tid(thread),
            "args": dict(args, tick=exact),
        }
        self.events.append(row)

    def _complete(
        self,
        thread: str,
        name: str,
        category: str,
        start: int,
        duration: int,
        args: dict,
    ) -> None:
        row = {
            "ph": "X",
            "name": name,
            "cat": category,
            "ts": _microseconds(start),
            "dur": _microseconds(duration),
            "pid": PROCESS_ID,
            "tid": self._tid(thread),
            "args": dict(args, tick=start),
        }
        self.events.append(row)

    def _counter(
        self, thread: str, name: str, values: dict, tick: int
    ) -> None:
        row = {
            "ph": "C",
            "name": name,
            "ts": _microseconds(tick),
            "pid": PROCESS_ID,
            "tid": self._tid(thread),
            "args": dict(values, tick=tick),
        }
        self.events.append(row)

    def _count(self, thread: str, name: str, step: int) -> None:
        value = self._counter_value.get(name, 0) + step
        self._counter_value[name] = value
        self._counter(thread, name, {"rounds": value}, self.engine.now)

    def _begin_residence(
        self, thread: str, key, name: str, category: str, args: dict
    ) -> None:
        self._open_residence[(thread, key)] = {
            "start": self.engine.now,
            "name": name,
            "cat": category,
            "args": dict(args),
        }

    def _end_residence(self, thread: str, key, closing: dict) -> None:
        open_row = self._open_residence.pop((thread, key), None)
        if open_row is None:
            return
        args = dict(open_row["args"], **closing)
        self._complete(
            thread,
            open_row["name"],
            open_row["cat"],
            open_row["start"],
            self.engine.now - open_row["start"],
            args,
        )

    def _flow_over_move(self, thread, attribution, transfer) -> None:
        """Start or step the flow of every round the move carries."""
        for round_key in _rounds_of(attribution):
            if round_key in self._flowing_rounds:
                phase = "t"
            else:
                phase = "s"
                self._flowing_rounds.add(round_key)
            self._flow(phase, thread, round_key, transfer.send_ticks)

    def _step_flow(self, thread: str, round_key) -> None:
        if round_key not in self._flowing_rounds:
            return
        self._flow("t", thread, round_key, self.engine.now)

    def _end_flow(self, thread: str, job: message.DecodeJob) -> None:
        for round_key in _job_round_keys(job):
            if round_key not in self._flowing_rounds:
                continue
            self._flow("f", thread, round_key, self.engine.now)
            self._flowing_rounds.discard(round_key)

    def _flow(self, phase: str, thread: str, round_key, tick: int) -> None:
        row = {
            "ph": phase,
            "name": f"round {round_key[1]}",
            "cat": "round",
            "id": round_text(round_key),
            "ts": _microseconds(tick),
            "pid": PROCESS_ID,
            "tid": self._tid(thread),
            "args": {"tick": tick},
        }
        if phase == "f":
            row["bp"] = "e"
        self.events.append(row)


def _microseconds(ticks: int) -> float:
    return ticks / config.TICKS_PER_MICROSECOND


def _by_tid(item) -> int:
    return item[1]


def _thread_name_event(tid: int, thread: str) -> dict:
    return {
        "ph": "M",
        "name": "thread_name",
        "pid": PROCESS_ID,
        "tid": tid,
        "args": {"name": thread},
    }


def _open_for(path: str):
    """gzip when the path says so, a plain file otherwise."""
    if path.endswith(".gz"):
        import gzip

        return gzip.open
    return open


def _copy_thread(target_name: str) -> str:
    """The thread a copy into this structure belongs on."""
    if target_name.startswith("Buffer"):
        return target_name
    if target_name == "controller intake":
        return "Controller"
    if target_name == "controller assembler":
        return "Controller"
    if target_name == "masked view":
        return "Window planner"
    return _unit_thread(target_name)


def _unit_thread(unit_name: str) -> str:
    return f"Decoder unit {unit_name}"


def _key_text(key) -> str:
    if isinstance(key, tuple) and len(key) == 2:
        return round_text(key)
    if isinstance(key, message.DecodeJob):
        return window_text((key.op_id, key.window_id))
    return str(key)


def _holder_text(holder) -> str:
    return type(holder).__name__


def _rounds_text(round_keys) -> str:
    indices = []
    for _operation_id, round_index in round_keys:
        indices.append(round_index)
    if not indices:
        return ""
    return f"{min(indices)}..{max(indices)}"


def _move_name(attribution) -> str:
    if attribution.window_id is not None:
        return f"move W{attribution.window_id}"
    if attribution.first_round is not None:
        return f"move round {attribution.first_round}"
    return "move"


def _rounds_of(attribution) -> tuple:
    """The round keys one move carries, from its attribution."""
    if attribution.first_round is None:
        return ()
    keys = []
    for index in range(attribution.first_round, attribution.last_round + 1):
        keys.append((attribution.operation_id, index))
    return tuple(keys)


def _job_round_keys(job: message.DecodeJob) -> tuple:
    """The rounds a landed job's input covers."""
    decoder_input: Optional[message.DecoderInput] = job.decoder_input
    if decoder_input is None:
        return ()
    keys = []
    for round_input in decoder_input.rounds:
        keys.append((round_input.operation_id, round_input.round_index))
    return tuple(keys)
