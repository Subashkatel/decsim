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
import decsim.records.rounds as round_records
import decsim.records.windows as window_records

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
# the controller's packing workspace draws on the controller's own lane
ASSEMBLER_THREAD = "Controller"
ASSEMBLER_COUNTER = "controller assembler rounds"


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
        f"{request_key.operation_id}:{request_key.window_id}:"
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
        # the rounds and windows whose flow has started, so a step never
        # precedes its start
        self._flowing_rounds: set = set()
        self._flowing_windows: set = set()
        # which unit is decoding each window, so a stage lands on its
        # lane: the stage record names the window, not the unit
        self._unit_thread_by_window: dict[tuple, str] = {}
        self._counter_value: dict[str, int] = {}
        self._unnamed_threads = 0

    # ---- the file

    def write(self, path: str) -> None:
        """Dump the metadata events and the event list as one JSON array."""
        document = self.document()
        opener = _open_for(path)
        with opener(path, "wt") as handle:
            json.dump(document, handle)

    def document(self) -> list:
        """The metadata events, then every event, in one list.

        A residence still open when the file is written ends at the
        clock's tick with that reason, so every X the reader sees has a
        length; the writer keeps it open, so a later call sees it again
        with the longer one.
        """
        process = self._process_name_event()
        rows = [process]
        threads = self._tid_by_thread.items()
        for thread, tid in sorted(threads, key=_by_tid):
            named = _thread_name_event(tid, thread)
            rows.append(named)
            sorted_event = _thread_sort_event(tid)
            rows.append(sorted_event)
        rows.extend(self.events)
        open_rows = self._still_open()
        rows.extend(open_rows)
        return rows

    def _still_open(self) -> list:
        """One X per residence the run never ended, closed at now."""
        rows = []
        for (thread, _key), open_row in self._open_residence.items():
            args = dict(open_row["args"])
            args["freed_reason"] = "end of run"
            tid = self._tid(thread)
            start = open_row["start"]
            duration = self.engine.now - start
            row = _complete_event(
                tid, open_row["name"], open_row["cat"], start, duration, args
            )
            rows.append(row)
        return rows

    # ---- the QPU and the controller

    def round_emitted(self, readout: round_records.QPUReadout) -> None:
        """One readout leaves the QPU."""
        round_key = (readout.operation_id, readout.round_index)
        args = {"round": round_text(round_key), "bits": readout.size_bits}
        name = f"emitted round {readout.round_index}"
        self._instant("QPU", name, "round", args)

    def copy_made(self, key, bits, source_name: str, target_name: str) -> None:
        """Bits duplicated into a structure the receiver owns."""
        thread = _copy_thread(target_name)
        args = _copied_identity(key)
        args["bits"] = bits
        args["from"] = source_name
        args["to"] = target_name
        args["transfer"] = "copy"
        name = f"{target_name} copy"
        self._instant(thread, name, "copy", args)
        self._learn_bits(thread, key, bits)

    def round_in_assembly(self, capacity, event) -> None:
        """The controller's packing workspace holds one more round, or one less.

        The workspace is the one bounded controller-side structure
        (controller.packing_rounds_in_flight, data_path.md's residence
        table): a round enters at its first fragment and leaves when it
        is packed, or when it is dropped for want of room.
        """
        round_key = (event.operation_id, event.round_index)
        if event.kind == "BINARY_AVAILABLE":
            self._begin_assembly(capacity, round_key, event)
            return
        if event.kind == "PACKED":
            self._end_assembly(round_key, "packed")
            return
        if event.kind == "DROPPED":
            self._end_assembly(round_key, "dropped, workspace full")

    def command_event(self, event) -> None:
        """A command arrived at the QPU or started on a boundary."""
        kind = event.kind.lower()
        name = f"command {kind}"
        args = {"operation": event.command.operation.id}
        self._instant("QPU", name, "command", args, tick=event.tick)

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
            window_key = (attribution.operation_id, attribution.window_id)
            args["window"] = window_text(window_key)
        if attribution.first_round is not None:
            first = attribution.first_round
            last = attribution.last_round
            args["rounds"] = f"{first}..{last}"
        name = _move_name(attribution)
        if attribution.window_id is None:
            category = "round,link"
        else:
            category = "window,link"
        start = transfer.send_ticks
        duration = transfer.delivery_ticks - start
        self._complete(thread, name, category, start, duration, args)
        self._flow_over_move(thread, attribution, transfer)
        if attribution.window_id is not None:
            window_key = (attribution.operation_id, attribution.window_id)
            self._step_window_flow(thread, window_key, start)

    # ---- the stores

    def round_stored(
        self, store_name: str, capacity, round_key, packet
    ) -> None:
        """A round takes a slot in the store."""
        args = {
            "round": round_text(round_key),
            "bits": _packet_bits(packet),
            "transfer": "copy",
            "capacity": capacity,
            "slot_taken": self.engine.now,
        }
        _operation_id, round_index = round_key
        name = f"round {round_index}"
        self._begin_residence(
            store_name, round_key, name, "round,residence", args
        )
        self._step_flow(store_name, round_key)
        counter = f"{store_name} rounds"
        self._count(store_name, counter, 1)

    def round_published(self, store_name: str, round_key, tick: int) -> None:
        """The round's bits are readable in the store."""
        readable = {"data_ready": tick}
        self._learn_on_residence(store_name, round_key, readable)

    def round_released(self, store_name: str, round_key) -> None:
        """The round's last holder let go; the slot is free."""
        closing = {
            "freed": self.engine.now,
            "freed_reason": "last hold released",
        }
        self._end_residence(store_name, round_key, closing)
        counter = f"{store_name} rounds"
        self._count(store_name, counter, -1)

    def hold_registered(self, store_name: str, holder, round_keys) -> None:
        """One consumer token keeps the listed rounds alive."""
        args = {
            "holder": _holder_text(holder),
            "rounds": _rounds_text(round_keys),
            "transfer": "reference",
        }
        self._instant(store_name, "hold registered", "hold", args)

    def hold_transferred(self, store_name: str, old_holder, new_holder) -> None:
        """A live hold moves to a new token, freeing nothing."""
        args = {
            "holder": _holder_text(old_holder),
            "to_holder": _holder_text(new_holder),
            "transfer": "reference",
        }
        self._instant(store_name, "hold transferred", "hold", args)

    def hold_released(self, store_name: str, holder) -> None:
        """A hold ends; rounds with no other holder are freed next."""
        args = {"holder": _holder_text(holder)}
        self._instant(store_name, "hold released", "hold", args)

    # ---- the window side

    def window_planned(self, window: window_records.Window) -> None:
        """A stream laid one more window."""
        args = {
            "window": window_text(window.key),
            "commit": f"{window.commit_lo}..{window.commit_hi}",
        }
        name = f"W{window.k} planned"
        self._instant("Window planner", name, "window", args)

    def window_ready(self, window: window_records.Window) -> None:
        """Every round the window reads is readable in its store."""
        args = {
            "window": window_text(window.key),
            "rounds": f"{window.start_round}..{window.buffer_hi}",
            "commit": f"{window.commit_lo}..{window.commit_hi}",
        }
        name = f"W{window.k} ready"
        self._instant("Window planner", name, "window", args)

    def job_enqueued(self, job: message.DecodeJob) -> None:
        """A window's decode request joins its ready queue."""
        window_key = (job.op_id, job.window_id)
        args = {
            "window": window_text(window_key),
            "request": request_text(job.request_key),
        }
        name = f"W{job.window_id} queued"
        self._begin_residence(
            "Window planner", job.request_key, name, "window,queue", args
        )
        self._start_window_flow("Window planner", window_key)

    def depth_changed(self, tick: int, depth: int) -> None:
        """The jobs waiting over every pool changed."""
        values = {"jobs": depth}
        self._counter("Window planner", "ready queue depth", values, tick)

    def verdict_given(self, window_key, request_key, verdict) -> None:
        """The policy answered one weak result."""
        args = {
            "window": window_text(window_key),
            "request": request_text(request_key),
            "verdict": _verdict_text(verdict),
        }
        self._instant("Window planner", "verdict", "window", args)

    def window_committed(
        self, window: window_records.Window, contribution
    ) -> None:
        """A window committed under the contribution that owns its rounds."""
        commit_lo = contribution.commit_lo
        commit_hi = contribution.commit_hi
        args = {
            "window": window_text(window.key),
            "owner": window_text(contribution.owner_key),
            "commit": f"{commit_lo}..{commit_hi}",
            "ownership": contribution.ownership_kind,
        }
        name = f"W{window.k} committed"
        self._instant("Window planner", name, "window", args)

    def window_absorbed(self, key, owner_key) -> None:
        """A strong window covers the window; the weak chain skips it."""
        args = {
            "window": window_text(key),
            "into": window_text(owner_key),
        }
        self._instant("Window planner", "window absorbed", "window", args)

    # ---- the decoder units

    def job_dispatched(self, job: message.DecodeJob, unit) -> None:
        """The job left the ready queue for a unit."""
        closing = {"unit": unit.name}
        self._end_residence("Window planner", job.request_key, closing)

    def input_landed(self, job: message.DecodeJob, unit) -> None:
        """The job's input is in the unit's own memory."""
        thread = _unit_thread(unit.name)
        window_key = (job.op_id, job.window_id)
        args = {
            "window": window_text(window_key),
            "request": request_text(job.request_key),
            "rounds": _job_rounds_text(job),
            "bits": _landed_bits(job),
            "transfer": "copy",
            "capacity": unit.memory.capacity_rounds,
            "slot_taken": _dispatch_tick(job),
            "data_ready": self.engine.now,
        }
        name = f"W{job.window_id} input in memory"
        self._begin_residence(
            thread, job.request_key, name, "window,residence", args
        )
        self._end_flow(thread, job)
        self._step_window_flow(thread, window_key, self.engine.now)

    def job_started(self, job: message.DecodeJob, unit) -> None:
        """The unit began this job's physical decode."""
        thread = _unit_thread(unit.name)
        window_key = (job.op_id, job.window_id)
        self._unit_thread_by_window[window_key] = thread
        args = {
            "request": request_text(job.request_key),
            "service": _service_text(job.service_key),
        }
        key = ("service", job.request_key)
        self._open_service[(thread, key)] = {
            "start": self.engine.now,
            "args": args,
        }

    def job_finished(self, job: message.DecodeJob, unit) -> None:
        """The unit's physical decode ended."""
        thread = _unit_thread(unit.name)
        key = ("service", job.request_key)
        open_row = self._open_service.pop((thread, key), None)
        if open_row is None:
            return
        start = open_row["start"]
        duration = self.engine.now - start
        name = f"W{job.window_id} service"
        args = open_row["args"]
        self._complete(thread, name, "window,service", start, duration, args)

    def stage_recorded(self, record) -> None:
        """One stage of one job on the lane of the unit that started it."""
        window_key = (record.op_id, record.window_id)
        thread = self._unit_thread_by_window.get(window_key)
        if thread is None:
            return
        args = {
            "window": window_text(window_key),
            "cycles": record.cycles,
        }
        start = record.start_ticks
        duration = record.end_ticks - start
        self._complete(thread, record.stage, "stage", start, duration, args)

    def memory_deposited(
        self, memory_name: str, _job: message.DecodeJob, decoder_input
    ) -> None:
        """One job's rounds landed in a unit's memory."""
        unit_name = _unit_of_memory(memory_name)
        thread = _unit_thread(unit_name)
        counter = f"{memory_name} rounds"
        rounds = len(decoder_input.rounds)
        self._count(thread, counter, rounds)

    def memory_taken(
        self, memory_name: str, job: message.DecodeJob, decoder_input
    ) -> None:
        """The unit's memory freed the job's rounds."""
        unit_name = _unit_of_memory(memory_name)
        thread = _unit_thread(unit_name)
        closing = {"freed": self.engine.now, "freed_reason": "decode done"}
        self._end_residence(thread, job.request_key, closing)
        counter = f"{memory_name} rounds"
        rounds = len(decoder_input.rounds)
        self._count(thread, counter, -rounds)

    # ---- the frame

    def correction_accepted(self, record) -> None:
        """The frame took one window's correction."""
        window = window_text(record.window_key)
        args = {
            "window": window,
            "tier": record.tier,
            "run_sequence": record.run_sequence,
            "accepted": record.accepted_ticks,
            "transfer": "copy",
        }
        key = (record.window_key, record.run_sequence)
        name = f"{window} correction"
        self._begin_residence("Frame", key, name, "window,residence", args)
        self._end_window_flow("Frame", record.window_key, self.engine.now)

    def correction_committed(self, record) -> None:
        """The frame's write for one window has landed."""
        window = window_text(record.window_key)
        key = (record.window_key, record.run_sequence)
        landed = {
            "committed": record.committed_ticks,
            "observables": list(record.logical_observables),
        }
        self._learn_on_residence("Frame", key, landed)
        args = {"window": window, "tier": record.tier}
        name = f"{window} committed"
        tick = record.committed_ticks
        self._instant("Frame", name, "window", args, tick=tick)

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
        """The thread's lane: its rank in THREAD_ORDER, or after them all."""
        known = self._tid_by_thread.get(thread)
        if known is not None:
            return known
        if thread in THREAD_ORDER:
            allocated = THREAD_ORDER.index(thread) + 1
        else:
            allocated = len(THREAD_ORDER) + 1 + self._unnamed_threads
            self._unnamed_threads += 1
        self._tid_by_thread[thread] = allocated
        return allocated

    def _instant(
        self, thread: str, name: str, category: str, args: dict, tick=None
    ) -> None:
        exact = tick
        if tick is None:
            exact = self.engine.now
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
        tid = self._tid(thread)
        row = _complete_event(tid, name, category, start, duration, args)
        self.events.append(row)

    def _counter(self, thread: str, name: str, values: dict, tick: int) -> None:
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

    def _begin_assembly(self, capacity, round_key, event) -> None:
        """The round's first fragment opens its place in the workspace."""
        if (ASSEMBLER_THREAD, round_key) in self._open_residence:
            return
        args = {
            "round": round_text(round_key),
            # the merge that packs the round tells the workspace its bits
            "bits": None,
            "transfer": "copy",
            "capacity": capacity,
            "slot_taken": event.tick,
        }
        name = f"assemble round {round_key[1]}"
        self._begin_residence(
            ASSEMBLER_THREAD, round_key, name, "round,residence", args
        )
        self._count(ASSEMBLER_THREAD, ASSEMBLER_COUNTER, 1)

    def _end_assembly(self, round_key, reason: str) -> None:
        """The round left the workspace, packed or dropped."""
        if (ASSEMBLER_THREAD, round_key) not in self._open_residence:
            return
        closing = {"freed": self.engine.now, "freed_reason": reason}
        self._end_residence(ASSEMBLER_THREAD, round_key, closing)
        self._count(ASSEMBLER_THREAD, ASSEMBLER_COUNTER, -1)

    def _begin_residence(
        self, thread: str, key, name: str, category: str, args: dict
    ) -> None:
        self._open_residence[(thread, key)] = {
            "start": self.engine.now,
            "name": name,
            "cat": category,
            "args": dict(args),
        }

    def _learn_bits(self, thread: str, key, bits) -> None:
        """A copy names the bits of the residence it lands in, when it can.

        A structure that knows its own payload keeps it; the controller's
        packing workspace holds raw measurement fragments and learns
        their size from the merge that packs them. Only a round names a
        residence: a copy of a whole window names its job, and a job's
        residence is keyed by its request.
        """
        if not isinstance(key, tuple):
            return
        open_row = self._open_residence.get((thread, key))
        if open_row is None:
            return
        held = open_row["args"].get("bits")
        if held is not None:
            return
        open_row["args"]["bits"] = bits

    def _learn_on_residence(self, thread: str, key, learned: dict) -> None:
        """Facts a residence hears after it opens and before it ends."""
        open_row = self._open_residence.get((thread, key))
        if open_row is None:
            return
        open_row["args"].update(learned)

    def _end_residence(self, thread: str, key, closing: dict) -> None:
        open_row = self._open_residence.pop((thread, key), None)
        if open_row is None:
            return
        args = dict(open_row["args"])
        args.update(closing)
        start = open_row["start"]
        duration = self.engine.now - start
        name = open_row["name"]
        category = open_row["cat"]
        self._complete(thread, name, category, start, duration, args)

    def _flow_over_move(self, thread, attribution, transfer) -> None:
        """Start or step the flow of every round the move carries.

        A window's move steps the flows of the rounds it reads and never
        starts one: a round's flow begins at its own hop out of the QPU
        and ends when its bits land in a unit's memory.
        """
        carries_window = attribution.window_id is not None
        for round_key in _rounds_of(attribution):
            is_flowing = round_key in self._flowing_rounds
            if not is_flowing and carries_window:
                continue
            phase = "s"
            if is_flowing:
                phase = "t"
            else:
                self._flowing_rounds.add(round_key)
            self._round_flow(phase, thread, round_key, transfer.send_ticks)

    def _step_flow(self, thread: str, round_key) -> None:
        if round_key not in self._flowing_rounds:
            return
        self._round_flow("t", thread, round_key, self.engine.now)

    def _end_flow(self, thread: str, job: message.DecodeJob) -> None:
        for round_key in _job_round_keys(job):
            if round_key not in self._flowing_rounds:
                continue
            self._round_flow("f", thread, round_key, self.engine.now)
            self._flowing_rounds.discard(round_key)

    def _flow(
        self,
        phase: str,
        thread: str,
        flow_id: str,
        name: str,
        category: str,
        tick: int,
    ) -> None:
        row = {
            "ph": phase,
            "name": name,
            "cat": category,
            "id": flow_id,
            "ts": _microseconds(tick),
            "pid": PROCESS_ID,
            "tid": self._tid(thread),
            "args": {"tick": tick},
        }
        if phase == "f":
            row["bp"] = "e"
        self.events.append(row)

    def _round_flow(
        self, phase: str, thread: str, round_key, tick: int
    ) -> None:
        """One hop of a round's own chain, from the QPU to a unit's memory."""
        flow_id = round_text(round_key)
        name = f"round {round_key[1]}"
        self._flow(phase, thread, flow_id, name, "round", tick)

    def _window_flow(
        self, phase: str, thread: str, window_key, tick: int
    ) -> None:
        """One hop of a window's chain, from its queue to the frame."""
        flow_id = _window_flow_id(window_key)
        name = f"W{window_key[1]}"
        self._flow(phase, thread, flow_id, name, "window", tick)

    def _start_window_flow(self, thread: str, window_key) -> None:
        """A window's chain begins where its request joins the queue.

        An escalated window joins the queue again under the same key.
        That is one more hop of the same chain: a Chrome flow has one
        start and then steps, so the second enqueue writes a `t`.
        """
        if window_key in self._flowing_windows:
            self._step_window_flow(thread, window_key, self.engine.now)
            return
        self._flowing_windows.add(window_key)
        self._window_flow("s", thread, window_key, self.engine.now)

    def _step_window_flow(self, thread: str, window_key, tick: int) -> None:
        """One more hop of a chain that has begun; an unstarted one waits."""
        if window_key not in self._flowing_windows:
            return
        self._window_flow("t", thread, window_key, tick)

    def _end_window_flow(self, thread: str, window_key, tick: int) -> None:
        """The chain ends where the frame holds the window's correction."""
        if window_key not in self._flowing_windows:
            return
        self._window_flow("f", thread, window_key, tick)
        self._flowing_windows.discard(window_key)


def _complete_event(
    tid: int,
    name: str,
    category: str,
    start: int,
    duration: int,
    args: dict,
) -> dict:
    """One X: a residence or a service, written with its length."""
    return {
        "ph": "X",
        "name": name,
        "cat": category,
        "ts": _microseconds(start),
        "dur": _microseconds(duration),
        "pid": PROCESS_ID,
        "tid": tid,
        "args": dict(args, tick=start),
    }


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


def _thread_sort_event(tid: int) -> dict:
    """The lane order a viewer draws: the tid is already the rank."""
    return {
        "ph": "M",
        "name": "thread_sort_index",
        "pid": PROCESS_ID,
        "tid": tid,
        "args": {"sort_index": tid},
    }


def _open_for(path: str):
    """A gzip file when the path says so, a plain one otherwise."""
    if path.endswith(".gz"):
        import gzip

        return gzip.open
    return open


def _copy_thread(target_name: str) -> str:
    """The thread a copy into this structure belongs on."""
    if target_name.startswith("Buffer"):
        return target_name
    if target_name in ("controller intake", "controller assembler"):
        return "Controller"
    if target_name == "masked view":
        return "Window planner"
    unit_name = _unit_of_memory(target_name)
    return _unit_thread(unit_name)


def _unit_of_memory(memory_name: str) -> str:
    """`unit default#0 memory` names the unit `default#0`."""
    without_prefix = memory_name.removeprefix("unit ")
    return without_prefix.removesuffix(" memory")


def _unit_thread(unit_name: str) -> str:
    return f"Decoder unit {unit_name}"


def _window_flow_id(window_key) -> str:
    """A window's flow id, kept apart from a round key's own text."""
    text = window_text(window_key)
    return f"window {text}"


def _dispatch_tick(job: message.DecodeJob) -> Optional[int]:
    """When the unit took the job's slot; None for a windowless job."""
    window = job.window
    if window is None:
        return None
    return window.t_dispatch


def _copied_identity(key) -> dict:
    """What a copy carried, named as every other event names it.

    A round key is a round; a decode job is a window, and the rounds it
    reads, so a round's own path can be followed through the copy into a
    unit's memory.
    """
    if isinstance(key, tuple) and len(key) == 2:
        return {"round": round_text(key)}
    if isinstance(key, message.DecodeJob):
        window_key = (key.op_id, key.window_id)
        window = window_text(window_key)
        rounds = _job_rounds_text(key)
        return {"window": window, "rounds": rounds}
    return {"key": str(key)}


def _job_rounds_text(job: message.DecodeJob) -> str:
    """The rounds a job's landed input holds, as `lo..hi`."""
    decoder_input = job.decoder_input
    if decoder_input is None:
        return ""
    indices = []
    for round_input in decoder_input.rounds:
        indices.append(round_input.round_index)
    if not indices:
        return ""
    low = min(indices)
    high = max(indices)
    return f"{low}..{high}"


def _holder_text(holder) -> str:
    """The hold token as the args name it: its kind and what it holds."""
    if isinstance(holder, tuple):
        return window_text(holder)
    kind = type(holder)
    kind_name = kind.__name__
    request_key = getattr(holder, "request_key", None)
    if request_key is not None:
        request = request_text(request_key)
        return f"{kind_name} {request}"
    window_key = getattr(holder, "window_key", None)
    if window_key is not None:
        window = window_text(window_key)
        return f"{kind_name} {window}"
    return kind_name


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
    last_round = attribution.last_round + 1
    for index in range(attribution.first_round, last_round):
        keys.append((attribution.operation_id, index))
    return tuple(keys)


def _packet_bits(packet) -> Optional[int]:
    """The bits one stored round holds, None when a fragment carries none."""
    total = 0
    for fragment in packet.fragments:
        if fragment.bits is None:
            return None
        total += len(fragment.bits)
    return total


def _landed_bits(job: message.DecodeJob) -> Optional[int]:
    """The bits of the input in the unit's memory, None when unknown."""
    decoder_input = job.decoder_input
    if decoder_input is None:
        return None
    total = 0
    for fragment in decoder_input.fragments():
        if fragment.bits is None:
            return None
        total += len(fragment.bits)
    return total


def _verdict_text(verdict) -> str:
    """The verdict as its own name."""
    name = getattr(verdict, "name", None)
    if name is None:
        return str(verdict)
    return name.lower()


def _service_text(service_key) -> str:
    """A service key as the args carry it: its run-wide ordinal."""
    if service_key is None:
        return ""
    return str(service_key.run_sequence)


def _job_round_keys(job: message.DecodeJob) -> tuple:
    """The rounds a landed job's input covers."""
    decoder_input: Optional[message.DecoderInput] = job.decoder_input
    if decoder_input is None:
        return ()
    keys = []
    for round_input in decoder_input.rounds:
        keys.append((round_input.operation_id, round_input.round_index))
    return tuple(keys)
