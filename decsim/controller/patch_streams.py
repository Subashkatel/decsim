"""Each patch life as one decode stream, on the controller's side.

A lattice-surgery decoder reads each patch's history: every round the
patch measures, an operation's or an idle one, from the operation that
starts the patch to the discard that ends it (frontends/patch_lives.py;
SWIPER 2412.05115, swiper/device_manager.py, which emits a round for
every active patch each cycle). An operation on k patches is a segment
of k streams, so its body emits one readout per patch stream a round,
and its result waits for every one of them (operation_results.py).
"""

import dataclasses
from typing import Any

import decsim.ports as ports
import decsim.records.program as program_records


class PatchStreams:
    """The run's patch streams: which live stream each patch extends.

    It implements the controller's Streams seam (feedback_streams.py) for
    a run whose decoding follows its patches; no protected region and no
    stream binding of the feedback kind exists in such a run.
    """

    runtime = ports.Port(ports.OperationRuntime)
    qpu = ports.Port(ports.Qpu)
    windows = ports.Port(ports.WindowInput)

    def __init__(
        self,
        lives: tuple[program_records.PatchLife, ...],
        owners: tuple[program_records.Operation, ...],
    ) -> None:
        self.table = _LifeTable(lives, owners)
        self.live_by_stream_id: dict = {}
        # patch -> the stream of the life it measures in now
        self.stream_by_patch: dict = {}

    def load(
        self,
        program: program_records.ExecutionProgram,
        source_round_limit_by_stream: dict,
    ) -> None:
        """Nothing to index: the lives were read at build."""
        del program, source_round_limit_by_stream

    def binding_for(
        self,
        operation_id: Any,  # an opaque identity
    ) -> None:
        """No operation is bound to a feedback stream."""
        del operation_id

    def blocks_start(self, operation: program_records.Operation) -> bool:
        """No protected region holds any operation."""
        del operation
        return False

    def begin(self, operation: program_records.Operation) -> None:
        """Nothing to activate."""
        del operation

    def bind_at_start(
        self, command: program_records.RunOperationBody
    ) -> program_records.RunOperationBody:
        """The body as a segment of each patch stream, at its next round.

        Its patches measure in their lives from now on. A body that emits
        no detector data, a T injection's, adds no round to a stream
        (swiper/window_builder.py build_windows decodes none of it).
        """
        operation = command.operation
        segments = []
        for patch, stream_id in self.table.streams_of(operation.id):
            self.stream_by_patch[patch] = stream_id
            if command.emits_detector_data:
                segment = self._segment(command, patch, stream_id)
                segments.append(segment)
        if not segments:
            return command
        return dataclasses.replace(
            command, segments=tuple(segments), source_round_count=0
        )

    def request_closes(self, operation: program_records.Operation) -> None:
        """A discard ends its patches' lives: their streams are sealed."""
        if operation.kind is not program_records.OpKind.DISCARD:
            return
        for patch, stream_id in self.table.streams_ended_by(operation.id):
            del self.stream_by_patch[patch]
            live = self._live(stream_id)
            self._seal(stream_id, live)

    def close_feedback_boundary(
        self, operation: program_records.Operation
    ) -> None:
        """No feedback boundary to close."""
        del operation

    def seal_finished_streams(self) -> None:
        """The workload is complete: seal every life no discard ended."""
        live_items = self.live_by_stream_id.items()
        for stream_id, live in list(live_items):
            if not live.is_sealed:
                self._seal(stream_id, live)

    def is_live_protected_patch(
        self,
        patch: Any,  # an opaque identity
    ) -> bool:
        """No patch is protected."""
        del patch
        return False

    def extend_live_stream(
        self,
        operation: program_records.Operation,
        patch: Any,  # an opaque identity
    ) -> bool:
        """An idle patch's round is its life's next stream round.

        An idle patch keeps its logical qubit in memory, so its windows
        read the round (Terhal 1302.3428; SWIPER's idle rounds of an
        active patch). A patch in no decoded life leaves the round to the
        idle policy.
        """
        del operation
        stream_id = self.stream_by_patch.get(patch)
        if stream_id is None:
            return False
        live = self._live(stream_id)
        live.next_round += 1
        owner = self.table.owner_by_stream[stream_id]
        self.qpu.emit_idle_stream_round(
            owner, stream_id, live.next_round, is_final=False
        )
        return True

    def _segment(self, command, patch, stream_id) -> program_records.Operation:
        """The body's view on one patch, bound at its stream's next round."""
        operation = command.operation
        live = self._live(stream_id)
        stream_offset = live.next_round
        live.next_round += command.round_count
        self.windows.bind_stream_operation(
            operation.id, stream_id, stream_offset
        )
        return dataclasses.replace(
            operation,
            qubits=(patch,),
            patches=(patch,),
            stream_id=stream_id,
            stream_offset=stream_offset,
        )

    def _seal(self, stream_id, live: "_LiveLife") -> None:
        """The QPU attests the stream's length, then the windows close it."""
        owner = self.table.owner_by_stream[stream_id]
        self.qpu.validate_stream_length(owner, live.next_round)
        self.windows.seal_stream(stream_id, live.next_round)
        live.is_sealed = True

    def _live(self, stream_id) -> "_LiveLife":
        live = self.live_by_stream_id.get(stream_id)
        if live is None:
            live = _LiveLife()
            self.live_by_stream_id[stream_id] = live
        return live


class _LifeTable:
    """The lives, as the operations that run in them read them."""

    def __init__(self, lives: tuple, owners: tuple) -> None:
        # operation id -> ((patch, stream id), ...) of the lives it runs in
        self.streams_by_operation: dict = {}
        # last operation id -> ((patch, stream id), ...) of the lives it
        # is last in; a discard ends them
        self.ended_by_operation: dict = {}
        self.owner_by_stream = {owner.id: owner for owner in owners}
        for life in lives:
            self._index(life)

    def streams_of(self, operation_id) -> tuple:
        """The (patch, stream) of each life the operation runs in."""
        return self.streams_by_operation.get(operation_id, ())

    def streams_ended_by(self, operation_id) -> tuple:
        """The (patch, stream) of each life the operation is last in."""
        return self.ended_by_operation.get(operation_id, ())

    def _index(self, life: program_records.PatchLife) -> None:
        pair = (life.patch, life.stream_id)
        for operation_id in life.operation_ids:
            pairs = self.streams_by_operation.get(operation_id, ())
            self.streams_by_operation[operation_id] = (*pairs, pair)
        last_id = life.operation_ids[-1]
        ended = self.ended_by_operation.get(last_id, ())
        self.ended_by_operation[last_id] = (*ended, pair)


@dataclasses.dataclass
class _LiveLife:
    """One life's stream so far: its last round, and whether it is sealed."""

    next_round: int = 0
    is_sealed: bool = False
