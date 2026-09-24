"""Stream bookkeeping on the controller's QPU-facing side.

A stream is a run of syndrome rounds shared by several operations
(segments) on one physical patch group. This object owns which operation
is bound to which stream round, the next free round of every stream, and the
protected regions: a protected region keeps one live stream on its owner group
between a start and an end operation, emits one round of it per QEC cycle
at the cycle boundary, holds operations that need the patch until that
boundary, and seals the stream only after its final round. Each stream's
state is one record (_LiveStream); the program's regions and the
resolved plan are one table. Nothing here schedules decoding; the window
manager learns about bindings, closed boundaries and seals through the
calls below.

NoFeedbackStreams is what a run without streams gets: every method is a
no-op with the same interface.

Streams is this package's own seam: the issuer and the idle accounting
are the only callers and both rows live here, so the controller may
change it alone (STYLE.md rule 7).
"""

import dataclasses
import functools
from typing import Any, Optional, Protocol, runtime_checkable

import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.identity as identity_records
import decsim.records.program as program_records

# Any denotes opaque operation, stream and patch identities in port calls.


@runtime_checkable
class Streams(Protocol):
    """The stream bookkeeping, as the rest of the controller sees it."""

    def binding_for(self, operation_id):
        """The stream and offset one operation is bound to, or None."""

    def blocks_start(self, operation) -> bool:
        """True while a protected region holds this operation's patch."""

    def begin(self, operation) -> None:
        """Open the region this operation starts, if it starts one."""

    def request_closes(self, operation) -> None:
        """Ask the region this operation ends to close at its boundary."""

    def close_feedback_boundary(self, operation) -> None:
        """Close the boundary the operation's last round reached."""

    def seal_finished_streams(self) -> None:
        """Seal every stream whose final round has been emitted."""

    def is_live_protected_patch(self, patch) -> bool:
        """True when a live protected region already emits this patch."""

    def extend_live_stream(
        self, operation: program_records.Operation, patch
    ) -> bool:
        """Emit one more round of the stream the idle patch holds."""


class NoFeedbackStreams:
    """A run whose operations share no streams and declare no regions."""

    # the row takes the wires the other row does: a seat has one shape
    runtime = ports.Port(ports.OperationRuntime)
    qpu = ports.Port(ports.Qpu)
    windows = ports.Port(ports.WindowInput)

    def load(self, program) -> None:
        """Nothing to index."""
        del program

    def binding_for(self, operation_id):
        """No operation is bound to a stream."""
        del operation_id
        return None

    def blocks_start(self, operation) -> bool:
        """No stream holds any operation."""
        del operation
        return False

    def begin(self, operation) -> None:
        """Nothing to activate."""
        del operation

    def request_closes(self, operation) -> None:
        """Nothing to close."""
        del operation

    def close_feedback_boundary(self, operation) -> None:
        """No boundary to close."""
        del operation

    def seal_finished_streams(self) -> None:
        """No stream to seal."""

    def is_live_protected_patch(self, patch) -> bool:
        """No patch is protected."""
        del patch
        return False

    def extend_live_stream(
        self, operation: program_records.Operation, patch
    ) -> bool:
        """No stream to extend."""
        del operation
        del patch
        return False


@dataclasses.dataclass
class _LiveStream:
    """One stream's state: its next round and, when protected, its cycle."""

    next_round: int = 0
    # the active protected region, None while the stream is unprotected
    region: object = None
    next_boundary_tick: Optional[int] = None
    is_boundary_open: bool = False
    is_close_requested: bool = False
    last_emission_tick: Optional[int] = None
    is_sealed: bool = False


class FeedbackStreams:
    """The stream bindings, the live streams and the protected cycle."""

    # a cadence change frees an operation the runtime is holding
    runtime = ports.Port(ports.OperationRuntime)
    qpu = ports.Port(ports.Qpu)
    windows = ports.Port(ports.WindowInput)

    def __init__(
        self,
        engine: engine_module.Engine,
        *,
        regions,
        resolved_operations,
        resolved_patches,
    ) -> None:
        self.engine = engine
        self.table = _StreamTable(
            regions, resolved_operations, resolved_patches
        )
        self.live_by_stream_id: dict = {}
        # operation id -> StreamBinding
        self.bindings: dict = {}
        # patch -> the stream of the last segment that ran on it
        self.stream_by_patch: dict = {}

    # ---- program load

    def load(self, program) -> None:
        """Index the protected regions and the declared stream bindings."""
        self.table.index(program)
        for operation in program.operations:
            has_stream = operation.stream_id is not None
            has_offset = operation.stream_offset is not None
            if has_stream and has_offset:
                self._bind(
                    operation.id, operation.stream_id, operation.stream_offset
                )

    def binding_for(self, operation_id):
        """The stream binding an operation was given, or None."""
        return self.bindings.get(operation_id)

    # ---- starting an operation

    def blocks_start(self, operation: program_records.Operation) -> bool:
        """True while a live protected group holds this operation."""
        live_patches = self._live_protected_patches()
        patches = set(operation.patches)
        active_patches = patches.intersection(live_patches)
        starting_regions = self.table.regions_starting_at(operation.id)
        for region in starting_regions:
            starting_patches = self.table.patches_of_stream(region.stream_id)
            if active_patches.intersection(starting_patches):
                return True
        if not active_patches:
            return False
        open_patches = self._boundary_open_patches()
        return not active_patches.issubset(open_patches)

    def begin(self, operation) -> None:
        """Activate the regions the operation starts; give it its rounds."""
        protected_stream_id = self._protected_feedback_stream(operation)
        self._activate_protected_regions(operation)
        if protected_stream_id is None:
            self._reserve_stream_rounds(operation)
        else:
            self._bind_protected_feedback_source(operation, protected_stream_id)
        self._hold_patches(operation)

    # ---- ending an operation

    def request_closes(self, operation) -> None:
        """A body finished: its ending regions close on the current boundary."""
        ending_regions = self.table.regions_ending_at(operation.id)
        for region in ending_regions:
            self._check_region_ends_on_boundary(region)
        for region in ending_regions:
            live = self._live(region.stream_id)
            live.is_close_requested = True

    def close_feedback_boundary(self, operation) -> None:
        """Close the stream boundary at the body measurement.

        Only in measurement_closed mode, and only for a feedback source:
        an operation some other operation is blocked by, the rule the
        round tracker applies to a finite source. Who waits, and through
        which operations, does not move a window's time boundary; the
        measurement does (Tan et al. 2209.09219 lines 898-903).
        """
        if operation.feedback_boundary_mode != "measurement_closed":
            return
        if not self.table.is_feedback_source(operation.id):
            return
        binding = self.bindings.get(operation.id)
        if binding is None:
            return
        round_count = self.table.round_count_of(operation.id)
        stream_round_count = binding.stream_offset + round_count
        self.windows.close_stream_boundary(
            binding.stream_id, stream_round_count
        )

    def seal_finished_streams(self) -> None:
        """The workload is complete: seal every unprotected dynamic stream."""
        live_streams = self.live_by_stream_id.items()
        for stream_id, live in list(live_streams):
            if self.table.is_protected(stream_id):
                continue
            if not self.windows.has_dynamic_stream(stream_id):
                continue
            self._seal(stream_id, live.next_round)

    # ---- idle rounds

    def is_live_protected_patch(self, patch) -> bool:
        """True while a protected stream is live on the patch."""
        live_patches = self._live_protected_patches()
        return patch in live_patches

    def extend_live_stream(
        self, operation: program_records.Operation, patch
    ) -> bool:
        """Advance the stream the idle patch holds, once per tick.

        The patch keeps its logical qubit in memory while it waits, so
        each idle cycle is the next round of that qubit's stream, and the
        stream's windows read those rounds as their buffer (Terhal
        1302.3428 lines 3176-3178; Skoric et al. 2209.08552 lines
        196-199). Every patch of the stream's group must be idle after
        the same operation.
        """
        stream_id = self._stream_held_after(operation, patch)
        if stream_id is None:
            return False
        if not self.windows.has_dynamic_stream(stream_id):
            return False
        patches = self.table.patches_of_stream(stream_id)
        if not self.qpu.are_patches_idle(operation.id, patches):
            raise RuntimeError(
                f"stream {stream_id!r} cannot extend a partially idle patch "
                f"group {patches!r} after operation {operation.id}"
            )
        live = self._live(stream_id)
        if live.last_emission_tick == self.engine.now:
            return True
        live.next_round += 1
        live.last_emission_tick = self.engine.now
        owner = self.table.owner_of(stream_id)
        self.qpu.emit_idle_stream_round(
            owner, stream_id, live.next_round, is_final=False
        )
        return True

    def _seal(self, stream_id, stream_round_count: int) -> None:
        """The QPU attests the stream's length, then the windows close it.

        The controller decides where a stream ends and only the QPU's
        source saw the rounds it executed, so the controller asks its
        neighbour on the data path before the seal travels on; a gem5
        port has one peer (gem5 src/sim/port.hh:59 and :82-84).
        """
        owner = self.table.owner_of(stream_id)
        self.qpu.validate_stream_length(owner, stream_round_count)
        self.windows.seal_stream(stream_id, stream_round_count)
        live = self._live(stream_id)
        live.is_sealed = True

    def _hold_patches(self, operation) -> None:
        """Record the stream each of the operation's patches holds after it.

        A segment's patches hold its stream. Any other detector-emitting
        operation clears them, since its rounds are its own detector
        history; an operation without detector data leaves them as they
        are.
        """
        patches = program_records.patches_of(operation)
        binding = self.bindings.get(operation.id)
        if binding is not None:
            for patch in patches:
                self.stream_by_patch[patch] = binding.stream_id
            return
        if not operation.emits_detector_data:
            return
        for patch in patches:
            self.stream_by_patch.pop(patch, None)

    def _stream_held_after(self, operation, patch):
        """The unsealed stream the patch holds while idle after the operation.

        A segment leaves its own stream until the next operation starts,
        even when that one is already issued; any other operation leaves
        what the patch holds.
        """
        stream_id = self.stream_by_patch.get(patch)
        binding = self.bindings.get(operation.id)
        if binding is not None:
            stream_id = binding.stream_id
        live = self.live_by_stream_id.get(stream_id)
        if live is None or live.is_sealed:
            return None
        return stream_id

    def _live(self, stream_id) -> _LiveStream:
        live = self.live_by_stream_id.get(stream_id)
        if live is None:
            live = _LiveStream()
            self.live_by_stream_id[stream_id] = live
        return live

    def _next_round(self, stream_id) -> int:
        live = self.live_by_stream_id.get(stream_id)
        if live is None:
            return 0
        return live.next_round

    def _live_protected_patches(self) -> set:
        patches = set()
        for stream_id, live in self.live_by_stream_id.items():
            if live.region is not None:
                owner_patches = self.table.patches_of_stream(stream_id)
                patches.update(owner_patches)
        return patches

    def _boundary_open_patches(self) -> set:
        patches = set()
        for stream_id, live in self.live_by_stream_id.items():
            if live.region is not None and live.is_boundary_open:
                owner_patches = self.table.patches_of_stream(stream_id)
                patches.update(owner_patches)
        return patches

    def _active_stream_id_of(self, patch):
        for stream_id, live in self.live_by_stream_id.items():
            if live.region is None:
                continue
            owner_patches = self.table.patches_of_stream(stream_id)
            if patch in owner_patches:
                return stream_id
        return None

    def _bind(self, operation_id, stream_id, stream_offset) -> None:
        binding = program_records.StreamBinding(stream_id, stream_offset)
        self.bindings[operation_id] = binding
        self.windows.bind_stream_operation(
            operation_id, stream_id, stream_offset
        )

    def _declared_stream(self, operation) -> tuple:
        """The operation's (stream_id, stream_offset), either may be None.

        The recorded binding wins over the operation's own declaration.
        """
        binding = self.bindings.get(operation.id)
        if binding is None:
            return operation.stream_id, operation.stream_offset
        return binding.stream_id, binding.stream_offset

    def _protected_feedback_stream(self, operation):
        """The one protected stream a feedback source feeds, or None."""
        if not self.table.is_feedback_source(operation.id):
            return None
        stream_ids = self._protected_stream_ids(operation)
        ordered_stream_ids = tuple(sorted(stream_ids))
        if len(ordered_stream_ids) > 1:
            raise ValueError(
                f"feedback source {operation.id} spans protected streams "
                f"{ordered_stream_ids}"
            )
        if not ordered_stream_ids:
            return None
        stream_id = ordered_stream_ids[0]
        self._check_protected_binding(operation, stream_id)
        return stream_id

    def _protected_stream_ids(self, operation) -> set:
        stream_ids = set()
        for patch in operation.patches:
            stream_id = self._active_stream_id_of(patch)
            if stream_id is not None:
                stream_ids.add(stream_id)
        starting_regions = self.table.regions_starting_at(operation.id)
        for region in starting_regions:
            stream_ids.add(region.stream_id)
        return stream_ids

    def _check_protected_binding(self, operation, stream_id) -> None:
        declared_stream_id, declared_offset = self._declared_stream(operation)
        if declared_stream_id not in (None, stream_id):
            raise ValueError(
                f"feedback source {operation.id} has conflicting stream_id"
            )
        current_round = self._next_round(stream_id)
        if declared_offset not in (None, current_round):
            raise ValueError(
                f"feedback source {operation.id} has conflicting stream_offset"
            )

    def _activate_protected_regions(self, operation) -> None:
        starting_regions = self.table.regions_starting_at(operation.id)
        live_patches = self._live_protected_patches()
        operation_patches = set(operation.patches)
        protected_patches = operation_patches.intersection(live_patches)
        for region in starting_regions:
            owner_patches = self.table.patches_of_stream(region.stream_id)
            protected_patches.update(owner_patches)
        if protected_patches and operation.emits_detector_data:
            raise ValueError(
                f"operation {operation.id} duplicates protected detector "
                "emission"
            )
        self._check_starting_regions(starting_regions, live_patches)
        for region in starting_regions:
            self._activate_region(region)

    def _check_starting_regions(self, starting_regions, live_patches) -> None:
        occupied_patches = set(live_patches)
        for region in starting_regions:
            owner_patches = self.table.patches_of_stream(region.stream_id)
            overlap = occupied_patches.intersection(owner_patches)
            if overlap:
                raise RuntimeError(
                    f"protected patch group {owner_patches!r} already has "
                    "an active stream"
                )
            occupied_patches.update(owner_patches)

    def _activate_region(self, region) -> None:
        live = self._live(region.stream_id)
        live.region = region
        self._schedule_next_boundary(region, boundary_round=1)

    def _bind_protected_feedback_source(self, operation, stream_id) -> None:
        stream_offset = self._next_round(stream_id)
        self._bind(operation.id, stream_id, stream_offset)
        round_count = self.table.round_count_of(operation.id)
        required_stream_end = stream_offset + max(round_count, 1)
        self.windows.bind_required_stream_end(operation.id, required_stream_end)

    def _reserve_stream_rounds(self, operation) -> None:
        """Give an unprotected stream segment its rounds.

        A segment that declares no offset is bound at the next free
        round; an offset already reserved is refused.
        """
        stream_id, stream_offset = self._declared_stream(operation)
        if stream_id is None:
            return
        next_round = self._next_round(stream_id)
        if operation.finalizes_stream_round:
            if stream_offset != next_round - 1:
                raise RuntimeError(
                    f"{operation.name} must finalize stream round {next_round}"
                )
            return
        if stream_offset is None:
            stream_offset = next_round
            self._bind(operation.id, stream_id, stream_offset)
        elif stream_offset < next_round:
            first_round = stream_offset + 1
            raise RuntimeError(
                f"{operation.name} starts at stream round {first_round}, "
                f"but stream {stream_id!r} has already reserved through "
                f"round {next_round}"
            )
        round_count = self.table.round_count_of(operation.id)
        operation_end = stream_offset + round_count
        live = self._live(stream_id)
        live.next_round = max(next_round, operation_end)

    # ---- private: the protected cycle, boundary, round, seal

    def _schedule_next_boundary(self, region, *, boundary_round: int) -> None:
        stream_id = region.stream_id
        cadence = self.table.round_ticks_of_stream(stream_id)
        live = self._live(stream_id)
        live.next_boundary_tick = self.engine.now + cadence
        open_boundary = functools.partial(
            self._open_protected_boundary, stream_id
        )
        self.engine.schedule(
            cadence,
            open_boundary,
            label=f"protected-boundary({stream_id},{boundary_round})",
        )

    def _open_protected_boundary(self, stream_id) -> None:
        """The cycle boundary: held operations may start, then the round."""
        live = self._live(stream_id)
        assert live.region is not None, (
            f"protected stream {stream_id} is not active"
        )
        assert live.next_boundary_tick == self.engine.now, (
            f"protected stream {stream_id} boundary tick mismatch: "
            f"{live.next_boundary_tick} != {self.engine.now}"
        )
        assert not live.is_boundary_open, (
            f"protected stream {stream_id} boundary already open"
        )
        live.is_boundary_open = True
        self.runtime.retry_ready_operations()
        next_round = live.next_round + 1
        emit_round = functools.partial(self._emit_protected_round, stream_id)
        self.engine.schedule(
            0,
            emit_round,
            label=f"protected-round({stream_id},{next_round})",
            priority=engine_module.Priority.PROTECTED_ROUND,
        )

    def _emit_protected_round(self, stream_id) -> None:
        """Emit the stream's next round on the QPU; then seal or reschedule.

        A close was requested: seal. Otherwise the next boundary is
        scheduled.
        """
        live = self._live(stream_id)
        region = live.region
        assert region is not None, f"protected stream {stream_id} is not active"
        assert live.is_boundary_open, (
            f"protected stream {stream_id} boundary is not open"
        )
        live.is_boundary_open = False
        live.next_round += 1
        owner = self.table.owner_of(stream_id)
        self.qpu.emit_idle_stream_round(
            owner,
            stream_id,
            live.next_round,
            is_final=live.is_close_requested,
        )
        live.last_emission_tick = self.engine.now
        if live.is_close_requested:
            live.next_boundary_tick = None
            self._seal(stream_id, live.next_round)
            release = functools.partial(
                self._release_protected_region, stream_id
            )
            self.engine.schedule(
                0,
                release,
                label=f"protected-release({stream_id})",
                priority=engine_module.Priority.PROTECTED_RELEASE,
            )
            return
        next_boundary_round = live.next_round + 1
        self._schedule_next_boundary(region, boundary_round=next_boundary_round)

    def _release_protected_region(self, stream_id) -> None:
        live = self._live(stream_id)
        assert live.region is not None, (
            f"protected stream {stream_id} is not active"
        )
        assert live.is_close_requested, (
            f"protected stream {stream_id} was not closed"
        )
        assert live.next_boundary_tick is None, (
            f"protected stream {stream_id} has a pending boundary"
        )
        assert live.last_emission_tick == self.engine.now, (
            f"protected stream {stream_id} lacks final-round evidence"
        )
        live.region = None
        live.is_close_requested = False
        live.last_emission_tick = None
        self.runtime.retry_ready_operations()

    def _check_region_ends_on_boundary(self, region) -> None:
        stream_id = region.stream_id
        live = self._live(stream_id)
        assert live.region is region, (
            f"protected stream {stream_id} ended while inactive"
        )
        assert live.next_boundary_tick == self.engine.now, (
            f"protected stream {stream_id} ended off boundary"
        )


class _StreamTable:
    """The program's protected regions and the resolved plan they read."""

    def __init__(self, regions, resolved_operations, resolved_patches) -> None:
        self.regions = tuple(regions)
        operation_by_id = {}
        for operation in resolved_operations:
            operation_by_id[operation.operation_id] = operation
        self.resolved_operation_by_id = operation_by_id
        patch_by_identity = {}
        for patch in resolved_patches:
            patch_by_identity[patch.patch_identity] = patch
        self.resolved_patch_by_identity = patch_by_identity
        # ("start" | "end", operation id) -> the regions there
        self.regions_by_endpoint: dict = {}
        self.owner_by_stream_id: dict = {}
        self.feedback_source_ids: set = set()

    def index(self, program: program_records.ExecutionProgram) -> None:
        """Index owner footprints, region endpoints and feedback sources."""
        self.owner_by_stream_id = {
            operation.id: operation for operation in program.dynamic_streams
        }
        self._index_regions(program.operations)
        self._index_feedback_sources(program.operations)
        executable_ids = {operation.id for operation in program.operations}
        self._reject_external_sources(
            "decode_ops", program.decode_operations, executable_ids
        )
        self._reject_external_sources(
            "dynamic_streams", program.dynamic_streams, executable_ids
        )

    def round_count_of(self, operation_id) -> int:
        return self.resolved_operation_by_id[operation_id].round_count

    def round_ticks_of_stream(self, stream_id: Any) -> int:
        """The common cadence validated for the owner's physical group."""
        patches = self.patches_of_stream(stream_id)
        first_patch = patches[0]
        return self.resolved_patch_by_identity[first_patch].round_ticks

    def patches_of_stream(self, stream_id: Any) -> tuple:
        """The owner's authoritative physical patch footprint."""
        return self.owner_by_stream_id[stream_id].patches

    def regions_starting_at(self, operation_id) -> tuple:
        return self.regions_by_endpoint.get(("start", operation_id), ())

    def regions_ending_at(self, operation_id) -> tuple:
        return self.regions_by_endpoint.get(("end", operation_id), ())

    def is_protected(self, stream_id: Any) -> bool:
        return any(region.stream_id == stream_id for region in self.regions)

    def owner_of(self, stream_id):
        return self.owner_by_stream_id[stream_id]

    def is_feedback_source(self, operation_id) -> bool:
        return operation_id in self.feedback_source_ids

    def _index_regions(self, operations) -> None:
        operations_by_id = {operation.id: operation for operation in operations}
        indexed_stream_ids = set()
        regions = sorted(self.regions, key=_protected_region_stream_id)
        for region in regions:
            if region.stream_id in indexed_stream_ids:
                raise ValueError(
                    f"protected stream {region.stream_id} has two protected "
                    "regions; a stream has at most one"
                )
            self._index_region(region, operations_by_id)
            indexed_stream_ids.add(region.stream_id)

    def _index_feedback_sources(self, operations) -> None:
        for operation in operations:
            if operation.blocked_by is not None:
                self.feedback_source_ids.add(operation.blocked_by)

    def _reject_external_sources(
        self, source_role: str, sources, executable_ids: set
    ) -> None:
        """Refuse a feedback source that is not itself executed.

        A decode operation or a dynamic stream may not be the feedback
        source of a protected stream or patch.
        """
        for source in sources:
            is_feedback_source = source.id in self.feedback_source_ids
            is_executable = source.id in executable_ids
            if is_feedback_source and not is_executable:
                self._reject_external_source(source_role, source)

    def _reject_external_source(self, source_role: str, source) -> None:
        touched_stream_ids, touched_patches = self._protected_contacts(source)
        if not touched_stream_ids:
            return
        ordered_streams = tuple(sorted(touched_stream_ids))
        ordered_patches = tuple(
            sorted(
                touched_patches, key=identity_records.stable_identity_order_key
            )
        )
        raise ValueError(
            f"external source {source.id} from {source_role} participates "
            f"in protected feedback for protected streams "
            f"{ordered_streams} and patches {ordered_patches}"
        )

    def _protected_contacts(self, source) -> tuple[set, set]:
        touched_stream_ids = set()
        touched_patches = set()
        source_patches = set(source.patches)
        for region in self.regions:
            owner_patches = self.patches_of_stream(region.stream_id)
            overlap = source_patches.intersection(owner_patches)
            touched_patches.update(overlap)
            if overlap or source.id == region.stream_id:
                touched_stream_ids.add(region.stream_id)
        return touched_stream_ids, touched_patches

    def _index_region(self, region, operations_by_id: dict) -> None:
        stream_id = region.stream_id
        owner = self.owner_by_stream_id.get(stream_id)
        if owner is None:
            raise ValueError(
                f"protected stream {stream_id} is none of the program's "
                "dynamic streams, so nothing owns its region"
            )
        owner_patches = tuple(owner.patches)
        self._check_group_footprint(stream_id, owner_patches)
        endpoints = (
            ("start", region.start_operation_id),
            ("end", region.end_operation_id),
        )
        for endpoint, operation_id in endpoints:
            operation = operations_by_id.get(operation_id)
            _check_region_endpoint(region, owner_patches, endpoint, operation)
            regions = self.regions_by_endpoint.setdefault(
                (endpoint, operation_id), []
            )
            regions.append(region)

    def _check_group_footprint(self, stream_id, patches: tuple) -> None:
        unique_patches = set(patches)
        if not patches or len(unique_patches) != len(patches):
            raise ValueError(
                f"protected stream {stream_id} requires nonempty unique patches"
            )
        periods = {
            self.resolved_patch_by_identity[patch].round_ticks
            for patch in patches
        }
        if len(periods) != 1:
            raise ValueError(
                f"protected stream {stream_id} patches require a common cadence"
            )


def _protected_region_stream_id(region):
    return region.stream_id


def _check_region_endpoint(
    region, owner_patches: tuple, endpoint: str, operation
) -> None:
    stream_id = region.stream_id
    if operation is None:
        raise ValueError(
            f"protected stream {stream_id}'s {endpoint} operation is not "
            "an operation of the program"
        )
    required_patches = set(owner_patches)
    if not required_patches.issubset(operation.patches):
        raise ValueError(
            f"protected stream {stream_id}'s {endpoint} operation does not "
            f"hold every patch of the stream {owner_patches!r}"
        )
