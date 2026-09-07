"""Stream bookkeeping on the controller's QPU-facing side.

A stream is a run of syndrome rounds shared by several operations
(segments) on one patch. This object owns which operation is bound to
which stream round, the next free round of every stream, and the
protected regions: a protected region keeps one live stream on a patch
between a start and an end operation, emits one round of it per QEC cycle
at the cycle boundary, holds operations that need the patch until that
boundary, and seals the stream only after its final round. Each stream's
state is one record (_LiveStream); the program's regions and the
resolved plan are one table. Nothing here schedules decoding; the window
manager learns about bindings, closed boundaries and seals through the
calls below.

NoFeedbackStreams is what a run without streams gets: every method is a
no-op with the same interface.
"""

import dataclasses
import functools
from typing import Optional

import decsim.message as message
import decsim.records.identity as identity_records


class NoFeedbackStreams:
    """A run whose operations share no streams and declare no regions."""

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

    def close_feedback_boundary(
        self, operation, waiting_blocked_successor: bool
    ) -> None:
        """No boundary to close."""
        del operation
        del waiting_blocked_successor

    def seal_finished_streams(self) -> None:
        """No stream to seal."""

    def is_live_protected_patch(self, patch) -> bool:
        """No patch is protected."""
        del patch
        return False

    def extend_live_stream(self, operation, patch) -> bool:
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


class FeedbackStreams:
    """The stream bindings, the live streams and the protected cycle."""

    def __init__(
        self,
        engine,
        *,
        qpu,
        window_manager,
        regions,
        resolved_operations,
        resolved_patches,
        retry_ready_operations,
    ):
        self.engine = engine
        self.qpu = qpu
        self.windows = window_manager
        self.table = _StreamTable(
            regions, resolved_operations, resolved_patches
        )
        self.live_by_stream_id: dict = {}
        # operation id -> StreamBinding
        self.bindings: dict = {}
        self.retry_ready_operations = retry_ready_operations

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

    def blocks_start(self, operation) -> bool:
        """True while a live protected stream holds the operation."""
        live_patches = self._live_protected_patches()
        patches = set(operation.patches)
        active_patches = patches.intersection(live_patches)
        starting_regions = self.table.regions_starting_at(operation.id)
        starts_on_live_patch = any(
            region.patch_id in active_patches for region in starting_regions
        )
        if starts_on_live_patch:
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

    # ---- ending an operation

    def request_closes(self, operation) -> None:
        """A body finished: its ending regions close on the current boundary."""
        ending_regions = self.table.regions_ending_at(operation.id)
        for region in ending_regions:
            self._check_region_ends_on_boundary(region)
        for region in ending_regions:
            live = self._live(region.stream_id)
            live.is_close_requested = True

    def close_feedback_boundary(
        self, operation, waiting_blocked_successor: bool
    ) -> None:
        """Close the stream boundary at the body measurement.

        Only in measurement_closed mode, and only for a feedback source
        that still blocks a successor.
        """
        if operation.feedback_boundary_mode != "measurement_closed":
            return
        if not waiting_blocked_successor:
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
            self.windows.seal_stream(stream_id, live.next_round)

    # ---- idle rounds

    def is_live_protected_patch(self, patch) -> bool:
        """True while a protected stream is live on the patch."""
        live_patches = self._live_protected_patches()
        return patch in live_patches

    def extend_live_stream(self, operation, patch) -> bool:
        """Emit an idle round of the patch as the next round of the stream.

        False when the operation's stream is not live.
        """
        binding = self.bindings.get(operation.id)
        if binding is None:
            return False
        stream_id = binding.stream_id
        if not self.windows.has_dynamic_stream(stream_id):
            return False
        live = self._live(stream_id)
        live.next_round += 1
        self.qpu.emit_idle_stream_round(
            operation, stream_id, live.next_round, patch
        )
        return True

    # ---- private: bindings and reservations

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
        for live in self.live_by_stream_id.values():
            if live.region is not None:
                patches.add(live.region.patch_id)
        return patches

    def _boundary_open_patches(self) -> set:
        patches = set()
        for live in self.live_by_stream_id.values():
            if live.region is not None and live.is_boundary_open:
                patches.add(live.region.patch_id)
        return patches

    def _active_stream_id_of(self, patch):
        for stream_id, live in self.live_by_stream_id.items():
            if live.region is not None and live.region.patch_id == patch:
                return stream_id
        return None

    def _bind(self, operation_id, stream_id, stream_offset) -> None:
        binding = message.StreamBinding(stream_id, stream_offset)
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
        live_stream_ids = set()
        for patch in operation.patches:
            stream_id = self._active_stream_id_of(patch)
            if stream_id is not None:
                live_stream_ids.add(stream_id)
        starting_regions = self.table.regions_starting_at(operation.id)
        starting_stream_ids = {
            region.stream_id
            for region in starting_regions
            if region.patch_id in operation.patches
        }
        stream_ids = live_stream_ids | starting_stream_ids
        ordered_stream_ids = tuple(sorted(stream_ids))
        if len(ordered_stream_ids) > 1:
            raise ValueError(
                f"feedback source {operation.id} spans protected streams "
                f"{ordered_stream_ids}"
            )
        if not ordered_stream_ids:
            return None
        stream_id = ordered_stream_ids[0]
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
        return stream_id

    def _activate_protected_regions(self, operation) -> None:
        starting_regions = self.table.regions_starting_at(operation.id)
        live_patches = self._live_protected_patches()
        patches = set(operation.patches)
        protected_patches = patches.intersection(live_patches)
        for region in starting_regions:
            protected_patches.add(region.patch_id)
        if protected_patches and operation.emits_detector_data:
            raise ValueError(
                f"operation {operation.id} duplicates protected detector "
                "emission"
            )
        pending_patches = set()
        for region in starting_regions:
            is_live = region.patch_id in live_patches
            is_pending = region.patch_id in pending_patches
            if is_live or is_pending:
                raise RuntimeError(
                    f"protected patch {region.patch_id!r} already has a stream"
                )
            pending_patches.add(region.patch_id)
        for region in starting_regions:
            self._activate_region(region)

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
        cadence = self.table.round_ticks_of(region.patch_id)
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
        self.retry_ready_operations()
        next_round = live.next_round + 1
        emit_round = functools.partial(self._emit_protected_round, stream_id)
        self.engine.schedule(
            0,
            emit_round,
            label=f"protected-round({stream_id},{next_round})",
            priority=1,
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
            owner, stream_id, live.next_round, region.patch_id
        )
        live.last_emission_tick = self.engine.now
        if live.is_close_requested:
            live.next_boundary_tick = None
            seal = functools.partial(self._seal_protected_region, stream_id)
            self.engine.schedule(
                0, seal, label=f"protected-seal({stream_id})", priority=2
            )
            return
        next_boundary_round = live.next_round + 1
        self._schedule_next_boundary(region, boundary_round=next_boundary_round)

    def _seal_protected_region(self, stream_id) -> None:
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
        self.windows.seal_stream(stream_id, live.next_round)
        live.region = None
        live.is_close_requested = False
        live.last_emission_tick = None
        self.retry_ready_operations()

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

    def index(self, program) -> None:
        """Index the program: feedback sources, then every region once."""
        operations = program.operations
        for operation in operations:
            if operation.blocked_by is not None:
                self.feedback_source_ids.add(operation.blocked_by)
        executable_ids = {operation.id for operation in operations}
        self._reject_external_sources(
            "decode_ops", program.decode_operations, executable_ids
        )
        self._reject_external_sources(
            "dynamic_streams", program.dynamic_streams, executable_ids
        )
        operations_by_id = {operation.id: operation for operation in operations}
        stream_owners = {
            operation.id: operation for operation in program.dynamic_streams
        }
        regions = sorted(self.regions, key=_protected_region_stream_id)
        for region in regions:
            self._index_region(region, operations_by_id, stream_owners)

    def round_count_of(self, operation_id) -> int:
        return self.resolved_operation_by_id[operation_id].round_count

    def round_ticks_of(self, patch_id) -> int:
        return self.resolved_patch_by_identity[patch_id].round_ticks

    def regions_starting_at(self, operation_id) -> tuple:
        return self.regions_by_endpoint.get(("start", operation_id), ())

    def regions_ending_at(self, operation_id) -> tuple:
        return self.regions_by_endpoint.get(("end", operation_id), ())

    def is_protected(self, stream_id) -> bool:
        return stream_id in self.owner_by_stream_id

    def owner_of(self, stream_id):
        return self.owner_by_stream_id[stream_id]

    def is_feedback_source(self, operation_id) -> bool:
        return operation_id in self.feedback_source_ids

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
        protected_stream_ids = {region.stream_id for region in self.regions}
        protected_patch_ids = {region.patch_id for region in self.regions}
        touched_patches = protected_patch_ids.intersection(source.patches)
        touched_stream_ids = {
            region.stream_id
            for region in self.regions
            if region.patch_id in touched_patches
        }
        if source.id in protected_stream_ids:
            touched_stream_ids.add(source.id)
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

    def _index_region(
        self, region, operations_by_id: dict, stream_owners: dict
    ) -> None:
        stream_id = region.stream_id
        if stream_id in self.owner_by_stream_id:
            raise ValueError(f"duplicate protected stream {stream_id}")
        owner = stream_owners.get(stream_id)
        if owner is None:
            raise ValueError(
                f"protected stream {stream_id} owner/patch mismatch"
            )
        owner_patches = tuple(owner.patches)
        if owner_patches != (region.patch_id,):
            raise ValueError(
                f"protected stream {stream_id} owner/patch mismatch"
            )
        endpoints = (
            ("start", region.start_operation_id),
            ("end", region.end_operation_id),
        )
        for endpoint, operation_id in endpoints:
            operation = operations_by_id.get(operation_id)
            _check_region_endpoint(region, endpoint, operation)
            regions = self.regions_by_endpoint.setdefault(
                (endpoint, operation_id), []
            )
            regions.append(region)
        self.owner_by_stream_id[stream_id] = owner


def _protected_region_stream_id(region):
    return region.stream_id


def _check_region_endpoint(region, endpoint: str, operation) -> None:
    stream_id = region.stream_id
    if operation is None:
        raise ValueError(f"protected stream {stream_id} invalid {endpoint}")
    if region.patch_id not in operation.patches:
        raise ValueError(f"protected stream {stream_id} invalid {endpoint}")
