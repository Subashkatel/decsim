"""Stream bookkeeping on the controller's QPU-facing side.

A stream is a run of syndrome rounds shared by several operations
(segments) on one patch. This object owns which operation is bound to
which stream round, the next free round of every stream, and the
protected regions: a protected region keeps one live stream on a patch
between a start and an end operation, emits one round of it per QEC cycle
at the cycle boundary, holds operations that need the patch until that
boundary, and seals the stream only after its final round. Nothing here
schedules decoding; the window manager learns about bindings, closed
boundaries and seals through the calls below.

NoFeedbackStreams is what a run without streams gets: every method is a
no-op with the same interface.
"""

import functools
import types

import decsim.message as message


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


class FeedbackStreams:
    """The stream bindings, the next free rounds and the protected cycle."""

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
        self.window_manager = window_manager
        self._retry_ready_operations = retry_ready_operations
        operation_by_id = {
            operation.operation_id: operation
            for operation in resolved_operations
        }
        self._resolved_operations = types.MappingProxyType(operation_by_id)
        patch_by_identity = {
            patch.patch_identity: patch for patch in resolved_patches
        }
        self._resolved_patches = types.MappingProxyType(patch_by_identity)
        self._protected_regions = tuple(regions)
        self._regions_starting_at = {}
        self._regions_ending_at = {}
        self._stream_owner_by_id = {}
        self._active_region_by_stream_id = {}
        self._active_stream_id_by_patch = {}
        self._next_boundary_tick_by_stream_id = {}
        self._close_requested_stream_ids = set()
        self._boundary_open_patches = set()
        self._last_emission_tick_by_stream_id = {}
        self._feedback_source_ids = set()
        self.stream_next_round: dict = {}
        self._binding_by_operation_id = {}

    # ---- program load

    def load(self, program) -> None:
        """Index the protected regions and the declared stream bindings."""
        operations = program.operations
        self._feedback_source_ids = {
            operation.blocked_by
            for operation in operations
            if operation.blocked_by is not None
        }
        self._reject_external_feedback_sources(program)
        self._index_protected_regions(operations, program.dynamic_streams)
        for operation in operations:
            has_stream = operation.stream_id is not None
            has_offset = operation.stream_offset is not None
            if has_stream and has_offset:
                self._bind(
                    operation.id, operation.stream_id, operation.stream_offset
                )

    def _reject_external_feedback_sources(self, program) -> None:
        """Refuse a feedback source that is not itself executed.

        A decode operation or a dynamic stream may not be the feedback
        source of a protected stream or patch.
        """
        executable_ids = {operation.id for operation in program.operations}
        self._reject_external_sources(
            "decode_ops", program.decode_operations, executable_ids
        )
        self._reject_external_sources(
            "dynamic_streams", program.dynamic_streams, executable_ids
        )

    def _reject_external_sources(
        self, source_role: str, sources, executable_ids: set
    ) -> None:
        for source in sources:
            is_feedback_source = source.id in self._feedback_source_ids
            is_executable = source.id in executable_ids
            if is_feedback_source and not is_executable:
                self._reject_external_source(source_role, source)

    def _reject_external_source(self, source_role: str, source) -> None:
        protected_stream_ids = {
            region.stream_id for region in self._protected_regions
        }
        protected_patch_ids = {
            region.patch_id for region in self._protected_regions
        }
        touched_patches = protected_patch_ids.intersection(source.patches)
        touched_stream_ids = {
            region.stream_id
            for region in self._protected_regions
            if region.patch_id in touched_patches
        }
        if source.id in protected_stream_ids:
            touched_stream_ids.add(source.id)
        if not touched_stream_ids:
            return
        ordered_streams = tuple(sorted(touched_stream_ids))
        ordered_patches = tuple(
            sorted(touched_patches, key=message.stable_identity_order_key)
        )
        raise ValueError(
            f"external source {source.id} from {source_role} participates "
            f"in protected feedback for protected streams "
            f"{ordered_streams} and patches {ordered_patches}"
        )

    def _index_protected_regions(self, operations, dynamic_streams) -> None:
        operations_by_id = {operation.id: operation for operation in operations}
        stream_owners = {
            operation.id: operation for operation in dynamic_streams
        }
        regions = sorted(
            self._protected_regions, key=_protected_region_stream_id
        )
        for region in regions:
            self._index_protected_region(
                region, operations_by_id, stream_owners
            )

    def _index_protected_region(
        self, region, operations_by_id: dict, stream_owners: dict
    ) -> None:
        stream_id = region.stream_id
        if stream_id in self._stream_owner_by_id:
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
        self._stream_owner_by_id[stream_id] = owner
        starting = self._regions_starting_at.setdefault(
            region.start_operation_id, []
        )
        starting.append(region)
        ending = self._regions_ending_at.setdefault(region.end_operation_id, [])
        ending.append(region)

    def _bind(self, operation_id, stream_id, stream_offset) -> None:
        binding = message.StreamBinding(stream_id, stream_offset)
        self._binding_by_operation_id[operation_id] = binding
        self.window_manager.bind_stream_operation(
            operation_id, stream_id, stream_offset
        )

    def binding_for(self, operation_id):
        """The stream binding an operation was given, or None."""
        return self._binding_by_operation_id.get(operation_id)

    def _declared_stream(self, operation) -> tuple:
        """The operation's (stream_id, stream_offset), either may be None.

        The recorded binding wins over the operation's own declaration.
        """
        binding = self._binding_by_operation_id.get(operation.id)
        if binding is None:
            return operation.stream_id, operation.stream_offset
        return binding.stream_id, binding.stream_offset

    def _active_region(self, stream_id):
        region = self._active_region_by_stream_id.get(stream_id)
        assert region is not None, f"protected stream {stream_id} is not active"
        return region

    # ---- starting an operation

    def blocks_start(self, operation) -> bool:
        """True while a live protected stream holds the operation."""
        patches = set(operation.patches)
        active_patches = patches.intersection(self._active_stream_id_by_patch)
        starting_regions = self._regions_starting_at.get(operation.id, ())
        starts_on_live_patch = any(
            region.patch_id in active_patches for region in starting_regions
        )
        if starts_on_live_patch:
            return True
        if not active_patches:
            return False
        return not active_patches.issubset(self._boundary_open_patches)

    def begin(self, operation) -> None:
        """Activate the regions the operation starts; give it its rounds."""
        protected_stream_id = self._protected_feedback_stream(operation)
        self._activate_protected_regions(operation)
        if protected_stream_id is None:
            self._reserve_stream_rounds(operation)
        else:
            self._bind_protected_feedback_source(operation, protected_stream_id)

    def _protected_feedback_stream(self, operation):
        """The one protected stream a feedback source feeds, or None."""
        if operation.id not in self._feedback_source_ids:
            return None
        live_stream_ids = {
            self._active_stream_id_by_patch[patch]
            for patch in operation.patches
            if patch in self._active_stream_id_by_patch
        }
        starting_regions = self._regions_starting_at.get(operation.id, ())
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
        current_round = self.stream_next_round.get(stream_id, 0)
        if declared_offset not in (None, current_round):
            raise ValueError(
                f"feedback source {operation.id} has conflicting stream_offset"
            )
        return stream_id

    def _activate_protected_regions(self, operation) -> None:
        starting_regions = self._regions_starting_at.get(operation.id, ())
        patches = set(operation.patches)
        protected_patches = patches.intersection(
            self._active_stream_id_by_patch
        )
        for region in starting_regions:
            protected_patches.add(region.patch_id)
        if protected_patches and operation.emits_detector_data:
            raise ValueError(
                f"operation {operation.id} duplicates protected detector "
                "emission"
            )
        pending_patches = set()
        for region in starting_regions:
            is_live = region.patch_id in self._active_stream_id_by_patch
            is_pending = region.patch_id in pending_patches
            if is_live or is_pending:
                raise RuntimeError(
                    f"protected patch {region.patch_id!r} already has a stream"
                )
            pending_patches.add(region.patch_id)
        for region in starting_regions:
            self._activate_region(region)

    def _activate_region(self, region) -> None:
        stream_id = region.stream_id
        self._active_region_by_stream_id[stream_id] = region
        self._active_stream_id_by_patch[region.patch_id] = stream_id
        self.stream_next_round.setdefault(stream_id, 0)
        self._schedule_next_boundary(
            stream_id, region.patch_id, boundary_round=1
        )

    def _bind_protected_feedback_source(self, operation, stream_id) -> None:
        stream_offset = self.stream_next_round.get(stream_id, 0)
        self._bind(operation.id, stream_id, stream_offset)
        resolved = self._resolved_operations[operation.id]
        round_count = max(resolved.round_count, 1)
        required_stream_end = stream_offset + round_count
        self.window_manager.bind_required_stream_end(
            operation.id, required_stream_end
        )

    def _reserve_stream_rounds(self, operation) -> None:
        """Give an unprotected stream segment its rounds.

        A segment that declares no offset is bound at the next free
        round; an offset already reserved is refused.
        """
        stream_id, stream_offset = self._declared_stream(operation)
        if stream_id is None:
            return
        next_round = self.stream_next_round.get(stream_id, 0)
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
        resolved = self._resolved_operations[operation.id]
        operation_end = stream_offset + resolved.round_count
        self.stream_next_round[stream_id] = max(next_round, operation_end)

    # ---- the protected cycle: boundary, round, seal

    def _schedule_next_boundary(
        self, stream_id, patch_id, *, boundary_round: int
    ) -> None:
        cadence = self._resolved_patches[patch_id].round_ticks
        boundary_tick = self.engine.now + cadence
        self._next_boundary_tick_by_stream_id[stream_id] = boundary_tick
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
        region = self._active_region(stream_id)
        expected_tick = self._next_boundary_tick_by_stream_id.get(stream_id)
        assert expected_tick == self.engine.now, (
            f"protected stream {stream_id} boundary tick mismatch: "
            f"{expected_tick} != {self.engine.now}"
        )
        assert region.patch_id not in self._boundary_open_patches, (
            f"protected stream {stream_id} boundary already open"
        )
        self._boundary_open_patches.add(region.patch_id)
        self._retry_ready_operations()
        next_round = self.stream_next_round.get(stream_id, 0) + 1
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
        region = self._active_region(stream_id)
        assert region.patch_id in self._boundary_open_patches, (
            f"protected stream {stream_id} boundary is not open"
        )
        self._boundary_open_patches.remove(region.patch_id)
        global_round = self.stream_next_round.get(stream_id, 0) + 1
        owner = self._stream_owner_by_id[stream_id]
        self.qpu.emit_idle_stream_round(
            owner, stream_id, global_round, region.patch_id
        )
        self.stream_next_round[stream_id] = global_round
        self._last_emission_tick_by_stream_id[stream_id] = self.engine.now
        if stream_id in self._close_requested_stream_ids:
            self._next_boundary_tick_by_stream_id.pop(stream_id, None)
            seal = functools.partial(self._seal_protected_region, stream_id)
            self.engine.schedule(
                0, seal, label=f"protected-seal({stream_id})", priority=2
            )
            return
        next_boundary_round = global_round + 1
        self._schedule_next_boundary(
            stream_id, region.patch_id, boundary_round=next_boundary_round
        )

    def _seal_protected_region(self, stream_id) -> None:
        region = self._active_region(stream_id)
        assert stream_id in self._close_requested_stream_ids, (
            f"protected stream {stream_id} was not closed"
        )
        assert stream_id not in self._next_boundary_tick_by_stream_id, (
            f"protected stream {stream_id} has a pending boundary"
        )
        last_emission = self._last_emission_tick_by_stream_id.get(stream_id)
        assert last_emission == self.engine.now, (
            f"protected stream {stream_id} lacks final-round evidence"
        )
        self.window_manager.seal_stream(
            stream_id, self.stream_next_round[stream_id]
        )
        self._active_region_by_stream_id.pop(stream_id)
        self._active_stream_id_by_patch.pop(region.patch_id)
        self._close_requested_stream_ids.remove(stream_id)
        self._last_emission_tick_by_stream_id.pop(stream_id, None)
        self._retry_ready_operations()

    # ---- ending an operation

    def request_closes(self, operation) -> None:
        """A body finished: its ending regions close on the current boundary."""
        ending_regions = self._regions_ending_at.get(operation.id, ())
        for region in ending_regions:
            self._check_region_ends_on_boundary(region)
        for region in ending_regions:
            self._close_requested_stream_ids.add(region.stream_id)

    def _check_region_ends_on_boundary(self, region) -> None:
        stream_id = region.stream_id
        active_region = self._active_region_by_stream_id.get(stream_id)
        assert active_region is region, (
            f"protected stream {stream_id} ended while inactive"
        )
        active_stream_id = self._active_stream_id_by_patch.get(region.patch_id)
        assert active_stream_id == stream_id, (
            f"protected stream {stream_id} lost patch ownership"
        )
        boundary_tick = self._next_boundary_tick_by_stream_id.get(stream_id)
        assert boundary_tick == self.engine.now, (
            f"protected stream {stream_id} ended off boundary"
        )

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
        binding = self._binding_by_operation_id.get(operation.id)
        if binding is None:
            return
        resolved = self._resolved_operations[operation.id]
        stream_round_count = binding.stream_offset + resolved.round_count
        self.window_manager.close_stream_boundary(
            binding.stream_id, stream_round_count
        )

    def seal_finished_streams(self) -> None:
        """The workload is complete: seal every unprotected dynamic stream."""
        stream_rounds = self.stream_next_round.items()
        for stream_id, total_rounds in list(stream_rounds):
            if stream_id in self._stream_owner_by_id:
                continue
            if not self.window_manager.has_dynamic_stream(stream_id):
                continue
            self.window_manager.seal_stream(stream_id, total_rounds)

    # ---- idle rounds

    def is_live_protected_patch(self, patch) -> bool:
        """True while a protected stream is live on the patch."""
        return patch in self._active_stream_id_by_patch

    def extend_live_stream(self, operation, patch) -> bool:
        """Emit an idle round of the patch as the next round of the stream.

        False when the operation's stream is not live.
        """
        binding = self._binding_by_operation_id.get(operation.id)
        if binding is None:
            return False
        stream_id = binding.stream_id
        if not self.window_manager.has_dynamic_stream(stream_id):
            return False
        global_round = self.stream_next_round.get(stream_id, 0) + 1
        self.stream_next_round[stream_id] = global_round
        self.qpu.emit_idle_stream_round(
            operation, stream_id, global_round, patch
        )
        return True


def _protected_region_stream_id(region):
    return region.stream_id


def _check_region_endpoint(region, endpoint: str, operation) -> None:
    stream_id = region.stream_id
    if operation is None:
        raise ValueError(f"protected stream {stream_id} invalid {endpoint}")
    if region.patch_id not in operation.patches:
        raise ValueError(f"protected stream {stream_id} invalid {endpoint}")
