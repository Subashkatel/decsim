"""Turn admitted operations and feedback decisions into QPU commands.

Execution admission belongs to ``ExecutionRuntime``. Syndrome production
belongs to ``QPUDevice``.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Callable, Optional

from .links import LinkPath, TrafficAttribution

from .message import (
    QPUReadout, RunOperationBody, StreamBinding, SyndromePacketRoute,
    normalize_binary_bits,
    SyndromePayload, ExecutionProgram, Decision, Operation,
    stable_identity_order_key)


class Controller:
    """Sequence admitted operations and issue commands to the QPU."""

    def __init__(self, engine, *, qpu, window_manager,
                 syndrome_ingress=None, binary_availability_ticks: int = 0,
                 links=None,
                 round_ticks: int, code_geometry, resolved_operations,
                 resolved_patches, idle_policy, protected_regions=()):
        self.engine = engine
        self.qpu = qpu
        self.window_manager = window_manager
        self.syndrome_ingress = syndrome_ingress
        self.binary_availability_ticks = binary_availability_ticks
        self.links = links
        self.runtime = None
        self.round_ticks = round_ticks
        self._code_geometry = code_geometry
        self.idle_policy = idle_policy
        self._resolved_operations = MappingProxyType({
            operation.operation_id: operation
            for operation in resolved_operations
        })
        self._resolved_patches = MappingProxyType({
            patch.patch_identity: patch
            for patch in resolved_patches
        })
        self._protected_regions = tuple(protected_regions)
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

        self.idle_rounds_emitted = 0
        self.stream_next_round: dict = {}
        self._stream_binding_by_operation_id = {}

    def round_ticks_for(self, operation: Operation) -> int:
        return self._resolved_operations[operation.id].round_ticks

    def _round_ticks_for_patch(self, patch) -> int:
        return self._resolved_patches[patch].round_ticks

    def _round_count_for(self, operation: Operation) -> int:
        return self._resolved_operations[operation.id].round_count

    def _index_protected_regions(self, ops, dynamic_streams) -> None:
        operations_by_id = {operation.id: operation for operation in ops}
        stream_owners = {operation.id: operation for operation in dynamic_streams}
        for region in sorted(self._protected_regions, key=lambda item: item.stream_id):
            stream_id = region.stream_id
            if stream_id in self._stream_owner_by_id:
                raise ValueError(f"duplicate protected stream {stream_id}")
            owner = stream_owners.get(stream_id)
            if owner is None or tuple(owner.patches) != (region.patch_id,):
                raise ValueError(f"protected stream {stream_id} owner/patch mismatch")
            for endpoint, operation_id in (
                ("start", region.start_operation_id),
                ("end", region.end_operation_id),
            ):
                operation = operations_by_id.get(operation_id)
                if operation is None or region.patch_id not in operation.patches:
                    raise ValueError(
                        f"protected stream {stream_id} invalid {endpoint}")
            self._stream_owner_by_id[stream_id] = owner
            self._regions_starting_at.setdefault(region.start_operation_id, []).append(region)
            self._regions_ending_at.setdefault(region.end_operation_id, []).append(region)

    def load_program(self, program: ExecutionProgram) -> None:
        """Load one immutable program, build dependencies, and start roots."""
        ops = program.operations
        decode_ops = program.decode_operations
        dynamic_streams = program.dynamic_streams
        feedback_source_ids = {op.blocked_by for op in ops if op.blocked_by is not None}
        self._feedback_source_ids = feedback_source_ids
        executable_ids = {op.id for op in ops}
        protected_stream_ids = {region.stream_id for region in self._protected_regions}
        protected_patch_ids = {region.patch_id for region in self._protected_regions}
        for source_role, sources in (("decode_ops", decode_ops), ("dynamic_streams", dynamic_streams)):
            for source in sources:
                if source.id not in feedback_source_ids or source.id in executable_ids:
                    continue
                intersecting_patches = protected_patch_ids.intersection(source.patches)
                relevant_stream_ids = {region.stream_id for region in self._protected_regions
                                       if region.patch_id in intersecting_patches}
                if source.id in protected_stream_ids:
                    relevant_stream_ids.add(source.id)
                if not relevant_stream_ids:
                    continue
                ordered_streams = tuple(sorted(relevant_stream_ids))
                ordered_patches = tuple(sorted(intersecting_patches, key=stable_identity_order_key))
                raise ValueError(
                    f"external source {source.id} from {source_role} participates "
                    f"in protected feedback for protected streams "
                    f"{ordered_streams} and patches {ordered_patches}")
        self._index_protected_regions(ops, dynamic_streams)
        for operation in ops:
            if operation.stream_id is not None and operation.stream_offset is not None:
                self._stream_binding_by_operation_id[operation.id] = StreamBinding(
                    operation.stream_id, operation.stream_offset)
                self.window_manager.bind_stream_operation(
                    operation.id, operation.stream_id, operation.stream_offset)
            self.window_manager.register_op(operation)
        self.runtime.load_program(program)

    def connect_runtime(self, runtime) -> None:
        self.runtime = runtime

    def can_start(self, operation: Operation) -> bool:
        return not self._must_wait_for_round_boundary(operation)

    def issue_operation(self, operation: Operation, idle_rounds: int) -> int:
        """Command the QPU; return the cycle boundary the operation starts on."""
        self._begin(operation, idle_rounds)
        return self.qpu.next_boundary()


    def _must_wait_for_round_boundary(self, operation: Operation) -> bool:
        active_patches = set(operation.patches).intersection(self._active_stream_id_by_patch)
        if any(region.patch_id in active_patches
               for region in self._regions_starting_at.get(operation.id, ())):
            return True
        if active_patches:
            return not active_patches.issubset(self._boundary_open_patches)
        return False

    def _protected_feedback_stream(self, operation: Operation):
        if operation.id not in self._feedback_source_ids:
            return None
        relevant_stream_ids = {
            self._active_stream_id_by_patch[patch]
            for patch in operation.patches
            if patch in self._active_stream_id_by_patch
        }
        relevant_stream_ids.update(
            region.stream_id
            for region in self._regions_starting_at.get(operation.id, ())
            if region.patch_id in operation.patches
        )
        ordered_stream_ids = tuple(sorted(relevant_stream_ids))
        if len(ordered_stream_ids) > 1:
            raise ValueError(
                f"feedback source {operation.id} spans protected streams "
                f"{ordered_stream_ids}")
        if not ordered_stream_ids:
            return None
        stream_id = ordered_stream_ids[0]
        binding = self._stream_binding_by_operation_id.get(operation.id)
        declared_stream_id = binding.stream_id if binding is not None else operation.stream_id
        declared_stream_offset = binding.stream_offset if binding is not None else operation.stream_offset
        if declared_stream_id not in (None, stream_id):
            raise ValueError(
                f"feedback source {operation.id} has conflicting stream_id")
        current_round = self.stream_next_round.get(stream_id, 0)
        if declared_stream_offset not in (None, current_round):
            raise ValueError(
                f"feedback source {operation.id} has conflicting stream_offset")
        return stream_id

    def _activate_protected_regions(self, operation: Operation) -> None:
        starting_regions = self._regions_starting_at.get(operation.id, ())
        protected_patches = set(operation.patches).intersection(self._active_stream_id_by_patch)
        protected_patches.update(region.patch_id for region in starting_regions)
        if protected_patches and operation.emits_detector_data:
            raise ValueError(
                f"operation {operation.id} duplicates protected detector emission")
        pending_patches = set()
        for region in starting_regions:
            if (region.patch_id in self._active_stream_id_by_patch
                    or region.patch_id in pending_patches):
                raise RuntimeError(
                    f"protected patch {region.patch_id!r} already has a stream")
            pending_patches.add(region.patch_id)
        for region in starting_regions:
            stream_id = region.stream_id
            self._active_region_by_stream_id[stream_id] = region
            self._active_stream_id_by_patch[region.patch_id] = stream_id
            self.stream_next_round.setdefault(stream_id, 0)
            cadence = self._round_ticks_for_patch(region.patch_id)
            self._next_boundary_tick_by_stream_id[stream_id] = self.engine.now + cadence
            self.engine.schedule(
                cadence,
                lambda active_stream_id=stream_id:
                    self._open_protected_boundary(active_stream_id),
                label=f"protected-boundary({stream_id},1)",
            )

    def _begin(self, operation: Operation, idle_rounds: int) -> None:
        """Consume idle rounds, reserve stream indices, and command the QPU."""
        protected_stream_id = self._protected_feedback_stream(operation)
        self._activate_protected_regions(operation)
        if idle_rounds:
            self.window_manager.prepend_idle_rounds(operation.id, idle_rounds)
        if protected_stream_id is None:
            self._reserve_stream_rounds(operation)
        else:
            self._bind_protected_feedback_source(operation, protected_stream_id)
        kind = "Clifford" if operation.clifford else "non-Clifford"
        release_note = "" if operation.blocked_by is None \
            else f" [unblocked by op#{operation.blocked_by}]"
        self.engine.log("Controller", f"START {operation.name}  ({kind}, qubits "
                                f"{operation.qubits}){release_note}")
        binding = self._stream_binding_by_operation_id.get(operation.id)
        effective_operation = operation if binding is None else replace(
            operation, stream_id=binding.stream_id, stream_offset=binding.stream_offset)
        source_operation_id = binding.stream_id if binding is not None else operation.id
        self.qpu.issue(RunOperationBody(
            operation=effective_operation,
            round_ticks=self.round_ticks_for(operation),
            round_count=self._round_count_for(operation),
            source_round_count=self._resolved_operations[source_operation_id].round_count,

            emits_detector_data=operation.emits_detector_data,
            finalizes_stream_round=operation.finalizes_stream_round,
        ))

    def _bind_protected_feedback_source(self, operation: Operation,
                                        stream_id: int) -> None:
        stream_offset = self.stream_next_round.get(stream_id, 0)
        binding = StreamBinding(stream_id, stream_offset)
        self._stream_binding_by_operation_id[operation.id] = binding
        self.window_manager.bind_stream_operation(operation.id, stream_id, stream_offset)
        resolved_round_count = self._round_count_for(operation)
        required_stream_end = stream_offset + max(resolved_round_count, 1)
        self.window_manager.bind_required_stream_end(operation.id, required_stream_end)

    def _open_protected_boundary(self, stream_id: int) -> None:
        region = self._active_region_by_stream_id.get(stream_id)
        if region is None:
            raise RuntimeError(f"protected stream {stream_id} is not active")
        expected_tick = self._next_boundary_tick_by_stream_id.get(stream_id)
        if expected_tick != self.engine.now:
            raise RuntimeError(
                f"protected stream {stream_id} boundary tick mismatch: "
                f"{expected_tick} != {self.engine.now}")
        if region.patch_id in self._boundary_open_patches:
            raise RuntimeError(
                f"protected stream {stream_id} boundary already open")
        self._boundary_open_patches.add(region.patch_id)
        self.runtime.retry_ready_operations()
        next_round = self.stream_next_round.get(stream_id, 0) + 1
        self.engine.schedule(
            0,
            lambda active_stream_id=stream_id:
                self._emit_protected_round(active_stream_id),
            label=f"protected-round({stream_id},{next_round})",
            priority=1,
        )

    def _emit_protected_round(self, stream_id: int) -> None:
        region = self._active_region_by_stream_id.get(stream_id)
        if region is None:
            raise RuntimeError(f"protected stream {stream_id} is not active")
        if region.patch_id not in self._boundary_open_patches:
            raise RuntimeError(
                f"protected stream {stream_id} boundary is not open")
        self._boundary_open_patches.remove(region.patch_id)
        global_round = self.stream_next_round.get(stream_id, 0) + 1
        owner = self._stream_owner_by_id[stream_id]
        self.qpu.emit_idle_stream_round(
            owner, stream_id, global_round, region.patch_id)
        self.stream_next_round[stream_id] = global_round
        self._last_emission_tick_by_stream_id[stream_id] = self.engine.now
        if stream_id in self._close_requested_stream_ids:
            self._next_boundary_tick_by_stream_id.pop(stream_id, None)
            self.engine.schedule(
                0,
                lambda active_stream_id=stream_id:
                    self._seal_protected_region(active_stream_id),
                label=f"protected-seal({stream_id})",
                priority=2,
            )
            return
        cadence = self._round_ticks_for_patch(region.patch_id)
        self._next_boundary_tick_by_stream_id[stream_id] = self.engine.now + cadence
        self.engine.schedule(
            cadence,
            lambda active_stream_id=stream_id:
                self._open_protected_boundary(active_stream_id),
            label=f"protected-boundary({stream_id},{global_round + 1})",
        )

    def _seal_protected_region(self, stream_id: int) -> None:
        region = self._active_region_by_stream_id.get(stream_id)
        if region is None:
            raise RuntimeError(f"protected stream {stream_id} is not active")
        if stream_id not in self._close_requested_stream_ids:
            raise RuntimeError(f"protected stream {stream_id} was not closed")
        if stream_id in self._next_boundary_tick_by_stream_id:
            raise RuntimeError(f"protected stream {stream_id} has a pending boundary")
        if self._last_emission_tick_by_stream_id.get(stream_id) != self.engine.now:
            raise RuntimeError(
                f"protected stream {stream_id} lacks final-round evidence")
        self.window_manager.seal_stream(stream_id, self.stream_next_round[stream_id])
        self._active_region_by_stream_id.pop(stream_id)
        self._active_stream_id_by_patch.pop(region.patch_id)
        self._close_requested_stream_ids.remove(stream_id)
        self._last_emission_tick_by_stream_id.pop(stream_id, None)
        self.runtime.retry_ready_operations()


    def _reserve_stream_rounds(self, operation: Operation) -> None:
        binding = self._stream_binding_by_operation_id.get(operation.id)
        if binding is None and operation.stream_id is None:
            return
        stream_id = binding.stream_id if binding is not None else operation.stream_id
        stream_offset = binding.stream_offset if binding is not None else operation.stream_offset
        next_round = self.stream_next_round.get(stream_id, 0)
        if operation.finalizes_stream_round:
            if stream_offset != next_round - 1:
                raise RuntimeError(
                    f"{operation.name} must finalize stream round {next_round}")
            return
        if stream_offset is None:
            stream_offset = next_round
            binding = StreamBinding(stream_id, stream_offset)
            self._stream_binding_by_operation_id[operation.id] = binding
            self.window_manager.bind_stream_operation(operation.id, stream_id, stream_offset)
        elif stream_offset < next_round:
            raise RuntimeError(
                f"{operation.name} starts at stream round {stream_offset + 1}, "
                f"but stream {stream_id!r} has already reserved through round {next_round}")
        operation_end = stream_offset + self._round_count_for(operation)
        self.stream_next_round[stream_id] = max(next_round, operation_end)

    def stream_binding_for(self, operation_id):
        return self._stream_binding_by_operation_id.get(operation_id)

    def _body_done(self, operation: Operation) -> None:
        self.runtime.body_done(operation)

    def before_successor_release(self, operation: Operation) -> None:
        self._request_protected_region_closes(operation)

    def after_successor_release(self, operation: Operation) -> None:
        self._close_feedback_boundary_if_needed(operation)
        self._seal_finished_streams_if_needed()
        if self.runtime.workload_complete:
            self.qpu.finish()

    def _request_protected_region_closes(self, operation: Operation) -> None:
        ending_regions = self._regions_ending_at.get(operation.id, ())
        for region in ending_regions:
            stream_id = region.stream_id
            if self._active_region_by_stream_id.get(stream_id) is not region:
                raise RuntimeError(
                    f"protected stream {stream_id} ended while inactive")
            if self._active_stream_id_by_patch.get(region.patch_id) != stream_id:
                raise RuntimeError(
                    f"protected stream {stream_id} lost patch ownership")
            if self._next_boundary_tick_by_stream_id.get(stream_id) != self.engine.now:
                raise RuntimeError(
                    f"protected stream {stream_id} ended off boundary")
        self._close_requested_stream_ids.update(
            region.stream_id for region in ending_regions
        )

    def _close_feedback_boundary_if_needed(self, operation: Operation) -> None:
        if operation.feedback_boundary_mode != "measurement_closed":
            return
        if not self.runtime.waiting_blocked_successor(operation.id):
            return
        binding = self._stream_binding_by_operation_id.get(operation.id)
        if binding is None:
            return
        stream_round_count = binding.stream_offset + self._round_count_for(operation)
        self.window_manager.close_stream_boundary(
            binding.stream_id,
            stream_round_count,
        )

    def _seal_finished_streams_if_needed(self) -> None:
        if not self.runtime.workload_complete:
            return
        protected_stream_ids = self._stream_owner_by_id
        for stream_id, total_rounds in list(self.stream_next_round.items()):
            if stream_id in protected_stream_ids:
                continue
            if not self.window_manager.has_dynamic_stream(stream_id):
                continue
            self.window_manager.seal_stream(stream_id, total_rounds)

    def emit_idle_round(self, op_id: int, patch, round_index: int) -> None:
        """One idle cycle of a patch nobody is operating on: syndrome extraction
        never stops, so the round is produced, transmitted and accounted; the
        window manager decides through the idle policy whether it costs decode
        work. Patches on a live protected stream emit through that stream."""
        if patch in self._active_stream_id_by_patch:
            return
        self._relay_idle_round(op_id, patch, round_index)
        self.runtime.record_idle_round(patch)
        self.idle_rounds_emitted += 1
        self.idle_policy.account(1, self.runtime.operations[op_id])
        self._submit_idle_decode_if_due(op_id, patch, round_index)

    def _relay_idle_round(self, op_id: int, patch, round_index: int) -> None:
        if self._relay_idle_round_to_live_stream(op_id, patch):
            return
        self.qpu.emit_feedback_memory_round(op_id, patch, round_index)

    def _relay_idle_round_to_live_stream(self, op_id: int, patch) -> bool:
        if self.idle_policy.mode != "extend_stream":
            return False
        operation = self.runtime.operations[op_id]
        binding = self._stream_binding_by_operation_id.get(operation.id)
        stream_id = None if binding is None else binding.stream_id
        if (
            stream_id is None
            or not self.window_manager.has_dynamic_stream(stream_id)
        ):
            return False
        global_round = self.stream_next_round.get(stream_id, 0) + 1
        self.stream_next_round[stream_id] = global_round
        self.qpu.emit_idle_stream_round(
            operation, stream_id, global_round, patch)
        return True


    def accept_qpu_readout(
        self, readout: QPUReadout, route: SyndromePacketRoute,
    ) -> None:
        """Turn one QPU readout into controller binary and hand it to ingress."""
        payload = SyndromePayload(
            operation_id=readout.operation_id,
            patch_id=readout.patch_id,
            round_index=readout.round_index,
            bits=normalize_binary_bits(readout.bits),
            code=readout.code,
            n_fragments=readout.n_fragments,
            fragment_index=readout.fragment_index,
            size_bits=readout.size_bits,
        )
        self.syndrome_ingress.relay_qpu_readout(
            payload, route, processing_ticks=self.binary_availability_ticks)


    def _submit_idle_decode_if_due(self, op_id: int, patch,
                                   round_index: int) -> None:
        if self.idle_policy.mode != "separate_decode_jobs":
            return
        patch_record = self._resolved_patches[patch]
        geometry = patch_record.code_geometry
        if round_index % geometry.commit_round_count == 0:
            self.window_manager.accept_idle_decode_demand(
                rounds=geometry.commit_round_count + geometry.buffer_round_count,
                code=geometry.code_name,
                spatial_nodes=patch_record.spatial_node_count,
                label=f"mem({self.runtime.operations[op_id].name},r{round_index})")

    def relay_instruction(self, decision: Decision,
                          deliver: Callable[[Decision], None]) -> None:
        """Model OC receipt followed by CQ instruction delivery."""
        if self.links is None:
            deliver(decision)
            return
        attribution = TrafficAttribution(
            operation_id=decision.target_operation_id, patch_ids=(),
            window_id=None, round_lo=None, round_hi=None)
        def at_controller():
            cq = self.links.reserve(
                LinkPath.CQ, payload_bits=None, now_ticks=self.engine.now,
                attribution=attribution).total_delay_ticks
            self.engine.schedule(
                cq, lambda: deliver(decision), label="controller->qpu")
        oc = self.links.reserve(
            LinkPath.OC, payload_bits=None, now_ticks=self.engine.now,
            attribution=attribution).total_delay_ticks
        self.engine.schedule(oc, at_controller, label="orchestrator->controller")
