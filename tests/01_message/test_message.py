from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from decsim import message


def make_fragment(**overrides):
    values = {
        "operation_id": 7,
        "patch_id": "patch-a",
        "round_index": 3,
        "bits": (0, 1),
        "code": "surface_code",
        "size_bits": 2,
        "fragment_index": 0,
    }
    values.update(overrides)
    return message.RetainedSyndromeFragment(**values)


def make_window_geometry(start_round):
    return message.WindowGeometry(
        buffer_lo=start_round,
        commit_lo=start_round,
        commit_hi=start_round,
        buffer_hi=start_round,
    )


def make_operation_window_plan(**overrides):
    values = {
        "operation_id": 9,
        "windows": (
            make_window_geometry(1),
            make_window_geometry(2),
            make_window_geometry(3),
        ),
        "internal_dependencies": ((0, 1), (1, 2)),
        "entry_window_indices": (0,),
        "exit_window_indices": (2,),
        "windowed": True,
        "batch_preceding_idle_rounds": False,
    }
    values.update(overrides)
    return message.OperationWindowPlan(**values)


def make_soft_output_source(**overrides):
    values = {
        "method": "matching_gap",
        "cluster_origin": "decoder",
        "growth_schedule": "uniform",
        "gap_units": "natural_log",
        "correction": "minimum_weight",
        "weight_step_natural_log": 0.5,
        "references": ("source",),
    }
    values.update(overrides)
    return message.SoftOutputSource(**values)


def make_operation(**overrides):
    values = {
        "id": 4,
        "name": "memory",
        "qubits": ("q0",),
    }
    values.update(overrides)
    return message.Operation(**values)


def assert_frozen(value):
    with pytest.raises(FrozenInstanceError):
        value.unexpected_attribute = "changed"


# Stable identity


def test_stable_string_accepts_exact_scalar_text_only():
    """Stable strings exclude subclasses, nonstrings, and surrogate code points."""
    class StringSubclass(str):
        pass

    assert message.is_stable_string("")
    assert message.is_stable_string("logical-π")
    assert not message.is_stable_string(StringSubclass("logical"))
    assert not message.is_stable_string(3)
    assert not message.is_stable_string("\ud800")


def test_stable_identity_accepts_recursive_exact_values_only():
    """Stable identities contain only exact integers, scalar strings, and tuples."""
    assert message.is_stable_identity((1, "patch", (2, "round")))
    assert message.is_stable_identity(())
    assert not message.is_stable_identity(True)
    assert not message.is_stable_identity([1, "patch"])
    assert not message.is_stable_identity((1, object()))


def test_stable_identity_comparison_rejects_cross_type_equality():
    """Stable identity comparison never aliases values of different runtime types."""
    assert message.same_stable_identity((1, "1"), (1, "1"))
    assert not message.same_stable_identity(1, True)
    assert not message.same_stable_identity((1,), (True,))
    assert not message.same_stable_identity((1,), (1, 2))


def test_canonical_identity_bytes_are_injective_across_identity_kinds():
    """Canonical identity bytes distinguish integer, string, and tuple structure."""
    identities = (
        1,
        "1",
        (1,),
        ("1",),
        (1, "1"),
        ((1,),),
    )
    encodings = tuple(message.stable_identity_bytes(value) for value in identities)
    assert len(set(encodings)) == len(identities)
    encoded_integer = b"I" + (1).to_bytes(8, "big") + b"1"
    encoded_string = b"S" + (1).to_bytes(8, "big") + b"1"
    encoded_tuple = (
        b"T"
        + (1).to_bytes(8, "big")
        + len(encoded_integer).to_bytes(8, "big")
        + encoded_integer
    )
    unicode_payload = "π".encode("utf-8")
    encoded_unicode = (
        b"S" + len(unicode_payload).to_bytes(8, "big") + unicode_payload
    )
    assert encodings[0] == encoded_integer
    assert encodings[1] == encoded_string
    assert encodings[2] == encoded_tuple
    assert message.stable_identity_bytes("π") == encoded_unicode


def test_stable_identity_ordering_and_json_preserve_typed_structure():
    """Stable identity ordering and JSON use the canonical typed representation."""
    identity = (3, "patch", (4,))
    assert message.stable_identity_order_key(identity) == message.stable_identity_bytes(identity)
    assert message.stable_identity_json(identity) == {
        "kind": "tuple",
        "value": None,
        "items": [
            {"kind": "integer", "value": "3", "items": None},
            {"kind": "string", "value": "patch", "items": None},
            {
                "kind": "tuple",
                "value": None,
                "items": [
                    {"kind": "integer", "value": "4", "items": None}
                ],
            },
        ],
    }


def test_seed_path_segments_have_distinct_framed_encodings():
    """Seed path segments encode field, string, none, and integer edges distinctly."""
    segments = (
        message.RunSeedPathSegment("field", "node"),
        message.RunSeedPathSegment("string_key", "node"),
        message.RunSeedPathSegment("none_key", None),
        message.RunSeedPathSegment("integer_key", 12),
    )
    encodings = tuple(segment.canonical_bytes() for segment in segments)
    assert len(set(encodings)) == len(encodings)
    assert encodings == (
        b"F" + (4).to_bytes(4, "big") + "node".encode("utf-8"),
        b"S" + (4).to_bytes(4, "big") + "node".encode("utf-8"),
        b"N" + (0).to_bytes(4, "big"),
        b"I" + (2).to_bytes(4, "big") + b"12",
    )
    for segment in segments:
        assert_frozen(segment)


def test_seed_path_segments_reject_unknown_kinds():
    """An unknown segment kind has no canonical encoding."""
    with pytest.raises(KeyError):
        message.RunSeedPathSegment("unknown", "value").canonical_bytes()


def test_seed_child_skips_container_validation_and_is_frozen():
    """Seed child edges retain supplied paths without constructor shape checks."""
    child = object()
    edge = message.RunSeedChild(relative_path=[], child=child)
    assert edge.relative_path == []
    assert edge.child is child
    assert_frozen(edge)


def test_seed_reservation_skips_source_seed_validation_and_is_frozen():
    """Seed reservations retain inconsistent proposals without constructor checks."""
    prepared_state = object()
    reservation = message.RunSeedReservation("entropy", 19, prepared_state)
    assert reservation.proposed_seed_source == "entropy"
    assert reservation.proposed_seed == 19
    assert reservation.prepared_state is prepared_state
    assert_frozen(reservation)


# Syndrome payloads and packets


def test_qpu_readout_skips_identity_round_and_fragment_validation():
    """QPU readout stores unconventional metadata without constructor validation."""
    readout = message.QPUReadout(
        operation_id=object(),
        patch_id=[],
        round_index=0,
        code="",
        n_fragments=0,
        fragment_index=-1,
        size_bits=-1,
    )
    assert readout.round_index == 0
    assert readout.n_fragments == 0
    assert readout.fragment_index == -1
    assert_frozen(readout)


def test_syndrome_payload_skips_fragment_validation_and_remains_mutable():
    """Syndrome payload stores unchecked fragment metadata and remains mutable."""
    payload = message.SyndromePayload(
        operation_id=1,
        patch_id=2,
        round_index=0,
        n_fragments=0,
        fragment_index=-1,
    )
    payload.fragment_index = 4
    assert payload.round_index == 0
    assert payload.n_fragments == 0
    assert payload.fragment_index == 4


def test_binary_bits_normalize_none_lists_and_tuples():
    """Binary bit normalization converts supported Python forms to immutable integers."""
    assert message.normalize_binary_bits(None) is None
    assert message.normalize_binary_bits([True, False, 1, 0]) == (1, 0, 1, 0)
    assert message.normalize_binary_bits((0, True, 1)) == (0, 1, 1)


def test_binary_bits_normalize_one_dimensional_boolean_arrays():
    """Binary bit normalization converts one-dimensional NumPy boolean arrays."""
    bits = np.array([True, False, True], dtype=bool)
    assert message.normalize_binary_bits(bits) == (1, 0, 1)


def test_retained_fragment_normalizes_payload_bits():
    """Retained fragments normalize payload bits while copying transport metadata."""
    payload = message.SyndromePayload(
        operation_id="operation",
        patch_id="patch",
        round_index=2,
        bits=[True, 0],
        code="code",
        fragment_index=3,
        size_bits=2,
    )
    fragment = message.RetainedSyndromeFragment.from_payload(payload)
    assert fragment.bits == (1, 0)
    assert fragment.operation_id == "operation"
    assert fragment.fragment_index == 3


def test_round_packet_preserves_supplied_fragment_order_and_is_frozen():
    """Round packets preserve supplied fragment order without sorting."""
    later_fragment = make_fragment(patch_id="patch-b", fragment_index=1)
    earlier_fragment = make_fragment(patch_id="patch-a", fragment_index=0)
    packet = message.SyndromeRoundPacket(
        operation_id=7,
        round_index=3,
        fragments=(later_fragment, earlier_fragment),
    )
    assert packet.fragments == (later_fragment, earlier_fragment)
    assert_frozen(packet)


# Window geometry and plans


def test_window_start_round_and_key_reflect_runtime_geometry():
    """Runtime windows expose their leading start round and operation-window key."""
    window = message.Window(4, 2, 3, 5, 6, 4, buffer_lo=1)
    assert window.start_round == 1
    assert window.key == (4, 2)

    window.buffer_lo = None
    assert window.start_round == 3


def test_window_info_snapshots_topology_and_detector_positions():
    """Window information copies mutable topology and detector-position inputs."""
    window = message.Window(4, 2, 3, 5, 6, 4, buffer_lo=1)
    window.deps.append((4, 1))
    window.dependents.append((4, 3))
    detector_positions = {7: (1, 2)}

    info = message.WindowInfo.from_window(
        window,
        detector_positions=detector_positions,
    )
    window.deps.append((4, 0))
    detector_positions[8] = (2, 3)

    assert info.start_round == 1
    assert info.deps == ((4, 1),)
    assert info.dependents == ((4, 3),)
    assert info.detector_positions == {7: (1, 2)}
    assert_frozen(info)


def test_resolved_planning_skips_noncount_type_validation():
    """Resolved planning accepts unchecked names, flags, identities, and geometry carriers."""
    geometry = message.ResolvedCodeGeometry(
        code_name=object(),
        distance=3,
        commit_round_count=2,
        buffer_round_count=0,
        minimum_leading_buffer_round_count=0,
        minimum_trailing_buffer_round_count=0,
        one_patch_spatial_node_count=5,
        buffer_floor_override_active="unchecked",
    )
    operation_plan = message.ResolvedOperationPlanning(object(), object(), 0, 1, 1)
    patch_plan = message.ResolvedPatchPlanning(object(), object(), 1, 1)
    assert geometry.code_name is not None
    assert operation_plan.code_geometry is not None
    assert patch_plan.patch_identity is not None
    assert_frozen(geometry)
    assert_frozen(operation_plan)
    assert_frozen(patch_plan)


def test_operation_window_plan_accepts_declared_nonchecks():
    """Operation window plans skip generic container and selected field type checks."""
    plan = make_operation_window_plan(
        operation_id=object(),
        windows=[
            make_window_geometry(1),
            make_window_geometry(2),
            make_window_geometry(3),
        ],
        internal_dependencies=[(0, 1), (1, 2)],
        windowed="unchecked",
        protocol=object(),
    )
    assert len(plan.windows) == 3
    assert plan.windowed == "unchecked"


def test_compiled_window_plan_carries_mutable_manager_mappings():
    """Compiled window plans carry manager mappings that remain mutable."""
    plan = message.WindowPlan(
        windows={},
        window_count={},
        op_windows={},
        successors={},
        spatial_nodes={},
        rounds_by_operation={},
        code_names={},
        total_windows=0,
        windowed_by_operation={},
        batch_preceding_idle_rounds_by_operation={},
    )
    plan.window_count[4] = 2
    assert plan.window_count == {4: 2}


def test_dependency_residual_skips_container_order_and_defects_validation():
    """Dependency residuals accept unsorted list identifiers and unchecked defects views."""
    defects = object()
    residual = message.DependencyResidual([2, 0, 1], defects)
    assert residual.detector_ids == [2, 0, 1]
    assert residual.defects is defects
    assert_frozen(residual)


def test_boundary_delivery_is_current_only_at_both_latest_revisions():
    """Boundary delivery freshness requires current source and delivery revisions."""
    values = {
        "source_key": (1, 0),
        "destination_key": (1, 1),
        "source_revision": 2,
        "delivery_revision": 3,
        "latest_source_revision": 2,
        "latest_delivery_revision": 3,
        "source_operation_round_count": 5,
        "dependency_released": False,
        "payload": object(),
    }
    current = message.BoundaryDelivery(**values)
    stale_delivery_values = dict(values)
    stale_delivery_values["latest_delivery_revision"] = 4
    stale_delivery = message.BoundaryDelivery(**stale_delivery_values)
    stale_source_values = dict(values)
    stale_source_values["latest_source_revision"] = 3
    stale_source = message.BoundaryDelivery(**stale_source_values)
    assert current.is_current
    assert not stale_delivery.is_current
    assert not stale_source.is_current
    assert_frozen(current)


def test_boundary_update_carries_policy_decisions_and_is_frozen():
    """Boundary updates carry policy state and acceptance and release decisions."""
    state = object()
    update = message.BoundaryUpdate(state, accepted=True, release_dependency=False)
    assert update.state is state
    assert update.accepted
    assert not update.release_dependency
    assert_frozen(update)


def test_strong_region_plan_skips_runtime_type_checks():
    """Strong region plans accept comparable noninteger bounds without type checks."""
    plan = message.StrongRegionPlan(2.0, 4.0, 1.0, 5.0, 1.0, None)
    assert plan.context_lo == 1.0
    assert plan.restart_buffer_lo == 1.0


def test_window_protocol_exposes_distinct_scientific_tags():
    """Window protocols distinguish generic and graphlike scientific contracts."""
    assert message.WindowProtocol.GENERIC is not message.WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE


# Decode jobs and results


def test_soft_output_source_normalizes_weight_step_without_text_checks():
    """Soft output sources accept unchecked provenance and normalize their weight step."""
    source = make_soft_output_source(
        method=object(),
        cluster_origin="",
        references=[],
        weight_step_natural_log=1,
    )
    absent_step_source = make_soft_output_source(weight_step_natural_log=None)
    assert source.weight_step_natural_log == 1.0
    assert source.references == []
    assert absent_step_source.weight_step_natural_log is None
    assert_frozen(source)
    assert_frozen(absent_step_source)


def test_soft_output_normalizes_numbers_without_source_instance_validation():
    """Soft output accepts any source and normalizes valid numeric fields to floats."""
    source = object()
    output = message.SoftOutput(gap=2, source=source, w_min=-3, w_comp=float("inf"))
    assert output.source is source
    assert output.gap == 2.0
    infinite_gap_output = message.SoftOutput(gap=float("inf"), source=source)
    assert output.w_min == -3.0
    assert output.w_comp == float("inf")
    assert infinite_gap_output.gap == float("inf")
    assert_frozen(output)
    assert_frozen(infinite_gap_output)


def test_decoder_keys_preserve_request_and_service_identity():
    """Decoder keys carry operation, window, tier, and run-sequence identity."""
    request_key = message.DecoderRequestKey((4, "op"), 2, message.DecoderTier.STRONG, 7)
    service_key = message.DecoderServiceKey(8)
    assert request_key.operation_id == (4, "op")
    assert request_key.window_id == 2
    assert request_key.tier is message.DecoderTier.STRONG
    assert request_key.run_sequence == 7
    assert service_key.run_sequence == 8
    assert_frozen(request_key)
    assert_frozen(service_key)


def test_decode_job_is_mutable_and_has_no_detector_count_property():
    """Decode jobs retain mutable lifecycle state without a detector-count convenience property."""
    job = message.DecodeJob(op_id=4, window_id=2, n_rounds=3)
    job.completed = True
    assert job.completed
    assert not hasattr(job, "detector_count")


def test_decode_result_supports_timing_only_and_richer_results():
    """Decode results allow timing-only defaults and optional correction data."""
    timing_only = message.DecodeResult(op_id=4, window_id=2)
    soft_output = message.SoftOutput(gap=1, source=object())
    boundary_data = object()
    rich = message.DecodeResult(
        op_id=4,
        window_id=2,
        correction="X",
        logical_observables=(1, 0),
        soft_output=soft_output,
        boundary_defects={3: 1},
        boundary_data=boundary_data,
    )
    assert timing_only.correction is None
    assert rich.logical_observables == (1, 0)
    assert rich.soft_output is soft_output
    assert rich.boundary_defects == {3: 1}
    assert rich.boundary_data is boundary_data


def test_strong_completion_accepts_matching_strong_identity():
    """Strong completion accepts exact carriers with matching strong request identity."""
    key = message.DecoderRequestKey(4, 2, message.DecoderTier.STRONG, 7)
    result = message.DecodeResult(op_id=4, window_id=2)
    completion = message.StrongDecodeCompletion(key, result)
    assert completion.request_key is key
    assert completion.result is result
    assert_frozen(completion)


def test_decode_outcome_pairs_result_with_job():
    job = message.DecodeJob(op_id=4, window_id=2, n_rounds=3)
    result = message.DecodeResult(op_id=4, window_id=2)
    outcome = message.DecodeOutcome(job, result)
    assert outcome.job is job
    assert outcome.result is result


def test_soft_output_does_not_enforce_weight_relationships():
    """Soft output accepts weights without enforcing a relationship to the gap."""
    output = message.SoftOutput(gap=10, source=object(), w_min=20, w_comp=-5)
    assert output.gap == 10.0
    assert output.w_min == 20.0
    assert output.w_comp == -5.0


# Operations


def test_runtime_artifacts_carry_fields_and_are_frozen():
    """Runtime artifacts preserve resource, decision, and program-load fields immutably."""
    claim = message.ResourceClaim("qubits", frozenset({"q0"}))
    decision = message.Decision(4, releases_operation=False)
    program = message.ExecutionProgram(["operation"], ["decode"], ["stream"], ["region"])
    assert claim.ids == frozenset({"q0"})
    assert not decision.releases_operation
    assert program.operations == ["operation"]
    for artifact in (claim, decision, program):
        assert_frozen(artifact)


def test_protected_region_skips_identity_and_endpoint_validation():
    """Protected regions store unchecked patch, stream, and endpoint identities."""
    region = message.ProtectedRegion(object(), "stream", [], {})
    assert region.stream_id == "stream"
    assert region.start_operation_id == []
    assert_frozen(region)


def test_readiness_messages_carry_local_successor_and_tail_state():
    """Readiness messages carry successor progress, memory progress, and tail state."""
    successor = message.SuccessorReadiness(5, rounds_arrived=2, round_count=3)
    readiness = message.WindowReadiness(1, 3, (successor,), 2, tail_closed=False)
    assert readiness.successors == (successor,)
    assert readiness.memory_rounds_arrived == 2
    assert not readiness.tail_closed
    assert_frozen(successor)
    assert_frozen(readiness)


def test_operation_skips_general_identity_and_exact_type_validation():
    """Operations accept unchecked identities, predecessor fields, flags, and blockers."""
    operation = make_operation(
        id=object(),
        qubits=[object()],
        patches=[object()],
        predecessors=["previous"],
        decoder_boundary_predecessors=["boundary"],
        stream_id=object(),
        scheduled_start_round=0.5,
        emits_detector_data="yes",
        finalizes_stream_round=0,
        blocked_by=object(),
    )
    assert operation.scheduled_start_round == 0.5
    assert operation.predecessors == ["previous"]


def test_operation_accepts_consistent_fragment_slots():
    """Operations accept paired in-range fragment slots on detector emitters."""
    operation = make_operation(
        syndrome_fragment_index=0,
        syndrome_fragment_count=2,
        emits_detector_data=True,
    )
    assert operation.syndrome_fragment_index == 0
    assert operation.syndrome_fragment_count == 2


def test_operation_accepts_complete_stream_finalizers():
    """Operations accept stream finalizers with every required field."""
    operation = make_operation(
        finalizes_stream_round=True,
        emits_detector_data=True,
        stream_id="stream",
        stream_offset=0,
        syndrome_fragment_index=0,
        syndrome_fragment_count=1,
    )
    assert operation.finalizes_stream_round


def test_operation_magic_state_need_uses_override_then_clifford_fallback():
    """Magic-state need follows an explicit override before the Clifford fallback."""
    assert not make_operation(clifford=True).needs_magic_state
    assert make_operation(clifford=False).needs_magic_state
    assert not make_operation(clifford=False, consumes_magic_state=False).needs_magic_state
    assert make_operation(clifford=True, consumes_magic_state=True).needs_magic_state


def test_operation_planning_view_snapshots_configuration_without_circuit():
    """Planning views snapshot tuple fields, omit circuits, and resolve feedback mode."""
    circuit = object()
    operation = make_operation(
        qubits=["q0"],
        patches=["patch"],
        predecessors=[1],
        decoder_boundary_predecessors=[2],
        circuit=circuit,
    )
    view = message.OperationPlanningView.from_operation(operation)
    operation.qubits.append("q1")

    assert view.qubits == ("q0",)
    assert view.patches == ("patch",)
    assert view.feedback_boundary_mode == "trailing_buffer"
    assert not hasattr(view, "circuit")
    assert not hasattr(view, "needs_magic_state")
    assert_frozen(view)


def test_operation_planning_view_preserves_explicit_feedback_mode():
    """Planning views preserve an operation's explicit feedback boundary mode."""
    operation = make_operation(feedback_boundary_mode="committed_region")
    view = message.OperationPlanningView.from_operation(
        operation,
        default_feedback_boundary_mode="trailing_buffer",
    )
    assert view.feedback_boundary_mode == "committed_region"
