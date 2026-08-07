#==================================================================
# TESTS FOR WORKLOAD GRAPH VALIDATION
# A malformed operation graph must be rejected with a named ValueError
# BEFORE planning or simulation: duplicate ids silently replace ops in
# dict lookups, unknown predecessors crash as KeyError, self-deps and
# cycles hang or no-op an empty event queue, and unknown blocked_by /
# duplicate stream ids leave ops waiting forever.
#==================================================================
import pytest
import numpy as np

from decsim.decoders import PresetLatencyDecoder
from decsim.message import Operation, ProtectedRegion
from decsim.planner import _validate_operation_graph
from decsim.run_spec import RunSpec


def _plan(operations):
    _validate_operation_graph(operations)
    return operations


def test_duplicate_operation_ids_are_rejected():
    first = Operation(7, "first", (0,))
    second = Operation(7, "second", (1,))
    with pytest.raises(ValueError) as err:
        _plan([first, second])
    message = str(err.value)
    assert "7" in message and "first" in message and "second" in message


def test_unknown_predecessor_is_a_named_error_not_a_keyerror():
    op = Operation(0, "op0", (0,), predecessors=(99,))
    with pytest.raises(ValueError) as err:
        _plan([op])
    message = str(err.value)
    assert "0" in message and "99" in message


def test_self_dependency_is_rejected():
    op = Operation(0, "op0", (0,), predecessors=(0,))
    with pytest.raises(ValueError) as err:
        _plan([op])
    assert "itself" in str(err.value)


def test_dependency_cycles_are_rejected():
    a = Operation(0, "a", (0,), predecessors=(1,))
    b = Operation(1, "b", (1,), predecessors=(0,))
    with pytest.raises(ValueError) as err:
        _plan([a, b])
    message = str(err.value)
    assert "cycle" in message and "0" in message and "1" in message


def test_valid_dag_still_plans():
    a = Operation(0, "a", (0,))
    b = Operation(1, "b", (1,), predecessors=(0,))
    plan = _plan([a, b])
    assert plan is not None


def test_unknown_blocked_by_is_rejected_at_validate():
    op = Operation(0, "op0", (0,), blocked_by=42)
    with pytest.raises(ValueError) as err:
        RunSpec(ops=[op]).build()
    message = str(err.value)
    assert "0" in message and "42" in message


def test_blocked_by_may_name_a_declared_decode_stream():
    stream = Operation(42, "stream", (0,))
    op = Operation(0, "op0", (0,), blocked_by=42)
    _validate_operation_graph(
        [op],
        validate_blockers=True,
        external_blocker_ids=[stream.id],
    )


def test_duplicate_ids_within_dynamic_streams_are_rejected():
    first = Operation(42, "first stream", (0,))
    second = Operation(42, "second stream", (1,))
    spec = RunSpec(ops=[Operation(0, "op0", (0,))],
                   dynamic_streams=[first, second])
    with pytest.raises(ValueError) as err:
        spec.build()
    message = str(err.value)
    assert "42" in message and "dynamic_streams" in message


def test_duplicate_ids_within_decode_ops_are_rejected():
    first = Operation(42, "first stream", (0,))
    second = Operation(42, "second stream", (1,))
    spec = RunSpec(ops=[Operation(0, "op0", (0,))],
                   decode_ops=[first, second])
    with pytest.raises(ValueError) as err:
        spec.build()
    message = str(err.value)
    assert "42" in message and "decode_ops" in message


@pytest.mark.parametrize("operation_id", [True, 1.0, np.int64(1), "1"])
def test_run_spec_requires_exact_integer_operation_ids(operation_id):
    with pytest.raises(TypeError, match="operation id.*exact built-in int"):
        RunSpec(
            ops=[Operation(operation_id, "invalid identity", (0,))],
        ).build()


@pytest.mark.parametrize(
    "qubits",
    [(True,), (np.int64(0),), (("patch", True),), (object(),)],
)
def test_run_spec_rejects_runtime_key_collisions_in_qubit_identities(qubits):
    with pytest.raises(TypeError, match="qubits.*stable built-in"):
        RunSpec(
            ops=[Operation(0, "invalid resources", qubits)],
        ).build()


def test_distinct_objects_cannot_share_an_id_across_workload_roles():
    executable = Operation(7, "executable", (0,))
    decode_owner = Operation(7, "decode owner", (0,))

    with pytest.raises(ValueError, match="operation id 7.*distinct objects"):
        RunSpec(
            ops=[executable],
            decode_ops=[decode_owner],
        ).build()


def test_same_object_may_have_executable_and_static_decode_membership():
    from decsim.planner import _validate_workload_identity

    operation = Operation(7, "shared owner", (0,))
    _validate_workload_identity([operation], [operation], [])


def test_executable_and_dynamic_stream_membership_cannot_alias():
    operation = Operation(7, "ambiguous dynamic owner", (0,))

    with pytest.raises(
        ValueError,
        match="ops and dynamic_streams",
    ):
        RunSpec(ops=[operation], dynamic_streams=[operation]).build()


def test_stream_reference_must_name_a_declared_stream_owner():
    operation = Operation(
        0,
        "orphan segment",
        (0,),
        stream_id=99,
    )

    with pytest.raises(ValueError, match="stream_id 99.*declared stream owner"):
        RunSpec(ops=[operation]).build()

def _protected_external_spec(source_role, stream_id):
    start = Operation(0, "alloc", (0,), patches=(0,), emits_detector_data=False)
    patch = 1 if stream_id < 0 else 0
    source = Operation(42, "external source", (patch,), patches=(patch,))
    consumer = Operation(2, "conditional", (1,), patches=(0,),
                         predecessors=(start.id,), blocked_by=source.id,
                         emits_detector_data=False)
    return RunSpec(
        ops=[start, consumer],
        protected_regions=(ProtectedRegion(0, 99 if stream_id < 0 else stream_id, start.id, consumer.id),),
        decoder=PresetLatencyDecoder(0),
        **{source_role: [source]},
    ), (start, source, consumer)

@pytest.mark.parametrize("source_role,stream_id",
                         [(role, stream) for role in ("decode_ops", "dynamic_streams")
                          for stream in (42, 99, -1)])
def test_external_protected_feedback_source_is_rejected_before_chip_admission(source_role, stream_id, monkeypatch):
    spec, operations = _protected_external_spec(source_role, stream_id)
    snapshots = tuple(operation.__dict__.copy() for operation in operations)
    if stream_id < 0:
        monkeypatch.setattr("decsim.chip.Chip._index_protected_regions", lambda *args: (_ for _ in ()).throw(RuntimeError("index reached")))
        with pytest.raises(RuntimeError, match="index reached"): spec.build()
        return
    with pytest.raises(ValueError, match=rf"external source 42.*protected stream.*{stream_id}"):
        spec.build()
    assert tuple(operation.__dict__ for operation in operations) == snapshots

def test_protected_region_has_exact_generation_identity():
    region = ProtectedRegion(("compute", 0), 7, 1, 4)
    assert (region.patch_id, region.stream_id) == (("compute", 0), 7)
    invalid_regions = (
        ((0, True, 1, 4), "stream_id.*exact built-in int"),
        ((object(), 7, 1, 4), "patch_id.*stable built-in identity"),
    )
    for arguments, message in invalid_regions:
        with pytest.raises(TypeError, match=message):
            ProtectedRegion(*arguments)

def test_executable_feedback_source_rejects_two_protected_streams_before_start():
    streams = (Operation(100, "first stream", (0,), patches=(0,)),
               Operation(200, "second stream", (1,), patches=(1,)))
    starts = (Operation(1, "first alloc", (0,), patches=(0,), emits_detector_data=False),
              Operation(2, "second alloc", (1,), patches=(1,), emits_detector_data=False))
    source = Operation(3, "joint measurement", (0, 1), patches=(0, 1),
                       predecessors=tuple(op.id for op in starts), emits_detector_data=False)
    consumer = Operation(4, "conditional", (2,), patches=(0,),
                         predecessors=(source.id,), blocked_by=source.id, emits_detector_data=False)
    regions = (ProtectedRegion(1, streams[1].id, starts[1].id, source.id),
               ProtectedRegion(0, streams[0].id, starts[0].id, source.id))
    with pytest.raises(
        ValueError, match=r"feedback source 3.*protected streams \(100, 200\)"
    ):
        RunSpec(ops=[*starts, source, consumer],
                dynamic_streams=list(reversed(streams)),
                protected_regions=regions,
                decoder=PresetLatencyDecoder(0)).build(verbose=False)
    assert (source.stream_id, source.stream_offset) == (None, None)
