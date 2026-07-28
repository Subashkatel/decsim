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

from decsim.codes import SurfaceCodeModel
from decsim.decoders import PresetLatencyDecoder
from decsim.layouts import UniformLayout
from decsim.message import IntrinsicMeasurement, Operation
from decsim.planner import FixedRounds, WindowPlanner
from decsim.run_spec import RunSpec
from decsim.schemes import SlidingWindowScheme


def _planner():
    return WindowPlanner(SlidingWindowScheme(),
                         UniformLayout(SurfaceCodeModel(d=3)),
                         FixedRounds(3))


def test_duplicate_operation_ids_are_rejected():
    first = Operation(7, "first", (0,))
    second = Operation(7, "second", (1,))
    with pytest.raises(ValueError) as err:
        _planner().plan([first, second])
    message = str(err.value)
    assert "7" in message and "first" in message and "second" in message


def test_unknown_predecessor_is_a_named_error_not_a_keyerror():
    op = Operation(0, "op0", (0,), predecessors=(99,))
    with pytest.raises(ValueError) as err:
        _planner().plan([op])
    message = str(err.value)
    assert "0" in message and "99" in message


def test_self_dependency_is_rejected():
    op = Operation(0, "op0", (0,), predecessors=(0,))
    with pytest.raises(ValueError) as err:
        _planner().plan([op])
    assert "itself" in str(err.value)


def test_dependency_cycles_are_rejected():
    a = Operation(0, "a", (0,), predecessors=(1,))
    b = Operation(1, "b", (1,), predecessors=(0,))
    with pytest.raises(ValueError) as err:
        _planner().plan([a, b])
    message = str(err.value)
    assert "cycle" in message and "0" in message and "1" in message


def test_valid_dag_still_plans():
    a = Operation(0, "a", (0,))
    b = Operation(1, "b", (1,), predecessors=(0,))
    plan = _planner().plan([a, b])
    assert plan is not None


def test_unknown_blocked_by_is_rejected_at_validate():
    op = Operation(0, "op0", (0,), blocked_by=42)
    with pytest.raises(ValueError) as err:
        RunSpec(ops=[op]).validate()
    message = str(err.value)
    assert "0" in message and "42" in message


def test_blocked_by_may_name_a_declared_decode_stream():
    stream = Operation(42, "stream", (0,))
    op = Operation(0, "op0", (0,), blocked_by=42)
    RunSpec(ops=[op], dynamic_streams=[stream]).validate()   # must not raise


def test_duplicate_ids_within_dynamic_streams_are_rejected():
    first = Operation(42, "first stream", (0,))
    second = Operation(42, "second stream", (1,))
    spec = RunSpec(ops=[Operation(0, "op0", (0,))],
                   dynamic_streams=[first, second])
    with pytest.raises(ValueError) as err:
        spec.validate()
    message = str(err.value)
    assert "42" in message and "dynamic_streams" in message


def test_duplicate_ids_within_decode_ops_are_rejected():
    first = Operation(42, "first stream", (0,))
    second = Operation(42, "second stream", (1,))
    spec = RunSpec(ops=[Operation(0, "op0", (0,))],
                   decode_ops=[first, second])
    with pytest.raises(ValueError) as err:
        spec.validate()
    message = str(err.value)
    assert "42" in message and "decode_ops" in message


def test_direct_operation_rejects_non_exact_logical_observable_index():
    operation = Operation(
        0,
        "op0",
        (0,),
        logical_observable_index=True,
    )

    with pytest.raises(TypeError, match="logical_observable_index"):
        RunSpec(ops=[operation]).validate()


def test_intrinsic_measurement_must_match_operation_and_stream_identity():
    operation = Operation(
        3,
        "op3",
        (0,),
        stream_id=3,
        intrinsic_measurement=IntrinsicMeasurement(
            operation_id=3,
            trajectory_id=4,
            value=0,
            source="controlled fixture",
        ),
    )

    with pytest.raises(ValueError, match="trajectory_id"):
        RunSpec(ops=[operation]).validate()


def test_run_spec_rejects_tuple_stream_identity_even_if_intrinsic_type_allows_it():
    operation = Operation(
        3,
        "op3",
        (0,),
        stream_id=("stream", (3, "branch")),
        intrinsic_measurement=IntrinsicMeasurement(
            operation_id=3,
            trajectory_id=("stream", (3, "branch")),
            value=0,
            source="controlled fixture",
        ),
    )

    with pytest.raises(TypeError, match="stream_id.*exact built-in int"):
        RunSpec(ops=[operation]).validate()


@pytest.mark.parametrize("operation_id", [True, 1.0, np.int64(1), "1"])
def test_run_spec_requires_exact_integer_operation_ids(operation_id):
    with pytest.raises(TypeError, match="operation id.*exact built-in int"):
        RunSpec(
            ops=[Operation(operation_id, "invalid identity", (0,))],
        ).validate()


@pytest.mark.parametrize(
    "qubits",
    [(True,), (np.int64(0),), (("patch", True),), (object(),)],
)
def test_run_spec_rejects_runtime_key_collisions_in_qubit_identities(qubits):
    with pytest.raises(TypeError, match="qubits.*stable built-in"):
        RunSpec(
            ops=[Operation(0, "invalid resources", qubits)],
        ).validate()


def test_distinct_objects_cannot_share_an_id_across_workload_roles():
    executable = Operation(7, "executable", (0,))
    decode_owner = Operation(7, "decode owner", (0,))

    with pytest.raises(ValueError, match="operation id 7.*distinct objects"):
        RunSpec(
            ops=[executable],
            decode_ops=[decode_owner],
        ).validate()


def test_same_object_may_have_executable_and_static_decode_membership():
    operation = Operation(7, "shared owner", (0,))

    RunSpec(ops=[operation], decode_ops=[operation]).validate()


def test_executable_and_dynamic_stream_membership_cannot_alias():
    operation = Operation(7, "ambiguous dynamic owner", (0,))

    with pytest.raises(
        ValueError,
        match="ops and dynamic_streams",
    ):
        RunSpec(ops=[operation], dynamic_streams=[operation]).validate()


def test_stream_reference_must_name_a_declared_stream_owner():
    operation = Operation(
        0,
        "orphan segment",
        (0,),
        stream_id=99,
    )

    with pytest.raises(ValueError, match="stream_id 99.*declared stream owner"):
        RunSpec(ops=[operation]).validate()


class _InvalidFeedbackFrontend:
    def build(self):
        return [
            Operation(
                0,
                "frontend op",
                (0,),
                logical_observable_index=-1,
            )
        ]


def test_frontend_materialized_operation_uses_same_feedback_validation():
    spec = RunSpec(
        frontend=_InvalidFeedbackFrontend(),
        decoder=PresetLatencyDecoder(0.0),
    )

    with pytest.raises(ValueError, match="logical_observable_index"):
        spec.build()
