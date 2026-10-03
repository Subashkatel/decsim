"""The program records of decsim/records/program.py.

An operation's magic-state need, its planning view's feedback mode,
and a segment's rounds folded into its stream.
"""

import decsim.records.program as program_records


def make_operation(**overrides):
    values = {
        "id": 4,
        "name": "memory",
        "qubits": ("q0",),
    }
    values.update(overrides)
    return program_records.Operation(**values)


def test_operation_magic_state_need_uses_override_then_clifford_fallback():
    """An explicit override wins; otherwise a non-Clifford needs a state."""
    clifford = make_operation(clifford=True)
    non_clifford = make_operation(clifford=False)
    refused = make_operation(clifford=False, consumes_magic_state=False)
    demanded = make_operation(clifford=True, consumes_magic_state=True)
    assert not clifford.needs_magic_state
    assert non_clifford.needs_magic_state
    assert not refused.needs_magic_state
    assert demanded.needs_magic_state


def test_operation_planning_view_preserves_explicit_feedback_mode():
    """An operation's own feedback boundary mode beats the run's default."""
    operation = make_operation(feedback_boundary_mode="committed_region")
    view = program_records.OperationPlanningView.from_operation(
        operation,
        default_feedback_boundary_mode="trailing_buffer",
    )
    assert view.feedback_boundary_mode == "committed_region"


def test_a_segments_rounds_fold_into_its_stream_at_its_offset():
    standalone = make_operation()
    segment = make_operation(stream_id="stream", stream_offset=6)
    assert program_records.decode_identity(standalone) == 4
    assert program_records.global_round(standalone, 3) == 3
    assert program_records.decode_identity(segment) == "stream"
    assert program_records.global_round(segment, 3) == 9
