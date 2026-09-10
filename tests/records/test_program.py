"""The program records of decsim/records/program.py.

A planning view is the operation frozen without its executable circuit,
so a planning collaborator cannot reach the Stim circuit or the live
tuples the front end still holds.
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


def test_operation_planning_view_snapshots_configuration_without_circuit():
    """The view copies the tuples, omits the circuit, resolves the mode."""
    circuit = object()
    operation = make_operation(
        qubits=["q0"],
        patches=["patch"],
        predecessors=[1],
        decoder_boundary_predecessors=[2],
        circuit=circuit,
    )
    view = program_records.OperationPlanningView.from_operation(operation)
    operation.qubits.append("q1")

    assert view.qubits == ("q0",)
    assert view.patches == ("patch",)
    assert view.predecessors == (1,)
    assert view.decoder_boundary_predecessors == (2,)
    assert view.feedback_boundary_mode == "trailing_buffer"
    assert not hasattr(view, "circuit")
    assert not hasattr(view, "needs_magic_state")


def test_operation_planning_view_preserves_explicit_feedback_mode():
    """An operation's own feedback boundary mode beats the run's default."""
    operation = make_operation(feedback_boundary_mode="committed_region")
    view = program_records.OperationPlanningView.from_operation(
        operation,
        default_feedback_boundary_mode="trailing_buffer",
    )
    assert view.feedback_boundary_mode == "committed_region"
