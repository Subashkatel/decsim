"""Workloads written by hand: an operation list, or a small text IR.

Both frontends fill each operation's patches and its predecessors from
program order on those patches, so two operations that share a patch
always carry a dependency edge between them. That rule is the whole
component: everything downstream reads the edges rather than the order.
"""

import pytest

import decsim.frontends.circuit_frontend as circuit_frontend
import decsim.records.program as program_records


def _by_id(operations) -> dict:
    indexed = {}
    for operation in operations:
        indexed[operation.id] = operation
    return indexed


def test_two_operations_that_share_a_patch_carry_an_edge_between_them():
    first = program_records.Operation(0, "Op0", (0, 1), clifford=True)
    second = program_records.Operation(1, "Op1", (1, 2), clifford=True)
    frontend = circuit_frontend.CircuitFrontend([first, second])

    operations = frontend.build()
    indexed = _by_id(operations)

    assert indexed[1].predecessors == (0,)


def test_two_operations_that_share_no_patch_carry_no_edge():
    first = program_records.Operation(0, "Op0", (0, 1), clifford=True)
    second = program_records.Operation(1, "Op1", (2, 3), clifford=True)
    frontend = circuit_frontend.CircuitFrontend([first, second])

    operations = frontend.build()
    indexed = _by_id(operations)

    assert indexed[0].predecessors == ()
    assert indexed[1].predecessors == ()


def test_the_decoder_boundary_edges_are_the_program_order_edges():
    operations = circuit_frontend.three_cnot_circuit()
    indexed = _by_id(operations)

    for operation in operations:
        assert operation.decoder_boundary_predecessors == operation.predecessors
    assert indexed[2].predecessors == (0, 1)


def test_an_operation_that_lists_one_qubit_twice_is_refused():
    twice = program_records.Operation(0, "Op0", (1, 1), clifford=True)
    frontend = circuit_frontend.CircuitFrontend([twice])

    with pytest.raises(ValueError) as refusal:
        frontend.build()

    assert "lists the same qubit more than once" in str(refusal.value)


def test_a_patch_map_that_leaves_a_used_qubit_unmapped_is_refused():
    operation = program_records.Operation(0, "Op0", (0, 1), clifford=True)
    frontend = circuit_frontend.CircuitFrontend(
        [operation], qubit_to_patch={0: "p0"}
    )

    with pytest.raises(ValueError) as refusal:
        frontend.build()

    assert "no patch for qubit(s) [1]" in str(refusal.value)


def test_a_patch_map_puts_two_qubits_on_one_patch_and_makes_the_edge():
    first = program_records.Operation(0, "Op0", (0,), clifford=True)
    second = program_records.Operation(1, "Op1", (1,), clifford=True)
    one_patch = {0: "p0", 1: "p0"}
    frontend = circuit_frontend.CircuitFrontend(
        [first, second], qubit_to_patch=one_patch
    )

    operations = frontend.build()
    indexed = _by_id(operations)

    assert indexed[1].predecessors == (0,)


def test_the_text_ir_lowers_one_line_per_gate():
    frontend = circuit_frontend.SurgeryIRFrontend("cnot q0 q1\ncnot q1 q2\n")

    operations = frontend.build()
    indexed = _by_id(operations)

    assert len(operations) == 2
    assert indexed[1].predecessors == (0,)


def test_the_text_ir_skips_a_comment_and_a_blank_line():
    frontend = circuit_frontend.SurgeryIRFrontend(
        "# the first gate\ncnot q0 q1\n\n   # nothing here\n"
    )

    operations = frontend.build()

    assert len(operations) == 1


def test_the_text_ir_reads_a_blocked_by_as_a_feedback_source():
    frontend = circuit_frontend.SurgeryIRFrontend(
        "cnot q0 q1\ncnot q2 q3 blocked_by 0\n"
    )

    operations = frontend.build()
    indexed = _by_id(operations)

    assert indexed[1].blocked_by == 0


def test_a_named_circuit_is_wired_before_it_is_handed_out():
    """The four named circuits are what the guides and slides run."""
    operations = circuit_frontend.cnot_plus_two_t_circuit()
    indexed = _by_id(operations)

    for operation in operations:
        assert operation.patches != ()
    assert indexed[1].predecessors == (0,)
