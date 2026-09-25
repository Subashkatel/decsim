"""An operation list wired in program order.

The wiring fills each operation's patches and its predecessors from
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

    operations = circuit_frontend._wire_circuit([first, second])
    indexed = _by_id(operations)

    assert indexed[1].predecessors == (0,)


def test_two_operations_that_share_no_patch_carry_no_edge():
    first = program_records.Operation(0, "Op0", (0, 1), clifford=True)
    second = program_records.Operation(1, "Op1", (2, 3), clifford=True)

    operations = circuit_frontend._wire_circuit([first, second])
    indexed = _by_id(operations)

    assert indexed[0].predecessors == ()
    assert indexed[1].predecessors == ()


def test_the_decoder_boundary_edges_are_the_program_order_edges():
    first = program_records.Operation(0, "Op0", (0, 1), clifford=True)
    second = program_records.Operation(1, "Op1", (2, 3), clifford=True)
    third = program_records.Operation(2, "Op2", (1, 3), clifford=True)

    operations = circuit_frontend._wire_circuit([first, second, third])
    indexed = _by_id(operations)

    for operation in operations:
        assert operation.decoder_boundary_predecessors == operation.predecessors
    assert indexed[2].predecessors == (0, 1)


def test_an_operation_that_lists_one_qubit_twice_is_refused():
    twice = program_records.Operation(0, "Op0", (1, 1), clifford=True)

    with pytest.raises(ValueError) as refusal:
        circuit_frontend._wire_circuit([twice])

    assert "lists the same qubit more than once" in str(refusal.value)
