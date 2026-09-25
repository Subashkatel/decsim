"""Workloads written as an operation list, wired in program order.

The wiring fills each operation's predecessors from program order on
its patches, so two operations that share a patch always carry a
dependency edge between them.
"""

import decsim.records.program as program_records


def _wire_circuit(
    operations: list[program_records.Operation],
) -> list[program_records.Operation]:
    """Fill operation patches and predecessors in schedule order."""
    _check_unique_qubits(operations)
    predecessors = _patch_order_predecessors(operations)
    for operation in operations:
        ordered = sorted(predecessors[operation.id])
        operation.predecessors = tuple(ordered)
        operation.decoder_boundary_predecessors = operation.predecessors
    return operations


def _check_unique_qubits(operations: list[program_records.Operation]) -> None:
    """Refuse an operation that lists the same qubit twice."""
    for operation in operations:
        distinct = set(operation.qubits)
        if len(distinct) != len(operation.qubits):
            raise ValueError(
                f"{operation.name} lists the same qubit more than once: "
                f"{operation.qubits}"
            )


def _operation_patches(operation: program_records.Operation) -> tuple:
    """The patches an operation touches: its own, or its qubits."""
    if operation.patches:
        return operation.patches
    return tuple(operation.qubits)


def _patch_order_predecessors(
    operations: list[program_records.Operation],
) -> dict:
    """Each operation's predecessors: the last earlier user of each patch."""
    last_operation_on_patch = {}
    predecessors = {}
    for operation in operations:
        predecessors[operation.id] = set()
    for operation in operations:
        operation.patches = _operation_patches(operation)
        earlier_users = _claim_patches(operation, last_operation_on_patch)
        predecessors[operation.id].update(earlier_users)
    return predecessors


def _claim_patches(
    operation: program_records.Operation, last_operation_on_patch: dict
) -> set:
    """Mark the operation as each patch's last user; the users it displaces."""
    earlier_users = set()
    for patch in operation.patches:
        previous_id = last_operation_on_patch.get(patch)
        if previous_id is not None:
            earlier_users.add(previous_id)
        last_operation_on_patch[patch] = operation.id
    return earlier_users
