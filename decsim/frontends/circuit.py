"""Small operation-list and text-IR frontends."""

from __future__ import annotations

import math
from typing import Optional

from ..message import Operation


def _validate_unique_qubits(operations: list[Operation]) -> None:
    """Reject operations that list the same qubit twice."""
    for operation in operations:
        if len(set(operation.qubits)) != len(operation.qubits):
            raise ValueError(
                f"{operation.name} lists the same qubit more than once: "
                f"{operation.qubits}")


def _validate_patch_mapping(operations: list[Operation],
                            qubit_to_patch: Optional[dict]) -> None:
    """Reject patch maps that do not cover every qubit used by the operations."""
    if qubit_to_patch is None:
        return
    missing = sorted({
        qubit
        for operation in operations
        for qubit in operation.qubits
        if qubit not in qubit_to_patch
    })
    if missing:
        raise ValueError(f"qubit_to_patch has no patch for qubit(s) {missing}")


def _operation_patches(operation: Operation, qubit_to_patch: Optional[dict]) -> tuple:
    """Return the patches touched by an operation."""
    if qubit_to_patch is not None:
        return tuple(dict.fromkeys(qubit_to_patch[qubit] for qubit in operation.qubits))
    if operation.patches:
        return operation.patches
    return tuple(operation.qubits)


def _wire_patch_dependencies(operations: list[Operation],
                             qubit_to_patch: Optional[dict]) -> tuple:
    """Return predecessor and successor maps from patch program order."""
    last_operation_on_patch = {}
    has_successor = {operation.id: False for operation in operations}
    predecessors = {operation.id: set() for operation in operations}

    for operation in operations:
        operation.patches = _operation_patches(operation, qubit_to_patch)
        for patch in operation.patches:
            if patch in last_operation_on_patch:
                previous_op_id = last_operation_on_patch[patch]
                predecessors[operation.id].add(previous_op_id)
                has_successor[previous_op_id] = True
            last_operation_on_patch[patch] = operation.id

    return predecessors, has_successor


def _wire_circuit(operations: list[Operation],
                  qubit_to_patch: Optional[dict] = None) -> list[Operation]:
    """Fill operation patches, predecessors, and successor flags in schedule order."""
    _validate_unique_qubits(operations)
    _validate_patch_mapping(operations, qubit_to_patch)
    predecessors, has_successor = _wire_patch_dependencies(operations, qubit_to_patch)

    for operation in operations:
        operation.predecessors = tuple(sorted(predecessors[operation.id]))
        operation.has_successor = has_successor[operation.id]
    return operations


def three_cnot_circuit() -> list[Operation]:
    """Three CNOTs where the first two can run before the third."""
    operations = [
        Operation(0, "Op0:CNOT(q0,q1)", (0, 1), clifford=True),
        Operation(1, "Op1:CNOT(q2,q3)", (2, 3), clifford=True),
        Operation(2, "Op2:CNOT(q1,q3)", (1, 3), clifford=True),
    ]
    return _wire_circuit(operations)


def cnot_plus_two_t_circuit() -> list[Operation]:
    """A CNOT followed by two dependent T operations."""
    operations = [
        Operation(0, "Op0:CNOT(q0,q1)", (0, 1), clifford=True),
        Operation(1, "Op1:T(q1)", (1,), clifford=False, blocked_by=None),
        Operation(2, "Op2:T(q1)", (1,), clifford=False, blocked_by=1),
    ]
    return _wire_circuit(operations)


def independent_t_circuit(n: int = 6) -> list[Operation]:
    """Independent T operations that only wait for magic-state supply."""
    operations = [
        Operation(index, f"T(q{index})", (index,), clifford=False, blocked_by=None)
        for index in range(n)
    ]
    return _wire_circuit(operations)


def three_cnot_six_qubits_circuit() -> list[Operation]:
    """Three independent CNOTs on six qubits."""
    operations = [
        Operation(0, "Op0:CNOT(q0,q1)", (0, 1), clifford=True),
        Operation(1, "Op1:CNOT(q2,q3)", (3, 4), clifford=True),
        Operation(2, "Op2:CNOT(q1,q3)", (2, 5), clifford=True),
    ]
    return _wire_circuit(operations)


class CircuitFrontend:
    """Frontend for a Python-built operation list."""

    def __init__(self, operations: list[Operation], qubit_to_patch: Optional[dict] = None):
        """Store operations and the optional qubit-to-patch map."""
        self.operations = operations
        self.qubit_to_patch = qubit_to_patch

    def build(self) -> list[Operation]:
        """Return operations with patch-order dependencies filled in."""
        return _wire_circuit(self.operations, self.qubit_to_patch)


CLIFFORD_GATES = {"cnot", "cx", "h", "x", "y", "z", "s", "sdg", "cz", "swap", "id"}
NON_CLIFFORD_GATES = {"t", "tdg", "ccz", "ccx", "toffoli"}
ROTATION_GATES = {"rz", "rx", "ry", "p", "u1"}
GENERAL_UNITARY_GATES = {"u2", "u3", "u"}


def _parse_angle(angle_expression) -> Optional[float]:
    """Parse a numeric or pi-fraction rotation angle into radians."""
    if angle_expression is None:
        return None
    if isinstance(angle_expression, (int, float)):
        return float(angle_expression)

    normalized_text = str(angle_expression).strip().lower().replace(" ", "")
    if not normalized_text:
        return None

    try:
        is_negative = normalized_text.startswith("-")
        if is_negative:
            normalized_text = normalized_text[1:]

        denominator = 1.0
        if "/" in normalized_text:
            numerator_text, denominator_text = normalized_text.split("/", 1)
            denominator = math.pi if "pi" in denominator_text else float(denominator_text)
            normalized_text = numerator_text

        coefficient = 1.0
        for factor_text in normalized_text.split("*"):
            coefficient *= math.pi if factor_text == "pi" else float(factor_text)

        angle = coefficient / denominator
        return -angle if is_negative else angle
    except (ValueError, ZeroDivisionError):
        return None


def _rotation_is_clifford(angle_expr: Optional[str]) -> bool:
    """Return whether a single-axis rotation is a Clifford rotation."""
    angle = _parse_angle(angle_expr)
    if angle is None:
        return False

    half_pi_units = angle / (math.pi / 2.0)
    return abs(half_pi_units - round(half_pi_units)) < 1e-9


def _gate_is_clifford(mnemonic: str, angle: Optional[str] = None) -> bool:
    """Classify a gate and fail loudly for unsupported gates."""
    mnemonic_lower = mnemonic.lower()
    if mnemonic_lower in CLIFFORD_GATES:
        return True
    if mnemonic_lower in NON_CLIFFORD_GATES or mnemonic_lower in GENERAL_UNITARY_GATES:
        return False
    if mnemonic_lower in ROTATION_GATES:
        return _rotation_is_clifford(angle)
    raise ValueError(f"unsupported gate '{mnemonic}'. Add it to CLIFFORD_GATES / "
                     f"NON_CLIFFORD_GATES / ROTATION_GATES / GENERAL_UNITARY_GATES")


def _operations_from_gate_list(gate_list: list,
                               qubit_to_patch: Optional[dict] = None) -> list[Operation]:
    """Lower parsed gates into wired operations."""
    operations = []
    for operation_index, gate in enumerate(gate_list):
        mnemonic, qubits, is_clifford, blocked_by = gate
        qubit_text = ",".join("q" + str(qubit) for qubit in qubits)
        operation = Operation(
            operation_index,
            f"Op{operation_index}:{mnemonic.upper()}({qubit_text})",
            tuple(qubits),
            clifford=is_clifford,
            blocked_by=blocked_by,
        )
        operations.append(operation)
    return _wire_circuit(operations, qubit_to_patch)


class SurgeryIRFrontend:
    """Frontend for the small line-based operation IR."""

    def __init__(self, text: str, qubit_to_patch: Optional[dict] = None):
        """Store source text and the optional qubit-to-patch map."""
        self.text = text
        self.qubit_to_patch = qubit_to_patch

    def build(self) -> list[Operation]:
        """Parse text gates and lower them into wired operations."""
        gate_list = []
        for raw_line in self.text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            mnemonic = tokens[0]
            blocked_by = None
            if "blocked_by" in tokens:
                token_index = tokens.index("blocked_by")
                blocked_by = int(tokens[token_index + 1])
                tokens = tokens[:token_index]

            qubits = tuple(
                int(token[1:])
                for token in tokens[1:]
                if token.lower().startswith("q")
            )
            is_clifford = _gate_is_clifford(mnemonic)
            gate_list.append((mnemonic, qubits, is_clifford, blocked_by))

        return _operations_from_gate_list(gate_list, self.qubit_to_patch)
