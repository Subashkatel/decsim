"""Workloads written by hand: an operation list, or a small text IR.

Both frontends fill each operation's patches and its predecessors from
program order on those patches, so two operations that share a patch
always carry a dependency edge between them. The four named circuits are
the examples the guides and slides run.
"""

import math
from typing import Optional

import decsim.message as message

CLIFFORD_GATES = {
    "cnot",
    "cx",
    "h",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "cz",
    "swap",
    "id",
}
NON_CLIFFORD_GATES = {"t", "tdg", "ccz", "ccx", "toffoli"}
ROTATION_GATES = {"rz", "rx", "ry", "p", "u1"}
GENERAL_UNITARY_GATES = {"u2", "u3", "u"}


def three_cnot_circuit() -> list[message.Operation]:
    """Three CNOTs where the first two can run before the third."""
    operations = [
        message.Operation(0, "Op0:CNOT(q0,q1)", (0, 1), clifford=True),
        message.Operation(1, "Op1:CNOT(q2,q3)", (2, 3), clifford=True),
        message.Operation(2, "Op2:CNOT(q1,q3)", (1, 3), clifford=True),
    ]
    return _wire_circuit(operations)


def cnot_plus_two_t_circuit() -> list[message.Operation]:
    """A CNOT followed by two dependent T operations."""
    operations = [
        message.Operation(0, "Op0:CNOT(q0,q1)", (0, 1), clifford=True),
        message.Operation(
            1, "Op1:T(q1)", (1,), clifford=False, blocked_by=None
        ),
        message.Operation(2, "Op2:T(q1)", (1,), clifford=False, blocked_by=1),
    ]
    return _wire_circuit(operations)


def independent_t_circuit(count: int = 6) -> list[message.Operation]:
    """Independent T operations that only wait for magic-state supply."""
    operations = []
    for index in range(count):
        operation = message.Operation(
            index, f"T(q{index})", (index,), clifford=False, blocked_by=None
        )
        operations.append(operation)
    return _wire_circuit(operations)


def three_cnot_six_qubits_circuit() -> list[message.Operation]:
    """Three independent CNOTs on six qubits."""
    operations = [
        message.Operation(0, "Op0:CNOT(q0,q1)", (0, 1), clifford=True),
        message.Operation(1, "Op1:CNOT(q3,q4)", (3, 4), clifford=True),
        message.Operation(2, "Op2:CNOT(q2,q5)", (2, 5), clifford=True),
    ]
    return _wire_circuit(operations)


class CircuitFrontend:
    """A workload from a Python-built operation list."""

    def __init__(
        self,
        operations: list[message.Operation],
        qubit_to_patch: Optional[dict] = None,
    ):
        self.operations = operations
        self.qubit_to_patch = qubit_to_patch

    def build(self) -> list[message.Operation]:
        """The operations with patch-order dependencies filled in."""
        return _wire_circuit(self.operations, self.qubit_to_patch)


class SurgeryIRFrontend:
    """A workload from the line-based text IR.

    Each line is a gate mnemonic, its qubits (q0 q1 ...), an optional
    rotation angle, and an optional ``blocked_by <operation id>``; a ``#``
    starts a comment.
    """

    def __init__(self, text: str, qubit_to_patch: Optional[dict] = None):
        self.text = text
        self.qubit_to_patch = qubit_to_patch

    def build(self) -> list[message.Operation]:
        """Parse the text and lower it into wired operations."""
        gates = []
        for raw_line in self.text.splitlines():
            gate = _parse_gate_line(raw_line)
            if gate is None:
                continue
            gates.append(gate)
        return _operations_from_gates(gates, self.qubit_to_patch)


def _wire_circuit(
    operations: list[message.Operation],
    qubit_to_patch: Optional[dict] = None,
) -> list[message.Operation]:
    """Fill operation patches and predecessors in schedule order."""
    _check_unique_qubits(operations)
    _check_patch_mapping(operations, qubit_to_patch)
    predecessors = _patch_order_predecessors(operations, qubit_to_patch)
    for operation in operations:
        ordered = sorted(predecessors[operation.id])
        operation.predecessors = tuple(ordered)
        operation.decoder_boundary_predecessors = operation.predecessors
    return operations


def _check_unique_qubits(operations: list[message.Operation]) -> None:
    """Refuse an operation that lists the same qubit twice."""
    for operation in operations:
        distinct = set(operation.qubits)
        if len(distinct) != len(operation.qubits):
            raise ValueError(
                f"{operation.name} lists the same qubit more than once: "
                f"{operation.qubits}"
            )


def _check_patch_mapping(
    operations: list[message.Operation], qubit_to_patch: Optional[dict]
) -> None:
    """Refuse a patch map that leaves a used qubit without a patch."""
    if qubit_to_patch is None:
        return
    missing = set()
    for operation in operations:
        unmapped = _unmapped_qubits(operation, qubit_to_patch)
        missing.update(unmapped)
    if missing:
        ordered = sorted(missing)
        raise ValueError(f"qubit_to_patch has no patch for qubit(s) {ordered}")


def _unmapped_qubits(operation: message.Operation, qubit_to_patch: dict) -> set:
    unmapped = set()
    for qubit in operation.qubits:
        if qubit not in qubit_to_patch:
            unmapped.add(qubit)
    return unmapped


def _operation_patches(
    operation: message.Operation, qubit_to_patch: Optional[dict]
) -> tuple:
    """The patches an operation touches, in first-use order."""
    if qubit_to_patch is not None:
        patches = []
        for qubit in operation.qubits:
            patches.append(qubit_to_patch[qubit])
        distinct = dict.fromkeys(patches)
        return tuple(distinct)
    if operation.patches:
        return operation.patches
    return tuple(operation.qubits)


def _patch_order_predecessors(
    operations: list[message.Operation], qubit_to_patch: Optional[dict]
) -> dict:
    """Each operation's predecessors: the last earlier user of each patch."""
    last_operation_on_patch = {}
    predecessors = {}
    for operation in operations:
        predecessors[operation.id] = set()
    for operation in operations:
        operation.patches = _operation_patches(operation, qubit_to_patch)
        earlier_users = _claim_patches(operation, last_operation_on_patch)
        predecessors[operation.id].update(earlier_users)
    return predecessors


def _claim_patches(
    operation: message.Operation, last_operation_on_patch: dict
) -> set:
    """Mark the operation as each patch's last user; the users it displaces."""
    earlier_users = set()
    for patch in operation.patches:
        previous_id = last_operation_on_patch.get(patch)
        if previous_id is not None:
            earlier_users.add(previous_id)
        last_operation_on_patch[patch] = operation.id
    return earlier_users


def _parse_gate_line(raw_line: str) -> Optional[tuple]:
    """One IR line as (mnemonic, qubits, is_clifford, blocked_by)."""
    code, _, _comment = raw_line.partition("#")
    line = code.strip()
    if not line:
        return None
    tokens = line.split()
    mnemonic = tokens[0]
    blocked_by = None
    if "blocked_by" in tokens:
        token_index = tokens.index("blocked_by")
        value_index = token_index + 1
        blocked_by = int(tokens[value_index])
        tokens = tokens[:token_index]
    qubits = _qubits_of(tokens)
    angle = None
    lowered = mnemonic.lower()
    if lowered in ROTATION_GATES:
        angle = _angle_token(tokens)
    is_clifford = _gate_is_clifford(mnemonic, angle)
    return (mnemonic, qubits, is_clifford, blocked_by)


def _qubits_of(tokens: list[str]) -> tuple[int, ...]:
    """The qubit indices named by q<index> tokens after the mnemonic."""
    qubits = []
    for token in tokens[1:]:
        if _is_qubit_token(token):
            qubits.append(int(token[1:]))
    return tuple(qubits)


def _angle_token(tokens: list[str]) -> Optional[str]:
    """The first token after the mnemonic that is not a qubit."""
    for token in tokens[1:]:
        if not _is_qubit_token(token):
            return token
    return None


def _is_qubit_token(token: str) -> bool:
    lowered = token.lower()
    return lowered.startswith("q")


def _operations_from_gates(
    gates: list, qubit_to_patch: Optional[dict] = None
) -> list[message.Operation]:
    """Lower parsed gates into wired operations."""
    operations = []
    for operation_index, gate in enumerate(gates):
        mnemonic, qubits, is_clifford, blocked_by = gate
        qubit_words = []
        for qubit in qubits:
            qubit_words.append(f"q{qubit}")
        qubit_text = ",".join(qubit_words)
        upper_mnemonic = mnemonic.upper()
        operation = message.Operation(
            operation_index,
            f"Op{operation_index}:{upper_mnemonic}({qubit_text})",
            tuple(qubits),
            clifford=is_clifford,
            blocked_by=blocked_by,
        )
        operations.append(operation)
    return _wire_circuit(operations, qubit_to_patch)


def _gate_is_clifford(mnemonic: str, angle: Optional[str] = None) -> bool:
    """Whether a gate is Clifford; an unknown mnemonic is refused."""
    lowered = mnemonic.lower()
    if lowered in CLIFFORD_GATES:
        return True
    if lowered in NON_CLIFFORD_GATES:
        return False
    if lowered in GENERAL_UNITARY_GATES:
        return False
    if lowered in ROTATION_GATES:
        return _rotation_is_clifford(angle)
    raise ValueError(
        f"unsupported gate '{mnemonic}'. Add it to CLIFFORD_GATES / "
        f"NON_CLIFFORD_GATES / ROTATION_GATES / GENERAL_UNITARY_GATES"
    )


def _rotation_is_clifford(angle_expression: Optional[str]) -> bool:
    """A single-axis rotation is Clifford at a multiple of a quarter turn."""
    angle = _parse_angle(angle_expression)
    if angle is None:
        return False
    quarter_turn = math.pi / 2.0
    quarter_turns = angle / quarter_turn
    nearest = round(quarter_turns)
    distance = quarter_turns - nearest
    return abs(distance) < 1e-9


def _parse_angle(angle_expression) -> Optional[float]:
    """A numeric or pi-fraction angle in radians; None when unreadable."""
    if angle_expression is None:
        return None
    if isinstance(angle_expression, (int, float)):
        return float(angle_expression)
    text = str(angle_expression)
    normalized = text.strip()
    normalized = normalized.lower()
    normalized = normalized.replace(" ", "")
    if not normalized:
        return None
    try:
        return _angle_from_text(normalized)
    except (ValueError, ZeroDivisionError):
        return None


def _angle_from_text(normalized: str) -> float:
    """[-]<factor>[*<factor>...][/<denominator>] with pi as a factor."""
    is_negative = normalized.startswith("-")
    if is_negative:
        normalized = normalized[1:]
    denominator = 1.0
    if "/" in normalized:
        numerator_text, denominator_text = normalized.split("/", 1)
        denominator = _denominator_value(denominator_text)
        normalized = numerator_text
    coefficient = 1.0
    for factor_text in normalized.split("*"):
        coefficient *= _factor_value(factor_text)
    angle = coefficient / denominator
    if is_negative:
        return -angle
    return angle


def _factor_value(factor_text: str) -> float:
    if factor_text == "pi":
        return math.pi
    return float(factor_text)


def _denominator_value(denominator_text: str) -> float:
    if "pi" in denominator_text:
        return math.pi
    return float(denominator_text)
