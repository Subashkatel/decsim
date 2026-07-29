#==================================================================
# TESTS FOR THE LINE-BASED SURGERY IR FRONTEND
# Rotation lines carry their angle as a bare token (rz pi/2 q0); the
# parser must hand that angle to _gate_is_clifford so pi/2 multiples
# come out Clifford while pi/4, pi/8, and symbolic angles stay
# non-Clifford. Helper circuit labels must describe the qubits the
# operation actually acts on.
#==================================================================
import re

import pytest

from decsim.frontends.circuit import (SurgeryIRFrontend,
                                      three_cnot_six_qubits_circuit)

CLIFFORD_ROTATIONS = """
rz pi/2 q0
rx -pi/2 q4
rx 3*pi/2 q4
rx 0 q4
rx 6.283185307179586 q4
"""

NON_CLIFFORD_ROTATIONS = """
rz pi/4 q0
rz pi/8 q1
rz theta q2
"""


def test_pi_over_two_multiples_parse_as_clifford():
    operations = SurgeryIRFrontend(CLIFFORD_ROTATIONS).build()
    assert [op.clifford for op in operations] == [True] * 5


def test_other_angles_stay_non_clifford():
    operations = SurgeryIRFrontend(NON_CLIFFORD_ROTATIONS).build()
    assert [op.clifford for op in operations] == [False] * 3


def test_rotation_qubits_survive_the_angle_token():
    operations = SurgeryIRFrontend(CLIFFORD_ROTATIONS).build()
    assert [op.qubits for op in operations] == [(0,), (4,), (4,), (4,), (4,)]


def test_unsupported_gates_fail_loudly():
    with pytest.raises(ValueError):
        SurgeryIRFrontend("frobnicate q0").build()


def test_six_qubit_helper_labels_match_their_qubit_tuples():
    for op in three_cnot_six_qubits_circuit():
        labeled = tuple(int(text[1:]) for text in re.findall(r"q\d+", op.name))
        assert labeled == op.qubits, f"{op.name} acts on {op.qubits}"
