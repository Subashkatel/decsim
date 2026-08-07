"""Licensed QUITS construction for the first BB-code experiment."""

import math
import sys


_CHECKS_PER_LAYER = 36


def build_bb72_memory_z(*, physical_error_rate, syndrome_round_count):
    """Build the Bravyi [[72,12,6]] Z-memory circuit and detector chronology."""
    if sys.version_info < (3, 10):
        raise RuntimeError("BB experiments require Python 3.10 or newer")
    if (type(physical_error_rate) not in (int, float)
            or not math.isfinite(physical_error_rate)
            or not 0 <= physical_error_rate <= 1):
        raise ValueError("physical_error_rate must be finite and between 0 and 1")
    if type(syndrome_round_count) is not int or syndrome_round_count <= 0:
        raise ValueError("syndrome_round_count must be a positive integer")

    try:
        from quits import BbCode, ErrorModel
    except ImportError as error:
        raise RuntimeError(
            "BB experiments require the optional quits==1.1.0 dependency"
        ) from error

    code = BbCode(
        l=6, m=6, A_x_pows=[3], A_y_pows=[1, 2],
        B_x_pows=[1, 2], B_y_pows=[3],
    )
    circuit = code.build_circuit(
        strategy="custom",
        num_rounds=syndrome_round_count,
        basis="Z",
        error_model=ErrorModel(
            idle_error=physical_error_rate, sqgate_error=physical_error_rate,
            tqgate_error=physical_error_rate, spam_error=physical_error_rate,
        ),
    )
    detector_layer_count = syndrome_round_count + 2
    expected_detectors = _CHECKS_PER_LAYER * detector_layer_count
    if (code.hx.shape != (36, 72) or code.hz.shape != (36, 72)
            or code.lx.shape != (12, 72) or code.lz.shape != (12, 72)
            or circuit.num_qubits != 144 or circuit.num_observables != 12
            or circuit.num_detectors != expected_detectors):
        raise ValueError("QUITS returned unexpected BB72 circuit dimensions")
    detector_rounds = {
        detector: detector // _CHECKS_PER_LAYER + 1
        for detector in range(expected_detectors)
    }
    return circuit, detector_rounds, detector_layer_count
