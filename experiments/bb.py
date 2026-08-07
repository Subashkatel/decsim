"""Finite BB catalog and licensed QUITS 1.1 circuit construction."""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
import math
import sys


@dataclass(frozen=True)
class _BbDefinition:
    l: int
    m: int
    a_x: tuple
    a_y: tuple
    b_x: tuple
    b_y: tuple
    logical_qubits: int


_BB_DEFINITIONS = {
    "bb-l6m6-a-x3_y1_y2-b-x1_x2_y3": _BbDefinition(6, 6, (3,), (1, 2), (1, 2), (3,), 12),
    "bb-l15m3-a-x9_y1_y2-b-1_x2_x7": _BbDefinition(15, 3, (9,), (1, 2), (0, 2, 7), (), 8),
    "bb-l3m21-a-1_y2_y10-b-x1_x2_y3": _BbDefinition(3, 21, (), (0, 2, 10), (1, 2), (3,), 8),
    "bb-l12m6-a-x3_y1_y2-b-x1_x2_y3": _BbDefinition(12, 6, (3,), (1, 2), (1, 2), (3,), 12),
    "bb-l3m27-a-1_y10_y14-b-x1_x2_y12": _BbDefinition(3, 27, (), (0, 10, 14), (1, 2), (12,), 8),
    "bb-l12m12-a-x3_y2_y7-b-x1_x2_y3": _BbDefinition(12, 12, (3,), (2, 7), (1, 2), (3,), 12),
    "bb-l28m14-a-x26_y6_y8-b-x9_x20_y7": _BbDefinition(28, 14, (26,), (6, 8), (9, 20), (7,), 24),
}
_NOISE_RATIOS = {
    "equal-rate": (1, 1, 1, 1),
    "reference-ionic-ratios": (0.01, 0.1, 1, 0.1),
}


def _require_quits_1_1_0():
    try:
        installed = package_version("quits")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "BB experiments require the optional quits==1.1.0 dependency"
        ) from error
    if installed != "1.1.0":
        raise RuntimeError(
            f"BB experiments require quits==1.1.0; installed quits version {installed}"
        )


def build_quits11_bb_memory(
    *,
    definition_id,
    basis,
    noise_profile,
    physical_error_rate,
    syndrome_round_count,
):
    """Build one detector-complete QUITS-1.1 custom BB memory circuit."""
    if sys.version_info < (3, 10):
        raise RuntimeError("BB experiments require Python 3.10 or newer")
    _require_quits_1_1_0()
    try:
        definition = _BB_DEFINITIONS[definition_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown BB definition_id {definition_id!r}") from error
    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")
    try:
        noise_ratios = _NOISE_RATIOS[noise_profile]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown BB noise_profile {noise_profile!r}") from error
    if (
        type(physical_error_rate) not in (int, float)
        or not math.isfinite(physical_error_rate)
        or not 0 <= physical_error_rate <= 1
    ):
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
        l=definition.l,
        m=definition.m,
        A_x_pows=list(definition.a_x),
        A_y_pows=list(definition.a_y),
        B_x_pows=list(definition.b_x),
        B_y_pows=list(definition.b_y),
    )
    error_rates = tuple(physical_error_rate * ratio for ratio in noise_ratios)
    circuit = code.build_circuit(
        strategy="custom",
        num_rounds=syndrome_round_count,
        basis=basis,
        error_model=ErrorModel(
            idle_error=error_rates[0],
            sqgate_error=error_rates[1],
            tqgate_error=error_rates[2],
            spam_error=error_rates[3],
        ),
    )
    checks_per_layer = definition.l * definition.m
    data_qubits = 2 * checks_per_layer
    detector_layer_count = syndrome_round_count + 2
    expected_detectors = checks_per_layer * detector_layer_count
    logical_shape = (definition.logical_qubits, data_qubits)
    if (
        code.hx.shape != (checks_per_layer, data_qubits)
        or code.hz.shape != (checks_per_layer, data_qubits)
        or code.lx.shape != logical_shape
        or code.lz.shape != logical_shape
        or circuit.num_qubits != 2 * data_qubits
        or circuit.num_observables != definition.logical_qubits
        or circuit.num_detectors != expected_detectors
    ):
        raise ValueError(f"QUITS returned unexpected dimensions for {definition_id}")
    detector_rounds = {
        detector: detector // checks_per_layer + 1
        for detector in range(expected_detectors)
    }
    return circuit, detector_rounds, detector_layer_count
