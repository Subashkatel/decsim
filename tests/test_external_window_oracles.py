"""Code-level oracles for published sliding-window implementations."""

import pathlib

import numpy as np
import stim

from decsim.detector_error_model import (
    PHYSICAL_FAULT_MODEL_REQUIRED,
    build_window_error_models,
)
from decsim.schemes import SlidingWindowScheme


_CHECKS_PER_ROUND = 36
_WINDOW_WIDTH = 6
_COMMIT_WIDTH = 3
_QUITS_PLAN = ((1, 3, 6), (4, 6, 9), (7, 12, 12))


def _quits_global_arrays(circuit):
    """Independently reproduce QUITS 1.1's physical DEM conversion."""
    detector_to_column = {}
    observable_sets = []
    priors = []
    for instruction in circuit.detector_error_model(
        decompose_errors=False
    ).flattened():
        if instruction.type != "error":
            continue
        detectors = frozenset(
            target.val
            for target in instruction.targets_copy()
            if target.is_relative_detector_id()
        )
        observables = frozenset(
            target.val
            for target in instruction.targets_copy()
            if target.is_logical_observable_id()
        )
        probability = float(instruction.args_copy()[0])
        if detectors in detector_to_column:
            column = detector_to_column[detectors]
            # QUITS keys only by detectors. Fail if this fixture would exercise
            # that known ambiguity instead of silently copying it.
            assert observable_sets[column] == observables
            previous = priors[column]
            priors[column] = previous * (1 - probability) + probability * (
                1 - previous
            )
        else:
            detector_to_column[detectors] = len(detector_to_column)
            observable_sets.append(observables)
            priors.append(probability)

    check = np.zeros(
        (circuit.num_detectors, len(detector_to_column)), dtype=np.uint8
    )
    observables = np.zeros(
        (circuit.num_observables, len(detector_to_column)), dtype=np.uint8
    )
    for detector_ids, column in detector_to_column.items():
        check[list(detector_ids), column] = 1
        observables[list(observable_sets[column]), column] = 1
    return check, observables, np.asarray(priors)


def _quits_window_arrays(circuit):
    """Reproduce QUITS 1.1 ``spacetime`` for W=6 and F=3."""
    check, observables, priors = _quits_global_arrays(circuit)
    windows = []
    first_column = 0
    for window_index in range(2):
        row_lo = window_index * _COMMIT_WIDTH * _CHECKS_PER_ROUND
        row_hi = (
            window_index * _COMMIT_WIDTH + _WINDOW_WIDTH
        ) * _CHECKS_PER_ROUND
        local_tail = check[row_lo:row_hi, first_column:]
        last_local_column = np.nonzero(local_tail.any(axis=0))[0][-1]
        local_check = local_tail[:, : last_local_column + 1]

        commit_rows = _COMMIT_WIDTH * _CHECKS_PER_ROUND
        last_owned_column = np.nonzero(
            local_check[:commit_rows].any(axis=0)
        )[0][-1]
        owned_count = last_owned_column + 1
        global_owned = slice(first_column, first_column + owned_count)
        next_row = (window_index + 1) * _COMMIT_WIDTH * _CHECKS_PER_ROUND
        windows.append((
            local_check,
            observables[:, global_owned],
            priors[first_column:first_column + local_check.shape[1]],
            check[next_row:next_row + _CHECKS_PER_ROUND, global_owned],
            owned_count,
        ))
        first_column += owned_count

    windows.append((
        check[2 * _COMMIT_WIDTH * _CHECKS_PER_ROUND:, first_column:],
        observables[:, first_column:],
        priors[first_column:],
        None,
        check.shape[1] - first_column,
    ))
    return windows


def test_decsim_exactly_matches_pinned_quits_window_matrices_and_handoff():
    """Match QUITS v1.1.0, commit 31ab782, on its BB-style circuit path.

    References:
      * Kang et al., Quantum 9, 1931 (2025), Sections 5.1-5.2,
        https://doi.org/10.22331/q-2025-12-05-1931
      * https://github.com/mkangquantum/quits/blob/31ab78252f5bbe169d11956450a5c9d2b1184d51/src/quits/decoder/base.py#L74-L190
      * https://github.com/mkangquantum/quits/blob/31ab78252f5bbe169d11956450a5c9d2b1184d51/src/quits/decoder/sliding_window.py#L104-L188
    """
    circuit = stim.Circuit.from_file(str(
        pathlib.Path(__file__).parent / "data/bb72_12_6_p003_r10.stim"
    ))
    detector_rounds = {
        detector_id: detector_id // _CHECKS_PER_ROUND + 1
        for detector_id in range(circuit.num_detectors)
    }
    planned = SlidingWindowScheme().plan_operation(
        0,
        12,
        commit_round_count=_COMMIT_WIDTH,
        buffer_round_count=_WINDOW_WIDTH - _COMMIT_WIDTH,
    ).windows
    plan = tuple(
        (window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in planned
    )
    assert plan == _QUITS_PLAN
    models = build_window_error_models(
        circuit,
        list(plan),
        round_count=12,
        detector_rounds=detector_rounds,
        fault_model_requirement=PHYSICAL_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )

    for model, oracle in zip(models, _quits_window_arrays(circuit)):
        expected_check, expected_observables, expected_priors, update, owned_count = oracle
        faults = model.physical_faults
        assert np.array_equal(faults.check, expected_check)
        assert np.allclose(faults.priors, expected_priors, rtol=0, atol=1e-15)
        assert np.array_equal(
            np.nonzero(faults.owned)[0], np.arange(owned_count)
        )
        assert np.array_equal(
            faults.observables[:, :owned_count], expected_observables
        )
        if update is None:
            assert not faults.future_flips
            continue

        next_round = model.commit_hi + 1
        next_detectors = tuple(
            detector_id
            for detector_id in range(circuit.num_detectors)
            if detector_rounds[detector_id] == next_round
        )
        local_row = {
            detector_id: row for row, detector_id in enumerate(next_detectors)
        }
        actual_update = np.zeros(update.shape, dtype=np.uint8)
        for column in range(owned_count):
            for detector_id in faults.future_flips.get(column, ()):
                if detector_id in local_row:
                    actual_update[local_row[detector_id], column] ^= 1
        assert np.array_equal(actual_update, update)
