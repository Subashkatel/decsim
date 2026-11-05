"""The truth decoder: each window commits exactly the faults the shot fired.

A hand-made window checks the selection: the owned columns whose identity
fired an odd number of times, and nothing else. A small switching machine
with the truth in both pools checks the whole chain: no shot fails, and
every window's committed answer equals its label (window_outcomes.py),
on Toshio's double window, where a strong region pins its far face on
the next window's decode.
"""

import dataclasses

import numpy
import pytest
import scipy.sparse

import decsim.decoders.truth as truth
import decsim.experiments.collect as collect
import decsim.experiments.measure as measure
import decsim.parameters.assignments as assignments_module
import decsim.parameters.tree as tree
import decsim.records.fault_model_contracts as fault_models
import decsim.records.program as program_records
import decsim.records.sampled_errors as sampled_errors
import tests.experiments.run_files as run_files

SHOTS = 12


def _fired(detectors, observables):
    return sampled_errors.FiredError(
        first_round=0, logical_observables=observables, detectors=detectors
    )


def _operation(operation_id):
    return program_records.Operation(
        id=operation_id, name="memory", qubits=(0,)
    )


def _window_faults():
    """Three columns on three detectors; the first two owned.

    Column 0 flips detectors 0 and 1 and the observable; column 1 flips
    detector 1 and a detector 7 of a later window; column 2, not owned,
    flips detector 2.
    """
    dense_check = numpy.array(
        [[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=numpy.uint8
    )
    check = scipy.sparse.csc_matrix(dense_check)
    dense_observables = numpy.array([[1, 0, 0]], dtype=numpy.uint8)
    observables = scipy.sparse.csc_matrix(dense_observables)
    priors = numpy.full(3, 0.01)
    owned = numpy.array([True, True, False])
    return fault_models.PlacedFaultModel(
        representation=fault_models.FaultRepresentation.PHYSICAL,
        check=check,
        priors=priors,
        observables=observables,
        owned=owned,
        source_fault_ids=(0, 1, 2),
        boundary_flips={0: (0, 1), 1: (1, 7)},
    )


def _selected(fired_errors):
    settings = truth.TruthDecoder.Settings()
    decoder = settings.build()
    operation = _operation(1)
    decoder.errors_sampled(operation, fired_errors)
    faults = _window_faults()
    backend = decoder.compile(faults, None)
    answer = decoder.decode_window(backend, None, faults, None)
    return tuple(int(bit) for bit in answer.selected_faults)


def test_the_owned_columns_that_fired_are_selected_and_no_others():
    fired = (_fired((1, 7), (0,)), _fired((2,), (0,)), _fired((5, 6), (1,)))

    assert _selected(fired) == (0, 1, 0)


def test_two_fired_errors_of_one_identity_cancel():
    fired = (_fired((0, 1), (1,)), _fired((0, 1), (1,)), _fired((1, 7), (0,)))

    assert _selected(fired) == (0, 1, 0)


def test_an_identity_needs_its_observables_too():
    """Detectors 0 and 1 without the observable are another mechanism."""
    fired = (_fired((0, 1), (0,)),)

    assert _selected(fired) == (0, 0, 0)


def test_two_columns_of_one_identity_select_one():
    """Linked catalogs keep a column per Stim instruction; both alike.

    Selecting both would cancel the fault and its boundary, so a fired
    identity selects only its first column.
    """
    dense_check = numpy.array([[1, 1], [1, 1]], dtype=numpy.uint8)
    check = scipy.sparse.csc_matrix(dense_check)
    dense_observables = numpy.array([[1, 1]], dtype=numpy.uint8)
    observables = scipy.sparse.csc_matrix(dense_observables)
    priors = numpy.full(2, 0.01)
    owned = numpy.array([True, True])
    faults = fault_models.PlacedFaultModel(
        representation=fault_models.FaultRepresentation.PHYSICAL,
        check=check,
        priors=priors,
        observables=observables,
        owned=owned,
        source_fault_ids=(0, 1),
        boundary_flips={0: (0, 1), 1: (0, 1)},
    )
    settings = truth.TruthDecoder.Settings()
    decoder = settings.build()
    operation = _operation(1)
    fired = (_fired((0, 1), (1,)),)
    decoder.errors_sampled(operation, fired)
    backend = decoder.compile(faults, None)

    answer = decoder.decode_window(backend, None, faults, None)

    assert tuple(int(bit) for bit in answer.selected_faults) == (1, 0)


def test_a_decoder_with_no_fired_errors_names_the_source_it_needs():
    settings = truth.TruthDecoder.Settings()
    decoder = settings.build()
    faults = _window_faults()
    backend = decoder.compile(faults, None)

    with pytest.raises(RuntimeError, match="error_model_stim"):
        decoder.decode_window(backend, None, faults, None)


def test_a_second_operation_in_one_shot_is_refused():
    settings = truth.TruthDecoder.Settings()
    decoder = settings.build()
    first = _operation(1)
    second = _operation(2)
    decoder.errors_sampled(first, ())

    with pytest.raises(RuntimeError, match="one operation"):
        decoder.errors_sampled(second, ())


def test_the_unit_is_held_for_the_stated_time():
    settings = truth.TruthDecoder.Settings(latency_microseconds=2.0)
    decoder = settings.build()

    assert decoder.ticks_after_decode(None, 999_999, 5) == 2_000_000


def _truth_machine(*extra):
    """The test switching machine: truth in both pools, double window.

    Windows switch at random (no confidence reads a truth decode), so
    strong regions and their restart windows occur in a few shots.
    """
    machine = run_files.switching_machine(rounds_per_shot=30)
    pairs = (
        ("qpu.source.type", "error_model_stim"),
        ("observation.record_window_outcomes", "true"),
        ("weak.algorithm.type", "truth"),
        ("strong.algorithm.type", "truth"),
        ("switching.confidence.type", "none"),
        ("switching.verdict.type", "bernoulli"),
        ("switching.verdict.rate", "0.3"),
        ("switching.strong_window.type", "double_window"),
        # the double window needs the weak chain to keep committing
        ("windows.boundary_policy.type", "Eager"),
        ("windows.terminal_policy", "lookahead"),
        ("workload.arguments.physical_error_probability", "0.01"),
    ) + extra
    texts = [f"{path}={value}" for path, value in pairs]
    written = assignments_module.from_texts(texts, "a test")
    applied = tree.apply(machine, written)
    return applied.settings


@dataclasses.dataclass(frozen=True)
class _TruthShot:
    """What one shot of the truth machine shows."""

    is_failed: bool
    strong_window_count: int
    fired_error_count: int
    wrong_windows: list


def _truth_shot(task: collect.Task, seed: int) -> _TruthShot:
    shot = collect.run_shot(task, seed)
    measurement = measure.measure_shot(shot)
    tiers = measurement.window_tiers
    strong_tiers = [tier for tier in tiers if tier == "strong"]
    outcomes = shot.machine.observation.window_outcomes
    rows_by_operation = outcomes.outcomes_by_operation
    errors_by_operation = outcomes.errors_by_operation
    fired = [len(errors_by_operation[key]) for key in rows_by_operation]
    wrong = _wrong_windows(rows_by_operation)
    return _TruthShot(
        measurement.logical_failure, len(strong_tiers), sum(fired), wrong
    )


def _wrong_windows(rows_by_operation: dict) -> list:
    """The window outcomes whose answer is not their label."""
    row_lists = rows_by_operation.values()
    return [
        row for rows in row_lists for row in rows if row.answer != row.label
    ]


def test_truth_in_both_pools_fails_no_shot_and_matches_every_label():
    machine = _truth_machine()
    task = collect.Task("truth", machine)

    shots = [_truth_shot(task, seed) for seed in range(SHOTS)]

    failures = [shot.is_failed for shot in shots]
    wrong_windows = [shot.wrong_windows for shot in shots]
    strong_windows = [shot.strong_window_count for shot in shots]
    fired = [shot.fired_error_count for shot in shots]
    assert not any(failures)
    assert not any(wrong_windows)
    # the check means something only when errors fire and regions switch
    assert sum(fired) > SHOTS
    assert sum(strong_windows) > 0


def test_a_source_that_keeps_no_errors_is_refused_at_the_build():
    machine = _truth_machine(("qpu.source.type", "stim_device"))
    task = collect.Task("truth", machine)

    with pytest.raises(ValueError, match="error_model_stim"):
        collect.run_shot(task, 0)
