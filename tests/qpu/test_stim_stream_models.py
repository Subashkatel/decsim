"""Growing window models agree with complete Stim circuits and WindowSlicer.

The exact slicer supplies ownership and handoff referents; full detector,
logical and linked decomposition effects identify columns across growth.
"""

import dataclasses

import numpy
import pytest

import decsim.detector_error_model.detector_formation as formation
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.detector_error_model.window_slicer as window_slicer
import decsim.frontends.deltakit as deltakit
import decsim.qpu.stim_stream_models as stream_models
import decsim.records.circuits as circuit_records
import decsim.records.windows as window_records
import tests.qpu.memory_programs as memory_programs

GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE
PHYSICAL = fault_models.FaultRepresentation.PHYSICAL
REPRESENTATIONS = frozenset({GRAPHLIKE, PHYSICAL})
LINKED = fault_models.DecoderFaultModelRequirement(
    REPRESENTATIONS, require_physical_to_graphlike_link=True
)


@pytest.fixture(params=["stim", "deltakit"])
def program(
    request: pytest.FixtureRequest,
) -> circuit_records.RepeatedStimCircuit:
    """Two producers provide the same ordinary Stim fragment contract."""
    if request.param == "stim":
        return memory_programs.memory_program()
    pytest.importorskip("deltakit_explorer")
    return deltakit.memory_rounds(
        "rotated_surface", 3, "Z", 0.003, round_period_microseconds=1.1
    )


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("round_count", [1, 2, 7, 13])
def test_assembled_fragments_preserve_stims_generated_circuit(
    distance: int, basis: str, round_count: int
) -> None:
    program = memory_programs.memory_program(distance, basis)
    assembled, measurement_rounds = program.assemble(round_count)
    expected = memory_programs.memory_circuit(round_count, distance, basis)
    actual_flattened = assembled.flattened()
    expected_flattened = expected.flattened()
    assert actual_flattened == expected_flattened
    assert len(measurement_rounds) == expected.num_measurements
    assert measurement_rounds[0] == 1
    last_measurement = expected.num_measurements - 1
    assert measurement_rounds[last_measurement] == round_count


@pytest.mark.parametrize("horizon", [13, 21])
def test_interior_window_effects_match_a_longer_complete_circuit(
    program: circuit_records.RepeatedStimCircuit, horizon: int
) -> None:
    growing = stream_models.GrowingStimModels(program, LINKED)
    first = window_records.Window(1, 0, 1, 3, 6, 6)
    actual = growing.for_window(first)
    referent = _referent(program, horizon)
    expected = _slice(referent, first)
    _assert_same_model(actual, expected)
    _assert_linked_detector_effects(actual)


def test_distance_five_x_memory_matches_the_complete_model() -> None:
    program = memory_programs.memory_program(5, "X")
    growing = stream_models.GrowingStimModels(program, LINKED)
    first = window_records.Window(1, 0, 1, 5, 10, 10)
    actual = growing.for_window(first)
    referent = _referent(program, 17)
    expected = _slice(referent, first)
    _assert_same_model(actual, expected)
    _assert_linked_detector_effects(actual)


def test_queued_fault_identities_survive_later_model_horizons(
    program: circuit_records.RepeatedStimCircuit,
) -> None:
    growing = stream_models.GrowingStimModels(program, LINKED)
    first = window_records.Window(1, 0, 1, 3, 6, 6)
    before = growing.for_window(first)
    first.queued = True
    second = window_records.Window(1, 1, 4, 6, 9, 6)
    growing.for_window(second)
    after = growing.for_window(first)
    _assert_same_model(before, after)
    _assert_stable_identities(before, after)


@pytest.mark.parametrize("round_count", [1, 2, 3])
def test_finishing_rebuilds_the_unqueued_initial_terminal_model(
    program: circuit_records.RepeatedStimCircuit, round_count: int
) -> None:
    growing = stream_models.GrowingStimModels(program, LINKED)
    pending = window_records.Window(1, 0, 1, 3, 6, 6)
    growing.for_window(pending)
    growing.finish(round_count)
    terminal = window_records.Window(
        1, 0, 1, round_count, round_count, round_count
    )
    actual = growing.for_window(terminal)
    referent = _referent(program, round_count)
    expected = _slice(referent, terminal, is_last=True)
    _assert_same_model(actual, expected)
    _assert_linked_detector_effects(actual)


@pytest.mark.parametrize("round_count", [7, 8, 9])
def test_finishing_refreshes_tail_models_without_changing_queued_ownership(
    program: circuit_records.RepeatedStimCircuit, round_count: int
) -> None:
    growing = stream_models.GrowingStimModels(program, LINKED)
    first = window_records.Window(1, 0, 1, 3, 6, 6)
    before = growing.for_window(first)
    first.queued = True
    pending = window_records.Window(1, 1, 4, 6, 9, 6)
    growing.for_window(pending)
    growing.finish(round_count)
    middle_round_count = round_count - 3
    middle = dataclasses.replace(
        pending, buffer_hi=round_count, round_count=middle_round_count
    )
    growing.for_window(middle)
    terminal_count = round_count - 6
    terminal = window_records.Window(
        1, 2, 7, round_count, round_count, terminal_count
    )
    actual = growing.for_window(terminal)
    referent = _referent(program, round_count)
    _slice(referent, first)
    _slice(referent, middle)
    expected = _slice(referent, terminal, is_last=True)
    _assert_same_model(actual, expected)
    unchanged = growing.for_window(first)
    _assert_same_model(before, unchanged)
    _assert_stable_identities(before, unchanged)


def test_strong_priors_use_stable_fault_ids_after_catalog_growth(
    program: circuit_records.RepeatedStimCircuit,
) -> None:
    growing = stream_models.GrowingStimModels(program, LINKED)
    first = window_records.Window(1, 0, 1, 3, 6, 6)
    second = window_records.Window(1, 1, 4, 6, 9, 6)
    first_model = growing.for_window(first)
    second_model = growing.for_window(second)
    priors = _owned_union(first_model, second_model)
    strong = window_records.Window(1, 2, 7, 9, 12, 6)
    actual = growing.for_strong_window(strong, (), priors)
    referent = _referent(program, 17)
    first_reference = _slice(referent, first)
    second_reference = _slice(referent, second)
    local_priors = _owned_union(first_reference, second_reference)
    assert priors != local_priors
    expected = _slice_with_priors(referent, strong, local_priors)
    _assert_same_model(actual, expected)
    _assert_excludes_priors(actual, priors)


def test_a_closed_weak_window_is_refused_at_the_model_boundary(
    program: circuit_records.RepeatedStimCircuit,
) -> None:
    growing = stream_models.GrowingStimModels(program, LINKED)
    closed = window_records.Window(
        1, 0, 1, 3, 6, 6, closed_temporal_boundaries=True
    )
    with pytest.raises(ValueError, match="trailing-buffer feedback"):
        growing.for_window(closed)


def test_a_closed_strong_window_is_refused_at_the_model_boundary(
    program: circuit_records.RepeatedStimCircuit,
) -> None:
    growing = stream_models.GrowingStimModels(program, LINKED)
    closed = window_records.Window(
        1, 0, 1, 3, 6, 6, closed_temporal_boundaries=True
    )
    with pytest.raises(ValueError, match="trailing-buffer feedback"):
        growing.for_strong_window(closed, (), None)


def _referent(
    program: circuit_records.RepeatedStimCircuit, round_count: int
) -> window_slicer.WindowSlicer:
    circuit, measurement_rounds = program.assemble(round_count)
    table = formation.build_formation_table(
        circuit, round_count, measurement_rounds=measurement_rounds
    )
    detector_rounds = table.detector_rounds()
    return window_slicer.WindowSlicer(
        circuit,
        round_count=round_count,
        detector_rounds=detector_rounds,
        fault_model_requirement=LINKED,
    )


def _slice(
    referent: window_slicer.WindowSlicer,
    window: window_records.Window,
    is_last: bool = False,
) -> fault_models.WindowErrorModel:
    return referent.slice_window(
        window.start_round,
        window.commit_lo,
        window.commit_hi,
        window.buffer_hi,
        is_last=is_last,
    )


def _slice_with_priors(
    referent: window_slicer.WindowSlicer,
    window: window_records.Window,
    priors: dict,
) -> fault_models.WindowErrorModel:
    return referent.slice_window(
        window.start_round,
        window.commit_lo,
        window.commit_hi,
        window.buffer_hi,
        is_last=False,
        explicitly_prior_faults=priors,
    )


def _assert_same_model(
    actual: fault_models.WindowErrorModel,
    expected: fault_models.WindowErrorModel,
) -> None:
    assert actual.detector_ids == expected.detector_ids
    assert actual.detector_coordinates == expected.detector_coordinates
    assert actual.defect_positions == expected.defect_positions
    assert actual.first_commit_round == expected.first_commit_round
    _assert_same_faults(actual, expected, GRAPHLIKE)
    _assert_same_faults(actual, expected, PHYSICAL)


def _assert_same_faults(
    actual: fault_models.WindowErrorModel,
    expected: fault_models.WindowErrorModel,
    representation: fault_models.FaultRepresentation,
) -> None:
    actual_columns = _columns_by_effect(actual, representation)
    expected_columns = _columns_by_effect(expected, representation)
    assert actual_columns.keys() == expected_columns.keys()
    for identity, probabilities in actual_columns.items():
        actual_priors = sorted(probabilities)
        expected_priors = sorted(expected_columns[identity])
        numpy.testing.assert_allclose(
            actual_priors, expected_priors, rtol=0, atol=1e-15
        )


def _columns_by_effect(
    model: fault_models.WindowErrorModel,
    representation: fault_models.FaultRepresentation,
) -> dict:
    faults = model.require_faults(representation)
    assert len(set(faults.source_fault_ids)) == len(faults.source_fault_ids)
    columns: dict = {}
    for column_index in range(len(faults.source_fault_ids)):
        signature = _column_effect(model, faults, column_index)
        components = _linked_components(model, representation, column_index)
        identity = (signature, components)
        probabilities = columns.setdefault(identity, [])
        probabilities.append(float(faults.priors[column_index]))
    return columns


def _column_effect(
    model: fault_models.WindowErrorModel,
    faults: fault_models.PlacedFaultModel,
    column_index: int,
) -> tuple:
    check_column = faults.check.getcol(column_index)
    detectors = tuple(model.detector_ids[row] for row in check_column.indices)
    observable_column = faults.observables.getcol(column_index)
    observables = tuple(observable_column.indices)
    owned = bool(faults.owned[column_index])
    handoff = faults.boundary_flips.get(column_index, ())
    return detectors, observables, owned, handoff


def _linked_components(
    model: fault_models.WindowErrorModel,
    representation: fault_models.FaultRepresentation,
    column_index: int,
) -> tuple:
    if representation is GRAPHLIKE:
        return ()
    projection = model.physical_to_graphlike_detector_projection
    assert projection is not None
    graphlike = model.require_faults(GRAPHLIKE)
    linked_column = projection.getcol(column_index)
    components = [
        _column_effect(model, graphlike, component)
        for component in linked_column.indices
    ]
    return tuple(sorted(components))


def _assert_linked_detector_effects(
    model: fault_models.WindowErrorModel,
) -> None:
    graphlike = model.require_faults(GRAPHLIKE)
    physical = model.require_faults(PHYSICAL)
    projection = model.physical_to_graphlike_detector_projection
    assert projection is not None
    projected = graphlike.check @ projection
    projected = projected.toarray()
    projected %= 2
    expected = physical.check.toarray()
    numpy.testing.assert_array_equal(projected, expected)
    column_counts = projection.getnnz(axis=0)
    assert max(column_counts) > 1


def _assert_stable_identities(
    before: fault_models.WindowErrorModel,
    after: fault_models.WindowErrorModel,
) -> None:
    earlier_graphlike = _columns_by_stable_id(before, GRAPHLIKE)
    later_graphlike = _columns_by_stable_id(after, GRAPHLIKE)
    earlier_physical = _columns_by_stable_id(before, PHYSICAL)
    later_physical = _columns_by_stable_id(after, PHYSICAL)
    assert earlier_graphlike == later_graphlike
    assert earlier_physical == later_physical


def _columns_by_stable_id(
    model: fault_models.WindowErrorModel,
    representation: fault_models.FaultRepresentation,
) -> dict:
    faults = model.require_faults(representation)
    columns = {}
    for column_index, identity in enumerate(faults.source_fault_ids):
        effect = _column_effect(model, faults, column_index)
        components = _linked_components(model, representation, column_index)
        columns[identity] = (effect, components)
    return columns


def _owned_union(
    first: fault_models.WindowErrorModel,
    second: fault_models.WindowErrorModel,
) -> dict:
    first_owned = first.owned_fault_ids()
    second_owned = second.owned_fault_ids()
    return {
        GRAPHLIKE: first_owned[GRAPHLIKE] | second_owned[GRAPHLIKE],
        PHYSICAL: first_owned[PHYSICAL] | second_owned[PHYSICAL],
    }


def _assert_excludes_priors(
    model: fault_models.WindowErrorModel, priors: dict
) -> None:
    graphlike = model.require_faults(GRAPHLIKE)
    physical = model.require_faults(PHYSICAL)
    assert priors[GRAPHLIKE].isdisjoint(graphlike.source_fault_ids)
    assert priors[PHYSICAL].isdisjoint(physical.source_fault_ids)
