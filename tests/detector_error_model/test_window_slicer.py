"""A window slice keeps the faults its rows can see and owns its commit rounds.

Sources: Skoric et al. 2209.08552, section II (a window is a commit
region followed by a buffer region; only the commit region's corrections
are final, and a chain cut at the boundary leaves an artificial defect
for the next window) and Tan et al. 2209.09219 (a committed fault's flips
are handed to the neighbouring window in either direction). The window
rule itself is compared against qLDPC's SlidingWindowDecoder in
test_window_model_builders; here every expected value is written out for
the distance-3 rotated surface code.
"""

import pytest
import stim

from decsim.detector_error_model import (
    detector_chronology,
    fault_model_contracts,
    window_slicer,
)

GRAPHLIKE = fault_model_contracts.FaultRepresentation.GRAPHLIKE
PHYSICAL = fault_model_contracts.FaultRepresentation.PHYSICAL


def surface_code_circuit(rounds):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=rounds,
        after_clifford_depolarization=0.001,
    )


def surface_code_slicer(
    rounds,
    requirement=fault_model_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED,
):
    circuit = surface_code_circuit(rounds)
    return window_slicer.WindowSlicer(
        circuit, round_count=rounds, fault_model_requirement=requirement
    )


def round_by_detector(rounds):
    circuit = surface_code_circuit(rounds)
    return detector_chronology.resolve_detector_rounds(circuit, None, rounds)


def detectors_in_rounds(rounds, chosen_rounds):
    """The detectors whose round is one of `chosen_rounds`, in index order."""
    rounds_of_detectors = round_by_detector(rounds)
    return tuple(
        detector_id
        for detector_id in sorted(rounds_of_detectors)
        if rounds_of_detectors[detector_id] in chosen_rounds
    )


def faults_touching(catalog, rows):
    """The catalog faults that flip at least one of `rows`."""
    rows = set(rows)
    return tuple(
        fault_index
        for fault_index, detectors in enumerate(catalog.detector_sets)
        if rows & set(detectors)
    )


def owned_faults(model):
    faults = model.require_faults(GRAPHLIKE)
    return [
        fault_index
        for fault_index, is_owned in zip(faults.source_fault_ids, faults.owned)
        if is_owned
    ]


def column_rows(matrix, column):
    """The sorted row indices of one column of a sparse matrix."""
    chosen = matrix[:, column]
    rows = chosen.indices.tolist()
    return sorted(rows)


def handed_off_outside(model):
    """The detectors the window's owned columns flip beyond its rows."""
    faults = model.require_faults(GRAPHLIKE)
    outside = set()
    for flips in faults.boundary_flips.values():
        outside.update(flips)
    return outside - set(model.detector_ids)


def columns_reaching_outside(model):
    """How many owned columns flip a detector beyond the window's rows."""
    faults = model.require_faults(GRAPHLIKE)
    rows = set(model.detector_ids)
    count = 0
    for flips in faults.boundary_flips.values():
        if set(flips) - rows:
            count += 1
    return count


def expected_check(catalog, model):
    """The check matrix the catalog implies for the window's rows, columns."""
    row_by_detector = {}
    for row, detector_id in enumerate(model.detector_ids):
        row_by_detector[detector_id] = row
    faults = model.require_faults(GRAPHLIKE)
    zero_row = [0] * len(faults.source_fault_ids)
    dense = [list(zero_row) for _ in model.detector_ids]
    for column, fault_index in enumerate(faults.source_fault_ids):
        detectors = catalog.detector_sets[fault_index]
        local_rows = [
            row_by_detector[detector_id]
            for detector_id in detectors
            if detector_id in row_by_detector
        ]
        for row in local_rows:
            dense[row][column] = 1
    return dense


def expected_observables(catalog, faults):
    """One row for observable 0: a one where the column flips it."""
    row = []
    for fault_index in faults.source_fault_ids:
        flips_observable = 0 in catalog.observable_sets[fault_index]
        row.append(int(flips_observable))
    return [row]


def expected_ownership(rounds, catalog, faults, commit_rounds):
    """A column is owned when one of its detectors lies in a commit round."""
    rounds_of_detectors = round_by_detector(rounds)
    owned = []
    for fault_index in faults.source_fault_ids:
        detectors = catalog.detector_sets[fault_index]
        fault_rounds = {rounds_of_detectors[detector] for detector in detectors}
        touches_commit_round = fault_rounds & commit_rounds
        owned.append(bool(touches_commit_round))
    return owned


def test_a_windows_rows_are_the_detectors_of_its_buffer_rounds():
    slicer = surface_code_slicer(6)
    model = slicer.slice_window(1, 1, 2, 3, is_last=False)
    assert model.detector_ids == detectors_in_rounds(6, {1, 2, 3})
    assert model.detector_ids == tuple(range(20))


def test_a_windows_columns_are_the_catalog_faults_that_flip_one_of_its_rows():
    slicer = surface_code_slicer(6)
    catalog = slicer.catalogs[GRAPHLIKE]
    model = slicer.slice_window(1, 1, 2, 3, is_last=False)
    faults = model.require_faults(GRAPHLIKE)
    assert faults.source_fault_ids == faults_touching(
        catalog, model.detector_ids
    )
    assert len(faults.source_fault_ids) == 80
    assert faults.source_fault_ids[59:63] == (59, 61, 62, 63)


def test_the_window_matrices_agree_with_the_catalog_fault_by_fault():
    slicer = surface_code_slicer(6)
    catalog = slicer.catalogs[GRAPHLIKE]
    model = slicer.slice_window(1, 1, 2, 3, is_last=False)
    faults = model.require_faults(GRAPHLIKE)
    dense_check = faults.check.toarray()
    dense_observables = faults.observables.toarray()
    assert dense_check.tolist() == expected_check(catalog, model)
    assert faults.priors.tolist() == [
        catalog.priors[fault_index] for fault_index in faults.source_fault_ids
    ]
    assert dense_observables.tolist() == expected_observables(catalog, faults)
    assert catalog.detector_sets[0] == (0,)
    assert column_rows(faults.check, 0) == [0]
    assert catalog.observable_sets[7] == (0,)
    assert dense_observables[0][7] == 1
    assert dense_observables[0][0] == 0


def test_a_fault_touching_a_commit_round_is_owned_and_a_buffer_one_is_not():
    slicer = surface_code_slicer(6)
    catalog = slicer.catalogs[GRAPHLIKE]
    model = slicer.slice_window(1, 1, 2, 3, is_last=False)
    faults = model.require_faults(GRAPHLIKE)
    expected_owned = expected_ownership(6, catalog, faults, {1, 2})
    assert faults.owned.tolist() == expected_owned
    assert faults.source_fault_ids[0] == 0
    assert faults.owned[0]
    assert faults.source_fault_ids[26] == 26
    assert not faults.owned[26]


def test_a_committed_fault_is_left_out_of_the_next_window():
    slicer = surface_code_slicer(6)
    first = slicer.slice_window(1, 1, 2, 3, is_last=False)
    second = slicer.slice_window(3, 3, 4, 5, is_last=False)
    second_faults = second.require_faults(GRAPHLIKE)
    first_owns = owned_faults(first)
    assert 0 in first_owns
    assert 26 not in first_owns
    assert second_faults.source_fault_ids == (
        (26, 31, 37, 46, 48, 49, 52, 55)
        + tuple(range(56, 124))
        + (125, 126, 127, 128, 130, 131, 132, 133, 134)
        + (136, 137, 138, 139, 140, 141, 143, 146, 147, 149, 150)
    )


def test_the_last_window_owns_everything_it_sees():
    slicer = surface_code_slicer(6)
    slicer.slice_window(1, 1, 2, 3, is_last=False)
    slicer.slice_window(3, 3, 4, 5, is_last=False)
    last = slicer.slice_window(5, 5, 6, 6, is_last=True)
    faults = last.require_faults(GRAPHLIKE)
    assert faults.owned.all()
    assert last.detector_ids == detectors_in_rounds(6, {5, 6})
    assert last.detector_ids == tuple(range(28, 48))


def test_every_fault_is_owned_by_exactly_one_window_of_a_sliding_plan():
    slicer = surface_code_slicer(6)
    catalog = slicer.catalogs[GRAPHLIKE]
    models = [
        slicer.slice_window(1, 1, 2, 3, is_last=False),
        slicer.slice_window(3, 3, 4, 5, is_last=False),
        slicer.slice_window(5, 5, 6, 6, is_last=True),
    ]
    owned_by_window = [owned_faults(model) for model in models]
    all_owned = sorted(sum(owned_by_window, []))
    assert len(catalog.detector_sets) == 174
    assert all_owned == list(range(174))


def test_a_last_window_takes_every_round_from_its_start_and_owns_it_all():
    slicer = surface_code_slicer(4)
    model = slicer.slice_window(2, 2, 2, 2, is_last=True)
    faults = model.require_faults(GRAPHLIKE)
    assert model.detector_ids == tuple(range(4, 32))
    assert len(faults.source_fault_ids) == 103
    assert faults.owned.all()


def test_an_owned_column_hands_off_its_whole_detector_effect():
    slicer = surface_code_slicer(4)
    catalog = slicer.catalogs[GRAPHLIKE]
    model = slicer.slice_window(1, 1, 3, 3, is_last=False)
    faults = model.require_faults(GRAPHLIKE)
    assert model.detector_ids == tuple(range(20))
    assert set(faults.boundary_flips) == set(range(80))
    assert faults.source_fault_ids[56] == 56
    assert catalog.detector_sets[56] == (12, 20)
    assert faults.boundary_flips[56] == (12, 20)
    assert column_rows(faults.check, 56) == [12]
    assert handed_off_outside(model) == {20, 21, 22, 23, 24, 25, 26, 27}
    assert columns_reaching_outside(model) == 18


def test_defect_positions_cover_the_rows_and_every_handed_off_detector():
    slicer = surface_code_slicer(4)
    model = slicer.slice_window(1, 1, 3, 3, is_last=False)
    assert set(model.defect_positions) == set(range(28))
    assert model.defect_positions[0] == (1, 0)
    assert model.defect_positions[11] == (2, 7)
    assert model.defect_positions[20] == (4, 0)
    assert model.defect_positions[27] == (4, 7)


def test_a_linked_window_projects_each_physical_column_onto_its_components():
    slicer = surface_code_slicer(
        4, fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED
    )
    graphlike_catalog = slicer.catalogs[GRAPHLIKE]
    physical_catalog = slicer.catalogs[PHYSICAL]
    model = slicer.slice_window(1, 1, 2, 3, is_last=False)
    graphlike = model.require_faults(GRAPHLIKE)
    physical = model.require_faults(PHYSICAL)
    projection = model.physical_to_graphlike_detector_projection
    assert graphlike.check.shape == (20, 80)
    assert physical.check.shape == (20, 336)
    assert projection.shape == (80, 336)
    assert physical.source_fault_ids[5] == 5
    assert physical_catalog.detector_sets[5] == (1, 4, 5)
    assert column_rows(physical.check, 5) == [1, 4, 5]
    assert physical.priors[5] == physical_catalog.priors[5]
    assert column_rows(projection, 5) == [4, 5]
    assert graphlike.source_fault_ids[4:6] == (4, 5)
    assert graphlike_catalog.detector_sets[4] == (1, 5)
    assert graphlike_catalog.detector_sets[5] == (4,)


def test_a_physical_only_window_keeps_hyperedges_and_has_no_projection():
    slicer = surface_code_slicer(
        4, fault_model_contracts.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    catalog = slicer.catalogs[PHYSICAL]
    model = slicer.slice_window(1, 1, 2, 3, is_last=False)
    physical = model.require_faults(PHYSICAL)
    assert model.graphlike_faults is None
    assert model.physical_to_graphlike_detector_projection is None
    assert physical.check.shape == (20, 262)
    assert physical.source_fault_ids[4] == 4
    assert catalog.detector_sets[4] == (1, 4, 5)
    assert column_rows(physical.check, 4) == [1, 4, 5]


def test_a_window_carries_stims_coordinates_for_its_rows():
    slicer = surface_code_slicer(4)
    model = slicer.slice_window(1, 1, 2, 3, is_last=False)
    assert len(model.detector_coordinates) == 20
    assert model.detector_coordinates[0] == (0.0, 4.0, 0.0)
    assert model.detector_coordinates[4] == (2.0, 0.0, 1.0)
    assert model.detector_coordinates[19] == (4.0, 6.0, 2.0)


def test_a_window_has_no_coordinates_when_a_detector_has_none():
    circuit = stim.Circuit(
        "R 0 1\nX_ERROR(0.1) 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n"
    )
    slicer = window_slicer.WindowSlicer(
        circuit,
        round_count=1,
        detector_rounds={0: 1, 1: 1},
        fault_model_requirement=fault_model_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED,
    )
    model = slicer.slice_window(1, 1, 1, 1, is_last=True)
    assert model.detector_ids == (0, 1)
    assert model.detector_coordinates is None


def test_explicit_owner_and_prior_maps_must_come_together():
    slicer = surface_code_slicer(4)
    with pytest.raises(ValueError, match="supplied together"):
        slicer.slice_window(
            1,
            1,
            2,
            3,
            is_last=False,
            explicitly_owned_faults={GRAPHLIKE: set()},
        )
