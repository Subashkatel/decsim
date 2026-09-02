"""A window slice keeps the faults its rows can see and owns its commit rounds.

Sources: Skoric et al. 2209.08552, section II (a window is a commit
region followed by a buffer region; only the commit region's corrections
are final, and a chain cut at the boundary leaves an artificial defect
for the next window); qLDPC's SlidingWindowDecoder (a window's detection
region is its rounds' detectors, and an error committed by one window is
removed from every later window); Tan et al. 2209.09219 (a committed
fault's flips are handed to the neighbouring window in either direction).
"""

import pytest
import stim

from decsim.detector_error_model import fault_model_contracts, window_slicer

GRAPHLIKE = fault_model_contracts.FaultRepresentation.GRAPHLIKE


def surface_code_slicer(rounds):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=rounds,
        after_clifford_depolarization=0.001,
    )
    return window_slicer.WindowSlicer(
        circuit,
        round_count=rounds,
        fault_model_requirement=fault_model_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED,
    )


def detectors_in_rounds(slicer, rounds):
    """The detectors whose round is one of `rounds`, in index order."""
    return tuple(
        detector_id
        for detector_id in sorted(slicer.round_by_detector)
        if slicer.round_by_detector[detector_id] in rounds
    )


def faults_touching(catalog, rows):
    """The catalog faults that flip at least one of `rows`."""
    rows = set(rows)
    return tuple(
        fault_index
        for fault_index, detectors in enumerate(catalog.detector_sets)
        if rows & set(detectors)
    )


def rounds_of_column(slicer, catalog, fault_index):
    detectors = catalog.detector_sets[fault_index]
    return {slicer.round_by_detector[detector_id] for detector_id in detectors}


def owned_faults(model):
    faults = model.require_faults(GRAPHLIKE)
    return [
        fault_index
        for fault_index, is_owned in zip(faults.source_fault_ids, faults.owned)
        if is_owned
    ]


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


def expected_ownership(slicer, catalog, faults, commit_rounds):
    """A column is owned when one of its detectors lies in a commit round."""
    owned = []
    for fault_index in faults.source_fault_ids:
        rounds = rounds_of_column(slicer, catalog, fault_index)
        touches_commit_round = rounds & commit_rounds
        owned.append(bool(touches_commit_round))
    return owned


def test_a_windows_rows_are_the_detectors_of_its_buffer_rounds():
    slicer = surface_code_slicer(6)
    model = slicer.slice_window(1, 1, 2, 3, is_last=False)
    assert model.detector_ids == detectors_in_rounds(slicer, {1, 2, 3})
    assert len(model.detector_ids) == 20


def test_a_windows_columns_are_the_catalog_faults_that_flip_one_of_its_rows():
    slicer = surface_code_slicer(6)
    catalog = slicer.catalogs[GRAPHLIKE]
    model = slicer.slice_window(1, 1, 2, 3, is_last=False)
    faults = model.require_faults(GRAPHLIKE)
    assert faults.source_fault_ids == faults_touching(
        catalog, model.detector_ids
    )


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


def test_a_fault_touching_a_commit_round_is_owned_and_a_buffer_one_is_not():
    slicer = surface_code_slicer(6)
    catalog = slicer.catalogs[GRAPHLIKE]
    model = slicer.slice_window(1, 1, 2, 3, is_last=False)
    faults = model.require_faults(GRAPHLIKE)
    expected_owned = expected_ownership(slicer, catalog, faults, {1, 2})
    assert faults.owned.tolist() == expected_owned
    assert True in expected_owned
    assert False in expected_owned


def test_a_committed_fault_is_left_out_of_the_next_window():
    slicer = surface_code_slicer(6)
    catalog = slicer.catalogs[GRAPHLIKE]
    first = slicer.slice_window(1, 1, 2, 3, is_last=False)
    second = slicer.slice_window(3, 3, 4, 5, is_last=False)
    second_faults = second.require_faults(GRAPHLIKE)
    first_owns = owned_faults(first)
    committed = set(first_owns)
    touching = faults_touching(catalog, second.detector_ids)
    still_open = tuple(
        fault_index for fault_index in touching if fault_index not in committed
    )
    assert second_faults.source_fault_ids == still_open
    assert committed & set(second_faults.source_fault_ids) == set()


def test_the_last_window_owns_everything_it_sees():
    slicer = surface_code_slicer(6)
    slicer.slice_window(1, 1, 2, 3, is_last=False)
    slicer.slice_window(3, 3, 4, 5, is_last=False)
    last = slicer.slice_window(5, 5, 6, 6, is_last=True)
    faults = last.require_faults(GRAPHLIKE)
    assert faults.owned.all()
    assert last.detector_ids == detectors_in_rounds(slicer, {5, 6})


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
    assert all_owned == list(range(len(catalog.detector_sets)))


def test_an_owned_column_hands_off_its_whole_detector_effect():
    slicer = surface_code_slicer(6)
    catalog = slicer.catalogs[GRAPHLIKE]
    model = slicer.slice_window(1, 1, 2, 3, is_last=False)
    faults = model.require_faults(GRAPHLIKE)
    expected_flips = {
        column: catalog.detector_sets[fault_index]
        for column, (fault_index, is_owned) in enumerate(
            zip(faults.source_fault_ids, faults.owned)
        )
        if is_owned
    }
    assert dict(faults.boundary_flips) == expected_flips


def test_defect_positions_cover_the_rows_and_every_handed_off_detector():
    slicer = surface_code_slicer(6)
    model = slicer.slice_window(1, 1, 2, 3, is_last=False)
    faults = model.require_faults(GRAPHLIKE)
    handed_off = set()
    for flips in faults.boundary_flips.values():
        handed_off.update(flips)
    assert set(model.defect_positions) == set(model.detector_ids) | handed_off
    assert model.defect_positions[0] == (1, 0)
    assert model.defect_positions[11] == (2, 7)


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


def test_an_inverted_exclusion_range_is_refused():
    slicer = surface_code_slicer(4)
    with pytest.raises(ValueError, match="is inverted"):
        slicer.slice_window(
            1, 1, 2, 3, is_last=False, fault_exclusion_ranges=((3, 1),)
        )
