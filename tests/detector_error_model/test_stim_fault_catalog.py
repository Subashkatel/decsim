"""The fault catalog reads Stim's error instructions the way Stim means them.

Source: Stim, doc/file_format_dem_detector_error_model.md: an error lists
the detectors and observables it flips, a target listed twice cancels,
and `^` separators suggest a decomposition into graphlike components.
Two errors with the same identity merge as independent errors,
p(1-q) + q(1-p), PyMatching's merge_strategy="independent".
"""

import pytest
import stim

from decsim.detector_error_model import (
    fault_model_contracts,
    stim_fault_catalog,
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


def xor_of_sets(detector_sets):
    """The detectors flipped an odd number of times across the sets."""
    odd = set()
    for detectors in detector_sets:
        odd ^= set(detectors)
    return tuple(sorted(odd))


def test_two_errors_with_the_same_identity_merge_as_independent_errors():
    model = stim.DetectorErrorModel("error(0.1) D0 D1\nerror(0.2) D1 D0\n")
    detector_sets, observable_sets, priors = (
        stim_fault_catalog.detector_error_model_to_faults(model)
    )
    assert detector_sets == [(0, 1)]
    assert observable_sets == [()]
    assert priors == [pytest.approx(0.26)]


def test_a_separator_splits_an_error_into_graphlike_components():
    model = stim.DetectorErrorModel("error(0.1) D0 D1 L0 ^ D1 D2\n")
    detector_sets, observable_sets, priors = (
        stim_fault_catalog.detector_error_model_to_faults(model)
    )
    assert detector_sets == [(0, 1), (1, 2)]
    assert observable_sets == [(0,), ()]
    assert priors == [0.1, 0.1]


def test_the_whole_identity_is_reduced_across_the_separators():
    model = stim.DetectorErrorModel("error(0.1) D0 D1 L0 ^ D1 D2\n")
    (record,) = stim_fault_catalog.canonical_error_instructions(model)
    assert record.aggregate_detectors == (0, 2)
    assert record.aggregate_logical_observables == (0,)
    assert record.probability == 0.1
    assert len(record.components) == 2


def test_a_target_listed_twice_cancels():
    model = stim.DetectorErrorModel("error(0.1) D0 D0 D1 L0 L0\n")
    (record,) = stim_fault_catalog.canonical_error_instructions(model)
    assert record.aggregate_detectors == (1,)
    assert record.aggregate_logical_observables == ()


def test_equal_components_of_one_error_cancel_before_merging():
    model = stim.DetectorErrorModel("error(0.1) D0 ^ D1 D2 ^ D1 D2\n")
    detector_sets, _, priors = (
        stim_fault_catalog.detector_error_model_to_faults(model)
    )
    assert detector_sets == [(0,)]
    assert priors == [0.1]


def test_an_error_that_flips_nothing_is_left_out_but_still_counted():
    model = stim.DetectorErrorModel("error(0.1) D0 D0\nerror(0.2) D1\n")
    (record,) = stim_fault_catalog.canonical_error_instructions(model)
    assert record.error_ordinal == 1
    assert record.aggregate_detectors == (1,)


def test_an_error_that_flips_an_observable_but_no_detector_is_refused():
    model = stim.DetectorErrorModel("error(0.1) L0\n")
    with pytest.raises(ValueError, match="error 0 is a detectorless"):
        stim_fault_catalog.canonical_error_instructions(model)


def test_only_the_requested_representations_are_built():
    circuit = surface_code_circuit(2)
    catalogs, link = stim_fault_catalog.prepare_fault_catalogs(
        circuit, fault_model_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED
    )
    assert set(catalogs) == {GRAPHLIKE}
    assert link is None
    assert catalogs[GRAPHLIKE].representation is GRAPHLIKE
    assert (
        max(len(detectors) for detectors in catalogs[GRAPHLIKE].detector_sets)
        == 2
    )


def test_the_physical_catalog_keeps_hyperedges():
    circuit = surface_code_circuit(2)
    catalogs, link = stim_fault_catalog.prepare_fault_catalogs(
        circuit, fault_model_contracts.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    assert set(catalogs) == {PHYSICAL}
    assert link is None
    assert (
        max(len(detectors) for detectors in catalogs[PHYSICAL].detector_sets)
        > 2
    )


def test_a_linked_physical_column_is_the_parity_of_its_graphlike_columns():
    circuit = surface_code_circuit(2)
    catalogs, link = stim_fault_catalog.prepare_fault_catalogs(
        circuit, fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED
    )
    graphlike = catalogs[GRAPHLIKE]
    physical = catalogs[PHYSICAL]
    assert link.shape == (
        len(graphlike.detector_sets),
        len(physical.detector_sets),
    )
    first_column = link[:, 0]
    component_sets = [
        graphlike.detector_sets[row] for row in first_column.indices
    ]
    assert xor_of_sets(component_sets) == physical.detector_sets[0]
