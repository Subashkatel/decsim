"""The fault catalog reads Stim's error instructions the way Stim means them.

Source: Stim, doc/file_format_dem_detector_error_model.md: an error lists
the detectors and observables it flips, a target listed twice cancels,
and `^` separators suggest a decomposition into graphlike components.
Two errors with the same identity merge as independent errors,
p(1-q) + q(1-p), PyMatching's merge_strategy="independent". The catalog
values below are those of Stim's distance-3, two-round rotated surface
code memory.
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


def link_rows(link, column):
    """The graphlike columns one physical column is made of."""
    chosen = link[:, column]
    rows = chosen.indices.tolist()
    return sorted(rows)


class TwoDisagreeingModels:
    """A stand-in circuit whose two Stim models describe different faults."""

    num_observables = 0

    def __init__(self, decomposed, undecomposed):
        self.decomposed = stim.DetectorErrorModel(decomposed)
        self.undecomposed = stim.DetectorErrorModel(undecomposed)

    def detector_error_model(self, *, decompose_errors):
        if decompose_errors:
            return self.decomposed
        return self.undecomposed


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


def test_a_component_that_flips_an_observable_but_no_detector_is_refused():
    model = stim.DetectorErrorModel("error(0.1) D0 ^ L0\n")
    with pytest.raises(
        ValueError, match="error 0 component 1 is a detectorless"
    ):
        stim_fault_catalog.canonical_error_instructions(model)


def test_only_the_requested_representations_are_built():
    circuit = surface_code_circuit(2)
    catalogs, link = stim_fault_catalog.prepare_fault_catalogs(
        circuit, fault_model_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED
    )
    assert set(catalogs) == {GRAPHLIKE}
    assert link is None
    graphlike = catalogs[GRAPHLIKE]
    assert graphlike.representation is GRAPHLIKE
    assert len(graphlike.detector_sets) == 44
    assert graphlike.detector_sets[:4] == ((0,), (0, 1), (0, 8), (1, 2))


def test_the_physical_catalog_keeps_hyperedges():
    circuit = surface_code_circuit(2)
    catalogs, link = stim_fault_catalog.prepare_fault_catalogs(
        circuit, fault_model_contracts.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    assert set(catalogs) == {PHYSICAL}
    assert link is None
    physical = catalogs[PHYSICAL]
    assert len(physical.detector_sets) == 107
    assert physical.detector_sets[4] == (1, 4, 5)


def test_a_linked_physical_column_is_the_parity_of_its_graphlike_columns():
    circuit = surface_code_circuit(2)
    catalogs, link = stim_fault_catalog.prepare_fault_catalogs(
        circuit, fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED
    )
    graphlike = catalogs[GRAPHLIKE]
    physical = catalogs[PHYSICAL]
    assert link.shape == (44, 134)
    assert physical.detector_sets[0] == (0,)
    assert link_rows(link, 0) == [0]
    assert graphlike.detector_sets[0] == (0,)
    assert physical.detector_sets[5] == (1, 4, 5)
    assert physical.observable_sets[5] == ()
    assert link_rows(link, 5) == [4, 5]
    assert graphlike.detector_sets[4] == (1, 5)
    assert graphlike.detector_sets[5] == (4,)
    assert physical.detector_sets[32] == (1, 4)
    assert physical.observable_sets[32] == (0,)
    assert link_rows(link, 32) == [5, 7]
    assert graphlike.detector_sets[7] == (1,)
    assert graphlike.observable_sets[7] == (0,)
    assert graphlike.observable_sets[5] == ()


def test_two_stim_models_that_disagree_on_a_physical_fault_are_refused():
    circuit = TwoDisagreeingModels(
        "error(0.1) D0 D1 ^ D1 D2\n", "error(0.1) D0 D1\n"
    )
    with pytest.raises(ValueError, match="disagree on physical faults"):
        stim_fault_catalog.prepare_fault_catalogs(
            circuit, fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED
        )


def test_two_stim_models_that_disagree_on_a_prior_are_refused():
    circuit = TwoDisagreeingModels(
        "error(0.1) D0 D1 ^ D1 D2\n", "error(0.2) D0 D2\n"
    )
    with pytest.raises(ValueError, match="disagree on physical faults"):
        stim_fault_catalog.prepare_fault_catalogs(
            circuit, fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED
        )
