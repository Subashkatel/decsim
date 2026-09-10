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


class StandInCircuit:
    """A stand-in circuit whose two Stim models are given as text."""

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
    circuit = StandInCircuit("error(0.1) D0 D1 ^ D1 D2\n", "error(0.1) D0 D1\n")
    with pytest.raises(ValueError, match="disagree on physical faults"):
        stim_fault_catalog.prepare_fault_catalogs(
            circuit, fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED
        )


def test_two_stim_models_that_disagree_on_a_prior_are_refused():
    circuit = StandInCircuit("error(0.1) D0 D1 ^ D1 D2\n", "error(0.2) D0 D2\n")
    with pytest.raises(ValueError, match="disagree on physical faults"):
        stim_fault_catalog.prepare_fault_catalogs(
            circuit, fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED
        )


def test_a_graphlike_catalog_refuses_a_hyperedge():
    circuit = StandInCircuit("error(0.1) D0 D1 D2\n", "error(0.1) D0 D1 D2\n")
    with pytest.raises(
        ValueError, match="graphlike catalog fault 0 is a detector hyperedge"
    ):
        stim_fault_catalog.prepare_fault_catalogs(
            circuit, fault_model_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED
        )


def test_two_physical_errors_with_the_same_identity_merge_as_independent():
    circuit = StandInCircuit("", "error(0.1) D0 D1 D2\nerror(0.2) D2 D1 D0\n")
    catalogs, link = stim_fault_catalog.prepare_fault_catalogs(
        circuit, fault_model_contracts.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    physical = catalogs[PHYSICAL]
    assert link is None
    assert physical.detector_sets == ((0, 1, 2),)
    assert physical.observable_sets == ((),)
    assert physical.priors == (pytest.approx(0.26),)


def test_the_component_columns_of_a_model_have_no_degree_bound():
    """detector_error_model_to_faults reads the model, it does not judge it.

    The graphlike bound belongs to the graphlike catalog, so a reader
    that wants every column of a model as Stim wrote it (the
    complementary-gap decoder builds its own matrices this way, and
    applies the bound itself) gets a three-detector column back.
    """
    model = stim.DetectorErrorModel("error(0.1) D0 D1 D2\n")
    detector_sets, _observable_sets, _priors = (
        stim_fault_catalog.detector_error_model_to_faults(model)
    )
    assert detector_sets == [(0, 1, 2)]


def test_two_faults_on_one_detector_pair_stay_apart_by_their_observables():
    """A column is keyed by its detectors and its observables together.

    Merging them would give one column the sum of two priors and lose
    the fact that only one of them flips the logical observable.
    """
    circuit = StandInCircuit(
        "error(0.1) D0 D1\nerror(0.2) D0 D1 L0\n",
        "error(0.1) D0 D1\nerror(0.2) D0 D1 L0\n",
    )
    catalogs, _link = stim_fault_catalog.prepare_fault_catalogs(
        circuit, fault_model_contracts.GRAPHLIKE_FAULT_MODEL_REQUIRED
    )
    graphlike = catalogs[GRAPHLIKE]
    assert graphlike.detector_sets == ((0, 1), (0, 1))
    assert graphlike.observable_sets == ((), (0,))
    assert graphlike.priors == (0.1, 0.2)


def test_two_decompositions_of_one_identity_stay_two_physical_columns():
    """A physical column is keyed by its identity and its components.

    Two mechanisms can flip the same detectors through different
    graphlike paths; each keeps its own prior, and the link says which
    graphlike columns each is made of. The undecomposed model sees them
    as one error whose prior is their independent merge,
    0.2(1-0.3) + 0.3(1-0.2) = 0.38.
    """
    circuit = StandInCircuit(
        "error(0.2) D1 D2 ^ D2 D3\nerror(0.3) D1 D0 ^ D0 D3\n",
        "error(0.38) D1 D3\n",
    )
    catalogs, link = stim_fault_catalog.prepare_fault_catalogs(
        circuit, fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED
    )
    physical = catalogs[PHYSICAL]
    assert physical.detector_sets == ((1, 3), (1, 3))
    assert physical.priors == (0.2, 0.3)
    assert link.shape[1] == 2
    assert link_rows(link, 0) == [0, 1]
    assert link_rows(link, 1) == [2, 3]


def test_the_two_models_priors_agree_to_a_fixed_absolute_bound():
    """The bound is absolute, not relative: 1e-15, from float rounding.

    The two Stim models are the same circuit read twice, so their priors
    differ only by the order the products were multiplied in; the bound
    is the size of that error and does not scale with the prior.
    """
    perturbed = 0.2 + 5e-16
    assert perturbed != 0.2
    circuit = StandInCircuit(
        "error(0.2) D1 D2 ^ D2 D3\n", f"error({perturbed!r}) D1 D3\n"
    )
    catalogs, _link = stim_fault_catalog.prepare_fault_catalogs(
        circuit, fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED
    )
    assert catalogs[PHYSICAL].priors == (0.2,)


def test_a_prior_disagreement_just_above_the_bound_is_refused():
    """1e-9 is far above float rounding, so it is a model disagreement.

    The accepting test above pins one side of the 1e-15 bound; this
    pins the other, so that widening the bound fails here.
    """
    perturbed = 0.2 + 1e-9
    circuit = StandInCircuit(
        "error(0.2) D1 D2 ^ D2 D3\n", f"error({perturbed!r}) D1 D3\n"
    )
    with pytest.raises(ValueError, match="disagree on physical faults"):
        stim_fault_catalog.prepare_fault_catalogs(
            circuit, fault_model_contracts.LINKED_FAULT_MODELS_REQUIRED
        )
